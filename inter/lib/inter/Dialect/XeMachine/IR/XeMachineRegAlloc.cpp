// Allocate XeMachine GRFs with transactional retries and ordered relief.

#include "inter/Dialect/XeMachine/IR/XeMachine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>

#define GET_OP_CLASSES
#include "inter/Dialect/XeMachine/IR/XeMachineTransformOps.cpp.inc"

using namespace mlir;
using namespace inter::xemachine;

namespace {

constexpr StringLiteral kStateAttr = "xemachine.regalloc_state";
constexpr StringLiteral kIterationAttr = "xemachine.regalloc_iterations";
constexpr StringLiteral kRematerializedAttr = "xemachine.rematerialized";
constexpr StringLiteral kSpilledAttr = "xemachine.spilled";
constexpr StringLiteral kScratchSetupAttr = "xemachine.scratch_setup";
constexpr StringLiteral kLoopIterationAttr =
    "xemachine.regalloc_loop_iteration";
constexpr StringLiteral kStageAttr = "stage";
constexpr StringLiteral kBuildStage = "alias-state";
constexpr StringLiteral kSuccessStage = "linear-scan-success";
constexpr StringLiteral kFailureStage = "linear-scan-failure";

struct AliasEdge {
  unsigned target;
  int64_t offset;
};

struct AllocationComponent {
  SmallVector<unsigned> nodes;
  int64_t minOffset = 0;
  int64_t maxOffset = 0;
  int64_t start = std::numeric_limits<int64_t>::max();
  int64_t end = 0;
  std::optional<int64_t> fixedBase;
  int64_t assignment = -1;
  bool allowFixedOverlap = false;

  int64_t widthGRFs() const {
    return llvm::divideCeil(maxOffset - minOffset, int64_t{16});
  }
};

struct AllocationState {
  SmallVector<Value> values;
  DenseMap<Value, unsigned> valueToNode;
  SmallVector<SmallVector<AliasEdge>> graph;
  SmallVector<int64_t> offsets;
  SmallVector<unsigned> nodeToComponent;
  SmallVector<AllocationComponent> components;
  DenseMap<Operation *, int64_t> positions;
};

static bool isRegister(Value value) { return isa<RegType>(value.getType()); }

static std::optional<uint64_t> getElementBytes(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type))
    return llvm::divideCeil(integer.getWidth(), 8u);
  if (type.isF32())
    return 4;
  return std::nullopt;
}

static int64_t getIntegerAttr(Operation *operation, StringRef name,
                              int64_t fallback) {
  if (IntegerAttr attr = operation->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return fallback;
}

static LogicalResult validateAluFootprint(Operation *operation) {
  if (!isa<MovOp, AddOp, SubOp, ShlOp, ShrOp, AndOp, OrOp, Add3Op, MulOp,
           CmpOp>(operation))
    return success();
  TypeAttr elementTypeAttr = operation->getAttrOfType<TypeAttr>("elemType");
  if (!elementTypeAttr)
    return operation->emitError("register operation requires elemType");
  int64_t executionSize = getIntegerAttr(operation, "execSize", 16);
  if (executionSize <= 0)
    return operation->emitError("execution size must be positive");

  if (operation->getNumResults() != 0) {
    auto destinationType = dyn_cast<RegType>(operation->getResult(0).getType());
    if (destinationType && destinationType.getWidthDwords() != 0) {
      std::optional<uint64_t> bytes =
          getElementBytes(elementTypeAttr.getValue());
      if (!bytes)
        return operation->emitError("unsupported destination element type");
      int64_t sub = getIntegerAttr(operation, "dstSub", 0);
      DstRegionAttr region =
          operation->getAttrOfType<DstRegionAttr>("dstRegion");
      int64_t stride = region ? region.getHstride() : 1;
      int64_t last = sub + (executionSize - 1) * stride;
      if (sub < 0 || stride < 0 ||
          static_cast<uint64_t>(last + 1) * *bytes >
              destinationType.getWidthDwords() * 4)
        return operation->emitError(
            "destination region exceeds declared register storage");
    }
  }

  constexpr std::array<StringLiteral, 3> regionNames = {
      "src0Region", "src1Region", "src2Region"};
  constexpr std::array<StringLiteral, 3> subNames = {"src0Sub", "src1Sub",
                                                     "src2Sub"};
  constexpr std::array<StringLiteral, 3> typeNames = {"src0Type", "src1Type",
                                                      "src2Type"};
  for (auto [index, operand] : llvm::enumerate(operation->getOperands())) {
    auto registerType = dyn_cast<RegType>(operand.getType());
    if (!registerType)
      continue;
    Type sourceType = elementTypeAttr.getValue();
    if (TypeAttr attr = operation->getAttrOfType<TypeAttr>(typeNames[index]))
      sourceType = attr.getValue();
    std::optional<uint64_t> bytes = getElementBytes(sourceType);
    if (!bytes)
      return operation->emitError("unsupported source element type");
    int64_t sub = getIntegerAttr(operation, subNames[index], 0);
    RegionAttr region =
        operation->getAttrOfType<RegionAttr>(regionNames[index]);
    int64_t vertical = region ? region.getVstride() : 1;
    int64_t width = region ? region.getWidth() : 1;
    int64_t horizontal = region ? region.getHstride() : 0;
    if (sub < 0 || vertical < 0 || width <= 0 || horizontal < 0)
      return operation->emitError("invalid source register region");
    int64_t lane = executionSize - 1;
    int64_t last = sub + lane / width * vertical + lane % width * horizontal;
    if (static_cast<uint64_t>(last + 1) * *bytes >
        registerType.getWidthDwords() * 4)
      return operation->emitError(
          "source region exceeds declared register storage");
  }
  return success();
}

static void addAlias(AllocationState &state, Value storage, Value alias,
                     int64_t offset) {
  if (!isRegister(storage) || !isRegister(alias))
    return;
  unsigned storageNode = state.valueToNode.lookup(storage);
  unsigned aliasNode = state.valueToNode.lookup(alias);
  state.graph[storageNode].push_back({aliasNode, offset});
  state.graph[aliasNode].push_back({storageNode, -offset});
}

static void addRegionAliases(AllocationState &state, Operation *operation) {
  auto addYields = [&](Region &region) {
    if (region.empty())
      return;
    auto yield = dyn_cast<YieldOp>(region.front().getTerminator());
    if (!yield)
      return;
    for (auto [result, yielded] :
         llvm::zip_equal(operation->getResults(), yield.getValues()))
      addAlias(state, result, yielded, 0);
  };

  if (auto ifOp = dyn_cast<ExecIfOp>(operation)) {
    addYields(ifOp.getThenRegion());
    addYields(ifOp.getElseRegion());
    return;
  }
  if (auto ifOp = dyn_cast<UniformIfOp>(operation)) {
    addYields(ifOp.getThenRegion());
    addYields(ifOp.getElseRegion());
    return;
  }
  auto loop = dyn_cast<UniformLoopOp>(operation);
  if (!loop || loop.getBody().empty())
    return;
  Block &body = loop.getBody().front();
  auto terminator = dyn_cast<ContinueIfOp>(body.getTerminator());
  if (!terminator)
    return;
  for (auto [init, argument, carried, result] :
       llvm::zip_equal(loop.getInits(), body.getArguments(),
                       terminator.getCarried(), loop.getResults())) {
    addAlias(state, result, init, 0);
    addAlias(state, result, argument, 0);
    addAlias(state, result, carried, 0);
  }
}

static LogicalResult assignComponentOffsets(func::FuncOp function,
                                            AllocationState &state) {
  state.offsets.assign(state.values.size(), 0);
  state.nodeToComponent.assign(state.values.size(),
                               std::numeric_limits<unsigned>::max());
  SmallVector<unsigned> worklist;

  for (unsigned root : llvm::seq<unsigned>(0, state.values.size())) {
    if (state.nodeToComponent[root] != std::numeric_limits<unsigned>::max())
      continue;
    unsigned componentIndex = state.components.size();
    state.components.emplace_back();
    state.nodeToComponent[root] = componentIndex;
    worklist.push_back(root);
    while (!worklist.empty()) {
      unsigned node = worklist.pop_back_val();
      state.components[componentIndex].nodes.push_back(node);
      for (AliasEdge edge : state.graph[node]) {
        int64_t expected = state.offsets[node] + edge.offset;
        if (state.nodeToComponent[edge.target] ==
            std::numeric_limits<unsigned>::max()) {
          state.nodeToComponent[edge.target] = componentIndex;
          state.offsets[edge.target] = expected;
          worklist.push_back(edge.target);
          continue;
        }
        if (state.nodeToComponent[edge.target] != componentIndex ||
            state.offsets[edge.target] != expected)
          return function.emitError("inconsistent register-storage aliases");
      }
    }
  }
  return success();
}

static int64_t getDefinitionPosition(Value value,
                                     const AllocationState &state) {
  if (Operation *definition = value.getDefiningOp())
    return state.positions.lookup(definition);
  Block *block = cast<BlockArgument>(value).getOwner();
  Operation *parent = block->getParentOp();
  return parent ? state.positions.lookup(parent) : 0;
}

static LogicalResult finalizeComponents(func::FuncOp function,
                                        AllocationState &state) {
  for (AllocationComponent &component : state.components) {
    component.minOffset = std::numeric_limits<int64_t>::max();
    component.maxOffset = std::numeric_limits<int64_t>::min();
    for (unsigned node : component.nodes) {
      Value value = state.values[node];
      RegType type = cast<RegType>(value.getType());
      if (Operation *definition = value.getDefiningOp())
        component.allowFixedOverlap |=
            definition->hasAttr(kAllowFixedOverlapAttrName);
      int64_t offset = state.offsets[node];
      component.minOffset = std::min(component.minOffset, offset);
      component.maxOffset =
          std::max(component.maxOffset, offset + type.getWidthDwords());
      component.start =
          std::min(component.start, getDefinitionPosition(value, state));
      component.end = std::max(component.end, component.start);
      for (OpOperand &use : value.getUses())
        component.end =
            std::max(component.end, state.positions.lookup(use.getOwner()));
    }

    for (unsigned node : component.nodes) {
      Value value = state.values[node];
      RegType type = cast<RegType>(value.getType());
      int64_t normalizedOffset = state.offsets[node] - component.minOffset;
      if (normalizedOffset % 16 != 0)
        return function.emitError(
            "register-storage alias is not GRF-aligned after selection");
      if (type.getBaseGRF() < 0)
        continue;
      int64_t candidate = type.getBaseGRF() - normalizedOffset / 16;
      if (component.fixedBase && *component.fixedBase != candidate)
        return function.emitError("conflicting physical register aliases");
      component.fixedBase = candidate;
    }
  }

  DenseMap<Value, int64_t> tokenEnds;
  SmallPtrSet<Value, 8> visiting;
  std::function<int64_t(Value)> getTokenEnd = [&](Value token) -> int64_t {
    auto cached = tokenEnds.find(token);
    if (cached != tokenEnds.end())
      return cached->second;
    if (!visiting.insert(token).second)
      return state.positions.size();
    int64_t end = 0;
    for (OpOperand &use : token.getUses()) {
      Operation *owner = use.getOwner();
      end = std::max(end, state.positions.lookup(owner));
      if (!owner->hasTrait<OpTrait::xemachine::NoMachineInst>())
        continue;
      bool forwarded = false;
      for (Value result : owner->getResults()) {
        if (!isa<MemTokenType>(result.getType()))
          continue;
        end = std::max(end, getTokenEnd(result));
        forwarded = true;
      }
      if (!forwarded && isa<YieldOp, ContinueIfOp>(owner))
        end = std::max<int64_t>(end, state.positions.size());
    }
    visiting.erase(token);
    tokenEnds[token] = end;
    return end;
  };

  function.walk([&](Operation *operation) {
    if (!isa<SendOp, LoadA64Op, StoreA64Op, LoadSLMOp, StoreSLMOp,
             AtomicIAddA64Op, LoadBlockA32Op, FenceSLMOp, BarrierSignalOp,
             EotOp>(operation))
      return;
    int64_t completion = state.positions.lookup(operation);
    for (Value result : operation->getResults()) {
      if (isa<MemTokenType>(result.getType()))
        completion = std::max(completion, getTokenEnd(result));
    }
    for (Value operand : operation->getOperands()) {
      if (!isRegister(operand))
        continue;
      unsigned node = state.valueToNode.lookup(operand);
      state.components[state.nodeToComponent[node]].end = std::max(
          state.components[state.nodeToComponent[node]].end, completion);
    }
  });

  function.walk([&](UniformLoopOp loop) {
    int64_t loopEnd = state.positions.lookup(loop);
    loop.getBody().walk([&](Operation *operation) {
      loopEnd = std::max(loopEnd, state.positions.lookup(operation));
    });
    loop.getBody().walk([&](Operation *operation) {
      for (Value operand : operation->getOperands()) {
        if (!isRegister(operand))
          continue;
        Operation *definition = operand.getDefiningOp();
        if (definition && loop->isProperAncestor(definition))
          continue;
        if (BlockArgument argument = dyn_cast<BlockArgument>(operand)) {
          Operation *parent = argument.getOwner()->getParentOp();
          if (parent && loop->isProperAncestor(parent))
            continue;
        }
        unsigned node = state.valueToNode.lookup(operand);
        AllocationComponent &component =
            state.components[state.nodeToComponent[node]];
        component.end = std::max(component.end, loopEnd);
      }
    });
  });
  return success();
}

static LogicalResult buildState(func::FuncOp function, AllocationState &state) {
  if (function
          .walk([&](Operation *operation) {
            return failed(validateAluFootprint(operation))
                       ? WalkResult::interrupt()
                       : WalkResult::advance();
          })
          .wasInterrupted())
    return failure();
  int64_t nextPosition = 0;
  function.walk([&](Operation *operation) {
    state.positions[operation] = nextPosition++;
    for (Value result : operation->getResults()) {
      if (!isRegister(result))
        continue;
      state.valueToNode[result] = state.values.size();
      state.values.push_back(result);
    }
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments()) {
          if (!isRegister(argument))
            continue;
          state.valueToNode[argument] = state.values.size();
          state.values.push_back(argument);
        }
  });
  state.graph.resize(state.values.size());

  DenseMap<int64_t, Value> architecturalRegisters;
  function.walk([&](Operation *operation) {
    if (auto aliases = dyn_cast<RegisterStorageAliasOpInterface>(operation)) {
      SmallVector<RegisterStorageAlias> relations;
      aliases.getRegisterStorageAliases(relations);
      for (const RegisterStorageAlias &relation : relations)
        addAlias(state, relation.storage, relation.alias, relation.offset);
    }
    if (auto archreg = dyn_cast<ArchRegOp>(operation)) {
      Value previous = architecturalRegisters.lookup(archreg.getIndex());
      if (previous)
        addAlias(state, previous, archreg.getResult(), 0);
      else
        architecturalRegisters[archreg.getIndex()] = archreg.getResult();
    }
    addRegionAliases(state, operation);
  });

  if (failed(assignComponentOffsets(function, state)))
    return failure();
  return finalizeComponents(function, state);
}

static bool intervalsOverlap(const AllocationComponent &lhs,
                             const AllocationComponent &rhs) {
  return lhs.start <= rhs.end && rhs.start <= lhs.end;
}

static bool registersOverlap(int64_t lhsBase, int64_t lhsWidth, int64_t rhsBase,
                             int64_t rhsWidth) {
  return lhsBase < rhsBase + rhsWidth && rhsBase < lhsBase + lhsWidth;
}

struct AllocationFailure {
  unsigned component;
  int64_t position;
};

static DictionaryAttr packState(Builder &builder, unsigned iteration,
                                StringRef stage, const AllocationState &state,
                                std::optional<AllocationFailure> failure = {}) {
  SmallVector<int64_t> intervals;
  for (const AllocationComponent &component : state.components) {
    intervals.push_back(component.start);
    intervals.push_back(component.end);
    intervals.push_back(component.minOffset);
    intervals.push_back(component.maxOffset);
    intervals.push_back(component.fixedBase.value_or(-1));
    intervals.push_back(component.assignment);
  }
  SmallVector<int64_t> aliases;
  for (unsigned node : llvm::seq<unsigned>(0, state.values.size())) {
    aliases.push_back(node);
    aliases.push_back(state.nodeToComponent[node]);
    aliases.push_back(state.offsets[node]);
  }
  SmallVector<NamedAttribute> attributes = {
      builder.getNamedAttr("iteration", builder.getI32IntegerAttr(iteration)),
      builder.getNamedAttr(kStageAttr, builder.getStringAttr(stage)),
      builder.getNamedAttr(
          "component_intervals",
          DenseI64ArrayAttr::get(builder.getContext(), intervals)),
      builder.getNamedAttr(
          "value_aliases",
          DenseI64ArrayAttr::get(builder.getContext(), aliases))};
  if (failure) {
    attributes.push_back(builder.getNamedAttr(
        "failed_component", builder.getI32IntegerAttr(failure->component)));
    attributes.push_back(builder.getNamedAttr(
        "failure_position", builder.getI64IntegerAttr(failure->position)));
  }
  return builder.getDictionaryAttr(attributes);
}

static std::optional<AllocationFailure> tryAllocate(AllocationState &state,
                                                    unsigned grfLimit,
                                                    unsigned reservedGrfCount) {
  SmallVector<unsigned> order;
  llvm::append_range(order, llvm::seq<unsigned>(0, state.components.size()));
  llvm::sort(order, [&](unsigned lhs, unsigned rhs) {
    const AllocationComponent &left = state.components[lhs];
    const AllocationComponent &right = state.components[rhs];
    if (left.start != right.start)
      return left.start < right.start;
    return left.fixedBase.has_value() > right.fixedBase.has_value();
  });

  SmallVector<unsigned> active;
  for (unsigned index : order) {
    AllocationComponent &component = state.components[index];
    llvm::erase_if(active, [&](unsigned other) {
      return state.components[other].end < component.start;
    });

    auto isAvailable = [&](int64_t base) {
      if (base < 0 || base + component.widthGRFs() > grfLimit)
        return false;
      for (unsigned otherIndex : active) {
        const AllocationComponent &other = state.components[otherIndex];
        if (component.fixedBase && other.fixedBase &&
            (component.allowFixedOverlap || other.allowFixedOverlap))
          continue;
        if (registersOverlap(base, component.widthGRFs(), other.assignment,
                             other.widthGRFs()))
          return false;
      }
      for (const AllocationComponent &fixed : state.components) {
        if (!fixed.fixedBase || &fixed == &component ||
            !intervalsOverlap(component, fixed))
          continue;
        if (component.fixedBase &&
            (component.allowFixedOverlap || fixed.allowFixedOverlap))
          continue;
        if (registersOverlap(base, component.widthGRFs(), *fixed.fixedBase,
                             fixed.widthGRFs()))
          return false;
      }
      return true;
    };

    if (component.fixedBase) {
      if (!isAvailable(*component.fixedBase))
        return AllocationFailure{index, component.start};
      component.assignment = *component.fixedBase;
    } else {
      for (int64_t base = reservedGrfCount;
           base + component.widthGRFs() <= grfLimit; ++base) {
        if (!isAvailable(base))
          continue;
        component.assignment = base;
        break;
      }
      if (component.assignment < 0)
        return AllocationFailure{index, component.start};
    }
    active.push_back(index);
  }
  return std::nullopt;
}

static bool isRematerializable(Operation *operation) {
  if (!operation || operation->hasAttr(kRematerializedAttr) ||
      operation->getNumResults() != 1)
    return false;
  if (llvm::any_of(operation->getOperandTypes(),
                   [](Type type) { return isa<ARFType>(type); }))
    return false;
  return isa<MovOp, AddOp, SubOp, ShlOp, AndOp, OrOp, Add3Op>(operation);
}

static bool rematerialize(AllocationState &state,
                          const AllocationFailure &failure) {
  AllocationComponent *candidate = nullptr;
  for (AllocationComponent &component : state.components) {
    if (component.fixedBase || component.nodes.size() != 1 ||
        component.start >= failure.position || component.end < failure.position)
      continue;
    Value value = state.values[component.nodes.front()];
    if (!isRematerializable(value.getDefiningOp()))
      continue;
    if (!candidate || candidate->end < component.end)
      candidate = &component;
  }
  if (!candidate)
    return false;

  Value value = state.values[candidate->nodes.front()];
  Operation *definition = value.getDefiningOp();
  SmallVector<OpOperand *> uses;
  for (OpOperand &use : value.getUses()) {
    if (state.positions.lookup(use.getOwner()) >= failure.position)
      uses.push_back(&use);
  }
  if (uses.empty())
    return false;

  definition->setAttr(kRematerializedAttr,
                      UnitAttr::get(definition->getContext()));
  DenseMap<Operation *, Value> clones;
  for (OpOperand *use : uses) {
    Operation *owner = use->getOwner();
    Value replacement = clones.lookup(owner);
    if (!replacement) {
      OpBuilder builder(owner);
      Operation *clone = builder.clone(*definition);
      clone->setAttr(kRematerializedAttr,
                     UnitAttr::get(definition->getContext()));
      replacement = clone->getResult(0);
      clones[owner] = replacement;
    }
    use->set(replacement);
  }
  return true;
}

static Value getScratchSurfaceOffset(func::FuncOp function) {
  Value surfaceOffset;
  function.walk([&](ShrOp shift) {
    if (shift->hasAttr(kScratchSetupAttr))
      surfaceOffset = shift.getDst();
  });
  if (surfaceOffset)
    return surfaceOffset;

  MLIRContext *context = function.getContext();
  Block &entry = function.getBody().front();
  OpBuilder builder = OpBuilder::atBlockBegin(&entry);
  Location location = function.getLoc();
  Type i32 = builder.getI32Type();
  RegionAttr uniform = RegionAttr::get(context, 0, 1, 0);
  DstRegionAttr canonical = DstRegionAttr::get(context, 1);
  Value r0 = ArchRegOp::create(builder, location, RegType::get(context, 16, 0),
                               builder.getI32IntegerAttr(0))
                 .getResult();
  Value four = ImmOp::create(builder, location, ImmType::get(context), 4, i32)
                   .getResult();
  Value mask =
      ImmOp::create(builder, location, ImmType::get(context), 0xFFFFFC00, i32)
          .getResult();
  AndOp maskSetup = AndOp::create(
      builder, location, ARFType::get(context, ARFFile::a0, 16, 0), i32,
      /*execSize=*/1, canonical, uniform, RegionAttr(),
      builder.getI32IntegerAttr(2), builder.getI32IntegerAttr(5), IntegerAttr(),
      TypeAttr(), TypeAttr(), /*noMask=*/true, /*maskOffset=*/0, r0, mask);
  maskSetup->setAttr(kScratchSetupAttr, builder.getUnitAttr());
  ShrOp setup = ShrOp::create(
      builder, location, ARFType::get(context, ARFFile::a0, 16, 0), i32,
      /*execSize=*/1, canonical, uniform, RegionAttr(),
      builder.getI32IntegerAttr(2), builder.getI32IntegerAttr(2), IntegerAttr(),
      TypeAttr(), TypeAttr(), /*noMask=*/true, /*maskOffset=*/0,
      maskSetup.getResult(), four);
  setup->setAttr(kScratchSetupAttr, builder.getUnitAttr());
  return setup.getDst();
}

static Value createScratchAddress(OpBuilder &builder, Location location,
                                  int64_t offset) {
  MLIRContext *context = builder.getContext();
  Type i32 = builder.getI32Type();
  Value immediate =
      ImmOp::create(builder, location, ImmType::get(context), offset, i32)
          .getResult();
  return MovOp::create(builder, location, RegType::get(context, 16, -1), i32,
                       /*execSize=*/1, DstRegionAttr::get(context, 1),
                       RegionAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                       /*noMask=*/true, /*maskOffset=*/0, immediate)
      .getResult();
}

static uint32_t getScratchDescriptor(uint32_t widthDwords, bool load) {
  uint32_t vectorEncoding = widthDwords == 16 ? 5 : 6;
  uint32_t descriptor = (1u << 30) | (1u << 25) | (2u << 7) | (2u << 9) |
                        (1u << 15) | (vectorEncoding << 12);
  if (load)
    descriptor |= llvm::divideCeil(widthDwords, 16u) << 20;
  else
    descriptor |= 4;
  return descriptor;
}

static SendOp createScratchSend(OpBuilder &builder, Location location,
                                Type destinationType, Value address, Value data,
                                Value surfaceOffset, Value dependency,
                                uint32_t descriptor) {
  MLIRContext *context = builder.getContext();
  return SendOp::create(builder, location, destinationType,
                        MemTokenType::get(context), SendFn::ugm, /*sfid=*/0,
                        descriptor, /*exdesc=*/0, /*execSize=*/1,
                        /*noMask=*/true, /*eot=*/false, address, data,
                        surfaceOffset, dependency, IntegerAttr());
}

static bool spillToScratch(func::FuncOp function, AllocationState &state,
                           const AllocationFailure &failure,
                           int64_t &nextScratchOffset) {
  bool conflictingAddressRegister = false;
  function.walk([&](Operation *operation) {
    if (operation->hasAttr(kScratchSetupAttr))
      return;
    for (Type type : operation->getResultTypes()) {
      auto arf = dyn_cast<ARFType>(type);
      conflictingAddressRegister |= arf && arf.getFile() == ARFFile::a0;
    }
  });
  if (conflictingAddressRegister)
    return false;

  AllocationComponent *candidate = nullptr;
  for (AllocationComponent &component : state.components) {
    if (component.fixedBase || component.nodes.size() != 1 ||
        component.start >= failure.position || component.end < failure.position)
      continue;
    Value value = state.values[component.nodes.front()];
    Operation *definition = value.getDefiningOp();
    RegType type = cast<RegType>(value.getType());
    if (!definition || definition->hasAttr(kSpilledAttr) ||
        isa<SendOp>(definition) ||
        (type.getWidthDwords() != 16 && type.getWidthDwords() != 32))
      continue;
    if (!candidate || candidate->end < component.end)
      candidate = &component;
  }
  if (!candidate)
    return false;

  Value value = state.values[candidate->nodes.front()];
  Operation *definition = value.getDefiningOp();
  RegType type = cast<RegType>(value.getType());
  SmallVector<OpOperand *> uses;
  for (OpOperand &use : value.getUses()) {
    if (state.positions.lookup(use.getOwner()) >= failure.position)
      uses.push_back(&use);
  }
  if (uses.empty())
    return false;

  int64_t slot = llvm::alignTo(nextScratchOffset, int64_t{64});
  nextScratchOffset = slot + type.getWidthDwords() * 4;
  Value surfaceOffset = getScratchSurfaceOffset(function);

  OpBuilder storeBuilder(definition);
  storeBuilder.setInsertionPointAfter(definition);
  Value storeAddress =
      createScratchAddress(storeBuilder, definition->getLoc(), slot);
  SendOp store = createScratchSend(
      storeBuilder, definition->getLoc(),
      RegType::get(function.getContext(), 0, -1), storeAddress, value,
      surfaceOffset, Value(),
      getScratchDescriptor(type.getWidthDwords(), /*load=*/false));
  SyncOp storeDrain =
      SyncOp::create(storeBuilder, definition->getLoc(),
                     MemTokenType::get(function.getContext()),
                     SyncKindAttr::get(function.getContext(), SyncKind::allrd),
                     store.getToken());
  definition->setAttr(kSpilledAttr, UnitAttr::get(function.getContext()));

  DenseMap<Operation *, Value> reloads;
  for (OpOperand *use : uses) {
    Operation *owner = use->getOwner();
    Value reload = reloads.lookup(owner);
    if (!reload) {
      OpBuilder loadBuilder(owner);
      Value loadAddress =
          createScratchAddress(loadBuilder, owner->getLoc(), slot);
      SendOp load = createScratchSend(
          loadBuilder, owner->getLoc(),
          RegType::get(function.getContext(), type.getWidthDwords(), -1),
          loadAddress, Value(), surfaceOffset, storeDrain.getToken(),
          getScratchDescriptor(type.getWidthDwords(), /*load=*/true));
      reload = load.getDst();
      reloads[owner] = reload;
    }
    use->set(reload);
  }
  function->setAttr(
      kScratchSizeAttrName,
      IntegerAttr::get(IntegerType::get(function.getContext(), 64),
                       nextScratchOffset));
  return true;
}

static void commitAllocation(AllocationState &state) {
  for (auto [node, value] : llvm::enumerate(state.values)) {
    const AllocationComponent &component =
        state.components[state.nodeToComponent[node]];
    int64_t offset = state.offsets[node] - component.minOffset;
    RegType oldType = cast<RegType>(value.getType());
    value.setType(RegType::get(value.getContext(), oldType.getWidthDwords(),
                               component.assignment + offset / 16));
  }
}

struct RegAllocConfig {
  unsigned grfLimit;
  unsigned reservedGrfCount;
  int64_t scratchSize;
};

static FailureOr<RegAllocConfig>
validateRegAllocFunction(func::FuncOp function) {
  FunctionType functionType = function.getFunctionType();
  auto isRegisterType = [](Type type) { return isa<RegType, ARFType>(type); };
  if (llvm::any_of(functionType.getInputs(), isRegisterType) ||
      llvm::any_of(functionType.getResults(), isRegisterType))
    return function.emitError("register allocation does not support "
                              "register-valued signatures"),
           failure();

  if (function
          .walk([&](Operation *operation) {
            SmallVector<Type> types(operation->getResultTypes());
            for (Region &region : operation->getRegions())
              for (Block &block : region)
                llvm::append_range(types, block.getArgumentTypes());
            for (Type type : types) {
              auto arf = dyn_cast<ARFType>(type);
              if (!arf || arf.getIndex() >= 0)
                continue;
              operation->emitError(
                  "GRF allocation requires ARF values to be allocated first");
              return WalkResult::interrupt();
            }
            return WalkResult::advance();
          })
          .wasInterrupted())
    return failure();

  IntegerAttr grfCount =
      function->getAttrOfType<IntegerAttr>(kGrfCountAttrName);
  if (!grfCount || grfCount.getInt() <= 0)
    return function.emitError("register allocation requires a positive ")
               << kGrfCountAttrName << " function attribute",
           failure();
  IntegerAttr reserved =
      function->getAttrOfType<IntegerAttr>(kReservedGrfCountAttrName);
  if (!reserved || reserved.getInt() < 0 ||
      reserved.getInt() > grfCount.getInt())
    return function.emitError("register allocation requires a valid ")
               << kReservedGrfCountAttrName << " function attribute",
           failure();
  IntegerAttr scratch =
      function->getAttrOfType<IntegerAttr>(kScratchSizeAttrName);
  if (scratch && scratch.getInt() < 0)
    return function.emitError("register allocation requires a nonnegative ")
               << kScratchSizeAttrName << " function attribute",
           failure();
  return RegAllocConfig{static_cast<unsigned>(grfCount.getInt()),
                        static_cast<unsigned>(reserved.getInt()),
                        scratch ? scratch.getInt() : 0};
}

static void collectFunctions(Operation *target,
                             SmallVectorImpl<func::FuncOp> &functions) {
  if (auto function = dyn_cast<func::FuncOp>(target)) {
    functions.push_back(function);
    return;
  }
  target->walk([&](func::FuncOp function) { functions.push_back(function); });
}

static unsigned getLoopIteration(func::FuncOp function) {
  IntegerAttr attr = function->getAttrOfType<IntegerAttr>(kLoopIterationAttr);
  return attr ? static_cast<unsigned>(attr.getInt()) : 1;
}

static LogicalResult buildTransformState(Operation *target) {
  SmallVector<func::FuncOp> functions;
  collectFunctions(target, functions);
  for (func::FuncOp function : functions) {
    if (failed(validateRegAllocFunction(function)))
      return failure();
    AllocationState state;
    if (failed(buildState(function, state)))
      return failure();
    Builder builder(function.getContext());
    function->setAttr(kStateAttr, packState(builder, getLoopIteration(function),
                                            kBuildStage, state));
  }
  return success();
}

static LogicalResult runTransformLinearScan(Operation *target) {
  SmallVector<func::FuncOp> functions;
  collectFunctions(target, functions);
  for (func::FuncOp function : functions) {
    FailureOr<RegAllocConfig> config = validateRegAllocFunction(function);
    if (failed(config))
      return failure();
    DictionaryAttr packed = function->getAttrOfType<DictionaryAttr>(kStateAttr);
    if (!packed ||
        packed.getAs<StringAttr>(kStageAttr).getValue() != kBuildStage)
      return function.emitError("linear scan requires alias-state input"),
             failure();
    AllocationState state;
    if (failed(buildState(function, state)))
      return failure();
    std::optional<AllocationFailure> allocationFailure =
        tryAllocate(state, config->grfLimit, config->reservedGrfCount);
    Builder builder(function.getContext());
    if (!allocationFailure) {
      commitAllocation(state);
      function->setAttr(
          kStateAttr,
          packState(builder, getLoopIteration(function), kSuccessStage, state));
      continue;
    }
    function->setAttr(kStateAttr,
                      packState(builder, getLoopIteration(function),
                                kFailureStage, state, allocationFailure));
  }
  return success();
}

static FailureOr<std::optional<AllocationFailure>>
readTransformFailure(func::FuncOp function) {
  DictionaryAttr packed = function->getAttrOfType<DictionaryAttr>(kStateAttr);
  if (!packed)
    return std::optional<AllocationFailure>();
  StringAttr stage = packed.getAs<StringAttr>(kStageAttr);
  if (!stage)
    return function.emitError("regalloc state is missing its stage"), failure();
  if (stage.getValue() != kFailureStage)
    return std::optional<AllocationFailure>();
  IntegerAttr component = packed.getAs<IntegerAttr>("failed_component");
  IntegerAttr position = packed.getAs<IntegerAttr>("failure_position");
  if (!component || !position)
    return function.emitError("regalloc failure state is incomplete"),
           failure();
  return std::optional<AllocationFailure>(AllocationFailure{
      static_cast<unsigned>(component.getInt()), position.getInt()});
}

static LogicalResult runTransformRematRelief(Operation *target) {
  SmallVector<func::FuncOp> functions;
  collectFunctions(target, functions);
  for (func::FuncOp function : functions) {
    FailureOr<std::optional<AllocationFailure>> failureRecord =
        readTransformFailure(function);
    if (failed(failureRecord))
      return failure();
    if (!*failureRecord)
      continue;
    AllocationState state;
    if (failed(buildState(function, state)))
      return failure();
    if (rematerialize(state, **failureRecord))
      function->removeAttr(kStateAttr);
  }
  return success();
}

static LogicalResult runTransformScratchRelief(Operation *target) {
  SmallVector<func::FuncOp> functions;
  collectFunctions(target, functions);
  for (func::FuncOp function : functions) {
    FailureOr<std::optional<AllocationFailure>> failureRecord =
        readTransformFailure(function);
    if (failed(failureRecord))
      return failure();
    if (!*failureRecord)
      continue;
    FailureOr<RegAllocConfig> config = validateRegAllocFunction(function);
    if (failed(config))
      return failure();
    AllocationState state;
    if (failed(buildState(function, state)))
      return failure();
    int64_t nextScratchOffset = config->scratchSize;
    if (spillToScratch(function, state, **failureRecord, nextScratchOffset))
      function->removeAttr(kStateAttr);
  }
  return success();
}

static SmallVector<transform::MappedValue>
buildHandleMapping(ArrayRef<Operation *> targets) {
  SmallVector<transform::MappedValue> mapping;
  for (Operation *target : targets)
    mapping.push_back(target);
  return mapping;
}

static DiagnosedSilenceableFailure
runNamedSequence(transform::TransformOpInterface caller,
                 transform::NamedSequenceOp callee,
                 MutableArrayRef<SmallVector<transform::MappedValue>> bindings,
                 transform::TransformState &state,
                 SmallVectorImpl<SmallVector<transform::MappedValue>> &out) {
  if (callee.isExternal())
    return caller.emitDefiniteFailure()
           << "named sequence `" << callee.getSymName()
           << "` is external; cannot invoke";
  Block &block = callee.getBody().front();
  if (block.getNumArguments() != bindings.size())
    return caller.emitDefiniteFailure()
           << "named sequence `" << callee.getSymName() << "` expects "
           << block.getNumArguments() << " arguments, got " << bindings.size();
  auto scope = state.make_region_scope(callee.getBody());
  for (auto [argument, mapping] :
       llvm::zip_equal(block.getArguments(), bindings))
    if (failed(state.mapBlockArgument(argument, mapping)))
      return DiagnosedSilenceableFailure::definiteFailure();
  for (Operation &operation : block.without_terminator()) {
    DiagnosedSilenceableFailure result =
        state.applyTransform(cast<transform::TransformOpInterface>(operation));
    if (!result.succeeded())
      return result;
  }
  transform::detail::prepareValueMappings(
      out, block.getTerminator()->getOperands(), state);
  return DiagnosedSilenceableFailure::success();
}

enum class LoopDecision { Restart, Done, Stalled };

static FailureOr<LoopDecision> classifyLoop(ArrayRef<Operation *> targets) {
  LoopDecision decision = LoopDecision::Done;
  SmallVector<func::FuncOp> functions;
  SmallVector<func::FuncOp> stalledFunctions;
  for (Operation *target : targets)
    collectFunctions(target, functions);
  if (functions.empty())
    return failure();
  for (func::FuncOp function : functions) {
    DictionaryAttr packed = function->getAttrOfType<DictionaryAttr>(kStateAttr);
    if (!packed) {
      decision = LoopDecision::Restart;
      continue;
    }
    StringAttr stage = packed.getAs<StringAttr>(kStageAttr);
    if (!stage)
      return failure();
    if (stage.getValue() == kFailureStage) {
      stalledFunctions.push_back(function);
      if (decision != LoopDecision::Restart)
        decision = LoopDecision::Stalled;
    } else if (stage.getValue() != kSuccessStage)
      return failure();
  }
  if (decision != LoopDecision::Stalled)
    return decision;
  for (func::FuncOp function : stalledFunctions) {
    IntegerAttr grfCount =
        function->getAttrOfType<IntegerAttr>(kGrfCountAttrName);
    function.emitError("register allocation exhausted ")
        << grfCount.getInt()
        << " GRFs and no rematerialization or scratch candidate can relieve "
           "pressure";
  }
  return decision;
}

static void clearLoopState(ArrayRef<Operation *> targets, unsigned iteration) {
  SmallVector<func::FuncOp> functions;
  for (Operation *target : targets)
    collectFunctions(target, functions);
  for (func::FuncOp function : functions) {
    function->removeAttr(kStateAttr);
    function->setAttr(
        kLoopIterationAttr,
        IntegerAttr::get(IntegerType::get(function.getContext(), 32),
                         iteration));
  }
}

static void finishLoop(ArrayRef<Operation *> targets, unsigned iterations) {
  SmallVector<func::FuncOp> functions;
  for (Operation *target : targets)
    collectFunctions(target, functions);
  for (func::FuncOp function : functions) {
    function->removeAttr(kStateAttr);
    function->removeAttr(kLoopIterationAttr);
    function->setAttr(
        kIterationAttr,
        IntegerAttr::get(IntegerType::get(function.getContext(), 32),
                         iterations));
  }
}

template <typename OpTy, typename ApplyFn>
static DiagnosedSilenceableFailure
applyStage(OpTy op, transform::TransformResults &results,
           transform::TransformState &state, ApplyFn apply) {
  SmallVector<Operation *> targets;
  for (Operation *target : state.getPayloadOps(op.getTarget())) {
    targets.push_back(target);
    if (failed(apply(target)))
      return op.emitDefiniteFailure() << "regalloc transform stage failed";
  }
  results.set(cast<OpResult>(op.getResult()), targets);
  return DiagnosedSilenceableFailure::success();
}

template <typename OpTy>
static void
getStageEffects(OpTy op,
                SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(op.getTargetMutable(), effects);
  transform::producesHandle(op->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

} // namespace

namespace inter::xemachine {

LogicalResult
TransformRegAllocLoopOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  transform::NamedSequenceOp body =
      symbolTable.lookupNearestSymbolFrom<transform::NamedSequenceOp>(
          getOperation(), getBodyAttr());
  if (!body)
    return emitOpError() << "body symbol `" << getBody()
                         << "` does not resolve to a transform.named_sequence";
  if (getMaxIterations() == 0 ||
      getMaxIterations() > std::numeric_limits<unsigned>::max())
    return emitOpError("requires max_iterations in the unsigned integer range");
  return success();
}

DiagnosedSilenceableFailure
TransformRegAllocLoopOp::apply(transform::TransformRewriter &,
                               transform::TransformResults &results,
                               transform::TransformState &state) {
  transform::NamedSequenceOp body =
      SymbolTable::lookupNearestSymbolFrom<transform::NamedSequenceOp>(
          getOperation(), getBodyAttr());
  SmallVector<Operation *> current;
  llvm::append_range(current, state.getPayloadOps(getTarget()));
  unsigned maxIterations = static_cast<unsigned>(getMaxIterations());
  for (unsigned iteration = 1; iteration <= maxIterations; ++iteration) {
    clearLoopState(current, iteration);
    SmallVector<SmallVector<transform::MappedValue>> bindings;
    bindings.push_back(buildHandleMapping(current));
    SmallVector<SmallVector<transform::MappedValue>> output;
    DiagnosedSilenceableFailure status =
        runNamedSequence(*this, body, bindings, state, output);
    if (!status.succeeded())
      return status;
    if (output.size() != 1)
      return emitDefiniteFailure(
          "regalloc loop body must yield one operation handle");
    SmallVector<Operation *> yielded;
    for (transform::MappedValue mapping : output.front()) {
      Operation *payload = mapping.dyn_cast<Operation *>();
      if (!payload)
        return emitDefiniteFailure(
            "regalloc loop body must yield operation handles");
      yielded.push_back(payload);
    }
    FailureOr<LoopDecision> decision = classifyLoop(yielded);
    if (failed(decision))
      return emitDefiniteFailure("failed to classify regalloc loop state");
    if (*decision == LoopDecision::Restart) {
      current = std::move(yielded);
      continue;
    }
    if (*decision == LoopDecision::Stalled)
      return emitDefiniteFailure(
          "register allocation stalled after all relief providers");
    finishLoop(yielded, iteration);
    results.set(cast<OpResult>(getResult()), yielded);
    return DiagnosedSilenceableFailure::success();
  }
  return emitDefiniteFailure() << "regalloc transform loop exceeded "
                               << getMaxIterations() << " iterations";
}

void TransformRegAllocLoopOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  getStageEffects(*this, effects);
}

DiagnosedSilenceableFailure
TransformRegAllocBuildStateOp::apply(transform::TransformRewriter &,
                                     transform::TransformResults &results,
                                     transform::TransformState &state) {
  return applyStage(*this, results, state, buildTransformState);
}

void TransformRegAllocBuildStateOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  getStageEffects(*this, effects);
}

DiagnosedSilenceableFailure
TransformRegAllocLinearScanOp::apply(transform::TransformRewriter &,
                                     transform::TransformResults &results,
                                     transform::TransformState &state) {
  return applyStage(*this, results, state, runTransformLinearScan);
}

void TransformRegAllocLinearScanOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  getStageEffects(*this, effects);
}

DiagnosedSilenceableFailure
TransformRegAllocRematReliefOp::apply(transform::TransformRewriter &,
                                      transform::TransformResults &results,
                                      transform::TransformState &state) {
  return applyStage(*this, results, state, runTransformRematRelief);
}

void TransformRegAllocRematReliefOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  getStageEffects(*this, effects);
}

DiagnosedSilenceableFailure
TransformRegAllocScratchReliefOp::apply(transform::TransformRewriter &,
                                        transform::TransformResults &results,
                                        transform::TransformState &state) {
  return applyStage(*this, results, state, runTransformScratchRelief);
}

void TransformRegAllocScratchReliefOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  getStageEffects(*this, effects);
}

} // namespace inter::xemachine
