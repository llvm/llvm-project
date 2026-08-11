// Xe2 hardware policy for the target-neutral machine scheduler.

#include "Xe2ScheduleModel.h"

#include "inter/Dialect/XeMachine/IR/XeMachine.h"
#include "inter/Transforms/MachineScheduler.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>

using namespace mlir;
using namespace inter::xemachine;

namespace {

static Xe2DependencyKind getXe2DependencyKind(inter::MachineHazardKind kind) {
  switch (kind) {
  case inter::MachineHazardKind::raw:
    return Xe2DependencyKind::raw;
  case inter::MachineHazardKind::war:
    return Xe2DependencyKind::war;
  case inter::MachineHazardKind::waw:
    return Xe2DependencyKind::waw;
  case inter::MachineHazardKind::order:
    return Xe2DependencyKind::order;
  }
  llvm_unreachable("unknown machine dependency kind");
}

static uint64_t getArfResource(ARFType type) {
  return (static_cast<uint64_t>(type.getFile()) << 32) |
         static_cast<uint32_t>(type.getIndex());
}

static uint64_t getGrfResource(unsigned grf) {
  return (uint64_t{1} << 63) | grf;
}

struct RegisterAliasEdge {
  Value target;
  int64_t offset;
};

struct RegisterAliasValueInfo {
  unsigned component;
  int64_t offset;
};

struct RegisterAliasComponentInfo {
  int64_t minOffset = 0;
  int64_t maxOffset = 0;
  std::optional<int64_t> fixedOrigin;
  SmallVector<Value, 4> members;
};

struct RegisterAliasInfo {
  DenseMap<Value, RegisterAliasValueInfo> values;
  SmallVector<RegisterAliasComponentInfo> components;
  DenseMap<Operation *, int64_t> positions;
};

static void addAlias(DenseMap<Value, SmallVector<RegisterAliasEdge, 4>> &graph,
                     Value storage, Value alias, int64_t offset) {
  if (!isa<RegType>(storage.getType()) || !isa<RegType>(alias.getType()))
    return;
  graph[storage].push_back({alias, offset});
  graph[alias].push_back({storage, -offset});
}

static void
addRegionAliases(DenseMap<Value, SmallVector<RegisterAliasEdge, 4>> &graph,
                 Operation *operation) {
  auto addYields = [&](Region &region) {
    if (region.empty())
      return;
    auto yield = dyn_cast<YieldOp>(region.front().getTerminator());
    if (!yield)
      return;
    for (auto [result, yielded] :
         llvm::zip_equal(operation->getResults(), yield.getValues()))
      addAlias(graph, result, yielded, 0);
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
    addAlias(graph, result, init, 0);
    addAlias(graph, result, argument, 0);
    addAlias(graph, result, carried, 0);
  }
}

static FailureOr<RegisterAliasInfo>
buildRegisterAliasInfo(func::FuncOp function) {
  SmallVector<Value> orderedValues;
  DenseMap<Value, SmallVector<RegisterAliasEdge, 4>> graph;
  DenseMap<Operation *, int64_t> positions;
  int64_t nextPosition = 0;
  function.walk([&](Operation *operation) {
    positions.try_emplace(operation, nextPosition++);
    for (Value result : operation->getResults())
      if (isa<RegType>(result.getType()) && graph.try_emplace(result).second)
        orderedValues.push_back(result);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          if (isa<RegType>(argument.getType()) &&
              graph.try_emplace(argument).second)
            orderedValues.push_back(argument);
  });

  DenseMap<int64_t, Value> architecturalRegisters;
  function.walk([&](Operation *operation) {
    if (auto aliases = dyn_cast<RegisterStorageAliasOpInterface>(operation)) {
      SmallVector<RegisterStorageAlias, 4> relations;
      aliases.getRegisterStorageAliases(relations);
      for (const RegisterStorageAlias &relation : relations)
        addAlias(graph, relation.storage, relation.alias, relation.offset);
    }
    if (auto archreg = dyn_cast<ArchRegOp>(operation)) {
      Value previous = architecturalRegisters.lookup(archreg.getIndex());
      if (previous)
        addAlias(graph, previous, archreg.getResult(), 0);
      else
        architecturalRegisters[archreg.getIndex()] = archreg.getResult();
    }
    addRegionAliases(graph, operation);
  });

  RegisterAliasInfo info;
  info.positions = std::move(positions);
  SmallVector<Value, 16> pending;
  for (Value root : orderedValues) {
    if (info.values.count(root))
      continue;
    unsigned componentIndex = info.components.size();
    RegisterAliasComponentInfo &component = info.components.emplace_back();
    info.values.try_emplace(root, RegisterAliasValueInfo{componentIndex, 0});
    pending.push_back(root);
    while (!pending.empty()) {
      Value value = pending.pop_back_val();
      int64_t valueOffset = info.values.lookup(value).offset;
      RegType type = cast<RegType>(value.getType());
      component.members.push_back(value);
      component.minOffset = std::min(component.minOffset, valueOffset);
      component.maxOffset =
          std::max(component.maxOffset,
                   valueOffset + static_cast<int64_t>(type.getWidthDwords()));
      if (type.getBaseGRF() >= 0) {
        int64_t origin =
            static_cast<int64_t>(type.getBaseGRF()) * 16 - valueOffset;
        if (component.fixedOrigin && *component.fixedOrigin != origin)
          return function.emitError("conflicting physical register aliases");
        component.fixedOrigin = origin;
      }
      for (RegisterAliasEdge edge : graph.lookup(value)) {
        int64_t expected = valueOffset + edge.offset;
        DenseMap<Value, RegisterAliasValueInfo>::iterator existing =
            info.values.find(edge.target);
        if (existing == info.values.end()) {
          info.values.try_emplace(
              edge.target, RegisterAliasValueInfo{componentIndex, expected});
          pending.push_back(edge.target);
          continue;
        }
        if (existing->second.component != componentIndex ||
            existing->second.offset != expected)
          return function.emitError("inconsistent register-storage aliases");
      }
    }
  }
  return info;
}

static void
collectAliasDefinitions(Value root,
                        const DenseMap<Operation *, unsigned> &nodes,
                        SmallVectorImpl<Operation *> &definitions) {
  SmallVector<Value, 4> pending{root};
  SmallPtrSet<Value, 16> visited;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    Operation *definition = value.getDefiningOp();
    if (!definition)
      continue;

    if (UpdateTupleOp update = dyn_cast<UpdateTupleOp>(definition)) {
      if (value == update.getResult()) {
        pending.push_back(update.getBase());
        llvm::append_range(pending, update.getUpdates());
        continue;
      }
    }
    if (auto alias = dyn_cast<RegisterStorageAliasOpInterface>(definition)) {
      SmallVector<RegisterStorageAlias, 4> aliases;
      alias.getRegisterStorageAliases(aliases);
      bool followed = false;
      for (const RegisterStorageAlias &constraint : aliases) {
        if (constraint.storage == value) {
          pending.push_back(constraint.alias);
          followed = true;
        } else if (constraint.alias == value) {
          pending.push_back(constraint.storage);
          followed = true;
        }
      }
      if (followed)
        continue;
    }
    if (!definition->hasTrait<OpTrait::xemachine::NoMachineInst>() &&
        nodes.count(definition) && !llvm::is_contained(definitions, definition))
      definitions.push_back(definition);
  }
}

static bool
requiresPinnedDefinition(Value root,
                         const DenseMap<Operation *, unsigned> &nodes) {
  SmallVector<Value, 4> pending{root};
  SmallPtrSet<Value, 16> visited;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (!nodes.count(user))
        return true;
      auto alias = dyn_cast<RegisterStorageAliasOpInterface>(user);
      if (!alias)
        continue;
      SmallVector<RegisterStorageAlias, 4> aliases;
      alias.getRegisterStorageAliases(aliases);
      for (const RegisterStorageAlias &constraint : aliases)
        if (constraint.destructive &&
            (constraint.storage == value || constraint.alias == value))
          return true;
      llvm::append_range(pending, user->getResults());
    }
  }
  return false;
}

struct PressureValue {
  SmallVector<unsigned, 4> definitions;
  SmallVector<unsigned, 4> uses;
  unsigned units = 0;
  bool liveIn = false;
  bool liveAfter = false;
};

struct PressureModel {
  SmallVector<PressureValue, 16> values;
};

static void collectPressureUses(Value value,
                                const DenseMap<Operation *, unsigned> &nodes,
                                const DenseMap<Operation *, int64_t> &positions,
                                int64_t regionBegin, int64_t regionEnd,
                                SmallVectorImpl<unsigned> &uses,
                                bool &liveBefore, bool &liveAfter,
                                SmallPtrSetImpl<Value> &visited) {
  if (!visited.insert(value).second)
    return;
  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();
    if (owner->hasTrait<OpTrait::xemachine::NoMachineInst>()) {
      bool forwarded = false;
      for (Value result : owner->getResults()) {
        if (!isa<RegType>(result.getType()))
          continue;
        collectPressureUses(result, nodes, positions, regionBegin, regionEnd,
                            uses, liveBefore, liveAfter, visited);
        forwarded = true;
      }
      if (forwarded)
        continue;
    }
    DenseMap<Operation *, unsigned>::const_iterator node = nodes.find(owner);
    if (node != nodes.end()) {
      if (!llvm::is_contained(uses, node->second))
        uses.push_back(node->second);
      continue;
    }
    int64_t position = positions.lookup(owner);
    liveBefore |= position < regionBegin;
    liveAfter |= position > regionEnd;
  }
}

static void collectTokenCompletionUses(
    Value token, const DenseMap<Operation *, unsigned> &nodes,
    const DenseMap<Operation *, int64_t> &positions, int64_t regionBegin,
    int64_t regionEnd, SmallVectorImpl<unsigned> &uses, bool &liveBefore,
    bool &liveAfter, SmallPtrSetImpl<Value> &visited) {
  if (!visited.insert(token).second) {
    liveAfter = true;
    return;
  }
  for (OpOperand &use : token.getUses()) {
    Operation *owner = use.getOwner();
    DenseMap<Operation *, unsigned>::const_iterator node = nodes.find(owner);
    if (node != nodes.end()) {
      if (!llvm::is_contained(uses, node->second))
        uses.push_back(node->second);
    } else {
      int64_t position = positions.lookup(owner);
      liveBefore |= position < regionBegin;
      liveAfter |= position > regionEnd;
    }
    if (!owner->hasTrait<OpTrait::xemachine::NoMachineInst>())
      continue;
    bool forwarded = false;
    for (Value result : owner->getResults()) {
      if (!isa<MemTokenType>(result.getType()))
        continue;
      collectTokenCompletionUses(result, nodes, positions, regionBegin,
                                 regionEnd, uses, liveBefore, liveAfter,
                                 visited);
      forwarded = true;
    }
    if (!forwarded && isa<YieldOp, ContinueIfOp>(owner))
      liveAfter = true;
  }
  visited.erase(token);
}

static bool extendsPayloadToTokenCompletion(Operation *operation) {
  return isa<SendOp, LoadA64Op, StoreA64Op, LoadSLMOp, StoreSLMOp,
             AtomicIAddA64Op, LoadBlockA32Op, FenceSLMOp, BarrierSignalOp,
             EotOp>(operation);
}

static PressureModel buildPressureModel(ArrayRef<Operation *> operations,
                                        const RegisterAliasInfo &aliases) {
  DenseMap<Operation *, unsigned> nodes;
  for (auto [index, operation] : llvm::enumerate(operations))
    nodes.try_emplace(operation, index);
  int64_t regionBegin = aliases.positions.lookup(operations.front());
  int64_t regionEnd = aliases.positions.lookup(operations.back());

  llvm::SmallSetVector<unsigned, 16> components;
  for (Operation *operation : operations) {
    for (Value result : operation->getResults()) {
      RegType reg = dyn_cast<RegType>(result.getType());
      if (!reg || reg.getWidthDwords() == 0)
        continue;
      components.insert(aliases.values.lookup(result).component);
    }
    for (Value operand : operation->getOperands()) {
      RegType reg = dyn_cast<RegType>(operand.getType());
      if (!reg || reg.getWidthDwords() == 0)
        continue;
      components.insert(aliases.values.lookup(operand).component);
    }
  }

  PressureModel model;
  for (unsigned componentIndex : components) {
    const RegisterAliasComponentInfo &component =
        aliases.components[componentIndex];
    PressureValue &pressure = model.values.emplace_back();
    pressure.units = llvm::divideCeil(component.maxOffset - component.minOffset,
                                      int64_t{16});
    for (Value value : component.members) {
      Operation *definition = value.getDefiningOp();
      if (definition &&
          !definition->hasTrait<OpTrait::xemachine::NoMachineInst>()) {
        DenseMap<Operation *, unsigned>::const_iterator node =
            nodes.find(definition);
        if (node != nodes.end()) {
          if (!llvm::is_contained(pressure.definitions, node->second))
            pressure.definitions.push_back(node->second);
        } else {
          int64_t position = aliases.positions.lookup(definition);
          pressure.liveIn |= position < regionBegin;
          pressure.liveAfter |= position > regionEnd;
        }
      } else if (!definition) {
        pressure.liveIn = true;
      }

      SmallPtrSet<Value, 16> visited;
      SmallVector<unsigned, 4> uses;
      bool liveBefore = false;
      bool liveAfter = false;
      collectPressureUses(value, nodes, aliases.positions, regionBegin,
                          regionEnd, uses, liveBefore, liveAfter, visited);
      for (OpOperand &use : value.getUses()) {
        Operation *owner = use.getOwner();
        if (!extendsPayloadToTokenCompletion(owner))
          continue;
        SmallPtrSet<Value, 8> visitedTokens;
        for (Value result : owner->getResults())
          if (isa<MemTokenType>(result.getType()))
            collectTokenCompletionUses(result, nodes, aliases.positions,
                                       regionBegin, regionEnd, uses, liveBefore,
                                       liveAfter, visitedTokens);
      }
      for (unsigned use : uses) {
        if (!llvm::is_contained(pressure.uses, use))
          pressure.uses.push_back(use);
        if (isa_and_nonnull<ArchRegOp>(definition) &&
            !llvm::is_contained(pressure.definitions, use))
          pressure.definitions.push_back(use);
      }
      pressure.liveIn |= liveBefore;
      pressure.liveAfter |= liveAfter;
    }
  }
  return model;
}

static unsigned getPeakPressure(ArrayRef<unsigned> order,
                                ArrayRef<PressureValue> values,
                                unsigned nodeCount) {
  BitVector scheduled(nodeCount);
  BitVector live(values.size());
  unsigned pressure = 0;
  auto makeLive = [&](unsigned index) {
    if (live.test(index))
      return;
    live.set(index);
    pressure += values[index].units;
  };
  auto makeDead = [&](unsigned index) {
    assert(live.test(index) && "pressure value is not live");
    live.reset(index);
    assert(pressure >= values[index].units && "pressure count underflow");
    pressure -= values[index].units;
  };
  for (auto [index, value] : llvm::enumerate(values)) {
    if (!value.liveIn)
      continue;
    makeLive(index);
  }
  unsigned peak = pressure;
  for (unsigned node : order) {
    for (auto [index, value] : llvm::enumerate(values)) {
      if (!llvm::is_contained(value.definitions, node))
        continue;
      makeLive(index);
    }
    peak = std::max(peak, pressure);
    scheduled.set(node);
    for (auto [index, value] : llvm::enumerate(values)) {
      if (!live.test(index) || value.liveAfter)
        continue;
      if (llvm::any_of(value.definitions, [&](unsigned definition) {
            return !scheduled.test(definition);
          }))
        continue;
      if (llvm::any_of(value.uses,
                       [&](unsigned use) { return !scheduled.test(use); }))
        continue;
      makeDead(index);
    }
  }
  return peak;
}

class Xe2RegionSession final : public inter::MachineScheduleRegionSession {
public:
  Xe2RegionSession(ArrayRef<Operation *> operations,
                   const RegisterAliasInfo &aliases)
      : pressureModel(buildPressureModel(operations, aliases)),
        nodeCount(operations.size()) {
    SmallVector<unsigned, 16> original;
    llvm::append_range(original, llvm::seq<unsigned>(nodeCount));
    originalPeak = getPeakPressure(original, pressureModel.values, nodeCount);
  }

  bool canSchedulePrefix(ArrayRef<unsigned> prefix) const override {
    return getPeakPressure(prefix, pressureModel.values, nodeCount) <=
           originalPeak;
  }

private:
  PressureModel pressureModel;
  unsigned nodeCount;
  unsigned originalPeak;
};

class Xe2ScheduleModel final : public inter::MachineScheduleModel {
public:
  explicit Xe2ScheduleModel(RegisterAliasInfo aliases)
      : aliases(std::move(aliases)) {}

  bool isSchedulable(Operation *operation) const override {
    if (!isa<InstructionIssueOpInterface>(operation) ||
        operation->hasTrait<OpTrait::xemachine::FullScoreboardDrain>())
      return false;
    SendOp send = dyn_cast<SendOp>(operation);
    return !send || !send.getEot();
  }

  bool isNoInstruction(Operation *operation) const override {
    return operation->hasTrait<OpTrait::xemachine::NoMachineInst>();
  }

  bool isSupportedRegionOperation(Operation *operation) const override {
    return isa<ExecIfOp, UniformIfOp, UniformLoopOp>(operation);
  }

  inter::MachineHazardKind
  classifyDataDependency(Value operand) const override {
    return isa<MemTokenType>(operand.getType())
               ? inter::MachineHazardKind::order
               : inter::MachineHazardKind::raw;
  }

  void getStorageAccesses(
      Operation *operation,
      SmallVectorImpl<inter::MachineStorageAccess> &accesses) const override {
    if (operation->hasTrait<OpTrait::xemachine::NoMachineInst>())
      return;
    auto record = [&](uint64_t resource, bool read, bool write) {
      for (inter::MachineStorageAccess &access : accesses) {
        if (access.resource != resource)
          continue;
        access.read |= read;
        access.write |= write;
        return;
      }
      accesses.push_back({resource, read, write});
    };
    auto recordValue = [&](Value value, bool read, bool write) {
      if (ARFType arf = dyn_cast<ARFType>(value.getType())) {
        record(getArfResource(arf), read, write);
        return;
      }
      RegType reg = dyn_cast<RegType>(value.getType());
      if (!reg || reg.getWidthDwords() == 0)
        return;
      int64_t firstDword = static_cast<int64_t>(reg.getBaseGRF()) * 16;
      DenseMap<Value, RegisterAliasValueInfo>::const_iterator valueInfo =
          aliases.values.find(value);
      if (valueInfo != aliases.values.end()) {
        const RegisterAliasComponentInfo &component =
            aliases.components[valueInfo->second.component];
        if (!component.fixedOrigin)
          return;
        firstDword = *component.fixedOrigin + valueInfo->second.offset;
      } else if (reg.getBaseGRF() < 0) {
        return;
      }
      int64_t firstGRF = firstDword / 16;
      int64_t grfEnd = llvm::divideCeil(
          firstDword + static_cast<int64_t>(reg.getWidthDwords()), int64_t{16});
      for (int64_t grf : llvm::seq(firstGRF, grfEnd))
        record(getGrfResource(grf), read, write);
    };
    for (Value operand : operation->getOperands())
      recordValue(operand, /*read=*/true, /*write=*/false);
    for (Value result : operation->getResults())
      recordValue(result, /*read=*/false, /*write=*/true);
  }

  void getExtraDependencies(ArrayRef<Operation *> operations,
                            SmallVectorImpl<inter::MachineExtraDependency>
                                &dependencies) const override {
    DenseMap<Operation *, unsigned> nodes;
    for (auto [index, operation] : llvm::enumerate(operations))
      nodes.try_emplace(operation, index);

    for (auto [index, operation] : llvm::enumerate(operations)) {
      if (operation->hasTrait<OpTrait::xemachine::NoMachineInst>() ||
          !llvm::any_of(operation->getResults(), [&](Value result) {
            return requiresPinnedDefinition(result, nodes);
          }))
        continue;
      for (unsigned predecessor : llvm::seq<unsigned>(index))
        dependencies.push_back({operations[predecessor], operation,
                                inter::MachineHazardKind::order});
    }

    UniformLoopOp loop =
        dyn_cast<UniformLoopOp>(operations.front()->getBlock()->getParentOp());
    if (!loop)
      return;
    ContinueIfOp terminator =
        dyn_cast<ContinueIfOp>(operations.front()->getBlock()->getTerminator());
    if (!terminator)
      return;
    for (auto [argument, carried] :
         llvm::zip_equal(operations.front()->getBlock()->getArguments(),
                         terminator.getCarried())) {
      if (!isa<RegType, ARFType>(argument.getType()))
        continue;
      SmallVector<Operation *, 4> definitions;
      collectAliasDefinitions(carried, nodes, definitions);
      if (definitions.empty())
        continue;

      SmallVector<Value, 4> pending{argument};
      SmallPtrSet<Value, 16> visited;
      while (!pending.empty()) {
        Value value = pending.pop_back_val();
        if (!visited.insert(value).second)
          continue;
        for (OpOperand &use : value.getUses()) {
          Operation *user = use.getOwner();
          if (isa<RegisterStorageAliasOpInterface>(user)) {
            llvm::append_range(pending, user->getResults());
            continue;
          }
          if (!nodes.count(user))
            continue;
          for (Operation *definition : definitions)
            if (user != definition)
              dependencies.push_back(
                  {user, definition, inter::MachineHazardKind::war});
        }
      }
    }
  }

  std::unique_ptr<inter::MachineScheduleRegionSession>
  createRegionSession(ArrayRef<Operation *> operations) const override {
    return std::make_unique<Xe2RegionSession>(operations, aliases);
  }

  std::unique_ptr<inter::MachineScheduleState> createState() const override;

private:
  RegisterAliasInfo aliases;
};

class Xe2ScheduleState final : public inter::MachineScheduleState {
public:
  FailureOr<inter::MachineScheduleIssue> previewIssue(
      Operation *operation,
      ArrayRef<inter::MachineScheduleDependency> dependencies) const override {
    FailureOr<Xe2InstructionTiming> timing = getXe2InstructionTiming(operation);
    if (failed(timing))
      return failure();

    int64_t dependencyReadyCycle = 0;
    for (const inter::MachineScheduleDependency &dependency : dependencies) {
      int64_t readyCycle = dependency.issue.cycle;
      if (dependency.issue.instruction) {
        FailureOr<Xe2InstructionTiming> producerTiming =
            getXe2InstructionTiming(dependency.producer);
        if (failed(producerTiming))
          return failure();
        readyCycle += getXe2RequiredGap(*producerTiming,
                                        getXe2DependencyKind(dependency.kind));
      }
      dependencyReadyCycle = std::max(dependencyReadyCycle, readyCycle);
    }

    inter::MachineScheduleIssue issue;
    issue.instruction = timing->issueClass != InstructionIssueClass::none;
    if (!issue.instruction) {
      issue.cycle = dependencyReadyCycle;
      issue.nextCycle = currentCycle;
      return issue;
    }

    unsigned pipe = static_cast<unsigned>(timing->pipe);
    issue.cycle =
        std::max({currentCycle, dependencyReadyCycle, pipeReadyCycle[pipe]});
    issue.nextCycle = issue.cycle + 1;
    issue.stallCycles = issue.cycle - currentCycle;
    return issue;
  }

  FailureOr<inter::MachineScheduleIssue> commitIssue(
      Operation *operation,
      ArrayRef<inter::MachineScheduleDependency> dependencies) override {
    FailureOr<inter::MachineScheduleIssue> issue =
        previewIssue(operation, dependencies);
    if (failed(issue) || !issue->instruction)
      return issue;
    FailureOr<Xe2InstructionTiming> timing = getXe2InstructionTiming(operation);
    if (failed(timing))
      return failure();
    currentCycle = issue->nextCycle;
    unsigned pipe = static_cast<unsigned>(timing->pipe);
    pipeReadyCycle[pipe] = issue->cycle + timing->occupancy;
    return issue;
  }

  FailureOr<bool>
  canFill(Operation *baseline,
          ArrayRef<inter::MachineScheduleDependency> baselineDependencies,
          Operation *candidate,
          ArrayRef<inter::MachineScheduleDependency> candidateDependencies)
      const override {
    FailureOr<inter::MachineScheduleIssue> baselineBefore =
        previewIssue(baseline, baselineDependencies);
    FailureOr<inter::MachineScheduleIssue> candidatePreview =
        previewIssue(candidate, candidateDependencies);
    if (failed(baselineBefore) || failed(candidatePreview))
      return failure();
    if (!candidatePreview->instruction || candidatePreview->stallCycles != 0)
      return false;

    Xe2ScheduleState trial = *this;
    if (failed(trial.commitIssue(candidate, candidateDependencies)))
      return failure();
    FailureOr<inter::MachineScheduleIssue> baselineAfter =
        trial.previewIssue(baseline, baselineDependencies);
    if (failed(baselineAfter))
      return failure();
    return baselineAfter->cycle <= baselineBefore->cycle;
  }

private:
  int64_t currentCycle = 0;
  std::array<int64_t, 4> pipeReadyCycle{};
};

std::unique_ptr<inter::MachineScheduleState>
Xe2ScheduleModel::createState() const {
  return std::make_unique<Xe2ScheduleState>();
}

} // namespace

FailureOr<std::unique_ptr<inter::MachineScheduleModel>>
inter::createXe2ScheduleModel(func::FuncOp function) {
  FailureOr<RegisterAliasInfo> aliases = buildRegisterAliasInfo(function);
  if (failed(aliases))
    return failure();
  std::unique_ptr<inter::MachineScheduleModel> model =
      std::make_unique<Xe2ScheduleModel>(std::move(*aliases));
  return model;
}
