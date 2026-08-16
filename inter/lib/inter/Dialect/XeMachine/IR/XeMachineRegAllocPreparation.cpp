#include "inter/Dialect/XeMachine/IR/XeMachineRegAllocPreparation.h"

#include "inter/Dialect/XeMachine/IR/XeMachine.h"
#include "inter/Dialect/XeMachine/IR/XeMachineRegionFlow.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>

using namespace mlir;
using namespace inter::xemachine;

namespace {

constexpr StringLiteral kRegisterCopyAttr = "xemachine.regalloc_copy";
constexpr StringLiteral kImmediateLegalizationAttr =
    "xemachine.immediate_legalization";

static bool isMarkedCopy(Operation *operation) {
  return operation && operation->hasAttr(kRegisterCopyAttr);
}

static void legalizeWideImmediates(func::FuncOp function) {
  SmallVector<OpOperand *> operands;
  function.walk([&](Operation *operation) {
    auto alu = dyn_cast<ALUOpInterface>(operation);
    if (!alu || !alu.getInstructionElementType().isInteger(64) ||
        cast<InstructionIssueOpInterface>(operation).getInstructionKind() ==
            MachineInstructionKind::mov)
      return;
    for (OpOperand &operand : operation->getOpOperands())
      if (operand.get().getDefiningOp<ImmOp>())
        operands.push_back(&operand);
  });
  for (OpOperand *operand : operands) {
    Operation *owner = operand->getOwner();
    OpBuilder builder(owner);
    MovOp move = MovOp::create(
        builder, owner->getLoc(), RegType::get(function.getContext(), 2, -1),
        builder.getI64Type(), /*execSize=*/1,
        DstRegionAttr::get(function.getContext(), 1), RegionAttr(),
        IntegerAttr(), IntegerAttr(), TypeAttr(), /*noMask=*/true,
        /*maskOffset=*/0, operand->get());
    move->setAttr(kImmediateLegalizationAttr, builder.getUnitAttr());
    operand->set(move.getDst());
    unsigned operandNumber = operand->getOperandNumber();
    ALUOpInterface alu = cast<ALUOpInterface>(owner);
    if (!alu.getSourceRegion(operandNumber))
      alu.setSourceRegion(operandNumber,
                          RegionAttr::get(function.getContext(), 0, 1, 0));
  }
}

static bool isMarkedCopy(Value value, StringRef kind) {
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  StringAttr marker = definition->getAttrOfType<StringAttr>(kRegisterCopyAttr);
  return marker && marker.getValue() == kind;
}

static Value getCopiedSource(Value value) {
  Operation *definition = value.getDefiningOp();
  if (!isMarkedCopy(definition))
    return {};
  if (MovOp move = dyn_cast<MovOp>(definition))
    return move.getSrc();

  TupleFromElementsOp tuple = dyn_cast<TupleFromElementsOp>(definition);
  if (!tuple || tuple.getElements().empty())
    return {};
  Value source;
  for (Value element : tuple.getElements()) {
    MovOp move = element.getDefiningOp<MovOp>();
    if (!move || !isMarkedCopy(move))
      return {};
    if (!source)
      source = move.getSrc();
    else if (source != move.getSrc())
      return {};
  }
  return source;
}

static FailureOr<Value>
materializeRegisterCopy(OpBuilder &builder, Location location, Value source,
                        RegType destinationType, StringRef kind,
                        int64_t destinationSub = 0, bool noMask = false) {
  RegType sourceType = dyn_cast<RegType>(source.getType());
  if (!sourceType)
    return emitError(location)
               << "cannot materialize XeMachine register copy from "
               << source.getType(),
           failure();

  uint32_t width = sourceType.getWidthDwords();
  bool singleInstructionWidth = width <= 32 && llvm::isPowerOf2_32(width);
  if (!singleInstructionWidth && (width == 0 || width % 16 != 0))
    return emitError(location)
               << "cannot materialize XeMachine register copy of " << width
               << " dwords; width must be a legal execution size or divisible "
                  "by 16",
           failure();
  if (destinationType.getWidthDwords() != width)
    return emitError(location)
               << "cannot materialize XeMachine register copy from " << width
               << " dwords to " << destinationType.getWidthDwords()
               << " dwords",
           failure();

  SmallVector<Value, 4> pieces;
  MLIRContext *context = builder.getContext();
  Type i32 = builder.getI32Type();
  for (uint32_t offset = 0; offset < width;) {
    uint32_t pieceWidth = std::min<uint32_t>(32, width - offset);
    int64_t pieceSub = destinationSub;
    int32_t destinationBase = destinationType.getBaseGRF();
    if (destinationBase >= 0) {
      destinationBase += offset / 16 + pieceSub / 16;
      pieceSub %= 16;
    }
    RegType pieceType = RegType::get(context, pieceWidth, destinationBase);
    IntegerAttr destinationSubAttr =
        pieceSub == 0 ? IntegerAttr() : builder.getI32IntegerAttr(pieceSub);
    IntegerAttr sourceSub =
        offset == 0 ? IntegerAttr() : builder.getI32IntegerAttr(offset);
    MovOp move = MovOp::create(builder, location, pieceType, i32,
                               /*execSize=*/pieceWidth, DstRegionAttr(),
                               RegionAttr(), destinationSubAttr, sourceSub,
                               TypeAttr(), noMask, /*maskOffset=*/0, source);
    move->setAttr(kRegisterCopyAttr, builder.getStringAttr(kind));
    pieces.push_back(move.getDst());
    offset += pieceWidth;
  }

  if (pieces.size() == 1)
    return pieces.front();
  TupleFromElementsOp tuple = TupleFromElementsOp::create(
      builder, location, destinationType, ValueRange(pieces));
  tuple->setAttr(kRegisterCopyAttr, builder.getStringAttr(kind));
  return tuple.getTuple();
}

static bool isPotentiallyAfter(Operation *candidate, Operation *anchor,
                               DominanceInfo &dominance) {
  if (candidate == anchor)
    return false;

  Block *anchorBlock = anchor->getBlock();
  Operation *ancestor = candidate;
  while (ancestor && ancestor->getBlock() != anchorBlock)
    ancestor = ancestor->getParentOp();
  if (ancestor)
    return ancestor != anchor && anchor->isBeforeInBlock(ancestor);
  if (dominance.properlyDominates(candidate, anchor))
    return false;
  return true;
}

static bool isLiveAfter(Value value, Operation *anchor,
                        DominanceInfo &dominance,
                        Operation *ignoredUser = nullptr) {
  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();
    if (owner == anchor || owner == ignoredUser)
      continue;
    if (isPotentiallyAfter(owner, anchor, dominance))
      return true;
  }
  return false;
}

struct AliasEdge {
  Value target;
  int64_t offset;
};

struct AliasValueInfo {
  unsigned component;
  int64_t offset;
};

class WeightedOverlapSummary {
public:
  LogicalResult build(func::FuncOp function, bool includeRegionAliases = true,
                      Operation *excludedRegionBranch = nullptr,
                      Operation *excludedAliasOperation = nullptr) {
    values.clear();
    graph.clear();
    info.clear();
    componentOrigins.clear();
    componentInconsistent.clear();

    auto remember = [&](Value value) {
      RegType type = dyn_cast<RegType>(value.getType());
      if (!type || graph.count(value))
        return;
      values.push_back(value);
      graph.try_emplace(value);
    };
    function.walk([&](Operation *operation) {
      for (Value result : operation->getResults())
        remember(result);
      for (Region &region : operation->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments())
            remember(argument);
    });

    auto connect = [&](Value storage, Value alias, int64_t offset) {
      remember(storage);
      remember(alias);
      graph[storage].push_back({alias, offset});
      graph[alias].push_back({storage, -offset});
    };
    function.walk([&](Operation *operation) {
      RegisterStorageAliasOpInterface interface =
          dyn_cast<RegisterStorageAliasOpInterface>(operation);
      if (interface && operation != excludedAliasOperation) {
        SmallVector<RegisterStorageAlias, 4> aliases;
        interface.getRegisterStorageAliases(aliases);
        for (const RegisterStorageAlias &relation : aliases)
          connect(relation.storage, relation.alias, relation.offset);
      }
    });
    if (includeRegionAliases) {
      RegionFlow regionFlow(function);
      for (const RegionFlow::Branch &branch : regionFlow.getBranches())
        if (branch.operation != excludedRegionBranch)
          for (const RegionFlow::Transfer &transfer : branch.transfers)
            connect(transfer.operand->get(), transfer.input, 0);
    }

    SmallVector<Value, 16> pending;
    for (Value root : values) {
      if (info.count(root))
        continue;
      unsigned component = componentOrigins.size();
      componentOrigins.push_back(std::nullopt);
      componentInconsistent.push_back(false);
      info.try_emplace(root, AliasValueInfo{component, 0});
      pending.push_back(root);
      while (!pending.empty()) {
        Value value = pending.pop_back_val();
        AliasValueInfo valueInfo = info.lookup(value);
        RegType type = cast<RegType>(value.getType());
        if (type.getBaseGRF() >= 0) {
          int64_t origin =
              static_cast<int64_t>(type.getBaseGRF()) * 16 - valueInfo.offset;
          std::optional<int64_t> &knownOrigin = componentOrigins[component];
          if (knownOrigin && *knownOrigin != origin) {
            componentInconsistent[component] = true;
            knownOrigin.reset();
          } else if (!componentInconsistent[component]) {
            knownOrigin = origin;
          }
        }
        for (AliasEdge edge : graph.lookup(value)) {
          int64_t expected = valueInfo.offset + edge.offset;
          auto existing = info.find(edge.target);
          if (existing == info.end()) {
            info.try_emplace(edge.target, AliasValueInfo{component, expected});
            pending.push_back(edge.target);
            continue;
          }
          if (existing->second.component != component ||
              existing->second.offset != expected)
            componentInconsistent[component] = true;
        }
      }
    }
    return success();
  }

  bool overlaps(Value lhs, Value rhs) const {
    if (lhs == rhs)
      return true;
    RegType lhsType = dyn_cast<RegType>(lhs.getType());
    RegType rhsType = dyn_cast<RegType>(rhs.getType());
    if (!lhsType || !rhsType || lhsType.getWidthDwords() == 0 ||
        rhsType.getWidthDwords() == 0)
      return false;

    auto lhsIt = info.find(lhs);
    auto rhsIt = info.find(rhs);
    if (lhsIt != info.end() && rhsIt != info.end() &&
        lhsIt->second.component == rhsIt->second.component)
      return componentInconsistent[lhsIt->second.component] ||
             rangesOverlap(lhsIt->second.offset, lhsType.getWidthDwords(),
                           rhsIt->second.offset, rhsType.getWidthDwords());

    std::optional<int64_t> lhsStart = getAbsoluteStart(lhs, lhsType);
    std::optional<int64_t> rhsStart = getAbsoluteStart(rhs, rhsType);
    return lhsStart && rhsStart &&
           rangesOverlap(*lhsStart, lhsType.getWidthDwords(), *rhsStart,
                         rhsType.getWidthDwords());
  }

  SmallVector<Value, 8> getOverlappingValues(Value value) const {
    SmallVector<Value, 8> overlapping;
    for (Value candidate : values)
      if (overlaps(value, candidate))
        overlapping.push_back(candidate);
    return overlapping;
  }

private:
  static bool rangesOverlap(int64_t lhsStart, int64_t lhsWidth,
                            int64_t rhsStart, int64_t rhsWidth) {
    return lhsStart < rhsStart + rhsWidth && rhsStart < lhsStart + lhsWidth;
  }

  std::optional<int64_t> getAbsoluteStart(Value value, RegType type) const {
    auto valueIt = info.find(value);
    if (valueIt != info.end()) {
      if (componentInconsistent[valueIt->second.component])
        return std::nullopt;
      std::optional<int64_t> origin =
          componentOrigins[valueIt->second.component];
      if (origin)
        return *origin + valueIt->second.offset;
    }
    if (type.getBaseGRF() >= 0)
      return static_cast<int64_t>(type.getBaseGRF()) * 16;
    return std::nullopt;
  }

  SmallVector<Value> values;
  DenseMap<Value, SmallVector<AliasEdge, 4>> graph;
  DenseMap<Value, AliasValueInfo> info;
  SmallVector<std::optional<int64_t>> componentOrigins;
  SmallVector<bool> componentInconsistent;
};

static LogicalResult copyTupleElement(TupleFromElementsOp tuple, unsigned index,
                                      StringRef kind) {
  Value source = tuple.getElements()[index];
  RegType sourceType = cast<RegType>(source.getType());
  RegType copyType =
      RegType::get(tuple.getContext(), sourceType.getWidthDwords(), -1);
  OpBuilder builder(tuple);
  FailureOr<Value> copy =
      materializeRegisterCopy(builder, tuple.getLoc(), source, copyType, kind);
  if (failed(copy))
    return failure();
  tuple->setOperand(index, *copy);
  return success();
}

static std::optional<int64_t> getTupleElementOffset(TupleToElementsOp split,
                                                    Value element) {
  int64_t offset = 0;
  for (Value candidate : split.getElements()) {
    if (candidate == element)
      return offset;
    offset += cast<RegType>(candidate.getType()).getWidthDwords();
  }
  return std::nullopt;
}

static std::optional<int64_t> getGRFCount(func::FuncOp function) {
  IntegerAttr count =
      function->getAttrOfType<IntegerAttr>("xemachine.grf_count");
  return count ? std::optional<int64_t>(count.getInt()) : std::nullopt;
}

static LogicalResult repairTupleSplits(func::FuncOp function) {
  std::optional<int64_t> grfCount = getGRFCount(function);
  SmallVector<TupleToElementsOp> splits;
  function.walk([&](TupleToElementsOp split) { splits.push_back(split); });
  for (TupleToElementsOp split : splits) {
    RegType sourceType = cast<RegType>(split.getTuple().getType());
    if (sourceType.getBaseGRF() >= 0)
      continue;
    int64_t offset = 0;
    std::optional<int64_t> inferredSourceOrigin;
    OpBuilder builder(split);
    builder.setInsertionPointAfter(split);
    for (Value element : split.getElements()) {
      RegType elementType = cast<RegType>(element.getType());
      int64_t inferredOrigin =
          static_cast<int64_t>(elementType.getBaseGRF()) * 16 - offset;
      bool invalidOrigin =
          elementType.getBaseGRF() >= 0 &&
          (inferredOrigin < 0 ||
           (grfCount &&
            inferredOrigin + sourceType.getWidthDwords() > *grfCount * 16) ||
           (inferredSourceOrigin && *inferredSourceOrigin != inferredOrigin));
      if (invalidOrigin) {
        element.setType(RegType::get(function.getContext(),
                                     elementType.getWidthDwords(), -1));
        FailureOr<Value> copy = materializeRegisterCopy(
            builder, split.getLoc(), element, elementType, "tuple-element");
        if (failed(copy))
          return failure();
        element.replaceAllUsesExcept(*copy, copy->getDefiningOp());
      } else if (elementType.getBaseGRF() >= 0) {
        inferredSourceOrigin = inferredOrigin;
      }
      offset += elementType.getWidthDwords();
    }
  }
  return success();
}

static bool isAlignedTupleView(TupleFromElementsOp tuple,
                               std::optional<int64_t> grfCount) {
  TupleToElementsOp split;
  std::optional<int64_t> sourceShift;
  int64_t consumerOffset = 0;
  for (Value element : tuple.getElements()) {
    TupleToElementsOp elementSplit = element.getDefiningOp<TupleToElementsOp>();
    if (!elementSplit || (split && split != elementSplit))
      return false;
    split = elementSplit;
    std::optional<int64_t> sourceOffset = getTupleElementOffset(split, element);
    if (!sourceOffset || *sourceOffset < consumerOffset)
      return false;
    int64_t shift = *sourceOffset - consumerOffset;
    if (sourceShift && *sourceShift != shift)
      return false;
    sourceShift = shift;
    consumerOffset += cast<RegType>(element.getType()).getWidthDwords();
  }
  if (!split || !sourceShift || *sourceShift % 16 != 0)
    return false;
  RegType sourceType = cast<RegType>(split.getTuple().getType());
  RegType resultType = cast<RegType>(tuple.getTuple().getType());
  if (*sourceShift + resultType.getWidthDwords() > sourceType.getWidthDwords())
    return false;
  if (sourceType.getBaseGRF() >= 0 && resultType.getBaseGRF() < 0)
    return false;
  if (sourceType.getBaseGRF() < 0 && resultType.getBaseGRF() >= 0) {
    int64_t inferredOrigin =
        static_cast<int64_t>(resultType.getBaseGRF()) * 16 - *sourceShift;
    if (inferredOrigin < 0 ||
        (grfCount &&
         inferredOrigin + sourceType.getWidthDwords() > *grfCount * 16))
      return false;
  }
  return sourceType.getBaseGRF() < 0 || resultType.getBaseGRF() < 0 ||
         resultType.getBaseGRF() == sourceType.getBaseGRF() + *sourceShift / 16;
}

static LogicalResult repairTupleSlots(func::FuncOp function) {
  if (failed(repairTupleSplits(function)))
    return failure();
  std::optional<int64_t> grfCount = getGRFCount(function);
  SmallVector<TupleFromElementsOp> tuples;
  function.walk([&](TupleFromElementsOp tuple) {
    if (!isMarkedCopy(tuple))
      tuples.push_back(tuple);
  });

  DenseMap<Value, int64_t> anchorSlots;
  function.walk([&](TupleToElementsOp split) {
    int64_t offset = 0;
    for (Value element : split.getElements()) {
      anchorSlots.try_emplace(element, offset);
      offset += cast<RegType>(element.getType()).getWidthDwords();
    }
  });
  DenseSet<Value> consumed;
  for (TupleFromElementsOp tuple : tuples) {
    RegType tupleType = cast<RegType>(tuple.getTuple().getType());
    bool preserveView = isAlignedTupleView(tuple, grfCount);
    int64_t offset = 0;
    for (unsigned index = 0; index < tuple.getElements().size(); ++index) {
      Value element = tuple.getElements()[index];
      RegType elementType = cast<RegType>(element.getType());
      bool fixedConflict =
          elementType.getBaseGRF() >= 0 &&
          (tupleType.getBaseGRF() < 0 || offset % 16 != 0 ||
           elementType.getBaseGRF() != tupleType.getBaseGRF() + offset / 16);
      DenseMap<Value, int64_t>::const_iterator anchor =
          anchorSlots.find(element);
      bool slotMismatch =
          anchor != anchorSlots.end() && anchor->second != offset;
      bool sourceDrag = element.getDefiningOp<TupleToElementsOp>() != nullptr;
      bool needsCopy =
          !preserveView && (fixedConflict || slotMismatch || sourceDrag ||
                            consumed.contains(element));
      if (needsCopy && failed(copyTupleElement(tuple, index, "tuple-element")))
        return failure();
      element = tuple.getElements()[index];
      if (!preserveView) {
        anchorSlots[element] = offset;
        consumed.insert(element);
      }
      offset += elementType.getWidthDwords();
    }
  }
  return success();
}

static bool isAvailableBefore(Value value, Operation *operation,
                              DominanceInfo &dominance) {
  Operation *definition = value.getDefiningOp();
  return definition != operation &&
         dominance.properlyDominates(value, operation);
}

static bool storageIsLiveAfter(const WeightedOverlapSummary &summary,
                               Value value, Operation *operation,
                               DominanceInfo &dominance,
                               Operation *ignoredUser = nullptr) {
  return llvm::any_of(summary.getOverlappingValues(value), [&](Value alias) {
    return isAvailableBefore(alias, operation, dominance) &&
           isLiveAfter(alias, operation, dominance, ignoredUser);
  });
}

static bool storageIsDefinedAfter(const WeightedOverlapSummary &summary,
                                  Value value, Value anchor,
                                  DominanceInfo &dominance) {
  Operation *anchorDefinition = anchor.getDefiningOp();
  if (!anchorDefinition)
    return false;
  Operation *definition = value.getDefiningOp();
  if (definition && isPotentiallyAfter(definition, anchorDefinition, dominance))
    return true;
  return llvm::any_of(summary.getOverlappingValues(value), [&](Value alias) {
    Operation *aliasDefinition = alias.getDefiningOp();
    return aliasDefinition &&
           isPotentiallyAfter(aliasDefinition, anchorDefinition, dominance);
  });
}

static bool isDefinedInside(Region &region, Value value);

static LogicalResult repairUpdateTuples(func::FuncOp function) {
  SmallVector<UpdateTupleOp> updates;
  function.walk([&](UpdateTupleOp update) { updates.push_back(update); });
  DominanceInfo dominance(function);
  RegionFlow regionFlow(function);
  for (UpdateTupleOp update : updates) {
    Value base = update.getBase();
    if (isMarkedCopy(base, "update-base") &&
        llvm::all_of(update.getUpdates(), [](Value replacement) {
          return isMarkedCopy(replacement, "update-value");
        }))
      continue;
    OpBuilder builder(update);
    WeightedOverlapSummary summary;
    if (failed(summary.build(function, /*includeRegionAliases=*/true,
                             /*excludedRegionBranch=*/nullptr, update)))
      return failure();

    Region *repetitiveRegion = regionFlow.getEnclosingRepetitiveRegion(update);
    bool repeatedExternalBase =
        repetitiveRegion && !isDefinedInside(*repetitiveRegion, base);
    if (repeatedExternalBase ||
        storageIsLiveAfter(summary, base, update, dominance)) {
      RegType baseType = cast<RegType>(base.getType());
      RegType copyType =
          RegType::get(function.getContext(), baseType.getWidthDwords(), -1);
      FailureOr<Value> copy = materializeRegisterCopy(
          builder, update.getLoc(), base, copyType, "update-base");
      if (failed(copy))
        return failure();
      update->setOperand(0, *copy);
    }

    for (unsigned index = 0; index < update.getUpdates().size(); ++index) {
      Value replacement = update.getUpdates()[index];
      bool uniqueStorage = llvm::all_of(
          llvm::enumerate(update.getUpdates()), [&](auto candidate) {
            return candidate.index() == index ||
                   !summary.overlaps(replacement, candidate.value());
          });
      SmallVector<Value, 8> aliases = summary.getOverlappingValues(replacement);
      bool unconstrainedStorage =
          aliases.size() == 1 && aliases.front() == replacement;
      Operation *definition = replacement.getDefiningOp();
      ALUOpInterface alu = dyn_cast_or_null<ALUOpInterface>(definition);
      int64_t offset = cast<IntegerAttr>(update.getOffsets()[index]).getInt();
      int64_t destinationSub = 0;
      bool representableOffset = offset % 16 == 0;
      if (alu && alu.getInstructionElementType().isIntOrFloat()) {
        unsigned elementBits =
            alu.getInstructionElementType().getIntOrFloatBitWidth();
        int64_t offsetBits = (offset % 16) * 32;
        representableOffset = offsetBits % elementBits == 0;
        destinationSub = offsetBits / elementBits;
      }
      if (uniqueStorage && unconstrainedStorage && representableOffset &&
          (offset % 16 == 0 || alu) && definition &&
          !storageIsDefinedAfter(summary, update.getBase(), replacement,
                                 dominance) &&
          !storageIsLiveAfter(summary, update.getBase(), definition, dominance,
                              update) &&
          !storageIsLiveAfter(summary, replacement, update, dominance)) {
        if (offset % 16 == 0 ||
            succeeded(alu.setDestinationSubregister(destinationSub)))
          continue;
      }

      RegType replacementType = cast<RegType>(replacement.getType());
      RegType replacementCopyType = RegType::get(
          function.getContext(), replacementType.getWidthDwords(), -1);
      FailureOr<Value> replacementCopy = materializeRegisterCopy(
          builder, update.getLoc(), replacement, replacementCopyType,
          "update-value", offset % 16);
      if (failed(replacementCopy))
        return failure();
      update->setOperand(index + 1, *replacementCopy);
    }
  }
  return success();
}

static bool isDefinedInside(Region &region, Value value) {
  if (Operation *definition = value.getDefiningOp())
    return region.isAncestor(definition->getParentRegion());
  BlockArgument argument = dyn_cast<BlockArgument>(value);
  return argument && region.isAncestor(argument.getOwner()->getParent());
}

static bool storageIsDefinedInside(const WeightedOverlapSummary &summary,
                                   Region &region, Value value,
                                   Operation *operation,
                                   DominanceInfo &dominance) {
  return llvm::all_of(summary.getOverlappingValues(value), [&](Value alias) {
    return !isAvailableBefore(alias, operation, dominance) ||
           isDefinedInside(region, alias);
  });
}

static bool storageHasUseInside(const WeightedOverlapSummary &summary,
                                Value value, Operation *container,
                                Operation *operation,
                                DominanceInfo &dominance) {
  return llvm::any_of(summary.getOverlappingValues(value), [&](Value alias) {
    if (!isAvailableBefore(alias, operation, dominance))
      return false;
    return llvm::any_of(alias.getUses(), [&](OpOperand &use) {
      Operation *owner = use.getOwner();
      return owner != container && container->isProperAncestor(owner);
    });
  });
}

static bool storageDiesAtBranch(const RegionFlow &flow,
                                const WeightedOverlapSummary &summary,
                                Value value, Operation *branch,
                                DominanceInfo &dominance) {
  Region *repetitiveRegion = flow.getEnclosingRepetitiveRegion(branch);
  if (repetitiveRegion && !isDefinedInside(*repetitiveRegion, value))
    return false;
  return !storageIsLiveAfter(summary, value, branch, dominance);
}

static bool
isUniqueJoinIncoming(const RegionFlow::Branch &branch,
                     const RegionFlow::Transfer &current, Value value,
                     const DenseMap<OpOperand *, Value> &incomingValues) {
  return llvm::none_of(branch.transfers,
                       [&](const RegionFlow::Transfer &other) {
                         return !other.target && other.input != current.input &&
                                incomingValues.lookup(other.operand) == value;
                       });
}

static bool
alternativesCanOverwrite(const RegionFlow &flow,
                         const RegionFlow::Branch &branch,
                         const RegionFlow::Transfer &current, Value value,
                         const DenseMap<OpOperand *, Value> &incomingValues,
                         const WeightedOverlapSummary &summary) {
  bool sawAlternative = false;
  for (const RegionFlow::Transfer &other : branch.transfers) {
    if (other.target || other.input != current.input || &other == &current)
      continue;
    if (!flow.areMutuallyExclusive(current.source, other.source))
      return false;
    sawAlternative = true;
    Value incoming = incomingValues.lookup(other.operand);
    if (summary.overlaps(incoming, value))
      continue;
    if (!isDefinedInside(*other.source, incoming))
      return false;
  }
  return sawAlternative;
}

static bool canAliasExternalJoinIncoming(
    const RegionFlow &flow, const RegionFlow::Branch &branch,
    const RegionFlow::Transfer &transfer, Value value,
    const DenseMap<OpOperand *, Value> &incomingValues,
    const WeightedOverlapSummary &summary, DominanceInfo &dominance) {
  RegType type = cast<RegType>(value.getType());
  return type.getBaseGRF() < 0 &&
         storageDiesAtBranch(flow, summary, value, branch.operation,
                             dominance) &&
         isUniqueJoinIncoming(branch, transfer, value, incomingValues) &&
         alternativesCanOverwrite(flow, branch, transfer, value, incomingValues,
                                  summary);
}

static LogicalResult
repairAcyclicExits(func::FuncOp function, const RegionFlow &flow,
                   const WeightedOverlapSummary &intrinsicSummary,
                   DominanceInfo &dominance) {
  for (const RegionFlow::Branch &branch : flow.getBranches()) {
    WeightedOverlapSummary surroundingSummary;
    if (failed(surroundingSummary.build(function, /*includeRegionAliases=*/true,
                                        branch.operation)))
      return failure();
    DenseMap<Operation *, SmallVector<const RegionFlow::Transfer *, 4>> groups;
    for (const RegionFlow::Transfer &transfer : branch.transfers)
      if (transfer.source && !transfer.target &&
          !flow.isRepetitive(transfer.source) &&
          isa<RegType>(transfer.input.getType()))
        groups[transfer.sourceOperation].push_back(&transfer);

    DenseMap<OpOperand *, Value> incomingValues;
    for (const RegionFlow::Transfer &transfer : branch.transfers)
      incomingValues.try_emplace(transfer.operand, transfer.operand->get());

    for (auto [terminator, transfers] : groups) {
      bool noMask = isa<UniformIfOp>(branch.operation);
      bool crossing = false;
      for (auto [lhsIndex, lhs] : llvm::enumerate(transfers)) {
        Value lhsSource = lhs->operand->get();
        if (!isa<RegType>(lhsSource.getType()))
          continue;
        for (auto [rhsIndex, rhs] : llvm::enumerate(transfers)) {
          if (lhsIndex == rhsIndex || lhsSource == rhs->operand->get())
            continue;
          crossing |= surroundingSummary.overlaps(lhsSource, rhs->input);
        }
      }

      OpBuilder builder(terminator);
      if (crossing) {
        SmallVector<Value, 4> snapshots(transfers.size());
        for (auto [index, transfer] : llvm::enumerate(transfers)) {
          Value source = transfer->operand->get();
          RegType sourceType = dyn_cast<RegType>(source.getType());
          if (!sourceType)
            continue;
          RegType snapshotType = RegType::get(function.getContext(),
                                              sourceType.getWidthDwords(), -1);
          FailureOr<Value> snapshot =
              materializeRegisterCopy(builder, terminator->getLoc(), source,
                                      snapshotType, "branch-snapshot",
                                      /*destinationSub=*/0, noMask);
          if (failed(snapshot))
            return failure();
          snapshots[index] = *snapshot;
        }
        for (auto [index, snapshot] : llvm::enumerate(snapshots)) {
          if (!snapshot)
            continue;
          RegType destinationType =
              cast<RegType>(transfers[index]->input.getType());
          FailureOr<Value> destination =
              materializeRegisterCopy(builder, terminator->getLoc(), snapshot,
                                      destinationType, "branch-yield",
                                      /*destinationSub=*/0, noMask);
          if (failed(destination))
            return failure();
          transfers[index]->operand->set(*destination);
        }
        continue;
      }

      DenseSet<Value> seen;
      for (const RegionFlow::Transfer *transfer : transfers) {
        Value source = incomingValues.lookup(transfer->operand);
        if (!isa<RegType>(source.getType()) ||
            isMarkedCopy(source, "branch-yield"))
          continue;
        bool local = storageIsDefinedInside(intrinsicSummary, *transfer->source,
                                            source, terminator, dominance);
        bool duplicate = !seen.insert(source).second;
        if (local && !duplicate)
          continue;
        if (!duplicate && canAliasExternalJoinIncoming(
                              flow, branch, *transfer, source, incomingValues,
                              intrinsicSummary, dominance))
          continue;
        RegType destinationType = cast<RegType>(transfer->input.getType());
        FailureOr<Value> destination =
            materializeRegisterCopy(builder, terminator->getLoc(), source,
                                    destinationType, "branch-yield",
                                    /*destinationSub=*/0, noMask);
        if (failed(destination))
          return failure();
        transfer->operand->set(*destination);
      }
    }
  }
  return success();
}

static LogicalResult
repairRepetitiveEntries(func::FuncOp function, const RegionFlow &flow,
                        const WeightedOverlapSummary &summary,
                        DominanceInfo &dominance) {
  for (const RegionFlow::Branch &branch : flow.getBranches()) {
    DenseMap<Region *, SmallVector<Value, 4>> previous;
    for (const RegionFlow::Transfer &transfer : branch.transfers) {
      if (transfer.source || !transfer.target ||
          !flow.isRepetitive(transfer.target))
        continue;
      Value init = transfer.operand->get();
      RegType inputType = dyn_cast<RegType>(transfer.input.getType());
      if (!inputType)
        continue;
      bool duplicate =
          llvm::any_of(previous[transfer.target], [&](Value value) {
            return summary.overlaps(value, init);
          });
      bool invariantRead = storageHasUseInside(summary, init, branch.operation,
                                               branch.operation, dominance);
      Region *enclosing = flow.getEnclosingRepetitiveRegion(branch.operation);
      bool nestedRepetitive =
          enclosing && !storageIsDefinedInside(summary, *enclosing, init,
                                               branch.operation, dominance);
      bool liveThrough =
          storageIsLiveAfter(summary, init, branch.operation, dominance) ||
          invariantRead || nestedRepetitive;
      if (duplicate || liveThrough) {
        OpBuilder builder(branch.operation);
        FailureOr<Value> copy = materializeRegisterCopy(
            builder, branch.operation->getLoc(), init, inputType, "loop-init",
            /*destinationSub=*/0, /*noMask=*/true);
        if (failed(copy))
          return failure();
        transfer.operand->set(*copy);
      }
      previous[transfer.target].push_back(init);
    }
  }
  return success();
}

static bool isPreparedBackedge(Value carried) {
  Value snapshot = getCopiedSource(carried);
  return snapshot && getCopiedSource(snapshot);
}

static bool hasInputUseAfterDefinition(const WeightedOverlapSummary &summary,
                                       Region &region, Value input,
                                       Operation *definition) {
  BlockArgument argument = dyn_cast<BlockArgument>(input);
  if (!argument)
    return true;
  Block *block = argument.getOwner();
  Operation *top = definition;
  while (top && top->getBlock() != block)
    top = top->getParentOp();
  if (!top)
    return true;
  return llvm::any_of(summary.getOverlappingValues(argument), [&](Value alias) {
    if (alias != argument) {
      if (BlockArgument aliasArgument = dyn_cast<BlockArgument>(alias)) {
        if (aliasArgument.getOwner() != block)
          return false;
      } else {
        Operation *aliasDefinition = alias.getDefiningOp();
        if (!aliasDefinition || aliasDefinition == definition ||
            !region.isAncestor(aliasDefinition->getParentRegion()))
          return false;
      }
    }
    return llvm::any_of(alias.getUses(), [&](OpOperand &use) {
      Operation *owner = use.getOwner();
      if (owner->hasTrait<OpTrait::IsTerminator>())
        return false;
      Operation *user = owner;
      while (user && user->getBlock() != block)
        user = user->getParentOp();
      return !user || (user != top && top->isBeforeInBlock(user));
    });
  });
}

static LogicalResult
repairRepetitiveCycles(func::FuncOp function, const RegionFlow &flow,
                       const WeightedOverlapSummary &summary) {
  for (const RegionFlow::Branch &branch : flow.getBranches()) {
    DenseMap<Operation *, SmallVector<const RegionFlow::Transfer *, 4>> groups;
    for (const RegionFlow::Transfer &transfer : branch.transfers)
      if (transfer.source && transfer.target &&
          flow.isRepetitive(transfer.target) &&
          flow.mayReach(transfer.target, transfer.source) &&
          isa<RegType>(transfer.input.getType()))
        groups[transfer.sourceOperation].push_back(&transfer);
    for (auto [terminator, transfers] : groups) {
      bool hazardous = false;
      for (auto [index, transfer] : llvm::enumerate(transfers)) {
        Value carried = transfer->operand->get();
        if (!isa<RegType>(carried.getType()))
          continue;
        if (carried != transfer->input && !isPreparedBackedge(carried)) {
          Operation *definition = carried.getDefiningOp();
          if (!definition ||
              !transfer->source->isAncestor(definition->getParentRegion()) ||
              hasInputUseAfterDefinition(summary, *transfer->source,
                                         transfer->input, definition))
            hazardous = true;
        }
        for (auto [otherIndex, other] : llvm::enumerate(transfers)) {
          if (otherIndex != index && summary.overlaps(carried, other->input))
            hazardous = true;
        }
        for (const RegionFlow::Transfer *earlier :
             ArrayRef(transfers).take_front(index))
          if (isa<RegType>(earlier->operand->get().getType()) &&
              summary.overlaps(carried, earlier->operand->get()))
            hazardous = true;
      }
      if (!hazardous)
        continue;

      bool alreadyPrepared =
          llvm::all_of(transfers, [&](const RegionFlow::Transfer *transfer) {
            Value carried = transfer->operand->get();
            return !isa<RegType>(carried.getType()) ||
                   isPreparedBackedge(carried);
          });
      if (alreadyPrepared)
        continue;

      // Complete every read before emitting any write tied to a destination.
      SmallVector<Value, 4> snapshots(transfers.size());
      OpBuilder builder(terminator);
      for (auto [index, transfer] : llvm::enumerate(transfers)) {
        Value carried = transfer->operand->get();
        RegType carriedType = dyn_cast<RegType>(carried.getType());
        if (!carriedType)
          continue;
        RegType snapshotType = RegType::get(function.getContext(),
                                            carriedType.getWidthDwords(), -1);
        FailureOr<Value> snapshot =
            materializeRegisterCopy(builder, terminator->getLoc(), carried,
                                    snapshotType, "loop-snapshot");
        if (failed(snapshot))
          return failure();
        snapshots[index] = *snapshot;
      }

      for (auto [index, snapshot] : llvm::enumerate(snapshots)) {
        if (!snapshot)
          continue;
        RegType destinationType =
            cast<RegType>(transfers[index]->input.getType());
        FailureOr<Value> destination =
            materializeRegisterCopy(builder, terminator->getLoc(), snapshot,
                                    destinationType, "loop-backedge");
        if (failed(destination))
          return failure();
        transfers[index]->operand->set(*destination);
      }
    }
  }
  return success();
}

} // namespace

bool inter::xemachine::hasPreparedUpdateBaseCopy(UpdateTupleOp update) {
  return update && isMarkedCopy(update.getBase(), "update-base");
}

LogicalResult
inter::xemachine::prepareRegisterAllocation(func::FuncOp function) {
  if (!function)
    return failure();

  legalizeWideImmediates(function);
  if (failed(repairTupleSlots(function)))
    return failure();
  if (failed(repairUpdateTuples(function)))
    return failure();

  WeightedOverlapSummary intrinsicSummary;
  if (failed(intrinsicSummary.build(function, /*includeRegionAliases=*/false)))
    return failure();
  RegionFlow regionFlow(function);
  {
    DominanceInfo dominance(function);
    if (failed(repairRepetitiveEntries(function, regionFlow, intrinsicSummary,
                                       dominance)))
      return failure();
  }

  WeightedOverlapSummary exitSummary;
  if (failed(exitSummary.build(function, /*includeRegionAliases=*/false)))
    return failure();
  RegionFlow updatedEntryFlow(function);
  {
    DominanceInfo dominance(function);
    if (failed(repairAcyclicExits(function, updatedEntryFlow, exitSummary,
                                  dominance)))
      return failure();
  }

  WeightedOverlapSummary backedgeSummary;
  if (failed(backedgeSummary.build(function, /*includeRegionAliases=*/false)))
    return failure();
  RegionFlow updatedFlow(function);
  return repairRepetitiveCycles(function, updatedFlow, backedgeSummary);
}
