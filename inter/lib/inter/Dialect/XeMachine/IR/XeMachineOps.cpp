#include "inter/Dialect/XeMachine/IR/XeMachine.h"

#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>

using namespace mlir;
using namespace inter::xemachine;

#define GET_OP_CLASSES
#include "inter/Dialect/XeMachine/IR/XeMachineOps.cpp.inc"

#define GET_OP_INTERFACE_CLASSES
#include "inter/Dialect/XeMachine/IR/XeMachineInterfaces.cpp.inc"

#define DEFINE_ALU_COMMON(Op)                                                  \
  Type Op::getInstructionElementType() { return getElemType(); }               \
  unsigned Op::getExecutionSize() { return getExecSize(); }

#define DEFINE_ALU_DESTINATION(Op)                                             \
  DstRegionAttr Op::getDestinationRegion() { return getDstRegionAttr(); }      \
  int64_t Op::getDestinationSubregister() {                                    \
    return getDstSubAttr() ? getDstSubAttr().getInt() : 0;                     \
  }                                                                            \
  LogicalResult Op::setDestinationSubregister(int64_t subregister) {           \
    setDstSubAttr(                                                             \
        IntegerAttr::get(IntegerType::get(getContext(), 32), subregister));    \
    return success();                                                          \
  }

#define DEFINE_SOURCE_SUBREGISTER(Op)                                          \
  int64_t Op::getSourceSubregister(unsigned index) {                           \
    if (index == 0)                                                            \
      return getSrc0SubAttr() ? getSrc0SubAttr().getInt() : 0;                 \
    return 0;                                                                  \
  }

#define DEFINE_SWSB_INTERFACE(Op)                                              \
  FinalSWSB Op::getFinalSWSB() {                                               \
    return {getSwsbPipe(), static_cast<int32_t>(getSwsbDistance()),            \
            static_cast<int32_t>(getSwsbToken()), getSwsbTokenMode()};         \
  }                                                                            \
  void Op::setFinalSWSB(const FinalSWSB &swsb) {                               \
    setSwsbPipe(swsb.pipe);                                                    \
    setSwsbDistance(swsb.distance);                                            \
    setSwsbToken(swsb.token);                                                  \
    setSwsbTokenMode(swsb.tokenMode);                                          \
  }

#define DEFINE_UNARY_ALU_INTERFACE(Op)                                         \
  DEFINE_ALU_COMMON(Op)                                                        \
  DEFINE_ALU_DESTINATION(Op)                                                   \
  DEFINE_SOURCE_SUBREGISTER(Op)                                                \
  RegionAttr Op::getSourceRegion(unsigned index) {                             \
    return index == 0 ? getSrc0RegionAttr() : RegionAttr();                    \
  }                                                                            \
  void Op::setSourceRegion(unsigned index, RegionAttr region) {                \
    if (index == 0)                                                            \
      setSrc0RegionAttr(region);                                               \
  }                                                                            \
  std::optional<Type> Op::getExplicitSourceElementType(unsigned index) {       \
    return index == 0 ? getSrc0Type() : std::nullopt;                          \
  }

#define DEFINE_BINARY_ALU_INTERFACE(Op)                                        \
  DEFINE_ALU_COMMON(Op)                                                        \
  DEFINE_ALU_DESTINATION(Op)                                                   \
  RegionAttr Op::getSourceRegion(unsigned index) {                             \
    if (index == 0)                                                            \
      return getSrc0RegionAttr();                                              \
    return index == 1 ? getSrc1RegionAttr() : RegionAttr();                    \
  }                                                                            \
  void Op::setSourceRegion(unsigned index, RegionAttr region) {                \
    if (index == 0)                                                            \
      setSrc0RegionAttr(region);                                               \
    else if (index == 1)                                                       \
      setSrc1RegionAttr(region);                                               \
  }                                                                            \
  std::optional<Type> Op::getExplicitSourceElementType(unsigned index) {       \
    if (index == 0)                                                            \
      return getSrc0Type();                                                    \
    return index == 1 ? getSrc1Type() : std::nullopt;                          \
  }                                                                            \
  int64_t Op::getSourceSubregister(unsigned index) {                           \
    if (index == 0)                                                            \
      return getSrc0SubAttr() ? getSrc0SubAttr().getInt() : 0;                 \
    if (index == 1)                                                            \
      return getSrc1SubAttr() ? getSrc1SubAttr().getInt() : 0;                 \
    return 0;                                                                  \
  }

DEFINE_UNARY_ALU_INTERFACE(MovOp)

Speculation::Speculatability MovOp::getSpeculatability() {
  return getNoMask() ? Speculation::Speculatable : Speculation::NotSpeculatable;
}
DEFINE_BINARY_ALU_INTERFACE(AddOp)
DEFINE_BINARY_ALU_INTERFACE(SubOp)
DEFINE_BINARY_ALU_INTERFACE(ShlOp)
DEFINE_BINARY_ALU_INTERFACE(ShrOp)
DEFINE_BINARY_ALU_INTERFACE(AndOp)
DEFINE_BINARY_ALU_INTERFACE(OrOp)
DEFINE_BINARY_ALU_INTERFACE(MulOp)

DEFINE_SWSB_INTERFACE(MovOp)
DEFINE_SWSB_INTERFACE(AddOp)
DEFINE_SWSB_INTERFACE(SubOp)
DEFINE_SWSB_INTERFACE(ShlOp)
DEFINE_SWSB_INTERFACE(ShrOp)
DEFINE_SWSB_INTERFACE(AndOp)
DEFINE_SWSB_INTERFACE(OrOp)
DEFINE_SWSB_INTERFACE(MulOp)
DEFINE_SWSB_INTERFACE(CmpOp)
DEFINE_SWSB_INTERFACE(SendOp)
DEFINE_SWSB_INTERFACE(SyncOp)
DEFINE_SWSB_INTERFACE(LoadA64Op)
DEFINE_SWSB_INTERFACE(StoreA64Op)
DEFINE_SWSB_INTERFACE(LoadSLMOp)
DEFINE_SWSB_INTERFACE(StoreSLMOp)
DEFINE_SWSB_INTERFACE(AtomicIAddA64Op)
DEFINE_SWSB_INTERFACE(LoadBlockA32Op)
DEFINE_SWSB_INTERFACE(FenceSLMOp)
DEFINE_SWSB_INTERFACE(FenceAwaitOp)
DEFINE_SWSB_INTERFACE(BarrierSignalOp)
DEFINE_SWSB_INTERFACE(EotOp)
DEFINE_SWSB_INTERFACE(DpasOp)

LogicalResult SyncOp::verify() {
  if (getSbidMask() != 0 && getKind() != SyncKind::allwr)
    return emitOpError("selective SBID mask requires allwr");
  FinalSWSB swsb = getFinalSWSB();
  if (swsb.token >= 0 && (getKind() != SyncKind::nop ||
                          (swsb.tokenMode != SWSBTokenMode::source &&
                           swsb.tokenMode != SWSBTokenMode::destination)))
    return emitOpError(
        "token wait requires sync.nop with source or destination mode");
  if (swsb.token < 0 && swsb.tokenMode != SWSBTokenMode::none)
    return emitOpError("token mode requires a nonnegative token");
  return success();
}

#define DEFINE_SEND_SCOREBOARD_INTERFACE(Op)                                   \
  AsyncScoreboardKind Op::getAsyncScoreboardKind() {                           \
    return AsyncScoreboardKind::send;                                          \
  }                                                                            \
  bool Op::hasAsyncDestination() {                                             \
    return llvm::any_of((*this)->getResultTypes(), [](Type type) {             \
      RegType reg = dyn_cast<RegType>(type);                                   \
      return reg && reg.getWidthDwords() != 0;                                 \
    });                                                                        \
  }

DEFINE_SEND_SCOREBOARD_INTERFACE(SendOp)
DEFINE_SEND_SCOREBOARD_INTERFACE(LoadA64Op)
DEFINE_SEND_SCOREBOARD_INTERFACE(StoreA64Op)
DEFINE_SEND_SCOREBOARD_INTERFACE(LoadSLMOp)
DEFINE_SEND_SCOREBOARD_INTERFACE(StoreSLMOp)
DEFINE_SEND_SCOREBOARD_INTERFACE(AtomicIAddA64Op)
DEFINE_SEND_SCOREBOARD_INTERFACE(LoadBlockA32Op)
DEFINE_SEND_SCOREBOARD_INTERFACE(FenceSLMOp)
DEFINE_SEND_SCOREBOARD_INTERFACE(BarrierSignalOp)
DEFINE_SEND_SCOREBOARD_INTERFACE(EotOp)

AsyncScoreboardKind DpasOp::getAsyncScoreboardKind() {
  return AsyncScoreboardKind::dpas;
}
bool DpasOp::hasAsyncDestination() { return true; }

#define DEFINE_TERNARY_ALU_INTERFACE(Op)                                       \
  DEFINE_ALU_COMMON(Op)                                                        \
  DEFINE_ALU_DESTINATION(Op)                                                   \
  RegionAttr Op::getSourceRegion(unsigned index) {                             \
    if (index == 0)                                                            \
      return getSrc0RegionAttr();                                              \
    if (index == 1)                                                            \
      return getSrc1RegionAttr();                                              \
    return index == 2 ? getSrc2RegionAttr() : RegionAttr();                    \
  }                                                                            \
  void Op::setSourceRegion(unsigned index, RegionAttr region) {                \
    if (index == 0)                                                            \
      setSrc0RegionAttr(region);                                               \
    else if (index == 1)                                                       \
      setSrc1RegionAttr(region);                                               \
    else if (index == 2)                                                       \
      setSrc2RegionAttr(region);                                               \
  }                                                                            \
  std::optional<Type> Op::getExplicitSourceElementType(unsigned index) {       \
    if (index == 0)                                                            \
      return getSrc0Type();                                                    \
    if (index == 1)                                                            \
      return getSrc1Type();                                                    \
    return index == 2 ? getSrc2Type() : std::nullopt;                          \
  }                                                                            \
  int64_t Op::getSourceSubregister(unsigned index) {                           \
    if (index == 0)                                                            \
      return getSrc0SubAttr() ? getSrc0SubAttr().getInt() : 0;                 \
    if (index == 1)                                                            \
      return getSrc1SubAttr() ? getSrc1SubAttr().getInt() : 0;                 \
    if (index == 2)                                                            \
      return getSrc2SubAttr() ? getSrc2SubAttr().getInt() : 0;                 \
    return 0;                                                                  \
  }

DEFINE_TERNARY_ALU_INTERFACE(Add3Op)
DEFINE_TERNARY_ALU_INTERFACE(CselOp)
DEFINE_SWSB_INTERFACE(Add3Op)
DEFINE_SWSB_INTERFACE(CselOp)

DEFINE_ALU_COMMON(CmpOp)
DstRegionAttr CmpOp::getDestinationRegion() { return {}; }
int64_t CmpOp::getDestinationSubregister() { return 0; }
LogicalResult CmpOp::setDestinationSubregister(int64_t subregister) {
  return success(subregister == 0);
}
RegionAttr CmpOp::getSourceRegion(unsigned index) {
  if (index == 0)
    return getSrc0RegionAttr();
  return index == 1 ? getSrc1RegionAttr() : RegionAttr();
}
void CmpOp::setSourceRegion(unsigned index, RegionAttr region) {
  if (index == 0)
    setSrc0RegionAttr(region);
  else if (index == 1)
    setSrc1RegionAttr(region);
}
std::optional<Type> CmpOp::getExplicitSourceElementType(unsigned index) {
  if (index == 0)
    return getSrc0Type();
  return index == 1 ? getSrc1Type() : std::nullopt;
}
int64_t CmpOp::getSourceSubregister(unsigned index) {
  if (index == 0)
    return getSrc0SubAttr() ? getSrc0SubAttr().getInt() : 0;
  if (index == 1)
    return getSrc1SubAttr() ? getSrc1SubAttr().getInt() : 0;
  return 0;
}

#undef DEFINE_UNARY_ALU_INTERFACE
#undef DEFINE_BINARY_ALU_INTERFACE
#undef DEFINE_TERNARY_ALU_INTERFACE
#undef DEFINE_SOURCE_SUBREGISTER
#undef DEFINE_ALU_DESTINATION
#undef DEFINE_ALU_COMMON
#undef DEFINE_SWSB_INTERFACE
#undef DEFINE_SEND_SCOREBOARD_INTERFACE

FailureOr<KernelResourceUsage>
inter::xemachine::analyzeKernelResources(func::FuncOp function,
                                         int64_t grfCount) {
  KernelResourceUsage usage{};
  auto observeType = [&](Type type, Location location) -> LogicalResult {
    RegType reg = dyn_cast<RegType>(type);
    if (!reg || reg.getWidthDwords() == 0)
      return success();
    if (reg.getBaseGRF() < 0)
      return emitError(location)
             << "resource info requires physical XeMachine registers";
    uint64_t end = static_cast<uint64_t>(reg.getBaseGRF()) +
                   llvm::divideCeil(reg.getWidthDwords(), 16u);
    if (end > static_cast<uint64_t>(grfCount))
      return emitError(location)
             << "physical register range ends at r" << end
             << " but the selected GRF mode has " << grfCount << " registers";
    usage.grfUsed = std::max(usage.grfUsed, end);
    return success();
  };

  WalkResult walk = function.walk([&](Operation *operation) {
    for (Value result : operation->getResults())
      if (failed(observeType(result.getType(), result.getLoc())))
        return WalkResult::interrupt();
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          if (failed(observeType(argument.getType(), argument.getLoc())))
            return WalkResult::interrupt();
    usage.hasGlobalAtomics |= isa<AtomicIAddA64Op>(operation);
    usage.hasDpas |= isa<DpasOp>(operation);
    usage.hasStatelessWrite |= isa<StoreA64Op, AtomicIAddA64Op>(operation);
    if (auto send = dyn_cast<SendOp>(operation))
      usage.hasStatelessWrite |=
          !send->hasAttr(kScratchAccessAttrName) && send.getDataPayload() &&
          (send.getFn() == SendFn::ugm || send.getFn() == SendFn::tgm);
    usage.barrierCount |= isa<BarrierSignalOp>(operation);
    return WalkResult::advance();
  });
  if (walk.wasInterrupted())
    return failure();
  return usage;
}

static void
getTupleElementStorageAliases(Value tuple, ValueRange elements,
                              SmallVectorImpl<RegisterStorageAlias> &aliases) {
  int64_t offset = 0;
  for (Value element : elements) {
    aliases.push_back({tuple, element, offset});
    offset += cast<RegType>(element.getType()).getWidthDwords();
  }
}

void TupleToElementsOp::getRegisterStorageAliases(
    SmallVectorImpl<RegisterStorageAlias> &aliases) {
  getTupleElementStorageAliases(getTuple(), getElements(), aliases);
}

void TupleFromElementsOp::getRegisterStorageAliases(
    SmallVectorImpl<RegisterStorageAlias> &aliases) {
  getTupleElementStorageAliases(getTuple(), getElements(), aliases);
}

void UpdateTupleOp::getRegisterStorageAliases(
    SmallVectorImpl<RegisterStorageAlias> &aliases) {
  aliases.push_back({getResult(), getBase(), 0, /*destructive=*/true});
  for (auto [value, offset] : llvm::zip_equal(getUpdates(), getOffsets()))
    aliases.push_back({getResult(), value, cast<IntegerAttr>(offset).getInt(),
                       /*destructive=*/true});
}

void DpasOp::getRegisterStorageAliases(
    SmallVectorImpl<RegisterStorageAlias> &aliases) {
  aliases.push_back({getDst(), getAcc(), 0, /*destructive=*/true});
}

LogicalResult DpasOp::verify() {
  RegType a = cast<RegType>(getA().getType());
  RegType b = cast<RegType>(getB().getType());
  RegType acc = cast<RegType>(getAcc().getType());
  RegType dst = cast<RegType>(getDst().getType());
  if (a.getWidthDwords() == 0 || b.getWidthDwords() == 0 ||
      acc.getWidthDwords() == 0 || dst.getWidthDwords() != acc.getWidthDwords())
    return emitOpError(
        "requires non-empty A/B packets and matching C/D widths");
  if (!getElemType().isF32())
    return emitOpError("requires an f32 accumulator and result");
  if (getSystolicDepth() != 8)
    return emitOpError("requires systolic depth 8 on Xe2");
  if (getRepeatCount() <= 0 || getRepeatCount() > 8)
    return emitOpError("requires a repeat count from 1 through 8");
  int64_t expectedA = getSystolicDepth() * getRepeatCount();
  int64_t expectedB = getSystolicDepth() * 16;
  int64_t expectedAccumulator = 16 * getRepeatCount();
  if (a.getWidthDwords() != expectedA || b.getWidthDwords() != expectedB ||
      acc.getWidthDwords() != expectedAccumulator)
    return emitOpError("packet widths must match Xe2 depth and repeat count");
  if (acc.getBaseGRF() >= 0 && dst.getBaseGRF() >= 0 &&
      acc.getBaseGRF() != dst.getBaseGRF())
    return emitOpError("physical destination must alias the accumulator");
  return success();
}

static int64_t sumElementWidths(ValueRange elements) {
  int64_t total = 0;
  for (Value element : elements)
    total += cast<RegType>(element.getType()).getWidthDwords();
  return total;
}

static LogicalResult verifyTupleElements(Operation *operation,
                                         RegType tupleType,
                                         ValueRange elements) {
  int64_t total = sumElementWidths(elements);
  if (tupleType.getWidthDwords() != total)
    return operation->emitOpError("element widths sum (")
           << total << ") must match tuple register width ("
           << tupleType.getWidthDwords() << ")";
  int64_t offset = 0;
  for (Value element : elements) {
    RegType elementType = cast<RegType>(element.getType());
    if (tupleType.getBaseGRF() >= 0 && elementType.getBaseGRF() >= 0 &&
        (offset % 16 != 0 ||
         elementType.getBaseGRF() != tupleType.getBaseGRF() + offset / 16))
      return operation->emitOpError(
          "physical element placement must match its tuple offset");
    if (offset % 16 != 0 || elementType.getWidthDwords() % 16 != 0)
      return operation->emitOpError(
          "tuple elements must occupy whole 16-dword GRFs");
    offset += elementType.getWidthDwords();
  }
  return success();
}

LogicalResult TupleToElementsOp::verify() {
  return verifyTupleElements(*this, cast<RegType>(getTuple().getType()),
                             getElements());
}

static bool canFoldTupleJoinSplit(TupleFromElementsOp joined,
                                  ValueRange splitElements) {
  if (joined.getElements().size() != splitElements.size())
    return false;
  RegType tupleType = cast<RegType>(joined.getTuple().getType());
  int64_t offset = 0;
  for (auto [source, result] :
       llvm::zip_equal(joined.getElements(), splitElements)) {
    if (source.getType() != result.getType())
      return false;
    RegType sourceType = cast<RegType>(source.getType());
    if (tupleType.getBaseGRF() >= 0 &&
        (offset % 16 != 0 ||
         sourceType.getBaseGRF() != tupleType.getBaseGRF() + offset / 16))
      return false;
    offset += sourceType.getWidthDwords();
  }
  return true;
}

LogicalResult TupleToElementsOp::fold(FoldAdaptor,
                                      SmallVectorImpl<OpFoldResult> &results) {
  if (getElements().size() == 1 &&
      getElements().front().getType() == getTuple().getType()) {
    results.push_back(getTuple());
    return success();
  }
  TupleFromElementsOp joined = getTuple().getDefiningOp<TupleFromElementsOp>();
  if (!joined || !canFoldTupleJoinSplit(joined, getElements()))
    return failure();
  llvm::append_range(results, joined.getElements());
  return success();
}

LogicalResult TupleFromElementsOp::verify() {
  return verifyTupleElements(*this, cast<RegType>(getTuple().getType()),
                             getElements());
}

static TupleToElementsOp getExactRoundTripSplit(Value element, unsigned index,
                                                size_t resultCount) {
  auto result = dyn_cast<OpResult>(element);
  if (!result || result.getResultNumber() != index)
    return {};
  auto split = dyn_cast_or_null<TupleToElementsOp>(result.getOwner());
  if (!split || split->getNumResults() != resultCount)
    return {};
  return split;
}

static Value getExactRoundTripSource(ValueRange elements) {
  Value sourceTuple;
  for (auto [index, element] : llvm::enumerate(elements)) {
    TupleToElementsOp split =
        getExactRoundTripSplit(element, index, elements.size());
    if (!split)
      return {};
    if (!sourceTuple) {
      sourceTuple = split.getTuple();
      continue;
    }
    if (sourceTuple != split.getTuple())
      return {};
  }
  return sourceTuple;
}

OpFoldResult TupleFromElementsOp::fold(FoldAdaptor) {
  if (getElements().size() == 1 &&
      getElements().front().getType() == getTuple().getType())
    return getElements().front();
  Value sourceTuple = getExactRoundTripSource(getElements());
  if (sourceTuple && sourceTuple.getType() == getTuple().getType())
    return sourceTuple;
  return {};
}

LogicalResult UpdateTupleOp::verify() {
  RegType baseType = cast<RegType>(getBase().getType());
  RegType resultType = cast<RegType>(getResult().getType());
  if (baseType.getWidthDwords() == 0)
    return emitOpError("tuple storage must be non-empty");
  if (baseType.getWidthDwords() != resultType.getWidthDwords())
    return emitOpError("base width must match result width");
  if (baseType.getBaseGRF() >= 0 && resultType.getBaseGRF() >= 0 &&
      baseType.getBaseGRF() != resultType.getBaseGRF())
    return emitOpError("physical base must match result storage");

  ArrayAttr offsets = getOffsets();
  if (offsets.size() != getUpdates().size())
    return emitOpError("offset count must match update count");

  int64_t lastEnd = 0;
  for (auto [offsetAttr, update] : llvm::zip_equal(offsets, getUpdates())) {
    auto offsetInt = dyn_cast<IntegerAttr>(offsetAttr);
    if (!offsetInt)
      return emitOpError("offsets must be integer attributes");
    int64_t offset = offsetInt.getInt();
    if (offset < 0)
      return emitOpError("offsets must be non-negative");
    if (offset < lastEnd)
      return emitOpError("offsets must be sorted and non-overlapping");
    RegType updateType = cast<RegType>(update.getType());
    int64_t end = offset + updateType.getWidthDwords();
    if (end > baseType.getWidthDwords())
      return emitOpError("update exceeds tuple width");
    if (resultType.getBaseGRF() >= 0 && updateType.getBaseGRF() >= 0 &&
        updateType.getBaseGRF() != resultType.getBaseGRF() + offset / 16)
      return emitOpError(
          "physical update placement must match its tuple offset");
    lastEnd = end;
  }
  return success();
}

static LogicalResult verifyA64AddressPayload(Operation *operation,
                                             Value address,
                                             int64_t executionSize) {
  if (executionSize != 8 && executionSize != 16 && executionSize != 32)
    return operation->emitOpError(
        "requires SIMD8, SIMD16, or SIMD32 execution");
  if (cast<RegType>(address.getType()).getWidthDwords() != executionSize * 2)
    return operation->emitOpError(
        "requires two address dwords per execution lane");
  return success();
}

LogicalResult LoadA64Op::verify() {
  return verifyA64AddressPayload(*this, getAddrPayload(), getExecSize());
}

LogicalResult StoreA64Op::verify() {
  return verifyA64AddressPayload(*this, getAddrPayload(), getExecSize());
}

LogicalResult AtomicIAddA64Op::verify() {
  return verifyA64AddressPayload(*this, getAddrPayload(), getExecSize());
}

//===----------------------------------------------------------------------===//
// Structured control flow region modeling.
//
// exec_if: parent -> arms/fallthrough; then -> else/parent; else -> parent.
// uniform_if: parent -> either arm/fallthrough; regions -> parent.
// uniform_loop: parent -> body; body -> body (back-edge) or parent (exit).
// payload_prologue: parent -> body or bypass; body -> parent.
//===----------------------------------------------------------------------===//

void PayloadPrologueOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.emplace_back(&getBody());
    regions.emplace_back(getOperation());
    return;
  }
  regions.emplace_back(getOperation());
}

void ExecIfOp::getSuccessorRegions(RegionBranchPoint point,
                                   SmallVectorImpl<RegionSuccessor> &regions) {
  bool hasElse = !getElseRegion().empty();
  if (point.isParent()) {
    regions.emplace_back(&getThenRegion());
    if (hasElse)
      regions.emplace_back(&getElseRegion());
    else if (getNumResults() == 0)
      regions.emplace_back(getOperation());
    return;
  }

  Region *source = point.getTerminatorPredecessorOrNull()->getParentRegion();
  if (source == &getThenRegion() && hasElse)
    regions.emplace_back(&getElseRegion());
  regions.emplace_back(getOperation());
}

void UniformIfOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.emplace_back(&getThenRegion());
    if (getElseRegion().empty())
      regions.emplace_back(getOperation());
    else
      regions.emplace_back(&getElseRegion());
    return;
  }
  regions.emplace_back(getOperation());
}

void UniformLoopOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.emplace_back(&getBody());
    return;
  }
  regions.emplace_back(&getBody());
  regions.emplace_back(getOperation());
}

SmallVector<Region *> UniformLoopOp::getLoopRegions() {
  SmallVector<Region *> regions{&getBody()};
  getBody().walk([&](Operation *operation) {
    if (isa<UniformLoopOp>(operation))
      return WalkResult::skip();
    for (Region &region : operation->getRegions())
      regions.push_back(&region);
    return WalkResult::advance();
  });
  return regions;
}

MutableArrayRef<OpOperand> UniformLoopOp::getInitsMutable() {
  return getOperation()->getOpOperands();
}

Block::BlockArgListType UniformLoopOp::getRegionIterArgs() {
  return getBody().front().getArguments();
}

std::optional<MutableArrayRef<OpOperand>>
UniformLoopOp::getYieldedValuesMutable() {
  auto terminator = cast<ContinueIfOp>(getBody().front().getTerminator());
  return terminator->getOpOperands().drop_front();
}

std::optional<ResultRange> UniformLoopOp::getLoopResults() {
  return getResults();
}

//===----------------------------------------------------------------------===//
// Operand -> successor-input mapping.
//===----------------------------------------------------------------------===//

ValueRange ExecIfOp::getSuccessorInputs(RegionSuccessor successor) {
  return successor.isOperation() ? ValueRange(getResults()) : ValueRange();
}

ValueRange PayloadPrologueOp::getSuccessorInputs(RegionSuccessor) { return {}; }

ValueRange UniformIfOp::getSuccessorInputs(RegionSuccessor successor) {
  return successor.isOperation() ? ValueRange(getResults()) : ValueRange();
}

ValueRange UniformLoopOp::getSuccessorInputs(RegionSuccessor successor) {
  return successor.isOperation() ? ValueRange(getResults())
                                 : getBody().getArguments();
}

OperandRange
UniformLoopOp::getEntrySuccessorOperands(RegionSuccessor successor) {
  return getInitialValues();
}

MutableOperandRange
YieldOp::getMutableSuccessorOperands(RegionSuccessor successor) {
  MutableOperandRange values = getValuesMutable();
  if (successor.isOperation())
    return values;
  return values.slice(0, 0);
}

MutableOperandRange
ContinueIfOp::getMutableSuccessorOperands(RegionSuccessor point) {
  // Operand 0 is the condition; only carried values flow to successors.
  return MutableOperandRange(getOperation(), /*start=*/1, getNumOperands() - 1);
}

MutableOperandRange
PayloadPrologueEndOp::getMutableSuccessorOperands(RegionSuccessor) {
  return MutableOperandRange(getOperation(), 0, 0);
}
