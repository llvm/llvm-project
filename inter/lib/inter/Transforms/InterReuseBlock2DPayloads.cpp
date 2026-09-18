// Reuse Xe2 block2D payloads through immediate X and Y send offsets.

#include "inter/Dialect/XeMachine/IR/XeMachine.h"
#include "inter/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <array>
#include <optional>

namespace inter {
#define GEN_PASS_DEF_REUSEBLOCK2DPAYLOADS
#include "inter/Transforms/Passes.h.inc"
} // namespace inter

using namespace mlir;
using namespace inter::xemachine;

namespace {

struct AffineCoordinate {
  Value base;
  int64_t offset;
};

static bool areEquivalentValues(Value lhs, Value rhs,
                                DenseSet<std::pair<Value, Value>> &visiting) {
  if (lhs == rhs)
    return true;
  Operation *lhsDefinition = lhs.getDefiningOp();
  Operation *rhsDefinition = rhs.getDefiningOp();
  if (!lhsDefinition || !rhsDefinition || lhsDefinition->getNumResults() != 1 ||
      rhsDefinition->getNumResults() != 1 ||
      lhsDefinition->getNumRegions() != 0 ||
      rhsDefinition->getNumRegions() != 0 ||
      !isMemoryEffectFree(lhsDefinition) || !isMemoryEffectFree(rhsDefinition))
    return false;
  std::pair<Value, Value> pair{lhs, rhs};
  if (!visiting.insert(pair).second)
    return true;
  bool equivalent = OperationEquivalence::isEquivalentTo(
      lhsDefinition, rhsDefinition,
      [&](Value lhsOperand, Value rhsOperand) {
        return success(areEquivalentValues(lhsOperand, rhsOperand, visiting));
      },
      nullptr,
      OperationEquivalence::IgnoreLocations |
          OperationEquivalence::IgnoreDiscardableAttrs);
  visiting.erase(pair);
  return equivalent;
}

static bool areEquivalentValues(Value lhs, Value rhs) {
  DenseSet<std::pair<Value, Value>> visiting;
  return areEquivalentValues(lhs, rhs, visiting);
}

static std::optional<int64_t> getImmediate(Value value) {
  if (ImmOp immediate = value.getDefiningOp<ImmOp>())
    return immediate.getValue();
  return std::nullopt;
}

static bool isCanonicalDestination(std::optional<DstRegionAttr> region) {
  return region && region->getHstride() == 1;
}

static bool isUniformSource(std::optional<RegionAttr> region) {
  return region && region->getVstride() == 0 && region->getWidth() == 1 &&
         region->getHstride() == 0;
}

static bool isCoordinateInteger(Type type) {
  return type.isInteger(32) || type.isInteger(64);
}

static bool isScalarIntegerMove(MovOp move) {
  if (move.getExecSize() != 1 || !move.getElemType().isInteger(32) ||
      !move.getNoMask() || move.getMaskOffset() != 0 || move.getDstSub() ||
      move.getSrc0Sub() || !isCanonicalDestination(move.getDstRegion()) ||
      !isUniformSource(move.getSrc0Region()))
    return false;
  std::optional<Type> explicitSourceType = move.getSrc0Type();
  if (explicitSourceType && !explicitSourceType->isInteger(32) &&
      !explicitSourceType->isInteger(64))
    return false;
  RegType resultType = dyn_cast<RegType>(move.getDst().getType());
  RegType sourceType = dyn_cast<RegType>(move.getSrc().getType());
  return resultType && sourceType && resultType.getWidthDwords() <= 2 &&
         sourceType.getWidthDwords() <= 2;
}

static std::optional<AffineCoordinate> decomposeCoordinate(Value value) {
  if (MovOp move = value.getDefiningOp<MovOp>())
    if (isScalarIntegerMove(move))
      return decomposeCoordinate(move.getSrc());

  if (AddOp add = value.getDefiningOp<AddOp>()) {
    if (add.getExecSize() == 1 && isCoordinateInteger(add.getElemType()) &&
        add.getNoMask() && add.getMaskOffset() == 0 && !add.getDstSub() &&
        !add.getSrc0Sub() && !add.getSrc1Sub() && !add.getSrc0Type() &&
        !add.getSrc1Type() && isCanonicalDestination(add.getDstRegion())) {
      Value source;
      std::optional<int64_t> immediate = getImmediate(add.getSrc0());
      if (immediate && isUniformSource(add.getSrc1Region()))
        source = add.getSrc1();
      else {
        immediate = getImmediate(add.getSrc1());
        if (immediate && isUniformSource(add.getSrc0Region()))
          source = add.getSrc0();
      }
      if (immediate) {
        std::optional<AffineCoordinate> coordinate =
            decomposeCoordinate(source);
        if (coordinate && !llvm::AddOverflow(coordinate->offset, *immediate,
                                             coordinate->offset))
          return coordinate;
      }
    }
  }
  return AffineCoordinate{value, 0};
}

static std::optional<Value> getTupleField(Value tuple, int64_t field) {
  UpdateTupleOp update = tuple.getDefiningOp<UpdateTupleOp>();
  if (!update)
    return std::nullopt;
  for (auto [offsetAttribute, value] :
       llvm::zip_equal(update.getOffsets(), update.getUpdates())) {
    int64_t offset = cast<IntegerAttr>(offsetAttribute).getInt();
    int64_t width = cast<RegType>(value.getType()).getWidthDwords();
    if (field == offset)
      return value;
    if (field >= offset && field < offset + width)
      return std::nullopt;
  }
  return getTupleField(update.getBase(), field);
}

static bool hasEquivalentInvariantFields(Value lhs, Value rhs) {
  constexpr std::array<int64_t, 5> invariantFields = {0, 2, 3, 4, 7};
  for (int64_t field : invariantFields) {
    std::optional<Value> lhsValue = getTupleField(lhs, field);
    std::optional<Value> rhsValue = getTupleField(rhs, field);
    if (!lhsValue || !rhsValue || !areEquivalentValues(*lhsValue, *rhsValue))
      return false;
  }
  return true;
}

static std::optional<int64_t> getCoordinateDelta(Value reference,
                                                 Value candidate) {
  std::optional<AffineCoordinate> referenceCoordinate =
      decomposeCoordinate(reference);
  std::optional<AffineCoordinate> candidateCoordinate =
      decomposeCoordinate(candidate);
  if (!referenceCoordinate || !candidateCoordinate ||
      !areEquivalentValues(referenceCoordinate->base,
                           candidateCoordinate->base))
    return std::nullopt;
  int64_t delta;
  if (llvm::SubOverflow(candidateCoordinate->offset,
                        referenceCoordinate->offset, delta))
    return std::nullopt;
  return delta;
}

static bool isLegalOffset(int64_t offset) {
  return offset >= -512 && offset <= 511;
}

static bool hasXOffsetWorkaround(int64_t offset) {
  return (static_cast<uint64_t>(offset) & 0xf) == 0xb;
}

static std::optional<uint32_t> getExdesc(Value reference, Value candidate,
                                         uint32_t descriptor) {
  std::optional<Value> referenceX = getTupleField(reference, 5);
  std::optional<Value> referenceY = getTupleField(reference, 6);
  std::optional<Value> candidateX = getTupleField(candidate, 5);
  std::optional<Value> candidateY = getTupleField(candidate, 6);
  if (!referenceX || !referenceY || !candidateX || !candidateY)
    return std::nullopt;
  std::optional<int64_t> x = getCoordinateDelta(*referenceX, *candidateX);
  std::optional<int64_t> y = getCoordinateDelta(*referenceY, *candidateY);
  if (!x || !y || !isLegalOffset(*x) || !isLegalOffset(*y) ||
      hasXOffsetWorkaround(*x))
    return std::nullopt;
  unsigned dataSizeEncoding = (descriptor >> 9) & 0x7;
  if (dataSizeEncoding > 3)
    return std::nullopt;
  unsigned dataSizeBits = 8u << dataSizeEncoding;
  if ((*x * dataSizeBits) % 32 != 0 || (*y * dataSizeBits) % 32 != 0)
    return std::nullopt;
  return ((static_cast<uint32_t>(*x) & 0x3ff) << 12) |
         ((static_cast<uint32_t>(*y) & 0x3ff) << 22);
}

static bool isEligibleBlock2D(SendOp send) {
  constexpr uint32_t operationMask = 0x3f;
  constexpr uint32_t loadBlock2D = 0x03;
  constexpr uint32_t addressTypeMask = 0x3;
  constexpr unsigned addressTypeShift = 29;
  return send.getFn() == SendFn::ugm &&
         (static_cast<uint32_t>(send.getDesc()) & operationMask) ==
             loadBlock2D &&
         ((static_cast<uint32_t>(send.getDesc()) >> addressTypeShift) &
          addressTypeMask) == 0 &&
         send.getExdesc() == 0 && !send.getExdescReg() &&
         send.getAddrPayload().getDefiningOp<UpdateTupleOp>();
}

static void eraseDeadProducerTree(Value value) {
  Operation *root = value.getDefiningOp();
  if (!root || !isOpTriviallyDead(root))
    return;
  SmallVector<Operation *> worklist{root};
  DenseSet<Operation *> erased;
  while (!worklist.empty()) {
    Operation *operation = worklist.pop_back_val();
    if (erased.contains(operation) || !isOpTriviallyDead(operation))
      continue;
    SmallVector<Operation *> producers;
    for (Value operand : operation->getOperands())
      if (Operation *producer = operand.getDefiningOp())
        producers.push_back(producer);
    operation->erase();
    erased.insert(operation);
    for (Operation *producer : producers)
      if (!erased.contains(producer) && isOpTriviallyDead(producer))
        worklist.push_back(producer);
  }
}

static void reusePayloads(Block &block) {
  SmallVector<SendOp> sends;
  for (Operation &operation : block)
    if (SendOp send = dyn_cast<SendOp>(operation);
        send && isEligibleBlock2D(send))
      sends.push_back(send);

  DenseSet<Operation *> rewritten;
  for (SendOp reference : sends) {
    if (rewritten.contains(reference))
      continue;
    Value referencePayload = reference.getAddrPayload();
    for (SendOp candidate : sends) {
      if (candidate == reference || rewritten.contains(candidate) ||
          !reference->isBeforeInBlock(candidate))
        continue;
      Value candidatePayload = candidate.getAddrPayload();
      if (!hasEquivalentInvariantFields(referencePayload, candidatePayload))
        continue;
      std::optional<uint32_t> exdesc =
          getExdesc(referencePayload, candidatePayload,
                    static_cast<uint32_t>(candidate.getDesc()));
      if (!exdesc)
        continue;
      candidate.getAddrPayloadMutable().set(referencePayload);
      candidate.setExdesc(static_cast<int32_t>(*exdesc));
      rewritten.insert(candidate);
      eraseDeadProducerTree(candidatePayload);
    }
  }
}

class ReuseBlock2DPayloadsPass
    : public inter::impl::ReuseBlock2DPayloadsBase<ReuseBlock2DPayloadsPass> {
public:
  void runOnOperation() override {
    getOperation().walk([&](Block *block) { reusePayloads(*block); });
  }
};

} // namespace
