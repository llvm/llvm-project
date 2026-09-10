#include "inter/Dialect/Inter/IR/XW.h"
#include "inter/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

namespace inter {
#define GEN_PASS_DEF_EXPANDARITHMETIC
#include "inter/Transforms/Passes.h.inc"
} // namespace inter

using namespace mlir;

namespace {

static Type getElementType(Type type) {
  if (xw::SimdType simd = dyn_cast<xw::SimdType>(type))
    return simd.getElementType();
  return type;
}

static Value constant(OpBuilder &builder, Location location, Type type,
                      const APInt &value) {
  Type elementType = getElementType(type);
  Value scalar = xw::ConstantOp::create(builder, location, elementType,
                                        IntegerAttr::get(elementType, value));
  if (isa<xw::SimdType>(type))
    return xw::SplatOp::create(builder, location, type, scalar);
  return scalar;
}

static Value binary(OpBuilder &builder, Location location, Type type,
                    xw::BinaryKind kind, Value lhs, Value rhs) {
  return xw::BinaryOp::create(builder, location, type, kind, lhs, rhs);
}

static Value matchShape(OpBuilder &builder, Location location, Type type,
                        Value value) {
  if (value.getType() == type)
    return value;
  if (value.getType() != cast<xw::SimdType>(type).getElementType())
    llvm_unreachable("verified XW binary operand must be broadcast-compatible");
  return xw::SplatOp::create(builder, location, type, value);
}

static std::pair<Value, Value> expandUnsignedDivRem(OpBuilder &builder,
                                                    xw::BinaryOp operation,
                                                    Value dividend,
                                                    Value divisor) {
  Location location = operation.getLoc();
  Type type = operation.getType();
  IntegerType elementType = cast<IntegerType>(getElementType(type));
  unsigned width = elementType.getWidth();
  Value zero = constant(builder, location, type, APInt(width, 0));
  Value one = constant(builder, location, type, APInt(width, 1));
  Value quotient = zero;
  Value remainder = zero;
  Type conditionType = builder.getI1Type();
  if (xw::SimdType simd = dyn_cast<xw::SimdType>(type))
    conditionType = xw::MaskType::get(type.getContext(), simd.getCardinality());

  for (unsigned step = 0; step < width; ++step) {
    unsigned bit = width - step - 1;
    Value bitValue = constant(builder, location, type, APInt(width, bit));
    Value quotientBit =
        constant(builder, location, type, APInt::getOneBitSet(width, bit));
    Value incoming = binary(builder, location, type, xw::BinaryKind::AndI,
                            binary(builder, location, type,
                                   xw::BinaryKind::ShRUI, dividend, bitValue),
                            one);
    Value extendedRemainder = binary(
        builder, location, type, xw::BinaryKind::OrI,
        binary(builder, location, type, xw::BinaryKind::ShLI, remainder, one),
        incoming);
    Value reduced = binary(builder, location, type, xw::BinaryKind::SubI,
                           extendedRemainder, divisor);
    Value condition = xw::CmpIOp::create(builder, location, conditionType,
                                         arith::CmpIPredicate::uge,
                                         extendedRemainder, divisor);
    remainder = xw::SelectOp::create(builder, location, type, condition,
                                     reduced, extendedRemainder);
    Value withBit = binary(builder, location, type, xw::BinaryKind::OrI,
                           quotient, quotientBit);
    quotient = xw::SelectOp::create(builder, location, type, condition, withBit,
                                    quotient);
  }
  return {quotient, remainder};
}

static Value signedNegative(OpBuilder &builder, Location location, Type type,
                            Value value) {
  IntegerType elementType = cast<IntegerType>(getElementType(type));
  Type conditionType = builder.getI1Type();
  if (xw::SimdType simd = dyn_cast<xw::SimdType>(type))
    conditionType = xw::MaskType::get(type.getContext(), simd.getCardinality());
  Value zero =
      constant(builder, location, type, APInt(elementType.getWidth(), 0));
  return xw::CmpIOp::create(builder, location, conditionType,
                            arith::CmpIPredicate::slt, value, zero);
}

static LogicalResult expandDivision(xw::BinaryOp operation) {
  if (operation.getKind() != xw::BinaryKind::DivUI &&
      operation.getKind() != xw::BinaryKind::RemUI &&
      operation.getKind() != xw::BinaryKind::DivSI &&
      operation.getKind() != xw::BinaryKind::RemSI)
    return failure();
  IntegerType elementType =
      dyn_cast<IntegerType>(getElementType(operation.getType()));
  if (!elementType || elementType.getWidth() > 64)
    return operation.emitOpError(
        "integer division supports element widths up to 64 bits");
  if (xw::SimdType simd = dyn_cast<xw::SimdType>(operation.getType());
      simd && simd.getCardinality() == 32 && elementType.getWidth() == 64)
    return operation.emitOpError(
        "SIMD32 i64 division/remainder has no exact two-half flag selection");

  OpBuilder builder(operation);
  Location location = operation.getLoc();
  Type type = operation.getType();
  Value dividend = matchShape(builder, location, type, operation.getLhs());
  Value divisor = matchShape(builder, location, type, operation.getRhs());
  bool isSigned = operation.getKind() == xw::BinaryKind::DivSI ||
                  operation.getKind() == xw::BinaryKind::RemSI;
  Value dividendNegative;
  Value quotientNegative;
  if (isSigned) {
    unsigned width = elementType.getWidth();
    Value zero = constant(builder, location, type, APInt(width, 0));
    dividendNegative = signedNegative(builder, location, type, dividend);
    Value divisorNegative = signedNegative(builder, location, type, divisor);
    Value negatedDividend =
        binary(builder, location, type, xw::BinaryKind::SubI, zero, dividend);
    Value negatedDivisor =
        binary(builder, location, type, xw::BinaryKind::SubI, zero, divisor);
    dividend = xw::SelectOp::create(builder, location, type, dividendNegative,
                                    negatedDividend, dividend);
    divisor = xw::SelectOp::create(builder, location, type, divisorNegative,
                                   negatedDivisor, divisor);
    if (isa<xw::MaskType>(dividendNegative.getType()))
      quotientNegative =
          xw::MaskXorOp::create(builder, location, dividendNegative.getType(),
                                dividendNegative, divisorNegative);
    else
      quotientNegative =
          binary(builder, location, dividendNegative.getType(),
                 xw::BinaryKind::XOrI, dividendNegative, divisorNegative);
  }

  std::pair<Value, Value> result =
      expandUnsignedDivRem(builder, operation, dividend, divisor);
  Value selected = operation.getKind() == xw::BinaryKind::DivUI ||
                           operation.getKind() == xw::BinaryKind::DivSI
                       ? result.first
                       : result.second;
  if (isSigned) {
    unsigned width = elementType.getWidth();
    Value zero = constant(builder, location, type, APInt(width, 0));
    Value sign = operation.getKind() == xw::BinaryKind::DivSI
                     ? quotientNegative
                     : dividendNegative;
    Value negated =
        binary(builder, location, type, xw::BinaryKind::SubI, zero, selected);
    selected =
        xw::SelectOp::create(builder, location, type, sign, negated, selected);
  }
  operation.replaceAllUsesWith(selected);
  operation.erase();
  return success();
}

class ExpandArithmetic
    : public inter::impl::ExpandArithmeticBase<ExpandArithmetic> {
public:
  void runOnOperation() override {
    SmallVector<xw::BinaryOp> divisions;
    getOperation().walk([&](xw::BinaryOp operation) {
      if (operation.getKind() == xw::BinaryKind::DivUI ||
          operation.getKind() == xw::BinaryKind::RemUI ||
          operation.getKind() == xw::BinaryKind::DivSI ||
          operation.getKind() == xw::BinaryKind::RemSI)
        divisions.push_back(operation);
    });
    for (xw::BinaryOp operation : divisions)
      if (failed(expandDivision(operation)))
        return signalPassFailure();
  }
};

} // namespace
