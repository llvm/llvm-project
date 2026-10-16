#include "inter/Dialect/Inter/IR/XW.h"

#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

#include <array>
#include <optional>

using namespace mlir;
using namespace xw;

namespace {
struct Shape {
  Type elementType;
  std::optional<int64_t> cardinality;
};

struct PointerShape {
  PtrType pointerType;
  std::optional<int64_t> cardinality;
};

static std::optional<int64_t> getTypeCardinality(Type type) {
  if (SimdType simd = dyn_cast<SimdType>(type))
    return simd.getCardinality();
  if (MaskType mask = dyn_cast<MaskType>(type))
    return mask.getCardinality();
  return std::nullopt;
}

static std::optional<int64_t> getEnclosingWidth(Operation *op) {
  for (Operation *scope = op; scope; scope = scope->getParentOp()) {
    IntegerAttr width =
        scope->getAttrOfType<IntegerAttr>(XWDialect::getSimdWidthAttrName());
    if (width)
      return width.getInt();
  }
  return std::nullopt;
}

static LogicalResult verifyCardinalities(Operation *op) {
  bool hasCardinality =
      llvm::any_of(
          op->getOperandTypes(),
          [](Type type) { return getTypeCardinality(type).has_value(); }) ||
      llvm::any_of(op->getResultTypes(), [](Type type) {
        return getTypeCardinality(type).has_value();
      });
  if (!hasCardinality)
    return success();
  std::optional<int64_t> width = getEnclosingWidth(op);
  if (!width)
    return op->emitOpError(
        "with SIMD or mask values requires an enclosing xw.simd_width");
  auto verify = [&](Type type) -> LogicalResult {
    std::optional<int64_t> cardinality = getTypeCardinality(type);
    if (cardinality && *width % *cardinality != 0)
      return op->emitOpError("cardinality ")
             << *cardinality << " does not divide enclosing xw.simd_width "
             << *width;
    return success();
  };
  for (Type type : op->getOperandTypes())
    if (failed(verify(type)))
      return failure();
  for (Type type : op->getResultTypes())
    if (failed(verify(type)))
      return failure();
  return success();
}

static Type getPayloadType(Type type) {
  if (SimdType simd = dyn_cast<SimdType>(type))
    return simd.getElementType();
  if (isa<MaskType>(type))
    return IntegerType::get(type.getContext(), 1);
  return type;
}

static FailureOr<Shape>
classifyInteger(Type type,
                function_ref<InFlightDiagnostic(const Twine &)> emitError) {
  std::optional<int64_t> cardinality;
  if (SimdType simd = dyn_cast<SimdType>(type)) {
    cardinality = simd.getCardinality();
    type = simd.getElementType();
  }
  if (type.isIndex())
    return Shape{type, cardinality};
  IntegerType integer = dyn_cast<IntegerType>(type);
  if (!integer || !integer.isSignless())
    return emitError("expected a signless integer or index type");
  return Shape{integer, cardinality};
}

static FailureOr<Shape>
classifyNumber(Type type,
               function_ref<InFlightDiagnostic(const Twine &)> emitError) {
  std::optional<int64_t> cardinality;
  if (SimdType simd = dyn_cast<SimdType>(type)) {
    cardinality = simd.getCardinality();
    type = simd.getElementType();
  }
  if (type.isIndex())
    return Shape{type, cardinality};
  if (IntegerType integer = dyn_cast<IntegerType>(type)) {
    if (!integer.isSignless())
      return emitError("integer type must be signless");
    return Shape{integer, cardinality};
  }
  if (isa<FloatType>(type))
    return Shape{type, cardinality};
  return emitError(
      "expected a signless integer, index, or floating-point type");
}

static FailureOr<PointerShape>
classifyPointer(Type type,
                function_ref<InFlightDiagnostic(const Twine &)> emitError) {
  std::optional<int64_t> cardinality;
  if (SimdType simd = dyn_cast<SimdType>(type)) {
    cardinality = simd.getCardinality();
    type = simd.getElementType();
  }
  PtrType pointer = dyn_cast<PtrType>(type);
  if (!pointer)
    return emitError("expected an XW pointer or SIMD of XW pointers");
  return PointerShape{pointer, cardinality};
}

static LogicalResult verifySameShape(Operation *op, Type lhs, Type rhs,
                                     Type result) {
  if (lhs != rhs || lhs != result)
    return op->emitOpError("operands and result must have the same type");
  return verifyCardinalities(op);
}

static LogicalResult verifyFloatShape(Operation *op, Type lhs, Type rhs,
                                      Type result) {
  if (failed(verifySameShape(op, lhs, rhs, result)))
    return failure();
  Type element = cast<SimdType>(lhs).getElementType();
  if (!isa<FloatType>(element))
    return op->emitOpError("SIMD element type must be floating-point");
  return success();
}

static LogicalResult verifyDimQuery(Operation *op, int64_t dim, bool varying) {
  if (dim < 0 || dim > 2)
    return op->emitOpError("dimension must be 0, 1, or 2");
  Type result = op->getResult(0).getType();
  if (varying && !isa<SimdType>(result))
    return op->emitOpError("result must be SIMD for a lane-varying query");
  if (!varying && isa<SimdType>(result))
    return op->emitOpError("result must be uniform for a launch query");
  auto emit = [op](const Twine &message) { return op->emitOpError(message); };
  if (failed(classifyInteger(result, emit)))
    return failure();
  return verifyCardinalities(op);
}

static LogicalResult
verifyPointerValueCardinality(Operation *op, Type pointerType, Type valueType) {
  auto emit = [op](const Twine &message) { return op->emitOpError(message); };
  FailureOr<PointerShape> pointer = classifyPointer(pointerType, emit);
  if (failed(pointer))
    return failure();
  std::optional<int64_t> valueCardinality = getTypeCardinality(valueType);
  if (pointer->cardinality &&
      (!valueCardinality || *pointer->cardinality != *valueCardinality))
    return op->emitOpError(
        "SIMD pointer cardinality must match value cardinality");
  return verifyCardinalities(op);
}

static OpFoldResult foldTokenMerge(ValueRange dependencies) {
  if (dependencies.size() == 1)
    return dependencies.front();
  return {};
}

static LogicalResult verifyBlock2D(Operation *operation, Value base,
                                   ValueRange geometry, int64_t blockWidth,
                                   int64_t blockHeight, int64_t blocks,
                                   int64_t elementBits, bool transpose,
                                   bool vnni, Value data, bool write) {
  PtrType pointer = dyn_cast<PtrType>(base.getType());
  if (!pointer || !isa<GlobalAddressSpaceAttr, ConstantAddressSpaceAttr>(
                      pointer.getAddressSpace()))
    return operation->emitOpError(
        "requires a uniform global or constant base pointer");
  for (Value operand : geometry)
    if (!operand.getType().isSignlessInteger(32))
      return operation->emitOpError(
          "surface geometry and coordinates must be uniform i32 values");
  if (blockWidth <= 0 || blockHeight <= 0 || blocks <= 0)
    return operation->emitOpError("block dimensions must be positive");
  if (elementBits != 8 && elementBits != 16 && elementBits != 32 &&
      elementBits != 64)
    return operation->emitOpError("element width must be 8, 16, 32, or 64");
  if (write && (blocks != 1 || transpose || vnni))
    return operation->emitOpError(
        "block2D writes require one untransformed block");
  if ((transpose || vnni) && blocks != 1)
    return operation->emitOpError(
        "transformed block2D reads require one block");
  if (data) {
    SimdType simd = dyn_cast<SimdType>(data.getType());
    VectorType vector =
        simd ? dyn_cast<VectorType>(simd.getElementType()) : VectorType();
    if (!vector || vector.getRank() != 1 || vector.isScalable())
      return operation->emitOpError(
          "data must be a SIMD value with a fixed 1-D vector payload");
  }
  return verifyCardinalities(operation);
}
} // namespace

#define GET_OP_CLASSES
#include "inter/Dialect/Inter/IR/XWOps.cpp.inc"

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  Attribute value;
  Type resultType;
  if (parser.parseAttribute(value, getValueAttrName(result.name),
                            result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseType(resultType))
      return failure();
  } else {
    TypedAttr typed = dyn_cast<TypedAttr>(value);
    if (!typed)
      return parser.emitError(parser.getNameLoc(),
                              "constant value must be typed");
    resultType = typed.getType();
  }
  result.addTypes(resultType);
  return success();
}

void ConstantOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getValue();
  printer.printOptionalAttrDict((*this)->getAttrs(), {getValueAttrName()});
  if (cast<TypedAttr>(getValue()).getType() != getType())
    printer << " -> " << getType();
}

bool ConstantOp::isBuildableWith(Attribute value, Type type) {
  TypedAttr typed = dyn_cast_if_present<TypedAttr>(value);
  return typed && typed.getType() == getPayloadType(type);
}

ConstantOp ConstantOp::materialize(OpBuilder &builder, Attribute value,
                                   Type type, Location loc) {
  if (!isBuildableWith(value, type))
    return nullptr;
  return ConstantOp::create(builder, loc, type, cast<TypedAttr>(value));
}

LogicalResult ConstantOp::verify() {
  if (!isBuildableWith(getValue(), getType()))
    return emitOpError("value type must match result payload type");
  return verifyCardinalities(getOperation());
}

OpFoldResult ConstantOp::fold(FoldAdaptor) { return getValue(); }

void ConstantOp::inferResultRanges(ArrayRef<ConstantIntRanges>,
                                   SetIntRangeFn setResultRange) {
  IntegerAttr value = dyn_cast<IntegerAttr>(getValue());
  Type type = getPayloadType(getType());
  if (!value || !type.isIntOrIndex())
    return;
  unsigned width = ConstantIntRanges::getStorageBitwidth(type);
  if (width == 0)
    return;
  setResultRange(getResult(), ConstantIntRanges::constant(
                                  value.getValue().sextOrTrunc(width)));
}

LogicalResult SplatOp::verify() {
  if (getSource().getType() !=
      cast<SimdType>(getResult().getType()).getElementType())
    return emitOpError("source type must match SIMD element type");
  return verifyCardinalities(getOperation());
}

OpFoldResult SplatOp::fold(FoldAdaptor adaptor) { return adaptor.getSource(); }

LogicalResult ReadFirstOp::verify() {
  if (cast<SimdType>(getSource().getType()).getElementType() !=
      getResult().getType())
    return emitOpError("result type must match SIMD element type");
  return verifyCardinalities(getOperation());
}

OpFoldResult ReadFirstOp::fold(FoldAdaptor adaptor) {
  if (SplatOp splat = getSource().getDefiningOp<SplatOp>())
    return splat.getSource();
  return adaptor.getSource();
}

LogicalResult ExpandOp::verify() {
  SimdType source = dyn_cast<SimdType>(getSource().getType());
  SimdType result = cast<SimdType>(getResult().getType());
  if (!source)
    return emitOpError("source must be SIMD; use xw.splat for uniform values");
  if (source.getElementType() != result.getElementType())
    return emitOpError("source and result element types must match");
  if (result.getCardinality() <= source.getCardinality() ||
      result.getCardinality() % source.getCardinality() != 0)
    return emitOpError(
        "result cardinality must be a larger multiple of source cardinality");
  return verifyCardinalities(getOperation());
}

OpFoldResult ExpandOp::fold(FoldAdaptor adaptor) { return adaptor.getSource(); }

LogicalResult FreezeOp::verify() {
  Type type = getSource().getType();
  if (type != getResult().getType())
    return emitOpError("source and result must have the same type");
  if (SimdType simd = dyn_cast<SimdType>(type))
    type = simd.getElementType();
  IntegerType integer = dyn_cast<IntegerType>(type);
  if ((!integer || !integer.isSignless()) && !type.isIndex() &&
      !isa<FloatType, PtrType>(type))
    return emitOpError(
        "requires a bare or SIMD signless integer, index, floating-point, or "
        "XW pointer payload");
  return verifyCardinalities(getOperation());
}

LogicalResult BinaryOp::verify() {
  if (getOverflowFlags() != arith::IntegerOverflowFlags::none &&
      getKind() != BinaryKind::AddI && getKind() != BinaryKind::SubI &&
      getKind() != BinaryKind::MulI && getKind() != BinaryKind::ShLI)
    return emitOpError(
        "overflow flags require addi, subi, muli, or shli operation");
  auto emit = [this](const Twine &message) { return emitOpError(message); };
  FailureOr<Shape> lhs = classifyInteger(getLhs().getType(), emit);
  FailureOr<Shape> rhs = classifyInteger(getRhs().getType(), emit);
  FailureOr<Shape> result = classifyInteger(getResult().getType(), emit);
  if (failed(lhs) || failed(rhs) || failed(result))
    return failure();
  if (lhs->elementType != rhs->elementType ||
      lhs->elementType != result->elementType)
    return emitOpError("operand and result element types must match");
  if (lhs->cardinality && rhs->cardinality &&
      lhs->cardinality != rhs->cardinality)
    return emitOpError(
        "SIMD operand cardinalities must match; use xw.expand explicitly");
  std::optional<int64_t> expected =
      lhs->cardinality ? lhs->cardinality : rhs->cardinality;
  if (result->cardinality != expected)
    return emitOpError("result shape must match the broadcast operand shape");
  return verifyCardinalities(getOperation());
}

namespace {
static unsigned getIntegerBitWidth(Type type) {
  return ConstantIntRanges::getStorageBitwidth(getPayloadType(type));
}

static ConstantIntRanges normalizeRange(const ConstantIntRanges &range,
                                        unsigned width) {
  if (range.smin().getBitWidth() == width)
    return range;
  return ConstantIntRanges::maxRange(width);
}

static intrange::OverflowFlags
convertOverflowFlags(arith::IntegerOverflowFlags flags) {
  intrange::OverflowFlags result = intrange::OverflowFlags::None;
  if (bitEnumContainsAny(flags, arith::IntegerOverflowFlags::nsw))
    result |= intrange::OverflowFlags::Nsw;
  if (bitEnumContainsAny(flags, arith::IntegerOverflowFlags::nuw))
    result |= intrange::OverflowFlags::Nuw;
  return result;
}
} // namespace

void BinaryOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                 SetIntRangeFn setResultRange) {
  unsigned width = getIntegerBitWidth(getType());
  if (width == 0 || argRanges.size() != 2)
    return;
  std::array<ConstantIntRanges, 2> ranges = {
      normalizeRange(argRanges[0], width), normalizeRange(argRanges[1], width)};
  intrange::OverflowFlags flags = convertOverflowFlags(getOverflowFlags());
  ConstantIntRanges result = ConstantIntRanges::maxRange(width);
  switch (getKind()) {
  case BinaryKind::AddI:
    result = intrange::inferAdd(ranges, flags);
    break;
  case BinaryKind::SubI:
    result = intrange::inferSub(ranges, flags);
    break;
  case BinaryKind::MulI:
    result = intrange::inferMul(ranges, flags);
    break;
  case BinaryKind::ShLI:
    result = intrange::inferShl(ranges, flags);
    break;
  case BinaryKind::ShRUI:
    result = intrange::inferShrU(ranges);
    break;
  case BinaryKind::ShRSI:
    result = intrange::inferShrS(ranges);
    break;
  case BinaryKind::AndI:
    result = intrange::inferAnd(ranges);
    break;
  case BinaryKind::OrI:
    result = intrange::inferOr(ranges);
    break;
  case BinaryKind::XOrI:
    result = intrange::inferXor(ranges);
    break;
  case BinaryKind::DivUI:
    result = intrange::inferDivU(ranges);
    break;
  case BinaryKind::DivSI:
    result = intrange::inferDivS(ranges);
    break;
  case BinaryKind::RemUI:
    result = intrange::inferRemU(ranges);
    break;
  case BinaryKind::RemSI:
    result = intrange::inferRemS(ranges);
    break;
  case BinaryKind::MulHUI:
    break;
  }
  setResultRange(getResult(), result);
}

OpFoldResult BinaryOp::fold(FoldAdaptor adaptor) {
  if (getKind() == BinaryKind::AddI) {
    if (matchPattern(adaptor.getRhs(), m_Zero()) &&
        getLhs().getType() == getType())
      return getLhs();
    if (matchPattern(adaptor.getLhs(), m_Zero()) &&
        getRhs().getType() == getType())
      return getRhs();
  }
  if (getKind() == BinaryKind::SubI && getLhs() == getRhs())
    return Builder(getContext()).getZeroAttr(getPayloadType(getType()));
  if (getKind() == BinaryKind::MulI) {
    if (matchPattern(adaptor.getRhs(), m_One()) &&
        getLhs().getType() == getType())
      return getLhs();
    if (matchPattern(adaptor.getLhs(), m_One()) &&
        getRhs().getType() == getType())
      return getRhs();
  }
  IntegerAttr lhs = dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  IntegerAttr rhs = dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
    return {};
  APInt a = lhs.getValue();
  APInt b = rhs.getValue();
  APInt value = a;
  switch (getKind()) {
  case BinaryKind::AddI:
    value = a + b;
    break;
  case BinaryKind::SubI:
    value = a - b;
    break;
  case BinaryKind::MulI:
    value = a * b;
    break;
  case BinaryKind::AndI:
    value = a & b;
    break;
  case BinaryKind::OrI:
    value = a | b;
    break;
  case BinaryKind::XOrI:
    value = a ^ b;
    break;
  case BinaryKind::ShLI:
    if (b.uge(b.getBitWidth()))
      return {};
    value = a.shl(b);
    break;
  case BinaryKind::ShRUI:
    if (b.uge(b.getBitWidth()))
      return {};
    value = a.lshr(b);
    break;
  case BinaryKind::ShRSI:
    if (b.uge(b.getBitWidth()))
      return {};
    value = a.ashr(b);
    break;
  case BinaryKind::DivUI:
    if (b.isZero())
      return {};
    value = a.udiv(b);
    break;
  case BinaryKind::DivSI: {
    bool overflow = false;
    if (b.isZero())
      return {};
    value = a.sdiv_ov(b, overflow);
    if (overflow)
      return {};
    break;
  }
  case BinaryKind::RemUI:
    if (b.isZero())
      return {};
    value = a.urem(b);
    break;
  case BinaryKind::RemSI:
    if (b.isZero())
      return {};
    value = a.srem(b);
    break;
  case BinaryKind::MulHUI:
    value = llvm::APIntOps::mulhu(a, b);
    break;
  }
  return IntegerAttr::get(getPayloadType(getType()), value);
}

namespace {
struct StrengthReducePowerOfTwo final : OpRewritePattern<BinaryOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryOp operation,
                                PatternRewriter &rewriter) const override {
    BinaryKind replacementKind;
    Value source = operation.getLhs();
    Value constant = operation.getRhs();
    switch (operation.getKind()) {
    case BinaryKind::MulI:
      replacementKind = BinaryKind::ShLI;
      if (!constant.getDefiningOp<ConstantOp>()) {
        std::swap(source, constant);
        if (!constant.getDefiningOp<ConstantOp>())
          return failure();
      }
      break;
    case BinaryKind::DivUI:
      replacementKind = BinaryKind::ShRUI;
      break;
    case BinaryKind::RemUI:
      replacementKind = BinaryKind::AndI;
      break;
    default:
      return failure();
    }

    ConstantOp constantOp = constant.getDefiningOp<ConstantOp>();
    if (!constantOp)
      return failure();
    IntegerAttr valueAttr = dyn_cast<IntegerAttr>(constantOp.getValue());
    if (!valueAttr || !valueAttr.getValue().isPowerOf2())
      return failure();

    APInt replacementValue = replacementKind == BinaryKind::AndI
                                 ? valueAttr.getValue() - 1
                                 : APInt(valueAttr.getValue().getBitWidth(),
                                         valueAttr.getValue().logBase2());
    Type constantType = getPayloadType(constant.getType());
    ConstantOp replacementConstant =
        ConstantOp::create(rewriter, operation.getLoc(), constantType,
                           IntegerAttr::get(constantType, replacementValue));
    BinaryOp replacement =
        BinaryOp::create(rewriter, operation.getLoc(), operation.getType(),
                         replacementKind, source, replacementConstant);
    if (replacementKind == BinaryKind::ShLI)
      replacement.setOverflowFlags(operation.getOverflowFlags());
    rewriter.replaceOp(operation, replacement);
    return success();
  }
};
} // namespace

void BinaryOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add<StrengthReducePowerOfTwo>(context);
}

namespace {
enum class NumberKind { Integer, Float };

static NumberKind getNumberKind(Type type) {
  return isa<FloatType>(type) ? NumberKind::Float : NumberKind::Integer;
}

static bool hasPolicyField(DictionaryAttr policy, StringRef name,
                           TypeID typeID) {
  Attribute value = policy ? policy.get(name) : Attribute();
  return value && value.getTypeID() == typeID;
}

static unsigned getNumberBitWidth(Type type) {
  if (type.isIndex())
    return 64;
  return type.getIntOrFloatBitWidth();
}
} // namespace

LogicalResult CastOp::verify() {
  auto emit = [this](const Twine &message) { return emitOpError(message); };
  FailureOr<Shape> source = classifyNumber(getSource().getType(), emit);
  FailureOr<Shape> result = classifyNumber(getResult().getType(), emit);
  if (failed(source) || failed(result))
    return failure();
  if (source->cardinality != result->cardinality)
    return emitOpError("source and result must have the same SIMD shape");
  if (getOverflowFlags() != arith::IntegerOverflowFlags::none &&
      (getKind() != CastKind::IntConvert ||
       getNumberBitWidth(result->elementType) >=
           getNumberBitWidth(source->elementType)))
    return emitOpError(
        "overflow flags require a narrowing intconvert operation");
  NumberKind sourceKind = getNumberKind(source->elementType);
  NumberKind resultKind = getNumberKind(result->elementType);
  switch (getKind()) {
  case CastKind::FpConvert:
    if (sourceKind != NumberKind::Float || resultKind != NumberKind::Float)
      return emitOpError("fpconvert requires float source and result");
    break;
  case CastKind::IntConvert:
    if (sourceKind != NumberKind::Integer || resultKind != NumberKind::Integer)
      return emitOpError("intconvert requires integer source and result");
    break;
  case CastKind::IntToFp:
    if (sourceKind != NumberKind::Integer || resultKind != NumberKind::Float)
      return emitOpError("int_to_fp requires integer source and float result");
    break;
  case CastKind::FpToInt:
    if (sourceKind != NumberKind::Float || resultKind != NumberKind::Integer)
      return emitOpError("fp_to_int requires float source and integer result");
    break;
  }
  DictionaryAttr policy = getPolicyAttr();
  if (policy)
    for (NamedAttribute field : policy)
      if (field.getName() != "rounding" && field.getName() != "signedness" &&
          field.getName() != "extension")
        return emitOpError("unknown cast policy field '")
               << field.getName() << "'";
  bool hasRounding =
      hasPolicyField(policy, "rounding", TypeID::get<CastRoundingPolicyAttr>());
  bool needsSignedness =
      getKind() == CastKind::IntToFp || getKind() == CastKind::FpToInt;
  bool hasSignedness = hasPolicyField(policy, "signedness",
                                      TypeID::get<CastSignednessPolicyAttr>());
  if (needsSignedness != hasSignedness)
    return emitOpError(needsSignedness
                           ? "signedness policy is required"
                           : "signedness policy is not valid for this cast");
  if (policy && policy.get("signedness") && !hasSignedness)
    return emitOpError("signedness policy has the wrong attribute type");
  if (policy && policy.get("rounding") && !hasRounding)
    return emitOpError("rounding policy has the wrong attribute type");
  if (hasRounding && getKind() != CastKind::FpConvert &&
      getKind() != CastKind::IntToFp)
    return emitOpError("rounding policy requires fpconvert or int_to_fp");
  bool wideningInteger = getKind() == CastKind::IntConvert &&
                         getNumberBitWidth(result->elementType) >
                             getNumberBitWidth(source->elementType);
  bool hasExtension = hasPolicyField(policy, "extension",
                                     TypeID::get<CastExtensionPolicyAttr>());
  if (policy && policy.get("extension") && !hasExtension)
    return emitOpError("extension policy has the wrong attribute type");
  if (wideningInteger != hasExtension)
    return emitOpError(
        wideningInteger
            ? "extension policy is required for widening intconvert"
            : "extension policy is only valid for widening intconvert");
  return verifyCardinalities(getOperation());
}

void CastOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               SetIntRangeFn setResultRange) {
  if (getKind() != CastKind::IntConvert || argRanges.empty())
    return;
  unsigned sourceWidth = getIntegerBitWidth(getSource().getType());
  unsigned resultWidth = getIntegerBitWidth(getType());
  if (sourceWidth == 0 || resultWidth == 0)
    return;
  ConstantIntRanges source = normalizeRange(argRanges.front(), sourceWidth);
  if (resultWidth < sourceWidth) {
    setResultRange(getResult(), intrange::truncRange(source, resultWidth));
    return;
  }
  if (resultWidth == sourceWidth) {
    setResultRange(getResult(), source);
    return;
  }
  DictionaryAttr policy = getPolicyAttr();
  auto extension =
      policy
          ? dyn_cast_or_null<CastExtensionPolicyAttr>(policy.get("extension"))
          : CastExtensionPolicyAttr();
  if (!extension)
    return;
  if (extension.getValue() == CastExtension::Zero)
    setResultRange(getResult(), intrange::extUIRange(source, resultWidth));
  else
    setResultRange(getResult(), intrange::extSIRange(source, resultWidth));
}

LogicalResult BitcastOp::verify() {
  auto getBits = [&](Type type) -> FailureOr<int64_t> {
    if (SimdType simd = dyn_cast<SimdType>(type))
      type = simd.getElementType();
    if (VectorType vector = dyn_cast<VectorType>(type)) {
      if (vector.isScalable())
        return failure();
      return vector.getNumElements() *
             vector.getElementType().getIntOrFloatBitWidth();
    }
    if (isa<IntegerType, FloatType>(type))
      return type.getIntOrFloatBitWidth();
    return failure();
  };
  FailureOr<int64_t> sourceBits = getBits(getSource().getType());
  FailureOr<int64_t> resultBits = getBits(getResult().getType());
  if (failed(sourceBits) || failed(resultBits) || *sourceBits != *resultBits)
    return emitOpError("source and result must have equal fixed bit widths");
  if (getTypeCardinality(getSource().getType()) !=
      getTypeCardinality(getResult().getType()))
    return emitOpError("source and result must have the same SIMD shape");
  return verifyCardinalities(getOperation());
}

OpFoldResult CastOp::fold(FoldAdaptor) {
  if (getSource().getType() == getType())
    return getSource();
  CastOp extension = getSource().getDefiningOp<CastOp>();
  if (getKind() == CastKind::IntConvert && extension &&
      getOverflowFlags() == arith::IntegerOverflowFlags::none &&
      extension.getKind() == CastKind::IntConvert &&
      getNumberBitWidth(getPayloadType(getType())) ==
          getNumberBitWidth(getPayloadType(extension.getSource().getType())) &&
      getNumberBitWidth(getPayloadType(getSource().getType())) >
          getNumberBitWidth(getPayloadType(getType())))
    return extension.getSource();
  return {};
}

LogicalResult FAddOp::verify() {
  return verifyFloatShape(getOperation(), getLhs().getType(),
                          getRhs().getType(), getResult().getType());
}
LogicalResult FSubOp::verify() {
  return verifyFloatShape(getOperation(), getLhs().getType(),
                          getRhs().getType(), getResult().getType());
}
LogicalResult FMulOp::verify() {
  return verifyFloatShape(getOperation(), getLhs().getType(),
                          getRhs().getType(), getResult().getType());
}
LogicalResult FMaxOp::verify() {
  return verifyFloatShape(getOperation(), getLhs().getType(),
                          getRhs().getType(), getResult().getType());
}
LogicalResult FmaOp::verify() {
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getAcc().getType())
    return emitOpError("all operands must have the same SIMD type");
  return verifyFloatShape(getOperation(), getLhs().getType(),
                          getRhs().getType(), getResult().getType());
}
LogicalResult FExp2Op::verify() {
  if (getSource().getType() != getResult().getType())
    return emitOpError("operand and result must have the same SIMD type");
  if (!isa<FloatType>(cast<SimdType>(getSource().getType()).getElementType()))
    return emitOpError("SIMD element type must be floating-point");
  return verifyCardinalities(getOperation());
}
LogicalResult FRcpOp::verify() {
  if (getSource().getType() != getResult().getType())
    return emitOpError("operand and result must have the same SIMD type");
  if (!isa<FloatType>(cast<SimdType>(getSource().getType()).getElementType()))
    return emitOpError("SIMD element type must be floating-point");
  return verifyCardinalities(getOperation());
}

namespace {
static LogicalResult verifyComparison(Operation *op, Type lhsType, Type rhsType,
                                      Type resultType, bool floating) {
  auto emit = [op](const Twine &message) { return op->emitOpError(message); };
  FailureOr<Shape> lhs =
      floating ? classifyNumber(lhsType, emit) : classifyInteger(lhsType, emit);
  FailureOr<Shape> rhs =
      floating ? classifyNumber(rhsType, emit) : classifyInteger(rhsType, emit);
  if (failed(lhs) || failed(rhs))
    return failure();
  if (floating &&
      (!isa<FloatType>(lhs->elementType) || !isa<FloatType>(rhs->elementType)))
    return op->emitOpError("operands must have floating-point elements");
  if (lhsType != rhsType)
    return op->emitOpError("operands must have the same type");
  if (lhs->cardinality) {
    MaskType mask = dyn_cast<MaskType>(resultType);
    if (!mask || mask.getCardinality() != *lhs->cardinality)
      return op->emitOpError("SIMD comparison result must be a matching mask");
  } else if (!resultType.isInteger(1)) {
    return op->emitOpError("uniform comparison result must be i1");
  }
  return verifyCardinalities(op);
}
} // namespace

LogicalResult CmpIOp::verify() {
  return verifyComparison(getOperation(), getLhs().getType(),
                          getRhs().getType(), getResult().getType(), false);
}

void CmpIOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               SetIntRangeFn setResultRange) {
  if (argRanges.size() != 2 || !getType().isInteger(1))
    return;
  intrange::CmpPredicate predicate =
      static_cast<intrange::CmpPredicate>(getPredicate());
  std::optional<bool> value =
      intrange::evaluatePred(predicate, argRanges[0], argRanges[1]);
  APInt minimum = APInt::getZero(1);
  APInt maximum = APInt::getAllOnes(1);
  if (value == true)
    minimum = maximum;
  else if (value == false)
    maximum = minimum;
  setResultRange(getResult(),
                 ConstantIntRanges::fromUnsigned(minimum, maximum));
}
LogicalResult CmpFOp::verify() {
  return verifyComparison(getOperation(), getLhs().getType(),
                          getRhs().getType(), getResult().getType(), true);
}

OpFoldResult CmpIOp::fold(FoldAdaptor adaptor) {
  IntegerAttr lhs = dyn_cast_or_null<IntegerAttr>(adaptor.getLhs());
  IntegerAttr rhs = dyn_cast_or_null<IntegerAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
    return {};
  return Builder(getContext())
      .getBoolAttr(arith::applyCmpPredicate(getPredicate(), lhs.getValue(),
                                            rhs.getValue()));
}

OpFoldResult CmpFOp::fold(FoldAdaptor) { return {}; }

LogicalResult SelectOp::verify() {
  Type condition = getCondition().getType();
  if (condition.isInteger(1))
    return verifyCardinalities(getOperation());
  MaskType mask = dyn_cast<MaskType>(condition);
  if (!mask)
    return emitOpError("condition must be i1 or an XW mask");
  std::optional<int64_t> resultCardinality = getTypeCardinality(getType());
  if (!resultCardinality || *resultCardinality != mask.getCardinality())
    return emitOpError(
        "mask condition requires a matching SIMD or mask result");
  return verifyCardinalities(getOperation());
}

OpFoldResult SelectOp::fold(FoldAdaptor adaptor) {
  if (getTrueValue() == getFalseValue())
    return getTrueValue();
  if (matchPattern(adaptor.getCondition(), m_One()))
    return getTrueValue();
  if (matchPattern(adaptor.getCondition(), m_Zero()))
    return getFalseValue();
  return {};
}

static LogicalResult verifyMaskOperation(Operation *op) {
  return verifyCardinalities(op);
}
LogicalResult MaskAndOp::verify() {
  return verifyMaskOperation(getOperation());
}
LogicalResult MaskOrOp::verify() { return verifyMaskOperation(getOperation()); }
LogicalResult MaskXorOp::verify() {
  return verifyMaskOperation(getOperation());
}
LogicalResult MaskNotOp::verify() {
  return verifyMaskOperation(getOperation());
}

OpFoldResult MaskAndOp::fold(FoldAdaptor adaptor) {
  if (getLhs() == getRhs())
    return getLhs();
  if (matchPattern(adaptor.getLhs(), m_Zero()))
    return adaptor.getLhs();
  if (matchPattern(adaptor.getRhs(), m_Zero()))
    return adaptor.getRhs();
  return {};
}
OpFoldResult MaskOrOp::fold(FoldAdaptor adaptor) {
  if (getLhs() == getRhs())
    return getLhs();
  if (matchPattern(adaptor.getLhs(), m_Zero()))
    return getRhs();
  if (matchPattern(adaptor.getRhs(), m_Zero()))
    return getLhs();
  return {};
}
OpFoldResult MaskXorOp::fold(FoldAdaptor adaptor) {
  if (getLhs() == getRhs())
    return Builder(getContext()).getBoolAttr(false);
  if (matchPattern(adaptor.getLhs(), m_Zero()))
    return getRhs();
  if (matchPattern(adaptor.getRhs(), m_Zero()))
    return getLhs();
  return {};
}
OpFoldResult MaskNotOp::fold(FoldAdaptor adaptor) {
  IntegerAttr value = dyn_cast_or_null<IntegerAttr>(adaptor.getInput());
  if (!value)
    return {};
  return Builder(getContext()).getBoolAttr(value.getValue().isZero());
}

LogicalResult BallotOp::verify() {
  int64_t cardinality = cast<MaskType>(getMask().getType()).getCardinality();
  unsigned expectedWidth = cardinality <= 8 ? 8 : cardinality <= 16 ? 16 : 32;
  if (getResult().getType().getWidth() != expectedWidth)
    return emitOpError(
        "result width must be 8, 16, or 32 and cover the mask cardinality");
  return verifyCardinalities(getOperation());
}

LogicalResult DpasOp::verify() {
  auto verifyPacket = [&](Type type, const Twine &name) -> LogicalResult {
    SimdType simd = dyn_cast<SimdType>(type);
    VectorType packet =
        simd ? dyn_cast<VectorType>(simd.getElementType()) : VectorType();
    if (!simd || !packet || packet.getRank() != 1 || packet.isScalable())
      return emitOpError() << name
                           << " must be a SIMD value with a fixed 1-D packet";
    return success();
  };
  if (failed(verifyPacket(getA().getType(), "A")) ||
      failed(verifyPacket(getB().getType(), "B")) ||
      failed(verifyPacket(getAcc().getType(), "accumulator")) ||
      failed(verifyPacket(getResult().getType(), "result")))
    return failure();
  if (getAcc().getType() != getResult().getType())
    return emitOpError("accumulator and result types must match");
  int64_t operandsPerDword = 2;
  if (getK() != getSystolicDepth() * operandsPerDword)
    return emitOpError("K must match systolic depth and source precision");
  VectorType resultPacket =
      cast<VectorType>(cast<SimdType>(getResult().getType()).getElementType());
  if (getRepeatCount() != static_cast<uint64_t>(resultPacket.getNumElements()))
    return emitOpError("repeat count must match the result packet length");
  if (getSystolicDepth() <= 0 || getRepeatCount() <= 0)
    return emitOpError("systolic depth and repeat count must be positive");
  return verifyCardinalities(getOperation());
}

OpFoldResult BallotOp::fold(FoldAdaptor adaptor) {
  IntegerAttr mask = dyn_cast_or_null<IntegerAttr>(adaptor.getMask());
  if (!mask)
    return {};
  unsigned width = getResult().getType().getWidth();
  APInt value =
      mask.getValue().isZero()
          ? APInt::getZero(width)
          : APInt::getLowBitsSet(
                width, cast<MaskType>(getMask().getType()).getCardinality());
  return IntegerAttr::get(getResult().getType(), value);
}

namespace {
static Type getVectorPayload(Type type) { return getPayloadType(type); }
} // namespace

LogicalResult PackOp::verify() {
  if (getInputs().empty())
    return emitOpError("requires at least one input");
  Type inputPayload = getPayloadType(getInputs().front().getType());
  VectorType resultVector = dyn_cast<VectorType>(getVectorPayload(getType()));
  if (!resultVector || resultVector.getRank() != 1 || resultVector.isScalable())
    return emitOpError("result payload must be a fixed 1-D vector");
  int64_t inputCount = 1;
  Type inputElement = inputPayload;
  if (VectorType vector = dyn_cast<VectorType>(inputPayload)) {
    if (vector.getRank() != 1 || vector.isScalable())
      return emitOpError("input vector payload must be fixed and 1-D");
    inputCount = vector.getNumElements();
    inputElement = vector.getElementType();
  }
  if (resultVector.getElementType() != inputElement ||
      resultVector.getNumElements() !=
          inputCount * static_cast<int64_t>(getInputs().size()))
    return emitOpError("inputs must exactly fill the result vector");
  if (getTypeCardinality(getInputs().front().getType()) !=
      getTypeCardinality(getType()))
    return emitOpError("input and result SIMD shapes must match");
  return verifyCardinalities(getOperation());
}

OpFoldResult PackOp::fold(FoldAdaptor) {
  if (getInputs().size() == 1 && getInputs().front().getType() == getType())
    return getInputs().front();
  return {};
}

LogicalResult ExtractOp::verify() {
  VectorType sourceVector =
      dyn_cast<VectorType>(getVectorPayload(getSource().getType()));
  if (!sourceVector || sourceVector.getRank() != 1)
    return emitOpError("source payload must be a 1-D vector");
  if (getIndex() >= static_cast<uint64_t>(sourceVector.getNumElements()))
    return emitOpError("index must be in source vector bounds");
  Type resultPayload = getPayloadType(getType());
  int64_t resultCount = 1;
  Type resultElement = resultPayload;
  if (VectorType vector = dyn_cast<VectorType>(resultPayload)) {
    resultCount = vector.getNumElements();
    resultElement = vector.getElementType();
  }
  if (resultElement != sourceVector.getElementType() ||
      getIndex() + static_cast<uint64_t>(resultCount) >
          static_cast<uint64_t>(sourceVector.getNumElements()))
    return emitOpError("result payload must be an in-bounds source slice");
  if (getTypeCardinality(getSource().getType()) !=
      getTypeCardinality(getType()))
    return emitOpError("source and result SIMD shapes must match");
  return verifyCardinalities(getOperation());
}

OpFoldResult ExtractOp::fold(FoldAdaptor) {
  PackOp pack = getSource().getDefiningOp<PackOp>();
  if (!pack || getIndex() >= pack.getInputs().size())
    return {};
  Value input = pack.getInputs()[getIndex()];
  return input.getType() == getType() ? OpFoldResult(input) : OpFoldResult();
}

LogicalResult Block2DPrefetchOp::verify() {
  std::array<Value, 5> geometry = {getSurfaceWidth(), getSurfaceHeight(),
                                   getSurfacePitch(), getX(), getY()};
  return verifyBlock2D(*this, getBase(), geometry, getBlockWidth(),
                       getBlockHeight(), getBlocks(), getElementBits(),
                       getTranspose(), getVnni(), Value(), false);
}

LogicalResult Block2DReadOp::verify() {
  std::array<Value, 5> geometry = {getSurfaceWidth(), getSurfaceHeight(),
                                   getSurfacePitch(), getX(), getY()};
  return verifyBlock2D(*this, getBase(), geometry, getBlockWidth(),
                       getBlockHeight(), getBlocks(), getElementBits(),
                       getTranspose(), getVnni(), getValue(), false);
}

LogicalResult Block2DWriteOp::verify() {
  std::array<Value, 5> geometry = {getSurfaceWidth(), getSurfaceHeight(),
                                   getSurfacePitch(), getX(), getY()};
  return verifyBlock2D(*this, getBase(), geometry, getBlockWidth(),
                       getBlockHeight(), getBlocks(), getElementBits(),
                       getTranspose(), getVnni(), getValue(), true);
}

Value PtrAddOp::getViewSource() { return getBase(); }
Value AddrspaceCastOp::getViewSource() { return getSource(); }

LogicalResult PtrAddOp::verify() {
  auto emit = [this](const Twine &message) { return emitOpError(message); };
  FailureOr<PointerShape> base = classifyPointer(getBase().getType(), emit);
  FailureOr<Shape> offset = classifyInteger(getOffset().getType(), emit);
  FailureOr<PointerShape> result = classifyPointer(getResult().getType(), emit);
  if (failed(base) || failed(offset) || failed(result))
    return failure();
  if (base->pointerType != result->pointerType)
    return emitOpError("result pointer type must match base pointer type");
  if (base->cardinality && offset->cardinality &&
      base->cardinality != offset->cardinality)
    return emitOpError("base and offset SIMD cardinalities must match");
  std::optional<int64_t> expected =
      base->cardinality ? base->cardinality : offset->cardinality;
  if (result->cardinality != expected)
    return emitOpError("result must have the broadcast pointer shape");
  return verifyCardinalities(getOperation());
}

LogicalResult AddrspaceCastOp::verify() {
  auto emit = [this](const Twine &message) { return emitOpError(message); };
  FailureOr<PointerShape> source = classifyPointer(getSource().getType(), emit);
  FailureOr<PointerShape> result = classifyPointer(getResult().getType(), emit);
  if (failed(source) || failed(result))
    return failure();
  if (source->cardinality != result->cardinality)
    return emitOpError("source and result pointer shapes must match");
  if (source->pointerType.getAddressSpace() ==
      result->pointerType.getAddressSpace())
    return emitOpError("source and result address spaces must differ");
  return verifyCardinalities(getOperation());
}

LogicalResult PtrToIntOp::verify() {
  auto emit = [this](const Twine &message) { return emitOpError(message); };
  FailureOr<PointerShape> source = classifyPointer(getSource().getType(), emit);
  FailureOr<Shape> result = classifyInteger(getResult().getType(), emit);
  if (failed(source) || failed(result))
    return failure();
  if (source->cardinality != result->cardinality)
    return emitOpError("source and result shapes must match");
  return verifyCardinalities(getOperation());
}

LogicalResult IntToPtrOp::verify() {
  auto emit = [this](const Twine &message) { return emitOpError(message); };
  FailureOr<Shape> source = classifyInteger(getSource().getType(), emit);
  FailureOr<PointerShape> result = classifyPointer(getResult().getType(), emit);
  if (failed(source) || failed(result))
    return failure();
  if (source->cardinality != result->cardinality)
    return emitOpError("source and result shapes must match");
  return verifyCardinalities(getOperation());
}

LogicalResult NullOp::verify() {
  auto emit = [this](const Twine &message) { return emitOpError(message); };
  if (failed(classifyPointer(getResult().getType(), emit)))
    return failure();
  return verifyCardinalities(getOperation());
}

LogicalResult PtrCmpOp::verify() {
  if (getPredicate() != arith::CmpIPredicate::eq &&
      getPredicate() != arith::CmpIPredicate::ne)
    return emitOpError("predicate must be eq or ne");
  auto emit = [this](const Twine &message) { return emitOpError(message); };
  FailureOr<PointerShape> lhs = classifyPointer(getLhs().getType(), emit);
  FailureOr<PointerShape> rhs = classifyPointer(getRhs().getType(), emit);
  if (failed(lhs) || failed(rhs))
    return failure();
  if (lhs->pointerType != rhs->pointerType ||
      lhs->cardinality != rhs->cardinality)
    return emitOpError("pointer operands must have the same type");
  if (lhs->cardinality) {
    MaskType result = dyn_cast<MaskType>(getResult().getType());
    if (!result || result.getCardinality() != *lhs->cardinality)
      return emitOpError("SIMD pointer comparison requires a matching mask");
  } else if (!getResult().getType().isInteger(1)) {
    return emitOpError("uniform pointer comparison requires i1 result");
  }
  return verifyCardinalities(getOperation());
}

OpFoldResult PtrCmpOp::fold(FoldAdaptor) {
  if (getLhs() != getRhs())
    return {};
  return Builder(getContext())
      .getBoolAttr(getPredicate() == arith::CmpIPredicate::eq);
}

LogicalResult WhereOp::verify() {
  if (failed(verifyCardinalities(getOperation())))
    return failure();
  auto verifyYield = [&](Region &region, StringRef name) -> LogicalResult {
    YieldOp yield = dyn_cast<YieldOp>(region.front().getTerminator());
    if (!yield)
      return emitOpError(name) << " region must terminate with xw.yield";
    if (yield.getValues().size() != getResults().size())
      return emitOpError(name) << " yield count must match result count";
    for (auto [value, result] : llvm::zip(yield.getValues(), getResults()))
      if (value.getType() != result.getType())
        return emitOpError(name) << " yield types must match result types";
    return success();
  };
  if (failed(verifyYield(getThenRegion(), "then")))
    return failure();
  if (!getElseRegion().empty())
    return verifyYield(getElseRegion(), "else");
  if (getNumResults() != 0)
    return emitOpError("with results requires an otherwise region");
  return success();
}

void WhereOp::getSuccessorRegions(RegionBranchPoint point,
                                  SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.emplace_back(&getThenRegion());
    if (!getElseRegion().empty())
      regions.emplace_back(&getElseRegion());
    else if (getNumResults() == 0)
      regions.emplace_back(getOperation());
    return;
  }
  regions.emplace_back(getOperation());
}

OperandRange WhereOp::getEntrySuccessorOperands(RegionSuccessor) {
  return OperandRange((*this)->operand_end(), (*this)->operand_end());
}
ValueRange WhereOp::getSuccessorInputs(RegionSuccessor successor) {
  return successor.isOperation() ? ValueRange(getResults()) : ValueRange();
}
MutableOperandRange
YieldOp::getMutableSuccessorOperands(RegionSuccessor successor) {
  MutableOperandRange values = getValuesMutable();
  return successor.isOperation() ? values : values.slice(0, 0);
}

LogicalResult LaneIdOp::verify() {
  if (!cast<SimdType>(getResult().getType())
           .getElementType()
           .isSignlessInteger())
    return emitOpError("result SIMD element must be a signless integer");
  return verifyCardinalities(getOperation());
}
LogicalResult SubgroupIdOp::verify() {
  if (!getResult().getType().isSignlessInteger())
    return emitOpError("result must be a bare signless integer");
  return success();
}
LogicalResult GlobalIdOp::verify() {
  return verifyDimQuery(getOperation(), getDim(), true);
}
LogicalResult LocalIdOp::verify() {
  return verifyDimQuery(getOperation(), getDim(), true);
}
LogicalResult GroupIdOp::verify() {
  return verifyDimQuery(getOperation(), getDim(), false);
}
LogicalResult GlobalSizeOp::verify() {
  return verifyDimQuery(getOperation(), getDim(), false);
}
LogicalResult LocalSizeOp::verify() {
  return verifyDimQuery(getOperation(), getDim(), false);
}
LogicalResult NumGroupsOp::verify() {
  return verifyDimQuery(getOperation(), getDim(), false);
}
LogicalResult LaunchGridSizeOp::verify() {
  return verifyDimQuery(getOperation(), getDim(), false);
}
LogicalResult LaunchBlockSizeOp::verify() {
  return verifyDimQuery(getOperation(), getDim(), false);
}

LogicalResult ShuffleOp::verify() {
  if (getSource().getType() != getResult().getType())
    return emitOpError("source and result SIMD types must match");
  auto emit = [this](const Twine &message) { return emitOpError(message); };
  FailureOr<Shape> lane = classifyInteger(getSourceLane().getType(), emit);
  if (failed(lane))
    return failure();
  if (lane->cardinality &&
      *lane->cardinality !=
          cast<SimdType>(getSource().getType()).getCardinality())
    return emitOpError("source lane cardinality must match source cardinality");
  return verifyCardinalities(getOperation());
}

OpFoldResult IssueTokenOp::fold(FoldAdaptor) {
  return foldTokenMerge(getDependencies());
}
OpFoldResult AfterOp::fold(FoldAdaptor) {
  return foldTokenMerge(getDependencies());
}
OpFoldResult JoinOp::fold(FoldAdaptor) {
  return foldTokenMerge(getDependencies());
}

LogicalResult LoadOp::verify() {
  return verifyPointerValueCardinality(getOperation(), getPtr().getType(),
                                       getValue().getType());
}
LogicalResult StoreOp::verify() {
  return verifyPointerValueCardinality(getOperation(), getPtr().getType(),
                                       getValue().getType());
}
LogicalResult AtomicRMWOp::verify() {
  if (getValue().getType() != getOld().getType())
    return emitOpError("value and old-value result types must match");
  return verifyPointerValueCardinality(getOperation(), getPtr().getType(),
                                       getValue().getType());
}

LogicalResult LocalMemoryBaseOp::verify() {
  if (!isa<LocalAddressSpaceAttr>(
          cast<PtrType>(getResult().getType()).getAddressSpace()))
    return emitOpError("result must use the local address space");
  if (getOffsetAttr().getInt() < 0)
    return emitOpError("offset must be non-negative");
  return success();
}

LogicalResult AllocOp::verify() {
  if (!isa<LocalAddressSpaceAttr>(
          cast<PtrType>(getResult().getType()).getAddressSpace()))
    return emitOpError("result must use the local address space");
  if (getBytesizeAttr().getInt() <= 0)
    return emitOpError("bytesize must be positive");
  int64_t align = getAlignAttr().getInt();
  if (align <= 0 || !llvm::isPowerOf2_64(static_cast<uint64_t>(align)))
    return emitOpError("align must be a positive power of two");
  if (IntegerAttr offsetAttr = getOffsetAttr()) {
    int64_t offset = offsetAttr.getInt();
    if (offset < 0)
      return emitOpError("offset must be non-negative");
    if (offset % align)
      return emitOpError("offset must satisfy alignment");
  }
  return success();
}

LogicalResult AllocReleaseOp::verify() {
  if (!isa<LocalAddressSpaceAttr>(
          cast<PtrType>(getAllocation().getType()).getAddressSpace()))
    return emitOpError("allocation must use the local address space");
  return success();
}
