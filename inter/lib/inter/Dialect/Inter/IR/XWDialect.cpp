#include "inter/Dialect/Inter/IR/XW.h"

#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace xw;

#include "inter/Dialect/Inter/IR/XWDialect.cpp.inc"
#include "inter/Dialect/Inter/IR/XWEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "inter/Dialect/Inter/IR/XWAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "inter/Dialect/Inter/IR/XWTypes.cpp.inc"

namespace {
static std::optional<int64_t> getCardinality(Type type) {
  if (SimdType simd = dyn_cast<SimdType>(type))
    return simd.getCardinality();
  if (MaskType mask = dyn_cast<MaskType>(type))
    return mask.getCardinality();
  return std::nullopt;
}

static LogicalResult verifyTypeCardinality(Type type, int64_t width,
                                           InFlightDiagnostic &diag) {
  std::optional<int64_t> cardinality = getCardinality(type);
  if (cardinality && width % *cardinality != 0)
    return diag << "type " << type << " has cardinality " << *cardinality
                << " which does not divide xw.simd_width " << width;
  return success();
}
} // namespace

void XWDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "inter/Dialect/Inter/IR/XWAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "inter/Dialect/Inter/IR/XWTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "inter/Dialect/Inter/IR/XWOps.cpp.inc"
      >();
}

Operation *XWDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                          Type type, Location loc) {
  return ConstantOp::materialize(builder, value, type, loc);
}

LogicalResult XWDialect::verifyOperationAttribute(Operation *op,
                                                  NamedAttribute attribute) {
  if (attribute.getName() != getSimdWidthAttrName())
    return success();
  IntegerAttr widthAttr = dyn_cast<IntegerAttr>(attribute.getValue());
  if (!widthAttr)
    return op->emitError("'xw.simd_width' must be an integer attribute");
  int64_t width = widthAttr.getInt();
  if (width != 8 && width != 16 && width != 32)
    return op->emitError("'xw.simd_width' must be 8, 16, or 32");

  FunctionOpInterface function = dyn_cast<FunctionOpInterface>(op);
  if (!function)
    return op->emitError("'xw.simd_width' is only valid on function-like ops");
  for (Type type : function.getArgumentTypes()) {
    InFlightDiagnostic diag = op->emitError("invalid function argument: ");
    if (failed(verifyTypeCardinality(type, width, diag)))
      return failure();
    diag.abandon();
  }
  for (Type type : function.getResultTypes()) {
    InFlightDiagnostic diag = op->emitError("invalid function result: ");
    if (failed(verifyTypeCardinality(type, width, diag)))
      return failure();
    diag.abandon();
  }
  return success();
}

LogicalResult SimdType::verify(function_ref<InFlightDiagnostic()> emitError,
                               Type elementType, int64_t cardinality) {
  if (!elementType)
    return emitError() << "SIMD element type must be non-null";
  if (cardinality <= 0)
    return emitError() << "SIMD cardinality must be positive";
  return success();
}

LogicalResult MaskType::verify(function_ref<InFlightDiagnostic()> emitError,
                               int64_t cardinality) {
  if (cardinality <= 0)
    return emitError() << "mask cardinality must be positive";
  return success();
}

LogicalResult PtrType::verify(function_ref<InFlightDiagnostic()> emitError,
                              Attribute addressSpace) {
  if (!isa_and_nonnull<PrivateAddressSpaceAttr, GlobalAddressSpaceAttr,
                       ConstantAddressSpaceAttr, LocalAddressSpaceAttr,
                       GenericAddressSpaceAttr>(addressSpace))
    return emitError() << "pointer address space must be an XW address space";
  return success();
}
