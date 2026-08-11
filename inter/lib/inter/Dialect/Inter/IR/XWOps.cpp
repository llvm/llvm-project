#include "inter/Dialect/Inter/IR/XW.h"

#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace xw;

#define GET_OP_CLASSES
#include "inter/Dialect/Inter/IR/XWOps.cpp.inc"

Value PtrAddOp::getViewSource() { return getBase(); }

LogicalResult PtrAddOp::verify() {
  constexpr uint32_t kInBounds = 1;
  constexpr uint32_t kNoUnsignedSignedWrap = 2;
  constexpr uint32_t kKnownFlags = 7;
  uint32_t flags = getGepFlags();
  if (flags & ~kKnownFlags)
    return emitOpError("has unknown LLVM GEP no-wrap flag bits");
  if ((flags & kInBounds) && !(flags & kNoUnsignedSignedWrap))
    return emitOpError("inbounds must imply nusw");
  return success();
}
