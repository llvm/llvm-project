#include "inter/Dialect/Inter/IR/XW.h"

#include "mlir/IR/DialectImplementation.h"

using namespace xw;

#include "inter/Dialect/Inter/IR/XWDialect.cpp.inc"

void XWDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "inter/Dialect/Inter/IR/XWOps.cpp.inc"
      >();
}
