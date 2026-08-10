#include "inter/Dialect/Inter/IR/XW.h"

#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace xw;

#define GET_OP_CLASSES
#include "inter/Dialect/Inter/IR/XWOps.cpp.inc"
