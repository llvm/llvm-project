#include "inter/Dialect/XeMachine/IR/XeMachine.h"

#include "mlir/IR/DialectImplementation.h"

using namespace inter::xemachine;

#include "inter/Dialect/XeMachine/IR/XeMachineDialect.cpp.inc"
#include "inter/Dialect/XeMachine/IR/XeMachineEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "inter/Dialect/XeMachine/IR/XeMachineTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "inter/Dialect/XeMachine/IR/XeMachineAttrs.cpp.inc"

void XeMachineDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "inter/Dialect/XeMachine/IR/XeMachineTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "inter/Dialect/XeMachine/IR/XeMachineAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "inter/Dialect/XeMachine/IR/XeMachineOps.cpp.inc"
      >();
}
