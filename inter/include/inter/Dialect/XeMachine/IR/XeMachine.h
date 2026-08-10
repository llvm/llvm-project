#ifndef INTER_DIALECT_XEMACHINE_IR_XEMACHINE_H
#define INTER_DIALECT_XEMACHINE_IR_XEMACHINE_H

#include "inter/Dialect/XeMachine/IR/XeMachineTraits.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "inter/Dialect/XeMachine/IR/XeMachineDialect.h.inc"
#include "inter/Dialect/XeMachine/IR/XeMachineEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "inter/Dialect/XeMachine/IR/XeMachineAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "inter/Dialect/XeMachine/IR/XeMachineTypes.h.inc"

#define GET_OP_INTERFACE_CLASSES
#include "inter/Dialect/XeMachine/IR/XeMachineInterfaces.h.inc"

#define GET_OP_CLASSES
#include "inter/Dialect/XeMachine/IR/XeMachineOps.h.inc"

#endif // INTER_DIALECT_XEMACHINE_IR_XEMACHINE_H
