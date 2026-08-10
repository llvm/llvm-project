#ifndef INTER_DIALECT_INTER_IR_XW_H
#define INTER_DIALECT_INTER_IR_XW_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "inter/Dialect/XeMachine/IR/XeMachine.h"

#include "inter/Dialect/Inter/IR/XWDialect.h.inc"

#define GET_OP_CLASSES
#include "inter/Dialect/Inter/IR/XWOps.h.inc"

#endif // INTER_DIALECT_INTER_IR_XW_H
