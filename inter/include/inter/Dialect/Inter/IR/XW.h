#ifndef INTER_DIALECT_INTER_IR_XW_H
#define INTER_DIALECT_INTER_IR_XW_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "inter/Dialect/Inter/IR/XWDialect.h.inc"
#include "inter/Dialect/Inter/IR/XWEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "inter/Dialect/Inter/IR/XWAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "inter/Dialect/Inter/IR/XWTypes.h.inc"

namespace xw {

class CacheStateResource
    : public mlir::SideEffects::Resource::Base<CacheStateResource> {
public:
  llvm::StringRef getName() const final { return "<XW cache state>"; }
  bool isAddressable() const final { return false; }
};

} // namespace xw

#define GET_OP_CLASSES
#include "inter/Dialect/Inter/IR/XWOps.h.inc"

#endif // INTER_DIALECT_INTER_IR_XW_H
