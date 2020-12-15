//===-- Optimizer/Dialect/FIROpsSupport.h -- FIR op support -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPTIMIZER_DIALECT_FIROPSSUPPORT_H
#define OPTIMIZER_DIALECT_FIROPSSUPPORT_H

#include "flang/Optimizer/Dialect/FIROps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"

namespace fir {

/// return true iff the Operation is a non-volatile LoadOp
inline bool nonVolatileLoad(mlir::Operation *op) {
  if (auto load = dyn_cast<fir::LoadOp>(op))
    return !load->getAttr("volatile");
  return false;
}

/// return true iff the Operation is a call
inline bool isaCall(mlir::Operation *op) {
  return isa<fir::CallOp>(op) || isa<fir::DispatchOp>(op) ||
         isa<mlir::CallOp>(op) || isa<mlir::CallIndirectOp>(op);
}

/// return true iff the Operation is a fir::CallOp, fir::DispatchOp,
/// mlir::CallOp, or mlir::CallIndirectOp and not pure
/// NB: this is not the same as `!pureCall(op)`
inline bool impureCall(mlir::Operation *op) {
  // Should we also auto-detect that the called function is pure if its
  // arguments are not references?  For now, rely on a "pure" attribute.
  return op && isaCall(op) && !op->getAttr("pure");
}

/// return true iff the Operation is a fir::CallOp, fir::DispatchOp,
/// mlir::CallOp, or mlir::CallIndirectOp and is also pure.
/// NB: this is not the same as `!impureCall(op)`
inline bool pureCall(mlir::Operation *op) {
  // Should we also auto-detect that the called function is pure if its
  // arguments are not references?  For now, rely on a "pure" attribute.
  return op && isaCall(op) && op->getAttr("pure");
}

/// Get or create a FuncOp in a module.
///
/// If `module` already contains FuncOp `name`, it is returned. Otherwise, a new
/// FuncOp is created, and that new FuncOp is returned.
mlir::FuncOp createFuncOp(mlir::Location loc, mlir::ModuleOp module,
                          llvm::StringRef name, mlir::FunctionType type,
                          llvm::ArrayRef<mlir::NamedAttribute> attrs = {});

/// Get or create a GlobalOp in a module.
fir::GlobalOp createGlobalOp(mlir::Location loc, mlir::ModuleOp module,
                             llvm::StringRef name, mlir::Type type,
                             llvm::ArrayRef<mlir::NamedAttribute> attrs = {});

} // namespace fir

#endif // OPTIMIZER_DIALECT_FIROPSSUPPORT_H
