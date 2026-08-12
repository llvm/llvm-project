//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIR_DIALECT_TRANSFORMS_PASSDETAIL_H
#define CIR_DIALECT_TRANSFORMS_PASSDETAIL_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ABI/TargetInfo.h"

namespace cir {

// Check the `nobuiltins` list. On a function, this list controls builtin
// calls in its body; the singular `nobuiltin` mark describes calls to the
// function and is intentionally not read here.
inline bool noBuiltinListDisables(mlir::Operation *op, llvm::StringRef name) {
  // Read the `nobuiltins` list of builtin names disabled on this operation.
  auto noBuiltins = op->getAttrOfType<mlir::ArrayAttr>(
      cir::CIRDialect::getNoBuiltinsAttrName());
  // No list means the list adds no restriction.
  if (!noBuiltins)
    return false;
  // An empty list disables every builtin, and a named list disables only
  // the builtins it contains.
  return noBuiltins.empty() ||
         llvm::any_of(noBuiltins, [name](mlir::Attribute entry) {
           auto builtinName = mlir::dyn_cast<mlir::StringAttr>(entry);
           return builtinName && builtinName.getValue() == name;
         });
}

// Check all no builtin state attached to a call.
inline bool isNoBuiltin(mlir::Operation *op, llvm::StringRef name) {
  // `builtin` wins over both `nobuiltin` and `nobuiltins` on this call.
  if (op->hasAttr(cir::CIRDialect::getBuiltinAttrName()))
    return false;
  // The singular `nobuiltin` mark blocks builtin handling for this call.
  if (op->hasAttr(cir::CIRDialect::getNoBuiltinAttrName()))
    return true;
  // Otherwise check the `nobuiltins` list for this name.
  return noBuiltinListDisables(op, name);
}

} // namespace cir

namespace mlir {
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

#define GEN_PASS_DECL
#include "clang/CIR/Dialect/Passes.h.inc"

} // namespace mlir

#endif // CIR_DIALECT_TRANSFORMS_PASSDETAIL_H
