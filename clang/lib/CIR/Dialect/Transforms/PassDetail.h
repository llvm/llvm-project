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

// A nobuiltin mark or list forbids `name`, and an empty list forbids all.
inline bool noBuiltinsForbid(mlir::Operation *op, llvm::StringRef name) {
  if (op->hasAttr(cir::CIRDialect::getNoBuiltinAttrName()))
    return true;
  auto noBuiltins = op->getAttrOfType<mlir::ArrayAttr>(
      cir::CIRDialect::getNoBuiltinsAttrName());
  if (!noBuiltins)
    return false;
  return noBuiltins.empty() ||
         llvm::any_of(noBuiltins, [name](mlir::Attribute entry) {
           auto builtinName = mlir::dyn_cast<mlir::StringAttr>(entry);
           return builtinName && builtinName.getValue() == name;
         });
}

// The call form, where a builtin mark wins over the nobuiltin state.
inline bool isNoBuiltin(mlir::Operation *op, llvm::StringRef name) {
  if (op->hasAttr(cir::CIRDialect::getBuiltinAttrName()))
    return false;
  return noBuiltinsForbid(op, name);
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
