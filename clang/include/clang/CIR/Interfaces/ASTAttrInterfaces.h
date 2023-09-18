//===- ASTAttrInterfaces.h - CIR AST Interfaces -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_CIR_AST_ATTR_INTERFACES_H_
#define MLIR_INTERFACES_CIR_AST_ATTR_INTERFACES_H_

#include "mlir/IR/Attributes.h"

#include "clang/AST/Attr.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Mangle.h"

namespace mlir {
namespace cir {

mlir::Attribute makeFuncDeclAttr(const clang::Decl *decl,
                                 mlir::MLIRContext *ctx);

} // namespace cir
} // namespace mlir

/// Include the generated interface declarations.
#include "clang/CIR/Interfaces/ASTAttrInterfaces.h.inc"

namespace mlir {
namespace cir {

template <typename T> bool hasAttr(ASTDeclInterface decl) {
  if constexpr (std::is_same_v<T, clang::OwnerAttr>)
    return decl.hasOwnerAttr();
  if constexpr (std::is_same_v<T, clang::PointerAttr>)
    return decl.hasPointerAttr();
  if constexpr (std::is_same_v<T, clang::InitPriorityAttr>)
    return decl.hasInitPriorityAttr();
}

} // namespace cir
} // namespace mlir

#endif // MLIR_INTERFACES_CIR_AST_ATAR_INTERFACES_H_
