//===------------------------- DeclOrExpr.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_DECLOREXPR_H
#define LLVM_CLANG_AST_INTERP_DECLOREXPR_H

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/TypeBase.h"
#include "llvm/ADT/PointerUnion.h"

namespace clang {
namespace interp {

struct DeclOrExpr {
  llvm::PointerUnion<const Decl *, const Expr *> V;

  DeclOrExpr() : V(nullptr) {}
  DeclOrExpr(std::nullptr_t) : V(nullptr) {}
  DeclOrExpr(const Decl *VD) : V(VD) {}
  DeclOrExpr(const Expr *E) : V(E) {}

  bool isExpr() const { return isa_and_nonnull<const Expr *>(V); }
  bool isDecl() const { return isa_and_nonnull<const Decl *>(V); }
  bool isValueDecl() const { return isa_and_nonnull<ValueDecl>(asDecl()); }

  const Expr *asExpr() const { return V.dyn_cast<const Expr *>(); }
  const Decl *asDecl() const { return V.dyn_cast<const Decl *>(); }
  const ValueDecl *asValueDecl() const {
    return dyn_cast_if_present<ValueDecl>(asDecl());
  }
  const VarDecl *asVarDecl() const {
    return dyn_cast_if_present<VarDecl>(asDecl());
  }

  const void *getOpaqueValue() const { return V.getOpaqueValue(); }

  bool operator==(DeclOrExpr O) const { return O.V == V; }
  bool operator!=(DeclOrExpr O) const { return O.V != V; }
  explicit operator bool() const { return !V.isNull(); }

  QualType getType() const {
    if (const auto *VD = asValueDecl())
      return VD->getType();
    return asExpr()->getType();
  }
};
static_assert(sizeof(DeclOrExpr) == sizeof(void *));

inline DeclOrExpr getSwappedBytes(DeclOrExpr F) { return F; }

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, DeclOrExpr D) {
  OS << D.getOpaqueValue();
  return OS;
}

} // namespace interp
} // namespace clang

#endif
