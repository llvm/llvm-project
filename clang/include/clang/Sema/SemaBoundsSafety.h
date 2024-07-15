//===---- SemaBoundsSafety.h - Bounds Safety specific routines-*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares semantic analysis functions specific to `-fbounds-safety`
/// (Bounds Safety) and also its attributes when used without `-fbounds-safety`
/// (e.g. `counted_by`)
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMABOUNDSSAFETY_H
#define LLVM_CLANG_SEMA_SEMABOUNDSSAFETY_H

#include "clang/Sema/SemaBase.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {
class CountAttributedType;
class Decl;
class Expr;
class FieldDecl;
class NamedDecl;
class ParsedAttr;
class TypeCoupledDeclRefInfo;

class SemaBoundsSafety : public SemaBase {
public:
  SemaBoundsSafety(Sema &S);

  bool CheckCountedByAttrOnField(
      FieldDecl *FD, Expr *E,
      llvm::SmallVectorImpl<TypeCoupledDeclRefInfo> &Decls, bool CountInBytes,
      bool OrNull);
};

} // namespace clang

#endif //  LLVM_CLANG_SEMA_SEMABOUNDSSAFETY_H
