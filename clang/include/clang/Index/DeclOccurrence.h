//===- DeclOccurrence.h - An occurrence of a decl within a file -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_DECLOCCURRENCE_H
#define LLVM_CLANG_INDEX_DECLOCCURRENCE_H

#include "clang/Basic/LLVM.h"
#include "clang/Index/IndexSymbol.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {
class Decl;

namespace index {

struct DeclOccurrence {
  SymbolRoleSet Roles;
  unsigned Offset;
  const Decl *Dcl;
  SmallVector<SymbolRelation, 3> Relations;

  DeclOccurrence(SymbolRoleSet R, unsigned Offset, const Decl *D,
                 ArrayRef<SymbolRelation> Relations)
      : Roles(R), Offset(Offset), Dcl(D),
        Relations(Relations.begin(), Relations.end()) {}

  friend bool operator<(const DeclOccurrence &LHS, const DeclOccurrence &RHS) {
    return LHS.Offset < RHS.Offset;
  }
};

} // namespace index
} // namespace clang

#endif // LLVM_CLANG_INDEX_DECLOCCURRENCE_H
