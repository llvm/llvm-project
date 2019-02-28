//===--- DeclOccurrence.h - An occurrence of a decl within a file ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

#endif
