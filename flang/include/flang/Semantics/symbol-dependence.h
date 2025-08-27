//===-- include/flang/Semantics/symbol-dependence.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_SYMBOL_DEPENDENCE_H_
#define FORTRAN_SEMANTICS_SYMBOL_DEPENDENCE_H_

#include "flang/Semantics/symbol.h"

namespace Fortran::semantics {

// For a set or scope of symbols, computes the transitive closure of their
// dependences due to their types, bounds, specific procedures, interfaces,
// initialization, storage association, &c. Includes the original symbol
// or members of the original set.  Does not include dependences from
// subprogram definitions, only their interfaces.
enum DependenceCollectionFlags {
  NoDependenceCollectionFlags = 0,
  IncludeOriginalSymbols = 1 << 0,
  FollowUseAssociations = 1 << 1,
  IncludeSpecificsOfGenerics = 1 << 2,
  IncludeUsesOfGenerics = 1 << 3,
  NotJustForOneModule = 1 << 4,
};

SymbolVector CollectAllDependences(const SymbolVector &,
    int = NoDependenceCollectionFlags, const Scope * = nullptr);
SymbolVector CollectAllDependences(
    const Scope &, int = NoDependenceCollectionFlags);

} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_SYMBOL_DEPENDENCE_H_
