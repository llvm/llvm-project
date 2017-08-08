//===--- RenamedSymbol.cpp - ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactor/RenamedSymbol.h"
#include "clang/AST/DeclObjC.h"
#include <algorithm>

using namespace clang;

namespace clang {
namespace tooling {
namespace rename {

Symbol::Symbol(const NamedDecl *FoundDecl, unsigned SymbolIndex,
               const LangOptions &LangOpts)
    : Name(FoundDecl->getNameAsString(), LangOpts), SymbolIndex(SymbolIndex),
      FoundDecl(FoundDecl) {
  if (const auto *MD = dyn_cast<ObjCMethodDecl>(FoundDecl))
    ObjCSelector = MD->getSelector();
}

bool operator<(const SymbolOccurrence &LHS, const SymbolOccurrence &RHS) {
  assert(!LHS.Locations.empty() && !RHS.Locations.empty());
  return LHS.Locations[0] < RHS.Locations[0];
}

bool operator==(const SymbolOccurrence &LHS, const SymbolOccurrence &RHS) {
  return LHS.Kind == RHS.Kind && LHS.SymbolIndex == RHS.SymbolIndex &&
         std::equal(LHS.Locations.begin(), LHS.Locations.end(),
                    RHS.Locations.begin());
}

} // end namespace rename
} // end namespace tooling
} // end namespace clang
