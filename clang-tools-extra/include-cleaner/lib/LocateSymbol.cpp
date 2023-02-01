//===--- LocateSymbol.cpp - Find locations providing a symbol -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInternal.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "llvm/Support/Casting.h"
#include <utility>
#include <vector>

namespace clang::include_cleaner {
namespace {

template <typename T> Hints completeIfDefinition(T *D) {
  return D->isThisDeclarationADefinition() ? Hints::CompleteSymbol
                                           : Hints::None;
}

Hints declHints(const Decl *D) {
  // Definition is only needed for classes and templates for completeness.
  if (auto *TD = llvm::dyn_cast<TagDecl>(D))
    return completeIfDefinition(TD);
  else if (auto *CTD = llvm::dyn_cast<ClassTemplateDecl>(D))
    return completeIfDefinition(CTD);
  else if (auto *FTD = llvm::dyn_cast<FunctionTemplateDecl>(D))
    return completeIfDefinition(FTD);
  // Any other declaration is assumed usable.
  return Hints::CompleteSymbol;
}

std::vector<Hinted<SymbolLocation>> locateDecl(const Decl &D) {
  std::vector<Hinted<SymbolLocation>> Result;
  // FIXME: Should we also provide physical locations?
  if (auto SS = tooling::stdlib::Recognizer()(&D))
    return {{*SS, Hints::CompleteSymbol}};
  // FIXME: Signal foreign decls, e.g. a forward declaration not owned by a
  // library. Some useful signals could be derived by checking the DeclContext.
  // Most incidental forward decls look like:
  //   namespace clang {
  //   class SourceManager; // likely an incidental forward decl.
  //   namespace my_own_ns {}
  //   }
  for (auto *Redecl : D.redecls())
    Result.push_back({Redecl->getLocation(), declHints(Redecl)});
  return Result;
}

} // namespace

std::vector<Hinted<SymbolLocation>> locateSymbol(const Symbol &S) {
  switch (S.kind()) {
  case Symbol::Declaration:
    return locateDecl(S.declaration());
  case Symbol::Macro:
    return {{S.macro().Definition, Hints::CompleteSymbol}};
  }
  llvm_unreachable("Unknown Symbol::Kind enum");
}

} // namespace clang::include_cleaner
