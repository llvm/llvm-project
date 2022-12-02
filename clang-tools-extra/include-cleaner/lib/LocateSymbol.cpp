//===--- LocateSymbol.cpp - Find locations providing a symbol -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInternal.h"
#include "clang/AST/DeclBase.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>
#include <vector>

namespace clang::include_cleaner {
namespace {

std::vector<SymbolLocation> locateDecl(const Decl &D) {
  std::vector<SymbolLocation> Result;
  // FIXME: Should we also provide physical locations?
  if (auto SS = tooling::stdlib::Recognizer()(&D))
    return {SymbolLocation(*SS)};
  for (auto *Redecl : D.redecls())
    Result.push_back(Redecl->getLocation());
  return Result;
}

} // namespace

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const SymbolLocation &S) {
  switch (S.kind()) {
  case SymbolLocation::Physical:
    // We can't decode the Location without SourceManager. Its raw
    // representation isn't completely useless (and distinguishes
    // SymbolReference from Symbol).
    return OS << "@0x"
              << llvm::utohexstr(
                     S.physical().getRawEncoding(), /*LowerCase=*/false,
                     /*Width=*/CHAR_BIT * sizeof(SourceLocation::UIntTy));
  case SymbolLocation::Standard:
    return OS << S.standard().scope() << S.standard().name();
  }
  llvm_unreachable("Unhandled Symbol kind");
}

std::vector<SymbolLocation> locateSymbol(const Symbol &S) {
  switch (S.kind()) {
  case Symbol::Declaration:
    return locateDecl(S.declaration());
  case Symbol::Macro:
    return {SymbolLocation(S.macro().Definition)};
  }
}

} // namespace clang::include_cleaner
