//===--- CollectMacros.cpp ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CollectMacros.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Lexer.h"

namespace clang {
namespace clangd {

void CollectMainFileMacros::add(const Token &MacroNameTok,
                                const MacroInfo *MI) {
  if (!InMainFile)
    return;
  auto Loc = MacroNameTok.getLocation();
  if (Loc.isInvalid() || Loc.isMacroID())
    return;

  auto Name = MacroNameTok.getIdentifierInfo()->getName();
  Out.Names.insert(Name);
  auto Range = halfOpenToRange(
      SM, CharSourceRange::getCharRange(Loc, MacroNameTok.getEndLoc()));
  if (auto SID = getSymbolID(Name, MI, SM))
    Out.MacroRefs[SID].push_back(Range);
  else
    Out.UnknownMacros.push_back(Range);
}
} // namespace clangd
} // namespace clang
