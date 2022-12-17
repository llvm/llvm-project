//===-- FindAllMacros.cpp - find all macros ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FindAllMacros.h"
#include "HeaderMapCollector.h"
#include "PathConfig.h"
#include "SymbolInfo.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Token.h"
#include "llvm/Support/Path.h"

namespace clang {
namespace find_all_symbols {

llvm::Optional<SymbolInfo>
FindAllMacros::CreateMacroSymbol(const Token &MacroNameTok,
                                 const MacroInfo *info) {
  std::string FilePath =
      getIncludePath(*SM, info->getDefinitionLoc(), Collector);
  if (FilePath.empty())
    return std::nullopt;
  return SymbolInfo(MacroNameTok.getIdentifierInfo()->getName(),
                    SymbolInfo::SymbolKind::Macro, FilePath, {});
}

void FindAllMacros::MacroDefined(const Token &MacroNameTok,
                                 const MacroDirective *MD) {
  if (auto Symbol = CreateMacroSymbol(MacroNameTok, MD->getMacroInfo()))
    ++FileSymbols[*Symbol].Seen;
}

void FindAllMacros::MacroUsed(const Token &Name, const MacroDefinition &MD) {
  if (!MD || !SM->isInMainFile(SM->getExpansionLoc(Name.getLocation())))
    return;
  if (auto Symbol = CreateMacroSymbol(Name, MD.getMacroInfo()))
    ++FileSymbols[*Symbol].Used;
}

void FindAllMacros::MacroExpands(const Token &MacroNameTok,
                                 const MacroDefinition &MD, SourceRange Range,
                                 const MacroArgs *Args) {
  MacroUsed(MacroNameTok, MD);
}

void FindAllMacros::Ifdef(SourceLocation Loc, const Token &MacroNameTok,
                          const MacroDefinition &MD) {
  MacroUsed(MacroNameTok, MD);
}

void FindAllMacros::Ifndef(SourceLocation Loc, const Token &MacroNameTok,
                           const MacroDefinition &MD) {
  MacroUsed(MacroNameTok, MD);
}

void FindAllMacros::EndOfMainFile() {
  Reporter->reportSymbols(SM->getFileEntryForID(SM->getMainFileID())->getName(),
                          FileSymbols);
  FileSymbols.clear();
}

} // namespace find_all_symbols
} // namespace clang
