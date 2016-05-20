//===-- FindAllMacros.cpp - find all macros ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FindAllMacros.h"
#include "HeaderMapCollector.h"
#include "SymbolInfo.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Token.h"

namespace clang {
namespace find_all_symbols {

void FindAllMacros::MacroDefined(const Token &MacroNameTok,
                                 const MacroDirective *MD) {
  SourceLocation Loc = SM->getExpansionLoc(MacroNameTok.getLocation());
  if (Loc.isInvalid() || SM->isInMainFile(Loc))
    return;

  llvm::StringRef FilePath = SM->getFilename(Loc);
  if (FilePath.empty())
    return;

  // If Collector is not nullptr, check pragma remapping header.
  if (Collector) {
    auto Iter = Collector->getHeaderMappingTable().find(FilePath);
    if (Iter != Collector->getHeaderMappingTable().end())
      FilePath = Iter->second;
  }

  SymbolInfo Symbol(MacroNameTok.getIdentifierInfo()->getName(),
                    SymbolInfo::SymbolKind::Macro, FilePath.str(),
                    SM->getSpellingLineNumber(Loc), {});

  Reporter->reportSymbol(SM->getFileEntryForID(SM->getMainFileID())->getName(),
                         Symbol);
}

} // namespace find_all_symbols
} // namespace clang
