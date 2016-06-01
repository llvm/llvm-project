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
#include "PathConfig.h"
#include "SymbolInfo.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Token.h"
#include "llvm/Support/Path.h"

namespace clang {
namespace find_all_symbols {

void FindAllMacros::MacroDefined(const Token &MacroNameTok,
                                 const MacroDirective *MD) {
  SourceLocation Loc = SM->getExpansionLoc(MacroNameTok.getLocation());
  std::string FilePath = getIncludePath(*SM, Loc, Collector);
  if (FilePath.empty()) return;

  SymbolInfo Symbol(MacroNameTok.getIdentifierInfo()->getName(),
                    SymbolInfo::SymbolKind::Macro, FilePath,
                    SM->getSpellingLineNumber(Loc), {});

  Reporter->reportSymbol(SM->getFileEntryForID(SM->getMainFileID())->getName(),
                         Symbol);
}

} // namespace find_all_symbols
} // namespace clang
