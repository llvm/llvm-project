//===-- PragmaCommentHandler.cpp - find all symbols -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PragmaCommentHandler.h"
#include "FindAllSymbols.h"
#include "HeaderMapCollector.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/Regex.h"

namespace clang {
namespace find_all_symbols {
namespace {
const char IWYUPragma[] = "// IWYU pragma: private, include ";
} // namespace

bool PragmaCommentHandler::HandleComment(Preprocessor &PP, SourceRange Range) {
  StringRef Text =
      Lexer::getSourceText(CharSourceRange::getCharRange(Range),
                           PP.getSourceManager(), PP.getLangOpts());
  size_t Pos = Text.find(IWYUPragma);
  if (Pos == StringRef::npos)
    return false;
  StringRef RemappingFilePath = Text.substr(Pos + std::strlen(IWYUPragma));
  Collector->addHeaderMapping(
      PP.getSourceManager().getFilename(Range.getBegin()),
      RemappingFilePath.trim("\"<>"));
  return false;
}

} // namespace find_all_symbols
} // namespace clang
