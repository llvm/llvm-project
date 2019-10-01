//===-- FindAllSymbolsAction.cpp - find all symbols action --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FindAllSymbolsAction.h"
#include "FindAllMacros.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/STLExtras.h"

namespace clang {
namespace find_all_symbols {

FindAllSymbolsAction::FindAllSymbolsAction(
    SymbolReporter *Reporter,
    const HeaderMapCollector::RegexHeaderMap *RegexHeaderMap)
    : Reporter(Reporter), Collector(RegexHeaderMap), Handler(&Collector),
      Matcher(Reporter, &Collector) {
  Matcher.registerMatchers(&MatchFinder);
}

std::unique_ptr<ASTConsumer>
FindAllSymbolsAction::CreateASTConsumer(CompilerInstance &Compiler,
                                        StringRef InFile) {
  Compiler.getPreprocessor().addCommentHandler(&Handler);
  Compiler.getPreprocessor().addPPCallbacks(llvm::make_unique<FindAllMacros>(
      Reporter, &Compiler.getSourceManager(), &Collector));
  return MatchFinder.newASTConsumer();
}

} // namespace find_all_symbols
} // namespace clang
