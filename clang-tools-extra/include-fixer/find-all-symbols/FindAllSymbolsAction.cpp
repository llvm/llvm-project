//===-- FindAllSymbolsAction.cpp - find all symbols action --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FindAllSymbolsAction.h"

namespace clang {
namespace find_all_symbols {

FindAllSymbolsAction::FindAllSymbolsAction(
    SymbolReporter *Reporter, const HeaderMapCollector::HeaderMap *PostfixMap)
    : Reporter(Reporter), Collector(PostfixMap), Handler(&Collector),
      Matcher(Reporter, &Collector) {
  Matcher.registerMatchers(&MatchFinder);
}

std::unique_ptr<clang::ASTConsumer>
FindAllSymbolsAction::CreateASTConsumer(clang::CompilerInstance &Compiler,
                                        StringRef InFile) {
  Compiler.getPreprocessor().addCommentHandler(&Handler);
  Compiler.getPreprocessor().addPPCallbacks(llvm::make_unique<FindAllMacros>(
      Reporter, &Compiler.getSourceManager(), &Collector));
  return MatchFinder.newASTConsumer();
}

} // namespace find_all_symbols
} // namespace clang
