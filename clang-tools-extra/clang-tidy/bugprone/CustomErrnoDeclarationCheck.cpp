//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CustomErrnoDeclarationCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

CustomErrnoDeclarationCheck::CustomErrnoDeclarationCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Inserter(Options.getLocalOrGlobal("IncludeStyle", utils::IncludeSorter::IS_LLVM), areDiagsSelfContained()) {}

void CustomErrnoDeclarationCheck::registerPPCallbacks(const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void CustomErrnoDeclarationCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(varDecl(anyOf(hasType(asString("int")), hasType(asString("int32_t")), hasType(asString("int16_t"))), hasName("errno"), hasExternalFormalLinkage()).bind("errnoDecl"), this);
}

void CustomErrnoDeclarationCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<VarDecl>("errnoDecl");
  const SourceManager &SM = *Result.SourceManager;
  const auto Location = MatchedDecl->getLocation();
  const auto FileID = SM.getFileID(Location);

  unsigned Line = SM.getSpellingLineNumber(MatchedDecl->getBeginLoc());
  StringRef Header = Result.Context->getLangOpts().CPlusPlus ? "<cerrno>" : "<errno.h>";

  diag(Location, "errno declaration detected, include cerrno instead")
      << FixItHint::CreateRemoval(CharSourceRange::getCharRange(SM.translateLineCol(FileID, Line, 1), SM.translateLineCol(FileID, Line + 1, 1)))
      << Inserter.createIncludeInsertion(FileID, Header);
}

} // namespace clang::tidy::bugprone
