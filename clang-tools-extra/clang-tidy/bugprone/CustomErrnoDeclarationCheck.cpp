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

void CustomErrnoDeclarationCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(varDecl(hasType(asString("int")), hasName("errno"), hasExternalFormalLinkage()).bind("errnoDecl"), this);
}

void CustomErrnoDeclarationCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<VarDecl>("errnoDecl"); // NULL?
  const SourceManager &SM = *Result.SourceManager;
  const auto Location = MatchedDecl->getLocation();
  const auto FileID = SM.getFileID(Location);

  unsigned Line = SM.getSpellingLineNumber(MatchedDecl->getBeginLoc());

  const auto Diag = diag(Location, "errno declaration detected, include cerrno instead")
      << FixItHint::CreateRemoval(CharSourceRange::getCharRange(SM.translateLineCol(FileID, Line, 1), SM.translateLineCol(FileID, Line + 1, 1)));

  if (alreadyInserted)
    return;

  Diag << FixItHint::CreateInsertion(SM.getLocForStartOfFile(FileID), "#include <cerrno>\n");
  alreadyInserted = true;
}

} // namespace clang::tidy::bugprone
