//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AvoidPragmaCommentCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

#include <string>

using namespace clang::ast_matchers;

namespace clang::tidy::portability {

static const internal::VariadicDynCastAllOfMatcher<Decl, PragmaCommentDecl>
    // All other node matchers declared in this way are camelCase
    // NOLINTNEXTLINE(readability-identifier-naming)
    pragmaCommentDecl;

void AvoidPragmaCommentCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(pragmaCommentDecl().bind("pragma"), this);
}

void AvoidPragmaCommentCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Pragma = Result.Nodes.getNodeAs<PragmaCommentDecl>("pragma");

  std::string Msg{"avoid 'pragma comment' directive"};

  // We can give specific advice about comments that add linker flags, but other
  // kinds are too generic
  const PragmaMSCommentKind &Kind = Pragma->getCommentKind();
  switch (Kind) {
  case PragmaMSCommentKind::PCK_Lib:
    Msg.append("; use the build system to link libraries");
    break;
  case PragmaMSCommentKind::PCK_Linker:
    Msg.append("; use the build system to set linker options");
    break;
  case PragmaMSCommentKind::PCK_Unknown:
    llvm_unreachable("unexpected pragma comment kind");
  default:
    break;
  }
  diag(Pragma->getBeginLoc(), Msg);
}

} // namespace clang::tidy::portability
