//===--- TwineLocalCheck.cpp - clang-tidy ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TwineLocalCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::llvm_check {

void TwineLocalCheck::registerMatchers(MatchFinder *Finder) {
  auto TwineType =
      qualType(hasDeclaration(cxxRecordDecl(hasName("::llvm::Twine"))));
  Finder->addMatcher(
      varDecl(unless(parmVarDecl()), hasType(TwineType)).bind("variable"),
      this);
}

void TwineLocalCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *VD = Result.Nodes.getNodeAs<VarDecl>("variable");
  auto Diag = diag(VD->getLocation(),
                   "twine variables are prone to use-after-free bugs");

  // If this VarDecl has an initializer try to fix it.
  if (VD->hasInit()) {
    // Peel away implicit constructors and casts so we can see the actual type
    // of the initializer.
    const Expr *C = VD->getInit()->IgnoreImplicit();

    while (isa<CXXConstructExpr>(C)) {
      if (cast<CXXConstructExpr>(C)->getNumArgs() == 0)
        break;
      C = cast<CXXConstructExpr>(C)->getArg(0)->IgnoreParenImpCasts();
    }

    SourceRange TypeRange =
        VD->getTypeSourceInfo()->getTypeLoc().getSourceRange();

    // A real Twine, turn it into a std::string.
    if (VD->getType()->getCanonicalTypeUnqualified() ==
        C->getType()->getCanonicalTypeUnqualified()) {
      SourceLocation EndLoc = Lexer::getLocForEndOfToken(
          VD->getInit()->getEndLoc(), 0, *Result.SourceManager, getLangOpts());
      Diag << FixItHint::CreateReplacement(TypeRange, "std::string")
           << FixItHint::CreateInsertion(VD->getInit()->getBeginLoc(), "(")
           << FixItHint::CreateInsertion(EndLoc, ").str()");
    } else {
      // Just an implicit conversion. Insert the real type.
      Diag << FixItHint::CreateReplacement(
          TypeRange,
          C->getType().getAsString(Result.Context->getPrintingPolicy()));
    }
  }
}

} // namespace clang::tidy::llvm_check
