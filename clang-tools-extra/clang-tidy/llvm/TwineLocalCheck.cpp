//===--- TwineLocalCheck.cpp - clang-tidy ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TwineLocalCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace llvm {

void TwineLocalCheck::registerMatchers(MatchFinder *Finder) {
  auto TwineType =
      qualType(hasDeclaration(recordDecl(hasName("::llvm::Twine"))));
  Finder->addMatcher(varDecl(hasType(TwineType)).bind("variable"), this);
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

    while (isa<CXXConstructExpr>(C))
      C = cast<CXXConstructExpr>(C)->getArg(0)->IgnoreParenImpCasts();

    SourceRange TypeRange =
        VD->getTypeSourceInfo()->getTypeLoc().getSourceRange();

    // A real Twine, turn it into a std::string.
    if (VD->getType()->getCanonicalTypeUnqualified() ==
        C->getType()->getCanonicalTypeUnqualified()) {
      SourceLocation EndLoc = Lexer::getLocForEndOfToken(
          VD->getInit()->getLocEnd(), 0, *Result.SourceManager, getLangOpts());
      Diag << FixItHint::CreateReplacement(TypeRange, "std::string")
           << FixItHint::CreateInsertion(VD->getInit()->getLocStart(), "(")
           << FixItHint::CreateInsertion(EndLoc, ").str()");
    } else {
      // Just an implicit conversion. Insert the real type.
      Diag << FixItHint::CreateReplacement(
          TypeRange,
          C->getType().getAsString(Result.Context->getPrintingPolicy()));
    }
  }
}

} // namespace llvm
} // namespace tidy
} // namespace clang
