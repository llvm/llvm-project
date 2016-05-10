//===--- DefaultArgumentsCheck.cpp - clang-tidy----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DefaultArgumentsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace google {

void DefaultArgumentsCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxMethodDecl(anyOf(isOverride(), isVirtual()),
                    hasAnyParameter(parmVarDecl(hasInitializer(expr()))))
          .bind("Decl"),
      this);
}

void DefaultArgumentsCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<CXXMethodDecl>("Decl");
  diag(MatchedDecl->getLocation(),
       "default arguments on virtual or override methods are prohibited");
}

} // namespace google
} // namespace tidy
} // namespace clang
