//===--- ExceptionBaseclassCheck.cpp - clang-tidy--------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ExceptionBaseclassCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

#include <iostream>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace hicpp {

void ExceptionBaseclassCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus)
    return;

  Finder->addMatcher(
      cxxThrowExpr(
          allOf(
              has(expr(unless(hasType(cxxRecordDecl(
                  isSameOrDerivedFrom(hasName("std::exception"))))))),
              eachOf(has(expr(hasType(namedDecl().bind("decl")))), anything())))
          .bind("bad_throw"),
      this);
}

void ExceptionBaseclassCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *BadThrow = Result.Nodes.getNodeAs<CXXThrowExpr>("bad_throw");
  diag(BadThrow->getLocStart(),
       "throwing an exception whose type is not derived from 'std::exception'")
      << BadThrow->getSourceRange();

  const auto *TypeDecl = Result.Nodes.getNodeAs<NamedDecl>("decl");
  if (TypeDecl != nullptr)
    diag(TypeDecl->getLocStart(), "type defined here", DiagnosticIDs::Note);
}

} // namespace hicpp
} // namespace tidy
} // namespace clang
