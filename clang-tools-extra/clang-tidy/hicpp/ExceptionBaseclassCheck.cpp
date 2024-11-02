//===--- ExceptionBaseclassCheck.cpp - clang-tidy--------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExceptionBaseclassCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace hicpp {

void ExceptionBaseclassCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      cxxThrowExpr(
          unless(has(expr(anyOf(isTypeDependent(), isValueDependent())))),
          // The thrown value is not derived from 'std::exception'.
          has(expr(unless(
              hasType(qualType(hasCanonicalType(hasDeclaration(cxxRecordDecl(
                  isSameOrDerivedFrom(hasName("::std::exception")))))))))),
          // This condition is always true, but will bind to the
          // template value if the thrown type is templated.
          anyOf(has(expr(
                    hasType(substTemplateTypeParmType().bind("templ_type")))),
                anything()),
          // Bind to the declaration of the type of the value that
          // is thrown. 'anything()' is necessary to always succeed
          // in the 'eachOf' because builtin types are not
          // 'namedDecl'.
          eachOf(has(expr(hasType(namedDecl().bind("decl")))), anything()))
          .bind("bad_throw"),
      this);
}

void ExceptionBaseclassCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *BadThrow = Result.Nodes.getNodeAs<CXXThrowExpr>("bad_throw");
  assert(BadThrow && "Did not match the throw expression");

  diag(BadThrow->getSubExpr()->getBeginLoc(), "throwing an exception whose "
                                              "type %0 is not derived from "
                                              "'std::exception'")
      << BadThrow->getSubExpr()->getType() << BadThrow->getSourceRange();

  if (const auto *Template =
          Result.Nodes.getNodeAs<SubstTemplateTypeParmType>("templ_type"))
    diag(BadThrow->getSubExpr()->getBeginLoc(),
         "type %0 is a template instantiation of %1", DiagnosticIDs::Note)
        << BadThrow->getSubExpr()->getType()
        << Template->getReplacedParameter();

  if (const auto *TypeDecl = Result.Nodes.getNodeAs<NamedDecl>("decl"))
    diag(TypeDecl->getBeginLoc(), "type defined here", DiagnosticIDs::Note);
}

} // namespace hicpp
} // namespace tidy
} // namespace clang
