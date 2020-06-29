//===--- UpgradeDurationConversionsCheck.cpp - clang-tidy -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UpgradeDurationConversionsCheck.h"
#include "DurationRewriter.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace abseil {

void UpgradeDurationConversionsCheck::registerMatchers(MatchFinder *Finder) {
  // For the arithmetic calls, we match only the uses of the templated operators
  // where the template parameter is not a built-in type. This means the
  // instantiation makes use of an available user defined conversion to
  // `int64_t`.
  //
  // The implementation of these templates will be updated to fail SFINAE for
  // non-integral types. We match them to suggest an explicit cast.

  // Match expressions like `a *= b` and `a /= b` where `a` has type
  // `absl::Duration` and `b` is not of a built-in type.
  Finder->addMatcher(
      cxxOperatorCallExpr(
          argumentCountIs(2),
          hasArgument(
              0, expr(hasType(cxxRecordDecl(hasName("::absl::Duration"))))),
          hasArgument(1, expr().bind("arg")),
          callee(functionDecl(
              hasParent(functionTemplateDecl()),
              unless(hasTemplateArgument(0, refersToType(builtinType()))),
              hasAnyName("operator*=", "operator/="))))
          .bind("OuterExpr"),
      this);

  // Match expressions like `a.operator*=(b)` and `a.operator/=(b)` where `a`
  // has type `absl::Duration` and `b` is not of a built-in type.
  Finder->addMatcher(
      cxxMemberCallExpr(
          callee(cxxMethodDecl(
              ofClass(cxxRecordDecl(hasName("::absl::Duration"))),
              hasParent(functionTemplateDecl()),
              unless(hasTemplateArgument(0, refersToType(builtinType()))),
              hasAnyName("operator*=", "operator/="))),
          argumentCountIs(1), hasArgument(0, expr().bind("arg")))
          .bind("OuterExpr"),
      this);

  // Match expressions like `a * b`, `a / b`, `operator*(a, b)`, and
  // `operator/(a, b)` where `a` has type `absl::Duration` and `b` is not of a
  // built-in type.
  Finder->addMatcher(
      callExpr(callee(functionDecl(
                   hasParent(functionTemplateDecl()),
                   unless(hasTemplateArgument(0, refersToType(builtinType()))),
                   hasAnyName("::absl::operator*", "::absl::operator/"))),
               argumentCountIs(2),
               hasArgument(0, expr(hasType(
                                  cxxRecordDecl(hasName("::absl::Duration"))))),
               hasArgument(1, expr().bind("arg")))
          .bind("OuterExpr"),
      this);

  // Match expressions like `a * b` and `operator*(a, b)` where `a` is not of a
  // built-in type and `b` has type `absl::Duration`.
  Finder->addMatcher(
      callExpr(callee(functionDecl(
                   hasParent(functionTemplateDecl()),
                   unless(hasTemplateArgument(0, refersToType(builtinType()))),
                   hasName("::absl::operator*"))),
               argumentCountIs(2), hasArgument(0, expr().bind("arg")),
               hasArgument(1, expr(hasType(
                                  cxxRecordDecl(hasName("::absl::Duration"))))))
          .bind("OuterExpr"),
      this);

  // For the factory functions, we match only the non-templated overloads that
  // take an `int64_t` parameter. Within these calls, we care about implicit
  // casts through a user defined conversion to `int64_t`.
  //
  // The factory functions will be updated to be templated and SFINAE on whether
  // the template parameter is an integral type. This complements the already
  // existing templated overloads that only accept floating point types.

  // Match calls like:
  //   `absl::Nanoseconds(x)`
  //   `absl::Microseconds(x)`
  //   `absl::Milliseconds(x)`
  //   `absl::Seconds(x)`
  //   `absl::Minutes(x)`
  //   `absl::Hours(x)`
  // where `x` is not of a built-in type.
  Finder->addMatcher(
      traverse(
          ast_type_traits::TK_AsIs,
          implicitCastExpr(anyOf(hasCastKind(CK_UserDefinedConversion),
                                 has(implicitCastExpr(
                                     hasCastKind(CK_UserDefinedConversion)))),
                           hasParent(callExpr(
                               callee(functionDecl(
                                   DurationFactoryFunction(),
                                   unless(hasParent(functionTemplateDecl())))),
                               hasArgument(0, expr().bind("arg")))))
              .bind("OuterExpr")),
      this);
}

void UpgradeDurationConversionsCheck::check(
    const MatchFinder::MatchResult &Result) {
  const llvm::StringRef Message =
      "implicit conversion to 'int64_t' is deprecated in this context; use an "
      "explicit cast instead";

  TraversalKindScope RAII(*Result.Context, ast_type_traits::TK_AsIs);

  const auto *ArgExpr = Result.Nodes.getNodeAs<Expr>("arg");
  SourceLocation Loc = ArgExpr->getBeginLoc();

  const auto *OuterExpr = Result.Nodes.getNodeAs<Expr>("OuterExpr");

  if (!match(isInTemplateInstantiation(), *OuterExpr, *Result.Context)
           .empty()) {
    if (MatchedTemplateLocations.count(Loc.getRawEncoding()) == 0) {
      // For each location matched in a template instantiation, we check if the
      // location can also be found in `MatchedTemplateLocations`. If it is not
      // found, that means the expression did not create a match without the
      // instantiation and depends on template parameters. A manual fix is
      // probably required so we provide only a warning.
      diag(Loc, Message);
    }
    return;
  }

  // We gather source locations from template matches not in template
  // instantiations for future matches.
  internal::Matcher<Stmt> IsInsideTemplate =
      hasAncestor(decl(anyOf(classTemplateDecl(), functionTemplateDecl())));
  if (!match(IsInsideTemplate, *ArgExpr, *Result.Context).empty())
    MatchedTemplateLocations.insert(Loc.getRawEncoding());

  DiagnosticBuilder Diag = diag(Loc, Message);
  CharSourceRange SourceRange = Lexer::makeFileCharRange(
      CharSourceRange::getTokenRange(ArgExpr->getSourceRange()),
      *Result.SourceManager, Result.Context->getLangOpts());
  if (SourceRange.isInvalid())
    // An invalid source range likely means we are inside a macro body. A manual
    // fix is likely needed so we do not create a fix-it hint.
    return;

  Diag << FixItHint::CreateInsertion(SourceRange.getBegin(),
                                     "static_cast<int64_t>(")
       << FixItHint::CreateInsertion(SourceRange.getEnd(), ")");
}

} // namespace abseil
} // namespace tidy
} // namespace clang
