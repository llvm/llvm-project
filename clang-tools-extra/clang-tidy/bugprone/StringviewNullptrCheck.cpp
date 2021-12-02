//===--- StringviewNullptrCheck.cpp - clang-tidy --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StringviewNullptrCheck.h"
#include "../utils/TransformerClangTidyCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Transformer/RangeSelector.h"
#include "clang/Tooling/Transformer/RewriteRule.h"
#include "clang/Tooling/Transformer/Stencil.h"
#include "llvm/ADT/StringRef.h"

namespace clang {
namespace tidy {
namespace bugprone {

using namespace ::clang::ast_matchers;
using namespace ::clang::transformer;

namespace {
AST_MATCHER_P(InitListExpr, initCountIs, unsigned, N) {
  return Node.getNumInits() == N;
}
} // namespace

RewriteRule StringviewNullptrCheckImpl() {
  auto construction_warning =
      cat("constructing basic_string_view from null is undefined; replace with "
          "the default constructor");
  auto assignment_warning =
      cat("assignment to basic_string_view from null is undefined; replace "
          "with the default constructor");
  auto relative_comparison_warning =
      cat("comparing basic_string_view to null is undefined; replace with the "
          "empty string");
  auto equality_comparison_warning =
      cat("comparing basic_string_view to null is undefined; replace with the "
          "emptiness query");
  auto StringViewConstructingFromNullExpr =
      cxxConstructExpr(
          hasType(hasUnqualifiedDesugaredType(recordType(hasDeclaration(
              cxxRecordDecl(hasName("::std::basic_string_view")))))),
          argumentCountIs(1),
          hasArgument(
              0, anyOf(ignoringParenImpCasts(cxxNullPtrLiteralExpr()),
                       initListExpr(initCountIs(1),
                                    hasInit(0, ignoringParenImpCasts(
                                                   cxxNullPtrLiteralExpr()))),
                       initListExpr(initCountIs(0)))),
          has(expr().bind("null_argument_expr")))
          .bind("construct_expr");

  auto HandleTemporaryCXXFunctionalCastExpr =
      makeRule(cxxFunctionalCastExpr(
                   hasSourceExpression(StringViewConstructingFromNullExpr)),
               remove(node("null_argument_expr")), construction_warning);

  auto HandleTemporaryCXXTemporaryObjectExprAndCompoundLiteralExpr =
      makeRule(cxxTemporaryObjectExpr(StringViewConstructingFromNullExpr),
               remove(node("null_argument_expr")), construction_warning);

  auto HandleTemporaryCStyleCastExpr = makeRule(
      cStyleCastExpr(hasSourceExpression(StringViewConstructingFromNullExpr)),
      changeTo(node("null_argument_expr"), cat("{}")), construction_warning);

  auto HandleTemporaryCXXStaticCastExpr = makeRule(
      cxxStaticCastExpr(
          hasSourceExpression(StringViewConstructingFromNullExpr)),
      changeTo(node("null_argument_expr"), cat("\"\"")), construction_warning);

  auto HandleStackCopyInitialization = makeRule(
      varDecl(hasInitializer(implicitCastExpr(
          ignoringImpCasts(StringViewConstructingFromNullExpr)))),
      changeTo(node("null_argument_expr"), cat("{}")), construction_warning);

  auto HandleStackDirectInitialization =
      makeRule(varDecl(hasInitializer(
                           cxxConstructExpr(StringViewConstructingFromNullExpr,
                                            unless(isListInitialization()))))
                   .bind("var_decl"),
               changeTo(node("construct_expr"), cat(name("var_decl"))),
               construction_warning);

  auto HandleStackDirectListAndCopyListInitialization = makeRule(
      varDecl(hasInitializer(cxxConstructExpr(
          StringViewConstructingFromNullExpr, isListInitialization()))),
      remove(node("null_argument_expr")), construction_warning);

  auto HandleFieldCopyInitialization = makeRule(
      fieldDecl(hasInClassInitializer(implicitCastExpr(
          ignoringImpCasts(StringViewConstructingFromNullExpr)))),
      changeTo(node("null_argument_expr"), cat("{}")), construction_warning);

  auto HandleFieldOtherInitialization = makeRule(
      fieldDecl(hasInClassInitializer(StringViewConstructingFromNullExpr)),
      remove(node("null_argument_expr")), construction_warning);

  auto HandleConstructorInitialization = makeRule(
      cxxCtorInitializer(withInitializer(StringViewConstructingFromNullExpr)),
      remove(node("null_argument_expr")), construction_warning);

  auto HandleDefaultArgumentInitialization = makeRule(
      parmVarDecl(hasInitializer(implicitCastExpr(
          hasSourceExpression(StringViewConstructingFromNullExpr)))),
      changeTo(node("null_argument_expr"), cat("{}")), construction_warning);

  auto HandleDefaultArgumentListInitialization =
      makeRule(parmVarDecl(hasInitializer(StringViewConstructingFromNullExpr)),
               remove(node("null_argument_expr")), construction_warning);

  auto HandleHeapInitialization = makeRule(
      cxxNewExpr(unless(isArray()), has(StringViewConstructingFromNullExpr)),
      remove(node("null_argument_expr")), construction_warning);

  auto HandleFunctionArgumentInitialization = makeRule(
      implicitCastExpr(hasSourceExpression(StringViewConstructingFromNullExpr),
                       hasParent(callExpr(unless(cxxOperatorCallExpr())))),
      changeTo(node("null_argument_expr"), cat("{}")), construction_warning);

  auto HandleFunctionArgumentListInitialization = makeRule(
      cxxConstructExpr(StringViewConstructingFromNullExpr,
                       hasParent(callExpr(unless(cxxOperatorCallExpr())))),
      remove(node("null_argument_expr")), construction_warning);

  auto HandleAssignment = makeRule(
      materializeTemporaryExpr(
          has(StringViewConstructingFromNullExpr),
          hasParent(cxxOperatorCallExpr(hasOverloadedOperatorName("=")))),
      changeTo(node("construct_expr"), cat("{}")), assignment_warning);

  auto HandleRelativeComparison =
      makeRule(implicitCastExpr(
                   hasSourceExpression(StringViewConstructingFromNullExpr),
                   hasParent(cxxOperatorCallExpr(
                       hasAnyOverloadedOperatorName("<", "<=", ">", ">=")))),
               changeTo(node("null_argument_expr"), cat("\"\"")),
               relative_comparison_warning);

  auto HandleEmptyEqualityComparison = makeRule(
      cxxOperatorCallExpr(
          hasOverloadedOperatorName("=="),
          hasOperands(traverse(clang::TK_IgnoreUnlessSpelledInSource,
                               expr().bind("string_view_instance")),
                      implicitCastExpr(hasSourceExpression(
                          StringViewConstructingFromNullExpr))))
          .bind("root"),
      changeTo(node("root"),
               cat(access("string_view_instance", cat("empty")), "()")),
      equality_comparison_warning);

  auto HandleNonEmptyEqualityComparison = makeRule(
      cxxOperatorCallExpr(
          hasOverloadedOperatorName("!="),
          hasOperands(traverse(clang::TK_IgnoreUnlessSpelledInSource,
                               expr().bind("string_view_instance")),
                      implicitCastExpr(hasSourceExpression(
                          StringViewConstructingFromNullExpr))))
          .bind("root"),
      changeTo(node("root"),
               cat("!", access("string_view_instance", cat("empty")), "()")),
      equality_comparison_warning);

  return applyFirst(
      {HandleTemporaryCXXFunctionalCastExpr,
       HandleTemporaryCXXTemporaryObjectExprAndCompoundLiteralExpr,
       HandleTemporaryCStyleCastExpr, HandleTemporaryCXXStaticCastExpr,
       HandleStackCopyInitialization, HandleStackDirectInitialization,
       HandleStackDirectListAndCopyListInitialization,
       HandleFieldCopyInitialization, HandleFieldOtherInitialization,
       HandleConstructorInitialization, HandleDefaultArgumentInitialization,
       HandleDefaultArgumentListInitialization, HandleHeapInitialization,
       HandleFunctionArgumentInitialization,
       HandleFunctionArgumentListInitialization, HandleAssignment,
       HandleRelativeComparison, HandleEmptyEqualityComparison,
       HandleNonEmptyEqualityComparison});
}

StringviewNullptrCheck::StringviewNullptrCheck(StringRef Name,
                                               ClangTidyContext *Context)
    : utils::TransformerClangTidyCheck(StringviewNullptrCheckImpl(), Name,
                                       Context) {}

} // namespace bugprone
} // namespace tidy
} // namespace clang
