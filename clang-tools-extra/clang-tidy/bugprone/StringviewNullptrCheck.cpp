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
#include "clang/AST/Decl.h"
#include "clang/AST/OperationKinds.h"
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

AST_MATCHER(clang::VarDecl, isDirectInitialization) {
  return Node.getInitStyle() != clang::VarDecl::InitializationStyle::CInit;
}

} // namespace

RewriteRule StringviewNullptrCheckImpl() {
  auto construction_warning =
      cat("constructing basic_string_view from null is undefined; replace with "
          "the default constructor");
  auto static_cast_warning =
      cat("casting to basic_string_view from null is undefined; replace with "
          "the empty string");
  auto argument_construction_warning =
      cat("passing null as basic_string_view is undefined; replace with the "
          "empty string");
  auto assignment_warning =
      cat("assignment to basic_string_view from null is undefined; replace "
          "with the default constructor");
  auto relative_comparison_warning =
      cat("comparing basic_string_view to null is undefined; replace with the "
          "empty string");
  auto equality_comparison_warning =
      cat("comparing basic_string_view to null is undefined; replace with the "
          "emptiness query");

  // Matches declarations and expressions of type `basic_string_view`
  auto HasBasicStringViewType = hasType(hasUnqualifiedDesugaredType(recordType(
      hasDeclaration(cxxRecordDecl(hasName("::std::basic_string_view"))))));

  // Matches `nullptr` and `(nullptr)` binding to a pointer
  auto NullLiteral = implicitCastExpr(
      hasCastKind(clang::CK_NullToPointer),
      hasSourceExpression(ignoringParens(cxxNullPtrLiteralExpr())));

  // Matches `{nullptr}` and `{(nullptr)}` binding to a pointer
  auto NullInitList = initListExpr(initCountIs(1), hasInit(0, NullLiteral));

  // Matches `{}`
  auto EmptyInitList = initListExpr(initCountIs(0));

  // Matches null construction without `basic_string_view` type spelling
  auto BasicStringViewConstructingFromNullExpr =
      cxxConstructExpr(
          HasBasicStringViewType, argumentCountIs(1),
          hasAnyArgument(/* `hasArgument` would skip over parens */ anyOf(
              NullLiteral, NullInitList, EmptyInitList)),
          unless(cxxTemporaryObjectExpr(/* filters out type spellings */)),
          has(expr().bind("null_arg_expr")))
          .bind("construct_expr");

  // `std::string_view(null_arg_expr)`
  auto HandleTemporaryCXXFunctionalCastExpr =
      makeRule(cxxFunctionalCastExpr(hasSourceExpression(
                   BasicStringViewConstructingFromNullExpr)),
               remove(node("null_arg_expr")), construction_warning);

  // `std::string_view{null_arg_expr}` and `(std::string_view){null_arg_expr}`
  auto HandleTemporaryCXXTemporaryObjectExprAndCompoundLiteralExpr = makeRule(
      cxxTemporaryObjectExpr(cxxConstructExpr(
          HasBasicStringViewType, argumentCountIs(1),
          hasAnyArgument(/* `hasArgument` would skip over parens */ anyOf(
              NullLiteral, NullInitList, EmptyInitList)),
          has(expr().bind("null_arg_expr")))),
      remove(node("null_arg_expr")), construction_warning);

  // `(std::string_view) null_arg_expr`
  auto HandleTemporaryCStyleCastExpr = makeRule(
      cStyleCastExpr(
          hasSourceExpression(BasicStringViewConstructingFromNullExpr)),
      changeTo(node("null_arg_expr"), cat("{}")), construction_warning);

  // `static_cast<std::string_view>(null_arg_expr)`
  auto HandleTemporaryCXXStaticCastExpr = makeRule(
      cxxStaticCastExpr(
          hasSourceExpression(BasicStringViewConstructingFromNullExpr)),
      changeTo(node("null_arg_expr"), cat("\"\"")), static_cast_warning);

  // `std::string_view sv = null_arg_expr;`
  auto HandleStackCopyInitialization = makeRule(
      varDecl(HasBasicStringViewType,
              hasInitializer(ignoringImpCasts(
                  cxxConstructExpr(BasicStringViewConstructingFromNullExpr,
                                   unless(isListInitialization())))),
              unless(isDirectInitialization())),
      changeTo(node("null_arg_expr"), cat("{}")), construction_warning);

  // `std::string_view sv = {null_arg_expr};`
  auto HandleStackCopyListInitialization =
      makeRule(varDecl(HasBasicStringViewType,
                       hasInitializer(cxxConstructExpr(
                           BasicStringViewConstructingFromNullExpr,
                           isListInitialization())),
                       unless(isDirectInitialization())),
               remove(node("null_arg_expr")), construction_warning);

  // `std::string_view sv(null_arg_expr);`
  auto HandleStackDirectInitialization =
      makeRule(varDecl(HasBasicStringViewType,
                       hasInitializer(cxxConstructExpr(
                           BasicStringViewConstructingFromNullExpr,
                           unless(isListInitialization()))),
                       isDirectInitialization())
                   .bind("var_decl"),
               changeTo(node("construct_expr"), cat(name("var_decl"))),
               construction_warning);

  // `std::string_view sv{null_arg_expr};`
  auto HandleStackDirectListInitialization =
      makeRule(varDecl(HasBasicStringViewType,
                       hasInitializer(cxxConstructExpr(
                           BasicStringViewConstructingFromNullExpr,
                           isListInitialization())),
                       isDirectInitialization()),
               remove(node("null_arg_expr")), construction_warning);

  // `struct S { std::string_view sv = null_arg_expr; };`
  auto HandleFieldInClassCopyInitialization = makeRule(
      fieldDecl(HasBasicStringViewType,
                hasInClassInitializer(ignoringImpCasts(
                    cxxConstructExpr(BasicStringViewConstructingFromNullExpr,
                                     unless(isListInitialization()))))),
      changeTo(node("null_arg_expr"), cat("{}")), construction_warning);

  // `struct S { std::string_view sv = {null_arg_expr}; };` and
  // `struct S { std::string_view sv{null_arg_expr}; };`
  auto HandleFieldInClassCopyListAndDirectListInitialization = makeRule(
      fieldDecl(HasBasicStringViewType,
                hasInClassInitializer(ignoringImpCasts(
                    cxxConstructExpr(BasicStringViewConstructingFromNullExpr,
                                     isListInitialization())))),
      remove(node("null_arg_expr")), construction_warning);

  // `class C { std::string_view sv; C() : sv(null_arg_expr) {} };`
  auto HandleConstructorDirectInitialization =
      makeRule(cxxCtorInitializer(forField(fieldDecl(HasBasicStringViewType)),
                                  withInitializer(cxxConstructExpr(
                                      BasicStringViewConstructingFromNullExpr,
                                      unless(isListInitialization())))),
               remove(node("null_arg_expr")), construction_warning);

  // `class C { std::string_view sv; C() : sv{null_arg_expr} {} };`
  auto HandleConstructorDirectListInitialization =
      makeRule(cxxCtorInitializer(forField(fieldDecl(HasBasicStringViewType)),
                                  withInitializer(cxxConstructExpr(
                                      BasicStringViewConstructingFromNullExpr,
                                      isListInitialization()))),
               remove(node("null_arg_expr")), construction_warning);

  // `void f(std::string_view sv = null_arg_expr);`
  auto HandleDefaultArgumentCopyInitialization = makeRule(
      parmVarDecl(HasBasicStringViewType,
                  hasInitializer(ignoringImpCasts(
                      cxxConstructExpr(BasicStringViewConstructingFromNullExpr,
                                       unless(isListInitialization()))))),
      changeTo(node("null_arg_expr"), cat("{}")), construction_warning);

  // `void f(std::string_view sv = {null_arg_expr});`
  auto HandleDefaultArgumentCopyListInitialization =
      makeRule(parmVarDecl(HasBasicStringViewType,
                           hasInitializer(cxxConstructExpr(
                               BasicStringViewConstructingFromNullExpr,
                               isListInitialization()))),
               remove(node("null_arg_expr")), construction_warning);

  // `new std::string_view(null_arg_expr)`
  auto HandleHeapDirectInitialization = makeRule(
      cxxNewExpr(has(cxxConstructExpr(BasicStringViewConstructingFromNullExpr,
                                      unless(isListInitialization()))),
                 unless(isArray()), unless(hasAnyPlacementArg(anything()))),
      remove(node("null_arg_expr")), construction_warning);

  // `new std::string_view{null_arg_expr}`
  auto HandleHeapDirectListInitialization = makeRule(
      cxxNewExpr(has(cxxConstructExpr(BasicStringViewConstructingFromNullExpr,
                                      isListInitialization())),
                 unless(isArray()), unless(hasAnyPlacementArg(anything()))),
      remove(node("null_arg_expr")), construction_warning);

  // `function(null_arg_expr)`
  auto HandleFunctionArgumentInitialization =
      makeRule(callExpr(hasAnyArgument(ignoringImpCasts(
                            BasicStringViewConstructingFromNullExpr)),
                        unless(cxxOperatorCallExpr())),
               changeTo(node("construct_expr"), cat("\"\"")),
               argument_construction_warning);

  // `sv = null_arg_expr`
  auto HandleAssignment = makeRule(
      cxxOperatorCallExpr(hasOverloadedOperatorName("="),
                          hasRHS(materializeTemporaryExpr(
                              has(BasicStringViewConstructingFromNullExpr)))),
      changeTo(node("construct_expr"), cat("{}")), assignment_warning);

  // `sv < null_arg_expr`
  auto HandleRelativeComparison = makeRule(
      cxxOperatorCallExpr(hasAnyOverloadedOperatorName("<", "<=", ">", ">="),
                          hasEitherOperand(ignoringImpCasts(
                              BasicStringViewConstructingFromNullExpr))),
      changeTo(node("construct_expr"), cat("\"\"")),
      relative_comparison_warning);

  // `sv == null_arg_expr`
  auto HandleEmptyEqualityComparison = makeRule(
      cxxOperatorCallExpr(
          hasOverloadedOperatorName("=="),
          hasOperands(ignoringImpCasts(BasicStringViewConstructingFromNullExpr),
                      traverse(clang::TK_IgnoreUnlessSpelledInSource,
                               expr().bind("instance"))))
          .bind("root"),
      changeTo(node("root"), cat(access("instance", cat("empty")), "()")),
      equality_comparison_warning);

  // `sv != null_arg_expr`
  auto HandleNonEmptyEqualityComparison = makeRule(
      cxxOperatorCallExpr(
          hasOverloadedOperatorName("!="),
          hasOperands(ignoringImpCasts(BasicStringViewConstructingFromNullExpr),
                      traverse(clang::TK_IgnoreUnlessSpelledInSource,
                               expr().bind("instance"))))
          .bind("root"),
      changeTo(node("root"), cat("!", access("instance", cat("empty")), "()")),
      equality_comparison_warning);

  // `return null_arg_expr;`
  auto HandleReturnStatement = makeRule(
      returnStmt(hasReturnValue(
          ignoringImpCasts(BasicStringViewConstructingFromNullExpr))),
      changeTo(node("construct_expr"), cat("{}")), construction_warning);

  // `T(null_arg_expr)`
  auto HandleConstructorInvocation =
      makeRule(cxxConstructExpr(
                   hasAnyArgument(/* `hasArgument` would skip over parens */
                                  ignoringImpCasts(
                                      BasicStringViewConstructingFromNullExpr)),
                   unless(HasBasicStringViewType)),
               changeTo(node("construct_expr"), cat("\"\"")),
               argument_construction_warning);

  return applyFirst(
      {HandleTemporaryCXXFunctionalCastExpr,
       HandleTemporaryCXXTemporaryObjectExprAndCompoundLiteralExpr,
       HandleTemporaryCStyleCastExpr,
       HandleTemporaryCXXStaticCastExpr,
       HandleStackCopyInitialization,
       HandleStackCopyListInitialization,
       HandleStackDirectInitialization,
       HandleStackDirectListInitialization,
       HandleFieldInClassCopyInitialization,
       HandleFieldInClassCopyListAndDirectListInitialization,
       HandleConstructorDirectInitialization,
       HandleConstructorDirectListInitialization,
       HandleDefaultArgumentCopyInitialization,
       HandleDefaultArgumentCopyListInitialization,
       HandleHeapDirectInitialization,
       HandleHeapDirectListInitialization,
       HandleFunctionArgumentInitialization,
       HandleAssignment,
       HandleRelativeComparison,
       HandleEmptyEqualityComparison,
       HandleNonEmptyEqualityComparison,
       HandleReturnStatement,
       HandleConstructorInvocation});
}

StringviewNullptrCheck::StringviewNullptrCheck(StringRef Name,
                                               ClangTidyContext *Context)
    : utils::TransformerClangTidyCheck(StringviewNullptrCheckImpl(), Name,
                                       Context) {}

} // namespace bugprone
} // namespace tidy
} // namespace clang
