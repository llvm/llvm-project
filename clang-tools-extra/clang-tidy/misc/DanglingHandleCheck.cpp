//===--- DanglingHandleCheck.cpp - clang-tidy------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DanglingHandleCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;
using namespace clang::tidy::matchers;

namespace clang {
namespace tidy {
namespace misc {

namespace {

ast_matchers::internal::BindableMatcher<Stmt>
handleFrom(const ast_matchers::internal::Matcher<RecordDecl> &IsAHandle,
           const ast_matchers::internal::Matcher<Expr> &Arg) {
  return cxxConstructExpr(hasDeclaration(cxxMethodDecl(ofClass(IsAHandle))),
                          hasArgument(0, Arg));
}

ast_matchers::internal::Matcher<Stmt> handleFromTemporaryValue(
    const ast_matchers::internal::Matcher<RecordDecl> &IsAHandle) {
  // If a ternary operator returns a temporary value, then both branches hold a
  // temporary value. If one of them is not a temporary then it must be copied
  // into one to satisfy the type of the operator.
  const auto TemporaryTernary =
      conditionalOperator(hasTrueExpression(cxxBindTemporaryExpr()),
                          hasFalseExpression(cxxBindTemporaryExpr()));

  return handleFrom(IsAHandle, anyOf(cxxBindTemporaryExpr(), TemporaryTernary));
}

ast_matchers::internal::Matcher<RecordDecl> isASequence() {
  return hasAnyName("::std::deque", "::std::forward_list", "::std::list",
                    "::std::vector");
}

ast_matchers::internal::Matcher<RecordDecl> isASet() {
  return hasAnyName("::std::set", "::std::multiset", "::std::unordered_set",
                    "::std::unordered_multiset");
}

ast_matchers::internal::Matcher<RecordDecl> isAMap() {
  return hasAnyName("::std::map", "::std::multimap", "::std::unordered_map",
                    "::std::unordered_multimap");
}

ast_matchers::internal::BindableMatcher<Stmt> makeContainerMatcher(
    const ast_matchers::internal::Matcher<RecordDecl> &IsAHandle) {
  // This matcher could be expanded to detect:
  //  - Constructors: eg. vector<string_view>(3, string("A"));
  //  - emplace*(): This requires a different logic to determine that
  //                the conversion will happen inside the container.
  //  - map's insert: This requires detecting that the pair conversion triggers
  //                  the bug. A little more complicated than what we have now.
  return callExpr(
      hasAnyArgument(
          ignoringParenImpCasts(handleFromTemporaryValue(IsAHandle))),
      anyOf(
          // For sequences: assign, push_back, resize.
          cxxMemberCallExpr(
              callee(functionDecl(hasAnyName("assign", "push_back", "resize"))),
              on(expr(hasType(recordDecl(isASequence()))))),
          // For sequences and sets: insert.
          cxxMemberCallExpr(
              callee(functionDecl(hasName("insert"))),
              on(expr(hasType(recordDecl(anyOf(isASequence(), isASet())))))),
          // For maps: operator[].
          cxxOperatorCallExpr(callee(cxxMethodDecl(ofClass(isAMap()))),
                              hasOverloadedOperatorName("[]"))));
}

} // anonymous namespace

DanglingHandleCheck::DanglingHandleCheck(StringRef Name,
                                         ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      HandleClasses(utils::options::parseStringList(Options.get(
          "HandleClasses",
          "std::basic_string_view;std::experimental::basic_string_view"))),
      IsAHandle(cxxRecordDecl(hasAnyName(std::vector<StringRef>(
                                  HandleClasses.begin(), HandleClasses.end())))
                    .bind("handle")) {}

void DanglingHandleCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "HandleClasses",
                utils::options::serializeStringList(HandleClasses));
}

void DanglingHandleCheck::registerMatchersForVariables(MatchFinder *Finder) {
  const auto ConvertedHandle = handleFromTemporaryValue(IsAHandle);

  // Find 'Handle foo(ReturnsAValue());'
  Finder->addMatcher(
      varDecl(hasType(cxxRecordDecl(IsAHandle)),
              hasInitializer(
                  exprWithCleanups(has(ignoringParenImpCasts(ConvertedHandle)))
                      .bind("bad_stmt"))),
      this);

  // Find 'Handle foo = ReturnsAValue();'
  Finder->addMatcher(
      varDecl(
          hasType(cxxRecordDecl(IsAHandle)), unless(parmVarDecl()),
          hasInitializer(exprWithCleanups(has(ignoringParenImpCasts(handleFrom(
                                              IsAHandle, ConvertedHandle))))
                             .bind("bad_stmt"))),
      this);
  // Find 'foo = ReturnsAValue();  // foo is Handle'
  Finder->addMatcher(
      cxxOperatorCallExpr(callee(cxxMethodDecl(ofClass(IsAHandle))),
                          hasOverloadedOperatorName("="),
                          hasArgument(1, ConvertedHandle))
          .bind("bad_stmt"),
      this);

  // Container insertions that will dangle.
  Finder->addMatcher(makeContainerMatcher(IsAHandle).bind("bad_stmt"), this);
}

void DanglingHandleCheck::registerMatchersForReturn(MatchFinder *Finder) {
  // Return a local.
  Finder->addMatcher(
      returnStmt(
          // The AST contains two constructor calls:
          //   1. Value to Handle conversion.
          //   2. Handle copy construction.
          // We have to match both.
          has(ignoringImplicit(handleFrom(
              IsAHandle,
              handleFrom(IsAHandle, declRefExpr(to(varDecl(
                                        // Is function scope ...
                                        hasAutomaticStorageDuration(),
                                        // ... and it is a local array or Value.
                                        anyOf(hasType(arrayType()),
                                              hasType(recordDecl(
                                                  unless(IsAHandle))))))))))),
          // Temporary fix for false positives inside lambdas.
          unless(hasAncestor(lambdaExpr())))
          .bind("bad_stmt"),
      this);

  // Return a temporary.
  Finder->addMatcher(
      returnStmt(
          has(ignoringParenImpCasts(exprWithCleanups(has(ignoringParenImpCasts(
              handleFrom(IsAHandle, handleFromTemporaryValue(IsAHandle))))))))
          .bind("bad_stmt"),
      this);
}

void DanglingHandleCheck::registerMatchers(MatchFinder *Finder) {
  registerMatchersForVariables(Finder);
  registerMatchersForReturn(Finder);
}

void DanglingHandleCheck::check(const MatchFinder::MatchResult &Result) {
  auto *Handle = Result.Nodes.getNodeAs<CXXRecordDecl>("handle");
  diag(Result.Nodes.getNodeAs<Stmt>("bad_stmt")->getLocStart(),
       "%0 outlives its value")
      << Handle->getQualifiedNameAsString();
}

} // namespace misc
} // namespace tidy
} // namespace clang
