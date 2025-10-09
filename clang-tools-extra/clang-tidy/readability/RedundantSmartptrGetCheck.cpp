//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantSmartptrGetCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {
internal::Matcher<Expr> callToGet(const internal::Matcher<Decl> &OnClass) {
  return expr(
             anyOf(cxxMemberCallExpr(
                       on(expr(anyOf(hasType(OnClass),
                                     hasType(qualType(pointsTo(
                                         decl(OnClass).bind("ptr_to_ptr"))))))
                              .bind("smart_pointer")),
                       unless(callee(
                           memberExpr(hasObjectExpression(cxxThisExpr())))),
                       callee(cxxMethodDecl(hasName("get"),
                                            returns(qualType(pointsTo(
                                                type().bind("getType"))))))),
                   cxxDependentScopeMemberExpr(
                       hasMemberName("get"),
                       hasObjectExpression(
                           expr(hasType(qualType(hasCanonicalType(
                                    templateSpecializationType(hasDeclaration(
                                        classTemplateDecl(has(cxxRecordDecl(
                                            OnClass,
                                            hasMethod(cxxMethodDecl(
                                                hasName("get"),
                                                returns(qualType(
                                                    pointsTo(type().bind(
                                                        "getType")))))))))))))))
                               .bind("smart_pointer")))))
      .bind("redundant_get");
}

internal::Matcher<Decl> knownSmartptr() {
  return recordDecl(hasAnyName("::std::unique_ptr", "::std::shared_ptr"));
}

void registerMatchersForGetArrowStart(MatchFinder *Finder,
                                      MatchFinder::MatchCallback *Callback) {
  const auto MatchesOpArrow =
      allOf(hasName("operator->"),
            returns(qualType(pointsTo(type().bind("op->Type")))));
  const auto MatchesOpStar =
      allOf(hasName("operator*"),
            returns(qualType(references(type().bind("op*Type")))));
  const auto HasRelevantOps =
      allOf(anyOf(hasMethod(MatchesOpArrow),
                  has(functionTemplateDecl(has(functionDecl(MatchesOpArrow))))),
            anyOf(hasMethod(MatchesOpStar),
                  has(functionTemplateDecl(has(functionDecl(MatchesOpStar))))));

  const auto QuacksLikeASmartptr =
      cxxRecordDecl(cxxRecordDecl().bind("duck_typing"), HasRelevantOps);

  // Make sure we are not missing the known standard types.
  const auto SmartptrAny = anyOf(knownSmartptr(), QuacksLikeASmartptr);
  const auto SmartptrWithDeref = anyOf(
      cxxRecordDecl(knownSmartptr(), HasRelevantOps), QuacksLikeASmartptr);

  // Catch 'ptr.get()->Foo()'
  Finder->addMatcher(
      memberExpr(expr().bind("memberExpr"), isArrow(),
                 hasObjectExpression(callToGet(SmartptrWithDeref))),
      Callback);

  // Catch '*ptr.get()' or '*ptr->get()'
  Finder->addMatcher(
      unaryOperator(hasOperatorName("*"),
                    hasUnaryOperand(callToGet(SmartptrWithDeref))),
      Callback);

  // Catch '!ptr.get()'
  const auto CallToGetAsBool = callToGet(
      recordDecl(SmartptrAny, has(cxxConversionDecl(returns(booleanType())))));
  Finder->addMatcher(
      unaryOperator(hasOperatorName("!"), hasUnaryOperand(CallToGetAsBool)),
      Callback);

  // Catch 'if(ptr.get())'
  Finder->addMatcher(ifStmt(hasCondition(CallToGetAsBool)), Callback);

  // Catch 'ptr.get() ? X : Y'
  Finder->addMatcher(conditionalOperator(hasCondition(CallToGetAsBool)),
                     Callback);

  Finder->addMatcher(cxxDependentScopeMemberExpr(hasObjectExpression(
                         callExpr(has(callToGet(SmartptrAny))))),
                     Callback);
}

void registerMatchersForGetEquals(MatchFinder *Finder,
                                  MatchFinder::MatchCallback *Callback) {
  // This one is harder to do with duck typing.
  // The operator==/!= that we are looking for might be member or non-member,
  // might be on global namespace or found by ADL, might be a template, etc.
  // For now, lets keep it to the known standard types.

  // Matches against nullptr.
  Finder->addMatcher(
      binaryOperator(hasAnyOperatorName("==", "!="),
                     hasOperands(anyOf(cxxNullPtrLiteralExpr(), gnuNullExpr(),
                                       integerLiteral(equals(0))),
                                 callToGet(knownSmartptr()))),
      Callback);

  // FIXME: Match and fix if (l.get() == r.get()).
}

} // namespace

void RedundantSmartptrGetCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
}

void RedundantSmartptrGetCheck::registerMatchers(MatchFinder *Finder) {
  registerMatchersForGetArrowStart(Finder, this);
  registerMatchersForGetEquals(Finder, this);
}

namespace {
bool allReturnTypesMatch(const MatchFinder::MatchResult &Result) {
  if (Result.Nodes.getNodeAs<Decl>("duck_typing") == nullptr)
    return true;
  // Verify that the types match.
  // We can't do this on the matcher because the type nodes can be different,
  // even though they represent the same type. This difference comes from how
  // the type is referenced (eg. through a typedef, a type trait, etc).
  const Type *OpArrowType =
      Result.Nodes.getNodeAs<Type>("op->Type")->getUnqualifiedDesugaredType();
  const Type *OpStarType =
      Result.Nodes.getNodeAs<Type>("op*Type")->getUnqualifiedDesugaredType();
  const Type *GetType =
      Result.Nodes.getNodeAs<Type>("getType")->getUnqualifiedDesugaredType();
  return OpArrowType == OpStarType && OpArrowType == GetType;
}
} // namespace

void RedundantSmartptrGetCheck::check(const MatchFinder::MatchResult &Result) {
  if (!allReturnTypesMatch(Result))
    return;

  bool IsPtrToPtr = Result.Nodes.getNodeAs<Decl>("ptr_to_ptr") != nullptr;
  bool IsMemberExpr = Result.Nodes.getNodeAs<Expr>("memberExpr") != nullptr;
  const auto *GetCall = Result.Nodes.getNodeAs<Expr>("redundant_get");
  if (GetCall->getBeginLoc().isMacroID() && IgnoreMacros)
    return;

  const auto *Smartptr = Result.Nodes.getNodeAs<Expr>("smart_pointer");

  if (IsPtrToPtr && IsMemberExpr) {
    // Ignore this case (eg. Foo->get()->DoSomething());
    return;
  }

  auto SR = GetCall->getSourceRange();
  // CXXDependentScopeMemberExpr source range does not include parens
  // Extend the source range of the get call to account for them.
  if (isa<CXXDependentScopeMemberExpr>(GetCall))
    SR.setEnd(Lexer::getLocForEndOfToken(SR.getEnd(), 0, *Result.SourceManager,
                                         getLangOpts())
                  .getLocWithOffset(1));

  StringRef SmartptrText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(Smartptr->getSourceRange()),
      *Result.SourceManager, getLangOpts());
  // Check if the last two characters are "->" and remove them
  if (SmartptrText.ends_with("->")) {
    SmartptrText = SmartptrText.drop_back(2);
  }
  // Replace foo->get() with *foo, and foo.get() with foo.
  std::string Replacement = Twine(IsPtrToPtr ? "*" : "", SmartptrText).str();
  diag(GetCall->getBeginLoc(), "redundant get() call on smart pointer")
      << FixItHint::CreateReplacement(SR, Replacement);
}

} // namespace clang::tidy::readability
