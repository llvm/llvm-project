//===--- RedundantSmartptrGetCheck.cpp - clang-tidy -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RedundantSmartptrGetCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace readability {

namespace {
internal::Matcher<Expr> callToGet(const internal::Matcher<Decl> &OnClass) {
  return cxxMemberCallExpr(
             on(expr(anyOf(hasType(OnClass),
                           hasType(qualType(
                               pointsTo(decl(OnClass).bind("ptr_to_ptr"))))))
                    .bind("smart_pointer")),
             unless(callee(memberExpr(hasObjectExpression(cxxThisExpr())))),
             callee(cxxMethodDecl(
                 hasName("get"),
                 returns(qualType(pointsTo(type().bind("getType")))))))
      .bind("redundant_get");
}

void registerMatchersForGetArrowStart(MatchFinder *Finder,
                                      MatchFinder::MatchCallback *Callback) {
  const auto QuacksLikeASmartptr = recordDecl(
      recordDecl().bind("duck_typing"),
      has(cxxMethodDecl(hasName("operator->"),
                        returns(qualType(pointsTo(type().bind("op->Type")))))),
      has(cxxMethodDecl(hasName("operator*"), returns(qualType(references(
                                                  type().bind("op*Type")))))));

  // Catch 'ptr.get()->Foo()'
  Finder->addMatcher(memberExpr(expr().bind("memberExpr"), isArrow(),
                                hasObjectExpression(ignoringImpCasts(
                                    callToGet(QuacksLikeASmartptr)))),
                     Callback);

  // Catch '*ptr.get()' or '*ptr->get()'
  Finder->addMatcher(
      unaryOperator(hasOperatorName("*"),
                    hasUnaryOperand(callToGet(QuacksLikeASmartptr))),
      Callback);
}

void registerMatchersForGetEquals(MatchFinder *Finder,
                                  MatchFinder::MatchCallback *Callback) {
  // This one is harder to do with duck typing.
  // The operator==/!= that we are looking for might be member or non-member,
  // might be on global namespace or found by ADL, might be a template, etc.
  // For now, lets keep a list of known standard types.

  const auto IsAKnownSmartptr = recordDecl(
      anyOf(hasName("::std::unique_ptr"), hasName("::std::shared_ptr")));

  // Matches against nullptr.
  Finder->addMatcher(
      binaryOperator(
          anyOf(hasOperatorName("=="), hasOperatorName("!=")),
          hasEitherOperand(ignoringImpCasts(cxxNullPtrLiteralExpr())),
          hasEitherOperand(callToGet(IsAKnownSmartptr))),
      Callback);
  // TODO: Catch ptr.get() == other_ptr.get()
}


}  // namespace

void RedundantSmartptrGetCheck::registerMatchers(MatchFinder *Finder) {
  // Only register the matchers for C++; the functionality currently does not
  // provide any benefit to other languages, despite being benign.
  if (!getLangOpts().CPlusPlus)
    return;

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
}  // namespace

void RedundantSmartptrGetCheck::check(const MatchFinder::MatchResult &Result) {
  if (!allReturnTypesMatch(Result)) return;

  bool IsPtrToPtr = Result.Nodes.getNodeAs<Decl>("ptr_to_ptr") != nullptr;
  bool IsMemberExpr = Result.Nodes.getNodeAs<Expr>("memberExpr") != nullptr;
  const Expr *GetCall = Result.Nodes.getNodeAs<Expr>("redundant_get");
  const Expr *Smartptr = Result.Nodes.getNodeAs<Expr>("smart_pointer");

  if (IsPtrToPtr && IsMemberExpr) {
    // Ignore this case (eg. Foo->get()->DoSomething());
    return;
  }

  StringRef SmartptrText = Lexer::getSourceText(
      CharSourceRange::getTokenRange(Smartptr->getSourceRange()),
      *Result.SourceManager, getLangOpts());
  // Replace foo->get() with *foo, and foo.get() with foo.
  std::string Replacement = Twine(IsPtrToPtr ? "*" : "", SmartptrText).str();
  diag(GetCall->getLocStart(), "redundant get() call on smart pointer")
      << FixItHint::CreateReplacement(GetCall->getSourceRange(), Replacement);
}

} // namespace readability
} // namespace tidy
} // namespace clang
