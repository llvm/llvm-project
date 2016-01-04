//===--- UnnecessaryCopyInitialization.cpp - clang-tidy--------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "UnnecessaryCopyInitialization.h"

#include "../utils/LexerUtils.h"
#include "../utils/Matchers.h"

namespace clang {
namespace tidy {
namespace performance {

using namespace ::clang::ast_matchers;

namespace {
AST_MATCHER(QualType, isPointerType) { return Node->isPointerType(); }
} // namespace

void UnnecessaryCopyInitialization::registerMatchers(
    ast_matchers::MatchFinder *Finder) {
  auto ConstReference = referenceType(pointee(qualType(isConstQualified())));
  auto ConstOrConstReference =
      allOf(anyOf(ConstReference, isConstQualified()),
            unless(allOf(isPointerType(), unless(pointerType(pointee(qualType(
                                              isConstQualified())))))));
  // Match method call expressions where the this argument is a const
  // type or const reference. This returned const reference is highly likely to
  // outlive the local const reference of the variable being declared.
  // The assumption is that the const reference being returned either points
  // to a global static variable or to a member of the called object.
  auto ConstRefReturningMethodCallOfConstParam = cxxMemberCallExpr(
      callee(cxxMethodDecl(returns(ConstReference))),
      on(declRefExpr(to(varDecl(hasType(qualType(ConstOrConstReference)))))));
  auto ConstRefReturningFunctionCall =
      callExpr(callee(functionDecl(returns(ConstReference))),
               unless(callee(cxxMethodDecl())));
  Finder->addMatcher(
      varDecl(
          hasLocalStorage(), hasType(isConstQualified()),
          hasType(matchers::isExpensiveToCopy()),
          hasInitializer(cxxConstructExpr(
              hasDeclaration(cxxConstructorDecl(isCopyConstructor())),
              hasArgument(0, anyOf(ConstRefReturningFunctionCall,
                                   ConstRefReturningMethodCallOfConstParam)))))
          .bind("varDecl"),
      this);
}

void UnnecessaryCopyInitialization::check(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const auto *Var = Result.Nodes.getNodeAs<VarDecl>("varDecl");
  SourceLocation AmpLocation = Var->getLocation();
  auto Token = lexer_utils::getPreviousNonCommentToken(*Result.Context,
                                                       Var->getLocation());
  if (!Token.is(tok::unknown)) {
    AmpLocation = Token.getLocation().getLocWithOffset(Token.getLength());
  }
  diag(Var->getLocation(),
       "the const qualified variable '%0' is copy-constructed from a "
       "const reference; consider making it a const reference")
      << Var->getName() << FixItHint::CreateInsertion(AmpLocation, "&");
}

} // namespace performance
} // namespace tidy
} // namespace clang
