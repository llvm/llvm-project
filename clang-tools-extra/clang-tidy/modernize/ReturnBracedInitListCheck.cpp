//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ReturnBracedInitListCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

void ReturnBracedInitListCheck::registerMatchers(MatchFinder *Finder) {
  auto SemanticallyDifferentContainer = allOf(
      hasDeclaration(
          // Container(size_type count, const T &value,
          //           const Allocator &alloc = Allocator());
          cxxConstructorDecl(parameterCountIs(3),
                             hasParameter(0, hasType(qualType(hasCanonicalType(
                                                 isInteger())))))),
      hasType(cxxRecordDecl(hasAnyName("::std::basic_string", "::std::vector",
                                       "::std::deque", "::std::forward_list",
                                       "::std::list"))));

  auto ConstructExpr =
      cxxConstructExpr(
          unless(anyOf(
              // Skip explicit constructor.
              hasDeclaration(cxxConstructorDecl(isExplicit())),
              // Skip list initialization and constructors with an initializer
              // list.
              isListInitialization(), hasDescendant(initListExpr()),
              // Skip container `vector(size_type, const T&, ...)`.
              SemanticallyDifferentContainer)))
          .bind("ctor");

  Finder->addMatcher(
      returnStmt(hasReturnValue(ConstructExpr),
                 forFunction(functionDecl(returns(unless(anyOf(builtinType(),
                                                               autoType()))))
                                 .bind("fn"))),
      this);
}

void ReturnBracedInitListCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedFunctionDecl = Result.Nodes.getNodeAs<FunctionDecl>("fn");
  const auto *MatchedConstructExpr =
      Result.Nodes.getNodeAs<CXXConstructExpr>("ctor");

  // Don't make replacements in macro.
  SourceLocation Loc = MatchedConstructExpr->getExprLoc();
  if (Loc.isMacroID())
    return;

  // Make sure that the return type matches the constructed type.
  const QualType ReturnType =
      MatchedFunctionDecl->getReturnType().getCanonicalType();
  const QualType ConstructType =
      MatchedConstructExpr->getType().getCanonicalType();
  if (ReturnType != ConstructType)
    return;

  auto Diag = diag(Loc, "avoid repeating the return type from the "
                        "declaration; use a braced initializer list instead");

  const SourceRange CallParensRange =
      MatchedConstructExpr->getParenOrBraceRange();

  // Make sure there is an explicit constructor call.
  if (CallParensRange.isInvalid())
    return;

  // Make sure that the ctor arguments match the declaration.
  for (unsigned I = 0, NumParams = MatchedConstructExpr->getNumArgs();
       I < NumParams; ++I) {
    if (const auto *VD = dyn_cast<VarDecl>(
            MatchedConstructExpr->getConstructor()->getParamDecl(I))) {
      if (MatchedConstructExpr->getArg(I)->getType().getCanonicalType() !=
          VD->getType().getCanonicalType())
        return;
    }
  }

  // Range for constructor name and opening brace.
  CharSourceRange CtorCallSourceRange = CharSourceRange::getTokenRange(
      Loc, CallParensRange.getBegin().getLocWithOffset(-1));

  Diag << FixItHint::CreateRemoval(CtorCallSourceRange)
       << FixItHint::CreateReplacement(CallParensRange.getBegin(), "{")
       << FixItHint::CreateReplacement(CallParensRange.getEnd(), "}");
}

} // namespace clang::tidy::modernize
