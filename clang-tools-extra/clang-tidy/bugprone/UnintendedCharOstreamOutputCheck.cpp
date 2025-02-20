//===--- UnintendedCharOstreamOutputCheck.cpp - clang-tidy ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnintendedCharOstreamOutputCheck.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {

// check if the type is unsigned char or signed char
AST_MATCHER(Type, isNumericChar) {
  return Node.isSpecificBuiltinType(BuiltinType::SChar) ||
         Node.isSpecificBuiltinType(BuiltinType::UChar);
}

// check if the type is char
AST_MATCHER(Type, isChar) {
  return Node.isSpecificBuiltinType(BuiltinType::Char_S) ||
         Node.isSpecificBuiltinType(BuiltinType::Char_U);
}

} // namespace

void UnintendedCharOstreamOutputCheck::registerMatchers(MatchFinder *Finder) {
  auto BasicOstream =
      cxxRecordDecl(hasName("::std::basic_ostream"),
                    // only basic_ostream<char, Traits> has overload operator<<
                    // with char / unsigned char / signed char
                    classTemplateSpecializationDecl(
                        hasTemplateArgument(0, refersToType(isChar()))));
  Finder->addMatcher(
      cxxOperatorCallExpr(
          hasOverloadedOperatorName("<<"),
          hasLHS(hasType(hasUnqualifiedDesugaredType(
              recordType(hasDeclaration(cxxRecordDecl(
                  anyOf(BasicOstream, isDerivedFrom(BasicOstream)))))))),
          hasRHS(hasType(hasUnqualifiedDesugaredType(isNumericChar()))))
          .bind("x"),
      this);
}

void UnintendedCharOstreamOutputCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CXXOperatorCallExpr>("x");
  const Expr *Value = Call->getArg(1);
  diag(Call->getOperatorLoc(),
       "%0 passed to 'operator<<' outputs as character instead of integer. "
       "cast to 'unsigned' to print numeric value or cast to 'char' to print "
       "as character")
      << Value->getType() << Value->getSourceRange();
}

} // namespace clang::tidy::bugprone
