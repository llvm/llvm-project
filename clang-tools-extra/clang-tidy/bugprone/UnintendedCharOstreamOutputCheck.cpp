//===--- UnintendedCharOstreamOutputCheck.cpp - clang-tidy ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UnintendedCharOstreamOutputCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Tooling/FixIt.h"

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

UnintendedCharOstreamOutputCheck::UnintendedCharOstreamOutputCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AllowedTypes(utils::options::parseStringList(
          Options.get("AllowedTypes", "unsigned char;signed char"))),
      CastTypeName(Options.get("CastTypeName")) {}
void UnintendedCharOstreamOutputCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "AllowedTypes",
                utils::options::serializeStringList(AllowedTypes));
  if (CastTypeName.has_value())
    Options.store(Opts, "CastTypeName", CastTypeName.value());
}

void UnintendedCharOstreamOutputCheck::registerMatchers(MatchFinder *Finder) {
  auto BasicOstream =
      cxxRecordDecl(hasName("::std::basic_ostream"),
                    // only basic_ostream<char, Traits> has overload operator<<
                    // with char / unsigned char / signed char
                    classTemplateSpecializationDecl(
                        hasTemplateArgument(0, refersToType(isChar()))));
  auto IsDeclRefExprFromAllowedTypes = declRefExpr(to(varDecl(
      hasType(matchers::matchesAnyListedTypeName(AllowedTypes, false)))));
  auto IsExplicitCastExprFromAllowedTypes = explicitCastExpr(hasDestinationType(
      matchers::matchesAnyListedTypeName(AllowedTypes, false)));
  Finder->addMatcher(
      cxxOperatorCallExpr(
          hasOverloadedOperatorName("<<"),
          hasLHS(hasType(hasUnqualifiedDesugaredType(
              recordType(hasDeclaration(cxxRecordDecl(
                  anyOf(BasicOstream, isDerivedFrom(BasicOstream)))))))),
          hasRHS(expr(hasType(hasUnqualifiedDesugaredType(isNumericChar())),
                      unless(ignoringParenImpCasts(
                          anyOf(IsDeclRefExprFromAllowedTypes,
                                IsExplicitCastExprFromAllowedTypes))))))
          .bind("x"),
      this);
}

void UnintendedCharOstreamOutputCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CXXOperatorCallExpr>("x");
  const Expr *Value = Call->getArg(1);
  const SourceRange SourceRange = Value->getSourceRange();

  DiagnosticBuilder Builder =
      diag(Call->getOperatorLoc(),
           "%0 passed to 'operator<<' outputs as character instead of integer. "
           "cast to 'unsigned int' to print numeric value or cast to 'char' to "
           "print as character")
      << Value->getType() << SourceRange;

  QualType T = Value->getType();
  const Type *UnqualifiedDesugaredType = T->getUnqualifiedDesugaredType();

  llvm::StringRef CastType = CastTypeName.value_or(
      UnqualifiedDesugaredType->isSpecificBuiltinType(BuiltinType::SChar)
          ? "int"
          : "unsigned int");

  Builder << FixItHint::CreateReplacement(
      SourceRange, ("static_cast<" + CastType + ">(" +
                    tooling::fixit::getText(*Value, *Result.Context) + ")")
                       .str());
}

} // namespace clang::tidy::bugprone
