//===--- NlohmannJsonExplicitConversionsCheck.cpp - clang-tidy ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NlohmannJsonExplicitConversionsCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

void NlohmannJsonExplicitConversionsCheck::registerMatchers(
    MatchFinder *Finder) {
  auto Matcher =
      cxxMemberCallExpr(
          on(expr().bind("arg")),
          callee(cxxConversionDecl(ofClass(hasName("nlohmann::basic_json")))
                     .bind("conversionDecl")))
          .bind("conversionCall");
  Finder->addMatcher(Matcher, this);
}

void NlohmannJsonExplicitConversionsCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Decl =
      Result.Nodes.getNodeAs<CXXConversionDecl>("conversionDecl");
  const auto *Call =
      Result.Nodes.getNodeAs<CXXMemberCallExpr>("conversionCall");
  const auto *Arg = Result.Nodes.getNodeAs<Expr>("arg");

  const QualType DestinationType = Decl->getConversionType();
  std::string DestinationTypeStr =
      DestinationType.getAsString(Result.Context->getPrintingPolicy());
  if (DestinationTypeStr == "std::basic_string<char>")
    DestinationTypeStr = "std::string";

  SourceRange ExprRange = Call->getSourceRange();
  if (!ExprRange.isValid())
    return;

  bool Deref = false;
  if (const auto *Op = llvm::dyn_cast<UnaryOperator>(Arg);
      Op && Op->getOpcode() == UO_Deref)
    Deref = true;
  else if (const auto *Op = llvm::dyn_cast<CXXOperatorCallExpr>(Arg);
           Op && Op->getOperator() == OO_Star)
    Deref = true;

  llvm::StringRef SourceText = clang::Lexer::getSourceText(
      clang::CharSourceRange::getTokenRange(ExprRange), *Result.SourceManager,
      Result.Context->getLangOpts());

  if (Deref)
    SourceText.consume_front("*");

  const std::string ReplacementText =
      (llvm::Twine(SourceText) + (Deref ? "->" : ".") + "get<" +
       DestinationTypeStr + ">()")
          .str();
  diag(Call->getExprLoc(),
       "implicit nlohmann::json conversion to %0 should be explicit")
      << DestinationTypeStr
      << FixItHint::CreateReplacement(CharSourceRange::getTokenRange(ExprRange),
                                      ReplacementText);
}

} // namespace clang::tidy::modernize
