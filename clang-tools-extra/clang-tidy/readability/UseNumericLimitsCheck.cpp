//===--- UseNumericLimitsCheck.cpp - clang-tidy ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseNumericLimitsCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Preprocessor.h"
#include <cmath>
#include <limits>

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

UseNumericLimitsCheck::UseNumericLimitsCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      SignedConstants{
          {std::numeric_limits<int8_t>::min(),
           "std::numeric_limits<int8_t>::min()"},
          {std::numeric_limits<int8_t>::max(),
           "std::numeric_limits<int8_t>::max()"},
          {std::numeric_limits<int16_t>::min(),
           "std::numeric_limits<int16_t>::min()"},
          {std::numeric_limits<int16_t>::max(),
           "std::numeric_limits<int16_t>::max()"},
          {std::numeric_limits<int32_t>::min(),
           "std::numeric_limits<int32_t>::min()"},
          {std::numeric_limits<int32_t>::max(),
           "std::numeric_limits<int32_t>::max()"},
          {std::numeric_limits<int64_t>::min(),
           "std::numeric_limits<int64_t>::min()"},
          {std::numeric_limits<int64_t>::max(),
           "std::numeric_limits<int64_t>::max()"},

      },
      UnsignedConstants{
          {std::numeric_limits<uint8_t>::max(),
           "std::numeric_limits<uint8_t>::max()"},
          {std::numeric_limits<uint16_t>::max(),
           "std::numeric_limits<uint16_t>::max()"},
          {std::numeric_limits<uint32_t>::max(),
           "std::numeric_limits<uint32_t>::max()"},
          {std::numeric_limits<uint64_t>::max(),
           "std::numeric_limits<uint64_t>::max()"},
      },
      Inserter(Options.getLocalOrGlobal("IncludeStyle",
                                        utils::IncludeSorter::IS_LLVM),
               areDiagsSelfContained()) {}

void UseNumericLimitsCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      expr(anyOf(unaryOperator(hasOperatorName("-"),
                               hasUnaryOperand(integerLiteral().bind(
                                   "negative-integer-literal"))),
                 unaryOperator(hasOperatorName("+"),
                               hasUnaryOperand(integerLiteral().bind(
                                   "positive-integer-literal"))),
                 integerLiteral(unless(hasParent(unaryOperator(
                                    hasAnyOperatorName("-", "+")))))
                     .bind("bare-integer-literal")))
          .bind("unary-op-exp"),
      this);
}

void UseNumericLimitsCheck::registerPPCallbacks(
    const SourceManager &SM, Preprocessor *PP, Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void UseNumericLimitsCheck::check(const MatchFinder::MatchResult &Result) {
  const IntegerLiteral *MatchedDecl = nullptr;

  const IntegerLiteral *NegativeMatchedDecl =
      Result.Nodes.getNodeAs<IntegerLiteral>("negative-integer-literal");
  const IntegerLiteral *PositiveMatchedDecl =
      Result.Nodes.getNodeAs<IntegerLiteral>("positive-integer-literal");
  const IntegerLiteral *BareMatchedDecl =
      Result.Nodes.getNodeAs<IntegerLiteral>("bare-integer-literal");

  if (NegativeMatchedDecl != nullptr) {
    MatchedDecl = NegativeMatchedDecl;
  } else if (PositiveMatchedDecl != nullptr) {
    MatchedDecl = PositiveMatchedDecl;
  } else if (BareMatchedDecl != nullptr) {
    MatchedDecl = BareMatchedDecl;
  }

  const llvm::APInt MatchedIntegerConstant = MatchedDecl->getValue();

  auto Fixer = [&](auto SourceValue, auto Value,
                   const std::string &Replacement) {
    SourceLocation Location = MatchedDecl->getExprLoc();
    SourceRange Range{MatchedDecl->getBeginLoc(), MatchedDecl->getEndLoc()};

    // Only valid if unary operator is present
    const UnaryOperator *UnaryOpExpr =
        Result.Nodes.getNodeAs<UnaryOperator>("unary-op-exp");

    if (MatchedDecl == NegativeMatchedDecl && -SourceValue == Value) {
      Range = SourceRange(UnaryOpExpr->getBeginLoc(), UnaryOpExpr->getEndLoc());
      Location = UnaryOpExpr->getExprLoc();
      SourceValue = -SourceValue;
    } else if (MatchedDecl == PositiveMatchedDecl && SourceValue == Value) {
      Range = SourceRange(UnaryOpExpr->getBeginLoc(), UnaryOpExpr->getEndLoc());
      Location = UnaryOpExpr->getExprLoc();
    } else if (MatchedDecl != BareMatchedDecl || SourceValue != Value) {
      return;
    }

    diag(Location,
         "The constant %0 is being utilized. Consider using %1 instead")
        << std::to_string(SourceValue) << Replacement
        << FixItHint::CreateReplacement(Range, Replacement)
        << Inserter.createIncludeInsertion(
               Result.SourceManager->getFileID(Location), "<limits>");
  };

  for (const auto &[Value, Replacement] : SignedConstants) {
    Fixer(MatchedIntegerConstant.getSExtValue(), Value, Replacement);
  }

  for (const auto &[Value, Replacement] : UnsignedConstants) {
    Fixer(MatchedIntegerConstant.getZExtValue(), Value, Replacement);
  }
}

} // namespace clang::tidy::readability
