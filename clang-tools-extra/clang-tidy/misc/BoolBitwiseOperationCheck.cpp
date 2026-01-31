//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BoolBitwiseOperationCheck.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/Casting.h"
#include <array>
#include <utility>

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

static constexpr std::array<std::pair<StringRef, StringRef>, 8U>
    OperatorsTransformation{{{"|", "||"},
                             {"|=", "||"},
                             {"&", "&&"},
                             {"&=", "&&"},
                             {"bitand", "and"},
                             {"and_eq", "and"},
                             {"bitor", "or"},
                             {"or_eq", "or"}}};

static StringRef translate(StringRef Value) {
  for (const auto &[Bitwise, Logical] : OperatorsTransformation)
    if (Value == Bitwise)
      return Logical;

  return {};
}

static bool isBitwiseOperation(StringRef Value) {
  return llvm::is_contained(llvm::make_first_range(OperatorsTransformation),
                            Value);
}

static std::optional<CharSourceRange>
getOperatorTokenRangeForFixIt(const BinaryOperator *BinOp,
                              const SourceManager &SM,
                              const LangOptions &LangOpts) {
  SourceLocation Loc = BinOp->getOperatorLoc();
  if (Loc.isInvalid() || Loc.isMacroID())
    return std::nullopt;

  Loc = SM.getSpellingLoc(Loc);
  if (Loc.isInvalid() || Loc.isMacroID())
    return std::nullopt;

  CharSourceRange TokenRange = CharSourceRange::getTokenRange(Loc);
  if (TokenRange.isInvalid())
    return std::nullopt;

  return TokenRange;
}

/// Checks if all leaf nodes in a bitwise expression satisfy a given condition.
///
/// \param Expr The bitwise expression to check.
/// \param Condition A function that checks if a leaf node satisfies the
///                  desired condition.
/// \returns true if the condition is satisfied according to the combiner logic.
template <typename F>
static bool leavesOfBitwiseSatisfy(const clang::Expr *Expr,
                                   const F &Condition) {
  // Strip away implicit casts and parentheses before checking the condition.
  // This is important for cases like:
  //   bool b1, b2;
  //   bool Deprecated = 0xFFFFFFF8 & (b1 & b2);
  // where the operands of the inner '&' are represented in the AST as
  //   ImplicitCastExpr <int> (ImplicitCastExpr <bool> (DeclRefExpr 'bool'))
  // and we still want to classify the leaves as boolean.
  Expr = Expr->IgnoreParenImpCasts();

  // For leaf nodes, check if the condition is satisfied after stripping
  // implicit casts/parens.
  if (Condition(Expr))
    return true;

  // If it's a binary operator, recursively check both operands.
  if (const auto *BinOp = dyn_cast<clang::BinaryOperator>(Expr)) {
    if (!isBitwiseOperation(BinOp->getOpcodeStr()))
      return false;
    return leavesOfBitwiseSatisfy(BinOp->getLHS(), Condition) &&
           leavesOfBitwiseSatisfy(BinOp->getRHS(), Condition);
  }

  return false;
}

namespace {

// FIXME: provide memoization for this matcher

/// Custom matcher that checks if all leaf nodes in an bitwise expression
/// satisfy the given inner matcher condition. This uses
/// leavesOfBitwiseSatisfy to recursively check.
///
/// Example usage:
///   expr(hasAllLeavesOfBitwiseSatisfying(hasType(booleanType())))
AST_MATCHER_P(Expr, hasAllLeavesOfBitwiseSatisfying,
              ast_matchers::internal::Matcher<Expr>, InnerMatcher) {
  auto Condition = [&](const clang::Expr *E) -> bool {
    return InnerMatcher.matches(*E, Finder, Builder);
  };
  return leavesOfBitwiseSatisfy(&Node, Condition);
}

AST_MATCHER_P(Expr, hasSideEffects, bool, IncludePossibleEffects) {
  auto &Ctx = Finder->getASTContext();
  return Node.HasSideEffects(Ctx, IncludePossibleEffects);
}

} // namespace

BoolBitwiseOperationCheck::BoolBitwiseOperationCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      UnsafeMode(Options.get("UnsafeMode", false)),
      IgnoreMacros(Options.get("IgnoreMacros", false)),
      StrictMode(Options.get("StrictMode", true)) {}

void BoolBitwiseOperationCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "UnsafeMode", UnsafeMode);
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
  Options.store(Opts, "StrictMode", StrictMode);
}

void BoolBitwiseOperationCheck::registerMatchers(MatchFinder *Finder) {
  auto BooleanLeaves = hasAllLeavesOfBitwiseSatisfying(hasType(booleanType()));
  auto NonVolatile = ignoringImpCasts(unless(hasType(isVolatileQualified())));

  auto FixItMatcher = binaryOperator(
      // Both operands must be non-volatile at the top level.
      hasOperands(NonVolatile, NonVolatile),
      hasRHS(unless(hasSideEffects(/*IncludePossibleEffects=*/!UnsafeMode))),
      anyOf(
          // Non-compound assignments: no additional LHS
          // restriction needed.
          hasAnyOperatorName("|", "&"),
          // Compound assignments ('|=' / '&='): require a simple
          // LHS so that we can safely duplicate it on the RHS.
          allOf(hasAnyOperatorName("|=", "&="),
                hasLHS(ignoringImpCasts(anyOf(declRefExpr(), memberExpr()))))));

  // Parentheses decision logic:
  // Case 1: | with parent && → parens needed around BinOp (the || result)
  //         e.g., a && b | c → a && (b || c)
  // Case 2: & with parent ^ or | → parens needed around BinOp (the && result)
  //         e.g., a ^ b & c → a ^ (b && c)
  // Case 3: &= with RHS || → parens needed around RHS
  //         e.g., a &= b || c → a = a && (b || c)
  //
  // If the expression is already wrapped in ParenExpr, no additional parens
  // are needed. For cases 1 & 2, if BinOp is in parens, its parent is
  // ParenExpr (not binaryOperator), so hasParent won't match.
  // For case 3, if RHS is in parens, hasRHS(binaryOperator(||)) won't match.

  // Case 1: | with && parent
  auto ParensCase1 = allOf(
      hasOperatorName("|"),
      hasParent(binaryOperator(hasOperatorName("&&")).bind("parensParent")));

  // Case 2: & with ^ or | parent
  auto ParensCase2 = allOf(
      hasOperatorName("&"),
      hasParent(
          binaryOperator(hasAnyOperatorName("^", "|")).bind("parensParent")));

  // Case 3: &= with || RHS
  auto ParensCase3 =
      allOf(hasOperatorName("&="),
            hasRHS(binaryOperator(hasOperatorName("||")).bind("parensExpr")));

  auto BaseMatcher = binaryOperator(
      hasAnyOperatorName("|", "&", "|=", "&="), hasLHS(BooleanLeaves),
      hasRHS(BooleanLeaves), optionally(FixItMatcher.bind("fixit")),
      optionally(anyOf(ParensCase1, ParensCase2)), optionally(ParensCase3));

  Finder->addMatcher(BaseMatcher.bind("binOp"), this);
}

void BoolBitwiseOperationCheck::emitWarningAndChangeOperatorsIfPossible(
    const BinaryOperator *BinOp, const Expr *ParensExpr,
    const clang::SourceManager &SM, clang::ASTContext &Ctx,
    bool CanApplyFixIt) {
  auto DiagEmitter = [BinOp, this] {
    return diag(BinOp->getOperatorLoc(),
                "use logical operator '%0' for boolean semantics instead of "
                "bitwise operator '%1'")
           << translate(BinOp->getOpcodeStr()) << BinOp->getOpcodeStr();
  };

  auto DiagEmitterForStrictMode = [&] {
    if (StrictMode)
      DiagEmitter();
    return true;
  };

  if (!CanApplyFixIt) {
    DiagEmitterForStrictMode();
    return;
  }

  const auto MaybeTokenRange =
      getOperatorTokenRangeForFixIt(BinOp, SM, Ctx.getLangOpts());
  if (!MaybeTokenRange) {
    IgnoreMacros || DiagEmitterForStrictMode();
    return;
  }
  const CharSourceRange TokenRange = *MaybeTokenRange;

  const StringRef FixSpelling =
      translate(Lexer::getSourceText(TokenRange, SM, Ctx.getLangOpts()));

  if (FixSpelling.empty()) {
    DiagEmitterForStrictMode();
    return;
  }

  FixItHint InsertEqual;
  if (BinOp->isCompoundAssignmentOp()) {
    const Expr *LHS = BinOp->getLHS()->IgnoreImpCasts();
    // the matcher ensures that `LHS` is a simple
    // declaration or member expression suitable for duplication.
    const SourceLocation LocLHS = LHS->getEndLoc();
    if (LocLHS.isInvalid() || LocLHS.isMacroID()) {
      IgnoreMacros || DiagEmitterForStrictMode();
      return;
    }
    const SourceLocation InsertLoc =
        clang::Lexer::getLocForEndOfToken(LocLHS, 0, SM, Ctx.getLangOpts());
    if (InsertLoc.isInvalid() || InsertLoc.isMacroID()) {
      IgnoreMacros || DiagEmitterForStrictMode();
      return;
    }
    auto SourceText = static_cast<std::string>(Lexer::getSourceText(
        CharSourceRange::getTokenRange(LHS->getSourceRange()), SM,
        Ctx.getLangOpts()));
    llvm::erase_if(SourceText,
                   [](unsigned char Ch) { return std::isspace(Ch); });
    InsertEqual = FixItHint::CreateInsertion(InsertLoc, " = " + SourceText);
  }

  auto ReplaceOperator = FixItHint::CreateReplacement(TokenRange, FixSpelling);

  // Generate parentheses fix-its if ParensExpr is provided.
  // The matcher already determined which expression needs parentheses
  // and skipped if already wrapped in ParenExpr.
  FixItHint InsertBrace1;
  FixItHint InsertBrace2;
  if (ParensExpr) {
    const SourceLocation InsertFirstLoc = ParensExpr->getBeginLoc();
    const SourceLocation InsertSecondLoc = clang::Lexer::getLocForEndOfToken(
        ParensExpr->getEndLoc(), 0, SM, Ctx.getLangOpts());
    if (InsertFirstLoc.isInvalid() || InsertFirstLoc.isMacroID() ||
        InsertSecondLoc.isInvalid() || InsertSecondLoc.isMacroID()) {
      IgnoreMacros || DiagEmitterForStrictMode();
      return;
    }
    InsertBrace1 = FixItHint::CreateInsertion(InsertFirstLoc, "(");
    InsertBrace2 = FixItHint::CreateInsertion(InsertSecondLoc, ")");
  }

  DiagEmitter() << InsertEqual << ReplaceOperator << InsertBrace1
                << InsertBrace2;
}

void BoolBitwiseOperationCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *BinOp = Result.Nodes.getNodeAs<BinaryOperator>("binOp");
  const auto *FixItBinOp = Result.Nodes.getNodeAs<BinaryOperator>("fixit");
  assert(BinOp);

  // Determine if parentheses are needed and around which expression.
  // - parensParent bound → cases 1 & 2: parens around BinOp itself
  // - parensExpr bound → case 3: parens around the bound RHS expression
  const auto *ParensParent =
      Result.Nodes.getNodeAs<BinaryOperator>("parensParent");
  const auto *ParensExprRHS = Result.Nodes.getNodeAs<Expr>("parensExpr");

  const Expr *PE = nullptr;
  if (ParensParent) {
    // Cases 1 & 2: parens around BinOp (| or &)
    PE = BinOp;
  } else if (ParensExprRHS) {
    // Case 3: parens around RHS (||)
    PE = ParensExprRHS;
  }

  if (isa_and_nonnull<ParenExpr>(PE)) {
    PE = nullptr;
  }

  const SourceManager &SM = *Result.SourceManager;
  ASTContext &Ctx = *Result.Context;

  const bool CanApplyFixIt = (FixItBinOp != nullptr && FixItBinOp == BinOp);
  emitWarningAndChangeOperatorsIfPossible(BinOp, PE, SM, Ctx,
                                          CanApplyFixIt);
}

} // namespace clang::tidy::misc
