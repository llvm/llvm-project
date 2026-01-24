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

static const Expr *getAcceptableCompoundsLHS(const BinaryOperator *BinOp) {
  assert(BinOp->isCompoundAssignmentOp());
  const Expr *LHS = BinOp->getLHS()->IgnoreImpCasts();
  return isa<DeclRefExpr, MemberExpr>(LHS) ? LHS : nullptr;
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
  Finder->addMatcher(
      binaryOperator(hasAnyOperatorName("|", "&", "|=", "&="),
                     optionally(hasParent(binaryOperator().bind("p"))),
                     allOf(hasLHS(BooleanLeaves), hasRHS(BooleanLeaves)))
          .bind("binOp"),
      this);
}

void BoolBitwiseOperationCheck::emitWarningAndChangeOperatorsIfPossible(
    const BinaryOperator *BinOp, const BinaryOperator *ParentBinOp,
    const clang::SourceManager &SM, clang::ASTContext &Ctx) {
  auto DiagEmitter = [BinOp, this] {
    return diag(BinOp->getOperatorLoc(),
                "use logical operator '%0' for boolean semantics instead of "
                "bitwise operator '%1'")
           << translate(BinOp->getOpcodeStr()) << BinOp->getOpcodeStr();
  };

  auto DiagEmitterForStrictMode = [&] {
    if (StrictMode)
      DiagEmitter();
  };

  // Helper lambda to check if location is valid and not in a macro
  auto IsValidLocation = [&](SourceLocation Loc) -> bool {
    if (Loc.isInvalid() || Loc.isMacroID()) {
      if (!IgnoreMacros)
        DiagEmitterForStrictMode();
      return false;
    }
    return true;
  };

  // Early validation: check for volatile operands
  const bool HasVolatileOperand = llvm::any_of(
      std::array{BinOp->getLHS(), BinOp->getRHS()}, [&](const Expr *E) {
        return E->IgnoreImpCasts()->getType().isVolatileQualified();
      });
  if (HasVolatileOperand) {
    DiagEmitterForStrictMode();
    return;
  }

  // Early validation: check for side effects
  const bool HasSideEffects = BinOp->getRHS()->HasSideEffects(
      Ctx, /*IncludePossibleEffects=*/!UnsafeMode);
  if (HasSideEffects) {
    DiagEmitterForStrictMode();
    return;
  }

  // Get and validate operator location
  SourceLocation OpLoc = BinOp->getOperatorLoc();
  if (!IsValidLocation(OpLoc))
    return;

  OpLoc = SM.getSpellingLoc(OpLoc);
  if (!IsValidLocation(OpLoc))
    return;

  // Generate fix-it hint for operator replacement
  const CharSourceRange TokenRange = CharSourceRange::getTokenRange(OpLoc);
  if (TokenRange.isInvalid()) {
    if (!IgnoreMacros)
      DiagEmitterForStrictMode();
    return;
  }

  const StringRef FixSpelling =
      translate(Lexer::getSourceText(TokenRange, SM, Ctx.getLangOpts()));
  if (FixSpelling.empty()) {
    DiagEmitterForStrictMode();
    return;
  }

  const FixItHint ReplaceOpHint =
      FixItHint::CreateReplacement(TokenRange, FixSpelling);

  // Generate fix-it hint for compound assignment (if applicable)
  FixItHint InsertEqualHint;
  if (BinOp->isCompoundAssignmentOp()) {
    const auto *LHS = getAcceptableCompoundsLHS(BinOp);
    if (!LHS) {
      DiagEmitterForStrictMode();
      return;
    }

    const SourceLocation LocLHS = LHS->getEndLoc();
    if (!IsValidLocation(LocLHS))
      return;

    const SourceLocation InsertLoc =
        clang::Lexer::getLocForEndOfToken(LocLHS, 0, SM, Ctx.getLangOpts());
    if (!IsValidLocation(InsertLoc))
      return;

    auto SourceText = static_cast<std::string>(Lexer::getSourceText(
        CharSourceRange::getTokenRange(LHS->getSourceRange()), SM,
        Ctx.getLangOpts()));
    llvm::erase_if(SourceText,
                   [](unsigned char Ch) { return std::isspace(Ch); });
    InsertEqualHint = FixItHint::CreateInsertion(InsertLoc, " = " + SourceText);
  }

  // Determine if parentheses are needed based on operator precedence
  const Expr *SurroundedExpr = nullptr;
  if (ParentBinOp) {
    const BinaryOperatorKind ParentOpcode = ParentBinOp->getOpcode();
    if ((BinOp->getOpcode() == BO_Or && ParentOpcode == BO_LAnd) ||
        (BinOp->getOpcode() == BO_And &&
         llvm::is_contained({BO_Xor, BO_Or}, ParentOpcode))) {
      const Expr *Side = ParentBinOp->getLHS()->IgnoreParenImpCasts() == BinOp
                             ? ParentBinOp->getLHS()
                             : ParentBinOp->getRHS();
      SurroundedExpr = Side->IgnoreImpCasts();
      assert(SurroundedExpr->IgnoreParens() == BinOp);
    }
  }

  if (!SurroundedExpr) {
    const auto *RHS =
        dyn_cast<BinaryOperator>(BinOp->getRHS()->IgnoreImpCasts());
    if (RHS && BinOp->getOpcode() == BO_AndAssign && RHS->getOpcode() == BO_LOr)
      SurroundedExpr = RHS;
  }

  if (isa_and_nonnull<ParenExpr>(SurroundedExpr))
    SurroundedExpr = nullptr;

  // Generate fix-it hints for parentheses (if needed)
  FixItHint InsertBrace1;
  FixItHint InsertBrace2;
  if (SurroundedExpr) {
    const SourceLocation InsertFirstLoc = SurroundedExpr->getBeginLoc();
    const SourceLocation InsertSecondLoc = clang::Lexer::getLocForEndOfToken(
        SurroundedExpr->getEndLoc(), 0, SM, Ctx.getLangOpts());
    if (!IsValidLocation(InsertFirstLoc) || !IsValidLocation(InsertSecondLoc))
      return;

    InsertBrace1 = FixItHint::CreateInsertion(InsertFirstLoc, "(");
    InsertBrace2 = FixItHint::CreateInsertion(InsertSecondLoc, ")");
  }

  DiagEmitter() << InsertEqualHint << ReplaceOpHint << InsertBrace1
                << InsertBrace2;
}

void BoolBitwiseOperationCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *BinOp = Result.Nodes.getNodeAs<BinaryOperator>("binOp");
  const auto *Parent = Result.Nodes.getNodeAs<BinaryOperator>("p");
  assert(BinOp);

  const SourceManager &SM = *Result.SourceManager;
  ASTContext &Ctx = *Result.Context;

  emitWarningAndChangeOperatorsIfPossible(BinOp, Parent, SM, Ctx);
}

} // namespace clang::tidy::misc
