//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BoolBitwiseOperationCheck.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
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

static std::optional<SourceLocation>
getSpellingLocationForFixIt(SourceLocation Loc, const SourceManager &SM) {
  if (Loc.isInvalid() || Loc.isMacroID())
    return std::nullopt;

  Loc = SM.getSpellingLoc(Loc);
  if (Loc.isInvalid() || Loc.isMacroID())
    return std::nullopt;

  return Loc;
}

static std::optional<SourceLocation>
getEndOfTokenLocationForFixIt(SourceLocation Loc, const SourceManager &SM,
                              const LangOptions &LangOpts) {
  if (Loc.isInvalid() || Loc.isMacroID())
    return std::nullopt;

  Loc = SM.getSpellingLoc(Loc);
  if (Loc.isInvalid() || Loc.isMacroID())
    return std::nullopt;

  SourceLocation EndLoc = Lexer::getLocForEndOfToken(Loc, 0, SM, LangOpts);
  if (EndLoc.isInvalid() || EndLoc.isMacroID())
    return std::nullopt;

  return EndLoc;
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
      StrictMode(Options.get("StrictMode", true)),
      ParenCompounds(Options.get("ParenCompounds", true)) {}

void BoolBitwiseOperationCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "UnsafeMode", UnsafeMode);
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
  Options.store(Opts, "StrictMode", StrictMode);
  Options.store(Opts, "ParenCompounds", ParenCompounds);
}

void BoolBitwiseOperationCheck::registerMatchers(MatchFinder *Finder) {
  auto BooleanLeaves = hasAllLeavesOfBitwiseSatisfying(hasType(booleanType()));
  auto NonVolatile = unless(hasType(isVolatileQualified()));
  auto CompoundOperator = hasAnyOperatorName("|=", "&=");
  auto ExprWithSideEffects = traverse(
      TK_AsIs, expr(hasSideEffects(/*IncludePossibleEffects=*/!UnsafeMode)));
  auto SimpleLhs = anyOf(declRefExpr(), memberExpr());

  auto FixItMatcher = binaryOperator(
      // Both operands must be non-volatile at the top level.
      hasOperands(NonVolatile, NonVolatile),
      hasRHS(unless(ExprWithSideEffects)),
      anyOf(
          // Non-compound assignments: no additional LHS
          // restriction needed.
          hasAnyOperatorName("|", "&"),
          // Compound assignments ('|=' / '&='): require a simple
          // LHS so that we can safely duplicate it on the RHS.
          allOf(CompoundOperator,
                hasLHS(anyOf(SimpleLhs,
                             unaryOperator(hasOperatorName("*"),
                                           hasUnaryOperand(SimpleLhs)))))));

  auto LhsOfCompoundMatcher = traverse(TK_AsIs, expr().bind("lhsOfCompound"));

  // Parentheses decision logic:
  // Case 1: | with parent && → parens needed around BinOp (the || result)
  //         e.g., a && b | c → a && (b || c)
  // Case 2: & with parent ^ or | → parens needed around BinOp (the && result)
  //         e.g., a ^ b & c → a ^ (b && c)
  // Case 3: &= with RHS || → parens needed around RHS
  //         e.g., a &= b || c → a = a && (b || c)

  // This matcher doesn't handle `ImplicitCastExpr` inside `ParenExpr` because
  // Clang's AST construction makes this case impossible:
  // - `ParenExpr` is syntactic (created during parsing)
  // - `ImplicitCastExpr` is semantic (added during Sema phase)
  // According to the Clang CFE Internals Manual, syntactic structure
  // (`ParenExpr`) is established before semantic transformations
  // (`ImplicitCastExpr`).
  auto NotAlreadyInParenExpr =
      traverse(TK_AsIs, unless(hasParent(parenExpr())));
  // Reference:
  // - "Faithfulness" design principle and AST layering
  // - "How to add an expression or statement" the order of establishing AST
  // nodes.
  // https://clang.llvm.org/docs/InternalsManual.html#faithfulness
  // https://clang.llvm.org/docs/InternalsManual.html#how-to-add-an-expression-or-statement

  // Case 1: | with && parent
  auto ParensCase1 = allOf(hasOperatorName("|"), NotAlreadyInParenExpr,
                           binaryOperator().bind("parensExpr"),
                           hasParent(binaryOperator(hasOperatorName("&&"))));

  // Case 2: & with ^ or | parent
  auto ParensCase2 =
      allOf(hasOperatorName("&"), NotAlreadyInParenExpr,
            binaryOperator().bind("parensExpr"),
            hasParent(binaryOperator(hasAnyOperatorName("^", "|"))));

  // Case 3: &= with || RHS
  auto ParensCase3 = allOf(
      hasOperatorName("&="),
      hasRHS(allOf(binaryOperator(hasOperatorName("||")).bind("parensExpr"),
                   NotAlreadyInParenExpr)));

  // Case 4: `ParenCompounds` option enabled and two different operators
  auto ParensCaseOpt = allOf(
      CompoundOperator,
      hasRHS(allOf(
          binaryOperator(/*operators checking later*/).bind("parensExprOpt"),
          NotAlreadyInParenExpr)));

  auto BaseMatcher = binaryOperator(
      hasAnyOperatorName("|", "&", "|=", "&="), hasLHS(BooleanLeaves),
      hasRHS(BooleanLeaves),
      optionally(allOf(CompoundOperator, hasLHS(LhsOfCompoundMatcher))),
      optionally(FixItMatcher.bind("fixit")),
      optionally(hasRHS(ExprWithSideEffects.bind("rhsWithSideEffects"))),
      optionally(anyOf(ParensCase1, ParensCase2, ParensCase3, ParensCaseOpt)));

  Finder->addMatcher(BaseMatcher.bind("binOp"), this);
}

DiagnosticBuilder
BoolBitwiseOperationCheck::createDiagBuilder(const BinaryOperator *BinOp) {
  return diag(BinOp->getOperatorLoc(),
              "use logical operator '%0' for boolean semantics instead of "
              "bitwise operator '%1'")
         << translate(BinOp->getOpcodeStr()) << BinOp->getOpcodeStr();
}

void BoolBitwiseOperationCheck::emitWarningAndChangeOperatorsIfPossible(
    const BinaryOperator *BinOp, const BinaryOperator *ParensExpr,
    const BinaryOperator *ParensExprOpt, const Expr *LhsOfCompound,
    const clang::SourceManager &SM, clang::ASTContext &Ctx,
    bool CanApplyFixIt) {
  // Early exit: the matcher proved that no fix-it possible
  if (!CanApplyFixIt) {
    if (StrictMode)
      createDiagBuilder(BinOp);
    return;
  }

  // Try to build fix-its, but fall back to warning-only if any step fails
  bool CanBuildFixIts = true;

  // Get operator token range
  const auto MaybeTokenRange =
      getOperatorTokenRangeForFixIt(BinOp, SM, Ctx.getLangOpts());
  if (!MaybeTokenRange)
    CanBuildFixIts = false;

  FixItHint ReplaceOperator;

  // Replace '&' to '&&' and so on.
  if (CanBuildFixIts) {
    const CharSourceRange TokenRange = *MaybeTokenRange;
    const StringRef FixSpelling =
        translate(Lexer::getSourceText(TokenRange, SM, Ctx.getLangOpts()));
    assert(!FixSpelling.empty());

    ReplaceOperator = FixItHint::CreateReplacement(TokenRange, FixSpelling);
  }

  FixItHint InsertEqual;

  // Insert ' = a' if it's needed
  if (CanBuildFixIts && LhsOfCompound) {
    const auto MaybeInsertLoc = getEndOfTokenLocationForFixIt(
        LhsOfCompound->getEndLoc(), SM, Ctx.getLangOpts());
    if (!MaybeInsertLoc)
      CanBuildFixIts = false;
    else {
      const SourceLocation InsertLoc = *MaybeInsertLoc;
      std::string SourceText{Lexer::getSourceText(
          CharSourceRange::getTokenRange(LhsOfCompound->getSourceRange()), SM,
          Ctx.getLangOpts())};
      llvm::erase_if(SourceText,
                     [](unsigned char Ch) { return std::isspace(Ch); });
      InsertEqual = FixItHint::CreateInsertion(InsertLoc, " = " + SourceText);
    }
  }

  // Handle the case which might lead to -WParens warning
  if (CanBuildFixIts && ParensExprOpt && !ParensExpr && ParenCompounds) {
    const StringRef RHSOpStr = ParensExprOpt->getOpcodeStr();
    const StringRef CompoundOpStr = BinOp->getOpcodeStr();
    const StringRef RHSLogicalOpStr = translate(RHSOpStr);
    const StringRef LogicalOpStr = translate(CompoundOpStr);
    const bool ShouldSkipRHSBrace =
        (RHSOpStr == LogicalOpStr ||
         (!RHSLogicalOpStr.empty() && RHSLogicalOpStr == LogicalOpStr));
    ParensExpr = ShouldSkipRHSBrace ? nullptr : ParensExprOpt;
  }

  FixItHint InsertBrace1, InsertBrace2;

  // Insert parentheses if it's needed
  if (CanBuildFixIts && ParensExpr) {
    const auto MaybeInsertFirstLoc =
        getSpellingLocationForFixIt(ParensExpr->getBeginLoc(), SM);
    const auto MaybeInsertSecondLoc = getEndOfTokenLocationForFixIt(
        ParensExpr->getEndLoc(), SM, Ctx.getLangOpts());
    if (!MaybeInsertFirstLoc || !MaybeInsertSecondLoc)
      CanBuildFixIts = false;
    else {
      InsertBrace1 = FixItHint::CreateInsertion(*MaybeInsertFirstLoc, "(");
      InsertBrace2 = FixItHint::CreateInsertion(*MaybeInsertSecondLoc, ")");
    }
  }

  // Emit diagnostic with or without fix-its
  if (CanBuildFixIts)
    createDiagBuilder(BinOp)
        << InsertEqual << ReplaceOperator << InsertBrace1 << InsertBrace2;
  else if (!IgnoreMacros && StrictMode)
    createDiagBuilder(BinOp);
}

void BoolBitwiseOperationCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *BinOp = Result.Nodes.getNodeAs<BinaryOperator>("binOp");
  const auto *FixItBinOp = Result.Nodes.getNodeAs<BinaryOperator>("fixit");
  const auto *ParensExpr = Result.Nodes.getNodeAs<BinaryOperator>("parensExpr");
  const auto *ParensExprOpt =
      Result.Nodes.getNodeAs<BinaryOperator>("parensExprOpt");
  const auto *LhsOfCompound = Result.Nodes.getNodeAs<Expr>("lhsOfCompound");
  const auto *RhsWithSideEffects =
      Result.Nodes.getNodeAs<Expr>("rhsWithSideEffects");
  assert(BinOp);

  const SourceManager &SM = *Result.SourceManager;
  ASTContext &Ctx = *Result.Context;

  const bool CanApplyFixIt = (FixItBinOp != nullptr && FixItBinOp == BinOp);
  emitWarningAndChangeOperatorsIfPossible(
      BinOp, ParensExpr, ParensExprOpt, LhsOfCompound, SM, Ctx, CanApplyFixIt);

  // Check if canceling the fix-it was caused by side effects.
  if (!CanApplyFixIt && RhsWithSideEffects)
    diag(RhsWithSideEffects->getExprLoc(),
         "extract the right operand to a variable", DiagnosticIDs::Note);
}

} // namespace clang::tidy::misc
