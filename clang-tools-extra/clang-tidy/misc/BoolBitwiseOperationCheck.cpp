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
#include <vector>

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
  for (const auto &[Bitwise, Logical] : OperatorsTransformation) {
    if (Value == Bitwise)
      return Logical;
  }

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

/// Checks if all leaf nodes in an bitwise expression satisfy a given condition. This
/// handles cases like `(a | b) & c` where we need to check that a, b, and c
/// all satisfy the condition.
///
/// \param Expr The bitwise expression to check.
/// \param Condition A function that checks if an leaf node satisfies the
///                  desired condition.
/// \returns true if all leaf nodes satisfy the condition, false otherwise.
template <typename F>
static bool allLeavesOfBitwiseSatisfy(const clang::Expr *Expr, const F& Condition) {
  // For leaf nodes, check if the condition is satisfied
  if (Condition(Expr))
    return true;

  Expr = Expr->IgnoreParenImpCasts();

  // If it's a binary operator, recursively check both operands
  if (const auto *BinOp = dyn_cast<clang::BinaryOperator>(Expr)) {
    if (!isBitwiseOperation(BinOp->getOpcodeStr()))
      return false;
    return allLeavesOfBitwiseSatisfy(BinOp->getLHS(), Condition) &&
            allLeavesOfBitwiseSatisfy(BinOp->getRHS(), Condition);
  }

  return false;
}

/// Custom matcher that checks if all leaf nodes in an bitwise expression satisfy
/// the given inner matcher condition. This uses allLeavesOfBitwiseSatisfy to recursively
///
/// Example usage:
///   expr(hasAllLeavesOfBitwiseSatisfying(hasType(booleanType())))
AST_MATCHER_P(Expr, hasAllLeavesOfBitwiseSatisfying,
              ast_matchers::internal::Matcher<Expr>, InnerMatcher) {
  auto Condition = [&](const clang::Expr *E) -> bool {
    return InnerMatcher.matches(*E, Finder, Builder);
  };
  return allLeavesOfBitwiseSatisfy(&Node, Condition);
}

BoolBitwiseOperationCheck::BoolBitwiseOperationCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      UnsafeMode(Options.get("UnsafeMode", false)),
      IgnoreMacros(Options.get("IgnoreMacros", false)) {}

void BoolBitwiseOperationCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "UnsafeMode", UnsafeMode);
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
}

void BoolBitwiseOperationCheck::registerMatchers(MatchFinder *Finder) {
  // Matcher for checking if all leaves in an expression are boolean type
  auto BooleanLeaves = hasAllLeavesOfBitwiseSatisfying(hasType(booleanType()));

  auto BitwiseOps = hasAnyOperatorName("|", "&", "|=", "&=");
  auto CompoundBitwiseOps = hasAnyOperatorName("|=", "&=");
  auto NotNestedInBitwise = unless(hasParent(binaryOperator(BitwiseOps)));
  auto OptionalParent = optionally(hasParent(binaryOperator().bind("p")));

  // Conditions that make it a boolean bitwise operation without ICE(*) context:
  // 1. Both LHS and RHS have all boolean leaves
  // 2. LHS has boolean leaves AND it's a compound assignment
  //
  // * ICE - Implicit cast expression
  auto BothBoolean = allOf(hasLHS(BooleanLeaves), hasRHS(BooleanLeaves));
  auto CompoundWithBoolLHS = allOf(hasLHS(BooleanLeaves), CompoundBitwiseOps);
  auto NoContextNeeded = anyOf(BothBoolean, CompoundWithBoolLHS);

  // At least one boolean operand (needs ICE context to be considered boolean
  // bitwise)
  auto AtLeastOneBoolean = anyOf(hasLHS(BooleanLeaves), hasRHS(BooleanLeaves));

  // Matcher for binop that doesn't need ICE context
  auto BinOpNoContext = traverse(
      TK_IgnoreUnlessSpelledInSource,
      binaryOperator(NotNestedInBitwise, BitwiseOps, OptionalParent,
                     NoContextNeeded)
          .bind("binOpRoot"));

  // Matcher for binop that needs ICE context (at least one boolean operand,
  // but not already covered by NoContextNeeded)
  auto BinOpNeedsContext = traverse(
      TK_IgnoreUnlessSpelledInSource,
      binaryOperator(NotNestedInBitwise, BitwiseOps, OptionalParent,
                     AtLeastOneBoolean, unless(NoContextNeeded))
          .bind("binOpRoot"));
  auto BooleanICE =
      implicitCastExpr(hasType(booleanType()), has(BinOpNeedsContext));

  Finder->addMatcher(expr(anyOf(BooleanICE, BinOpNoContext)), this);
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

  // Helper lambda to check if location is valid and not in a macro
  auto IsValidLocation = [&](SourceLocation Loc) -> bool {
    if (Loc.isInvalid() || Loc.isMacroID()) {
      if (!IgnoreMacros)
        DiagEmitter();
      return false;
    }
    return true;
  };

  // Early validation: check for volatile operands
  const bool HasVolatileOperand = llvm::any_of(
      std::array{BinOp->getLHS(), BinOp->getRHS()}, [&](const Expr *E) {
        return E->IgnoreImpCasts()
            ->getType()
            .isVolatileQualified();
      });
  if (HasVolatileOperand) {
    DiagEmitter();
    return;
  }

  // Early validation: check for side effects
  const bool HasSideEffects = BinOp->getRHS()->HasSideEffects(
    Ctx, /*IncludePossibleEffects=*/!UnsafeMode);
  if (HasSideEffects) {
    DiagEmitter();
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
      DiagEmitter();
    return;
  }

  const StringRef FixSpelling =
      translate(Lexer::getSourceText(TokenRange, SM, Ctx.getLangOpts()));
  if (FixSpelling.empty()) {
    DiagEmitter();
    return;
  }

  FixItHint ReplaceOpHint = FixItHint::CreateReplacement(TokenRange, FixSpelling);

  // Generate fix-it hint for compound assignment (if applicable)
  FixItHint InsertEqualHint;
  if (BinOp->isCompoundAssignmentOp()) {
    const auto *LHS = getAcceptableCompoundsLHS(BinOp);
    if (!LHS) {
      DiagEmitter();
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
    const auto *RHS = dyn_cast<BinaryOperator>(BinOp->getRHS()->IgnoreImpCasts());
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

namespace {
class BinaryOperatorVisitor : public clang::DynamicRecursiveASTVisitor {
  clang::tidy::misc::BoolBitwiseOperationCheck &Check;
  const clang::SourceManager &SM;
  clang::ASTContext &Ctx;
  const clang::BinaryOperator *const ParentRoot;
  // Stack to track parent binary operators during traversal
  std::vector<const clang::BinaryOperator *> ParentStack;

  /// Checks if BinOp is a direct child of the parent binary operator in the
  /// stack (ignoring parentheses and implicit casts).
  bool isDirectChildOfParent(const clang::BinaryOperator *BinOp) const {
    if (ParentStack.empty())
      return true;

    const clang::BinaryOperator *Parent = ParentStack.back();
    const std::array<const Expr *, 2> ParentOperands = {Parent->getLHS(),
                                                        Parent->getRHS()};

    return llvm::any_of(ParentOperands, [&](const Expr *E) {
      return E->IgnoreParenImpCasts() == BinOp;
    });
  }

public:
  BinaryOperatorVisitor(clang::tidy::misc::BoolBitwiseOperationCheck &Check,
                        const clang::SourceManager &SM, clang::ASTContext &Ctx,
                        const clang::BinaryOperator *ParentRoot)
      : Check(Check), SM(SM), Ctx(Ctx), ParentRoot(ParentRoot) {}

  bool TraverseBinaryOperator(clang::BinaryOperator *BinOp) override {
    if (!BinOp)
      return true;

    if (!isDirectChildOfParent(BinOp))
      return true;

    // Track this binary operator as a parent for its children.
    ParentStack.push_back(BinOp);
    const bool Result =
        clang::DynamicRecursiveASTVisitor::TraverseBinaryOperator(BinOp);
    ParentStack.pop_back();

    return Result;
  }

  bool VisitBinaryOperator(clang::BinaryOperator *BinOp) override {
    if (!BinOp)
      return true;

    if (!isBitwiseOperation(BinOp->getOpcodeStr()))
      return true;

    const clang::BinaryOperator *ParentBinOp =
        ParentStack.size() < 2 ? ParentRoot
                               : ParentStack[ParentStack.size() - 2];

    Check.emitWarningAndChangeOperatorsIfPossible(BinOp, ParentBinOp, SM, Ctx);

    return true;
  }
};
} // namespace

void BoolBitwiseOperationCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *BinOpRoot = Result.Nodes.getNodeAs<BinaryOperator>("binOpRoot");
  const auto *ParentRoot = Result.Nodes.getNodeAs<BinaryOperator>("p");
  assert(BinOpRoot);

  const SourceManager &SM = *Result.SourceManager;
  ASTContext &Ctx = *Result.Context;

  BinaryOperatorVisitor Visitor(*this, SM, Ctx, ParentRoot);
  // TraverseStmt requires non-const pointer, but we're only reading
  Visitor.TraverseStmt(const_cast<BinaryOperator *>(BinOpRoot));
}

} // namespace clang::tidy::misc
