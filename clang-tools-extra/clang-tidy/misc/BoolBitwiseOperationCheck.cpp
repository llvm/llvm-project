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
#include <optional>
#include <utility>
#include <vector>

using namespace clang::ast_matchers;

namespace clang::tidy::misc {

static const DynTypedNode *ignoreParensTowardsTheRoot(const DynTypedNode *N,
                                                      ASTContext *AC) {
  if (const auto *S = N->get<Stmt>(); isa_and_nonnull<ParenExpr>(S)) {
    auto Parents = AC->getParents(*S);
    // FIXME: do we need to consider all `Parents` ?
    if (!Parents.empty())
      return ignoreParensTowardsTheRoot(&Parents[0], AC);
  }
  return N;
}

static bool assignsToBoolean(const BinaryOperator *BinOp, ASTContext *AC) {
  const TraversalKindScope RAII(*AC, TK_AsIs);
  auto Parents = AC->getParents(*BinOp);

  return llvm::any_of(Parents, [&](const DynTypedNode &Parent) {
    const auto *S = ignoreParensTowardsTheRoot(&Parent, AC)->get<Stmt>();
    const auto *ICE = dyn_cast_if_present<ImplicitCastExpr>(S);
    return ICE ? ICE->getType().getDesugaredType(*AC)->isBooleanType() : false;
  });
}

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
  Finder->addMatcher(
      binaryOperator(unless(isExpansionInSystemHeader()),
                     unless(hasParent(binaryOperator())) // ignoring parenExpr
                     )
          .bind("binOpRoot"),
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

  const bool HasVolatileOperand = llvm::any_of(
      std::array{BinOp->getLHS(), BinOp->getRHS()}, [&](const Expr *E) {
        return E->IgnoreImpCasts()
            ->getType()
            .getDesugaredType(Ctx)
            .isVolatileQualified();
      });
  if (HasVolatileOperand) {
    DiagEmitter();
    return;
  }

  const bool HasSideEffects = BinOp->getRHS()->HasSideEffects(
      Ctx, /*IncludePossibleEffects=*/!UnsafeMode);
  if (HasSideEffects) {
    DiagEmitter();
    return;
  }

  SourceLocation Loc = BinOp->getOperatorLoc();

  if (Loc.isInvalid() || Loc.isMacroID()) {
    IgnoreMacros || DiagEmitter();
    return;
  }

  Loc = SM.getSpellingLoc(Loc);
  if (Loc.isInvalid() || Loc.isMacroID()) {
    IgnoreMacros || DiagEmitter();
    return;
  }

  const CharSourceRange TokenRange = CharSourceRange::getTokenRange(Loc);
  if (TokenRange.isInvalid()) {
    IgnoreMacros || DiagEmitter();
    return;
  }

  const StringRef FixSpelling =
      translate(Lexer::getSourceText(TokenRange, SM, Ctx.getLangOpts()));

  if (FixSpelling.empty()) {
    DiagEmitter();
    return;
  }

  FixItHint InsertEqual;
  if (BinOp->isCompoundAssignmentOp()) {
    const auto *LHS = getAcceptableCompoundsLHS(BinOp);
    if (!LHS) {
      DiagEmitter();
      return;
    }
    const SourceLocation LocLHS = LHS->getEndLoc();
    if (LocLHS.isInvalid() || LocLHS.isMacroID()) {
      IgnoreMacros || DiagEmitter();
      return;
    }
    const SourceLocation InsertLoc =
        clang::Lexer::getLocForEndOfToken(LocLHS, 0, SM, Ctx.getLangOpts());
    if (InsertLoc.isInvalid() || InsertLoc.isMacroID()) {
      IgnoreMacros || DiagEmitter();
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

  std::optional<BinaryOperatorKind> ParentOpcode;
  if (ParentBinOp)
    ParentOpcode = ParentBinOp->getOpcode();

  const auto *RHS = dyn_cast<BinaryOperator>(BinOp->getRHS()->IgnoreImpCasts());
  std::optional<BinaryOperatorKind> RHSOpcode;
  if (RHS)
    RHSOpcode = RHS->getOpcode();

  const Expr *SurroundedExpr = nullptr;
  if ((BinOp->getOpcode() == BO_Or && ParentOpcode == BO_LAnd) ||
      (BinOp->getOpcode() == BO_And &&
       llvm::is_contained({BO_Xor, BO_Or}, ParentOpcode))) {
    const Expr *Side = ParentBinOp->getLHS()->IgnoreParenImpCasts() == BinOp
                           ? ParentBinOp->getLHS()
                           : ParentBinOp->getRHS();
    SurroundedExpr = Side->IgnoreImpCasts();
    assert(SurroundedExpr->IgnoreParens() == BinOp);
  } else if (BinOp->getOpcode() == BO_AndAssign && RHSOpcode == BO_LOr)
    SurroundedExpr = RHS;

  if (isa_and_nonnull<ParenExpr>(SurroundedExpr))
    SurroundedExpr = nullptr;

  FixItHint InsertBrace1;
  FixItHint InsertBrace2;
  if (SurroundedExpr) {
    const SourceLocation InsertFirstLoc = SurroundedExpr->getBeginLoc();
    const SourceLocation InsertSecondLoc = clang::Lexer::getLocForEndOfToken(
        SurroundedExpr->getEndLoc(), 0, SM, Ctx.getLangOpts());
    if (InsertFirstLoc.isInvalid() || InsertFirstLoc.isMacroID() ||
        InsertSecondLoc.isInvalid() || InsertSecondLoc.isMacroID()) {
      IgnoreMacros || DiagEmitter();
      return;
    }
    InsertBrace1 = FixItHint::CreateInsertion(InsertFirstLoc, "(");
    InsertBrace2 = FixItHint::CreateInsertion(InsertSecondLoc, ")");
  }

  DiagEmitter() << InsertEqual << ReplaceOperator << InsertBrace1
                << InsertBrace2;
}

namespace {
class BinaryOperatorVisitor : public clang::DynamicRecursiveASTVisitor {
  clang::tidy::misc::BoolBitwiseOperationCheck &Check;
  const clang::SourceManager &SM;
  clang::ASTContext &Ctx;
  /// Three-state boolean to track whether the root binary operator in the
  /// expression tree assigns to a boolean type and one of operands is boolean.
  /// This is used to propagate the assignment context through nested binary
  /// operations.
  std::optional<bool> RootAssignsToBoolean;
  // Stack to track parent binary operators during traversal
  std::vector<const clang::BinaryOperator *> ParentStack;

  void setRootAssignsToBoolean(bool Value, bool IsRoot) {
    if (!IsRoot)
      return;
    RootAssignsToBoolean = RootAssignsToBoolean.value_or(Value);
  }

  /// Checks if an expression is boolean type, either directly or recursively
  /// through nested binary operators. This handles cases like `(a | b) & c`
  /// where the LHS is itself a boolean bitwise operation.
  ///
  /// \param Expr The expression to check.
  /// \returns true if the expression is boolean type or is a boolean bitwise
  ///          operation, false otherwise.
  bool isBooleanType(const clang::Expr *Expr) {
    if (Expr->IgnoreImpCasts()
            ->getType()
            .getDesugaredType(Ctx)
            ->isBooleanType())
      return true;
    return isBooleanBitwise(
        dyn_cast<clang::BinaryOperator>(Expr->IgnoreParenImpCasts()),
        /*IsRoot=*/false);
  }

  /// Checks if a binary operator is a bitwise operation that should be treated
  /// as a boolean operation (i.e., should use logical operators instead).
  ///
  /// This function determines whether a bitwise operation (|, &, |=, &=, etc.)
  /// is being used in a boolean context, which typically indicates the case
  /// where logical operators (||, &&) should have been used instead.
  ///
  /// \param BinOp The binary operator to check. Must not be null.
  /// \param RootAssignsToBoolean An output parameter that
  ///
  /// \returns true if the operation is a bitwise operation used in a boolean
  ///          context (both operands are boolean, or it assigns to boolean),
  ///          false otherwise.
  bool isBooleanBitwise(const clang::BinaryOperator *BinOp,
                        bool IsRoot = true) {
    if (!BinOp)
      return false;

    if (!isBitwiseOperation(BinOp->getOpcodeStr()))
      return false;

    assert(!IsRoot || RootAssignsToBoolean == std::nullopt);

    // If we've already determined that the root assigns to boolean (from a
    // nested operation), then this is a boolean bitwise operation.
    if (RootAssignsToBoolean.value_or(false))
      return true;

    const bool IsBooleanLHS = isBooleanType(BinOp->getLHS());
    const bool IsBooleanRHS = isBooleanType(BinOp->getRHS());

    // If both operands are boolean, this is definitely a boolean bitwise
    // operation. Preserve the existing RootAssignsToBoolean value if set,
    // otherwise set it to false (no assignment context).
    if (IsBooleanLHS && IsBooleanRHS) {
      setRootAssignsToBoolean(false, IsRoot);
      return true;
    }

    // Check if this operation assigns to a boolean type. This includes:
    // 1. Operations where at least one operand is boolean and the result is
    //    assigned to boolean (e.g., `bool x = a | b` where a is int and b is
    //    boolean)
    // 2. Compound assignments where the LHS is boolean (e.g., `x |= y` where y
    // is int and x is boolean)
    const bool IsRelevantAssignmentToBoolean =
        ((IsBooleanLHS || IsBooleanRHS) && assignsToBoolean(BinOp, &Ctx)) ||
        (IsBooleanLHS && BinOp->isCompoundAssignmentOp());

    // If this operation assigns to boolean, then this is a boolean bitwise
    // operation. Set RootAssignsToBoolean to true to propagate this information
    // up the call stack.
    if (IsRelevantAssignmentToBoolean) {
      setRootAssignsToBoolean(true, IsRoot);
      return true;
    }

    return false;
  }

  /// Checks if BinOp is a direct child of the parent binary operator in the
  /// stack (ignoring parentheses and implicit casts).
  bool isDirectChildOfParent(const clang::BinaryOperator *BinOp) const {
    if (ParentStack.empty())
      return true;

    const clang::BinaryOperator *Parent = ParentStack.back();
    const std::array<const Expr *, 2> ParentOperands = {
        Parent->getLHS(), Parent->getRHS()};

    return llvm::any_of(ParentOperands, [&](const Expr *E) {
      return E->IgnoreParenImpCasts() == BinOp;
    });
  }

public:
  BinaryOperatorVisitor(clang::tidy::misc::BoolBitwiseOperationCheck &Check,
                        const clang::SourceManager &SM, clang::ASTContext &Ctx)
      : Check(Check), SM(SM), Ctx(Ctx) {}

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

    const clang::BinaryOperator *ParentBinOp =
        ParentStack.size() < 2 ? nullptr : ParentStack[ParentStack.size() - 2];

    if (isBooleanBitwise(BinOp)) {
      assert(RootAssignsToBoolean.has_value());
      Check.emitWarningAndChangeOperatorsIfPossible(BinOp, ParentBinOp, SM,
                                                    Ctx);
    }

    return true;
  }
};
} // namespace

void BoolBitwiseOperationCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *BinOpRoot = Result.Nodes.getNodeAs<BinaryOperator>("binOpRoot");
  assert(BinOpRoot);

  const SourceManager &SM = *Result.SourceManager;
  ASTContext &Ctx = *Result.Context;

  BinaryOperatorVisitor Visitor(*this, SM, Ctx);
  // TraverseStmt requires non-const pointer, but we're only reading
  Visitor.TraverseStmt(const_cast<BinaryOperator *>(BinOpRoot));
}

} // namespace clang::tidy::misc
