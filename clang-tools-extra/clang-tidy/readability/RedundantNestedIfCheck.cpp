//===--- RedundantNestedIfCheck.cpp - clang-tidy -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantNestedIfCheck.h"
#include "../utils/LexerUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/FixIt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <optional>
#include <string>
#include <vector>

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

static constexpr llvm::StringLiteral AllowUserDefinedBoolConversionStr =
    "AllowUserDefinedBoolConversion";
static constexpr llvm::StringLiteral MergeableIfDiag =
    "nested 'if' statements can be merged together";
static constexpr llvm::StringLiteral NestedIfNote =
    "nested 'if' statement to merge declared here";

namespace {

using IfChain = llvm::SmallVector<const IfStmt *>;

enum class WarningType {
  None,
  WarnOnly,
  WarnAndFix,
};

enum class CombinedConditionBuildStatus {
  Success,
  UnsupportedCommentPlacement,
  Failure,
};

struct CombinedConditionBuildResult {
  CombinedConditionBuildStatus Status = CombinedConditionBuildStatus::Failure;
  std::string Text;
};

} // namespace

// Conjoining conditions with `&&` can change behavior when a condition relies
// on contextual user-defined conversion to `bool`.
static bool containsUserDefinedBoolConversion(const Expr *Expression) {
  assert(Expression);

  if (const auto *Cast = dyn_cast<ImplicitCastExpr>(Expression);
      Cast && Cast->getCastKind() == CK_UserDefinedConversion)
    return true;

  return llvm::any_of(Expression->children(), [](const Stmt *Child) {
    const Expr *ChildExpr = dyn_cast_or_null<Expr>(Child);
    return ChildExpr && containsUserDefinedBoolConversion(ChildExpr);
  });
}

static bool conditionNeedsBoolCast(const Expr *Condition) {
  assert(Condition);

  if (!containsUserDefinedBoolConversion(Condition))
    return false;

  const Expr *const Unwrapped =
      Condition->IgnoreImplicitAsWritten()->IgnoreParens();
  const QualType ConditionType = Unwrapped->getType();
  return ConditionType.isNull() || !ConditionType->isScalarType();
}

static bool
isConditionExpressionMergeable(const Expr *Condition,
                               bool AllowUserDefinedBoolConversion) {
  assert(Condition);

  if (Condition->isTypeDependent())
    return false;

  if (containsUserDefinedBoolConversion(Condition))
    return AllowUserDefinedBoolConversion;

  const Expr *const Unwrapped = Condition->IgnoreParenImpCasts();
  const QualType ConditionType = Unwrapped->getType();
  return !ConditionType.isNull() && ConditionType->isScalarType();
}

static std::optional<CharSourceRange>
getIfConditionRange(const IfStmt *If, const SourceManager &SM,
                    const LangOptions &LangOpts) {
  assert(If);

  const SourceLocation ConditionBegin =
      Lexer::getLocForEndOfToken(If->getLParenLoc(), 0, SM, LangOpts);
  if (ConditionBegin.isInvalid() || If->getRParenLoc().isInvalid())
    return std::nullopt;

  const CharSourceRange FileRange = Lexer::makeFileCharRange(
      CharSourceRange::getCharRange(ConditionBegin, If->getRParenLoc()), SM,
      LangOpts);
  if (FileRange.isInvalid())
    return std::nullopt;

  return FileRange;
}

static std::optional<std::string>
getIfConditionText(const IfStmt *If, const SourceManager &SM,
                   const LangOptions &LangOpts) {
  const std::optional<CharSourceRange> ConditionRange =
      getIfConditionRange(If, SM, LangOpts);
  if (!ConditionRange)
    return std::nullopt;

  bool Invalid = false;
  const StringRef ConditionText =
      Lexer::getSourceText(*ConditionRange, SM, LangOpts, &Invalid);
  return Invalid || ConditionText.empty()
             ? std::nullopt
             : std::optional<std::string>(ConditionText.str());
}

static bool isLocationInCharRange(SourceLocation Loc, CharSourceRange Range,
                                  const SourceManager &SM) {
  return Loc.isValid() && Range.isValid() &&
         !SM.isBeforeInTranslationUnit(Loc, Range.getBegin()) &&
         SM.isBeforeInTranslationUnit(Loc, Range.getEnd());
}

// Keep fix-its only when comments in removed nested headers stay inside the
// preserved condition text. Other comment placements keep the diagnostic but
// suppress the rewrite.
static bool hasOnlyPayloadCommentsInNestedHeader(const IfStmt *Nested,
                                                 const SourceManager &SM,
                                                 const LangOptions &LangOpts) {
  assert(Nested);

  const CharSourceRange HeaderFileRange = Lexer::makeFileCharRange(
      CharSourceRange::getCharRange(Nested->getBeginLoc(),
                                    Nested->getThen()->getBeginLoc()),
      SM, LangOpts);
  if (HeaderFileRange.isInvalid())
    return false;

  const std::optional<CharSourceRange> PayloadRange =
      getIfConditionRange(Nested, SM, LangOpts);
  if (!PayloadRange)
    return false;

  const std::vector<utils::lexer::CommentToken> Comments =
      utils::lexer::getCommentsInRange(HeaderFileRange, SM, LangOpts);
  return llvm::all_of(Comments, [&](const utils::lexer::CommentToken &Comment) {
    return isLocationInCharRange(Comment.Loc, *PayloadRange, SM);
  });
}

// Only an outer condition variable can be rewritten safely by moving the
// declaration into an init-statement and conjoining the condition variable.
static bool
canRewriteOuterConditionVariable(const IfStmt *If,
                                 bool AllowUserDefinedBoolConversion) {
  assert(If);
  assert(If->hasVarStorage());

  // `if (init; bool X = cond())` cannot generally become
  // `if (init; bool X = cond(); X)`.
  // Same-type declaration merging, like `if (bool Y = init(), X = cond(); X)`,
  // is possible but too narrow to be worth supporting.
  if (If->hasInitStorage())
    return false;

  const VarDecl *const ConditionVariable = If->getConditionVariable();
  const DeclStmt *const ConditionVariableDeclStmt =
      If->getConditionVariableDeclStmt();
  if (!ConditionVariable || !ConditionVariableDeclStmt ||
      !ConditionVariableDeclStmt->isSingleDecl() ||
      ConditionVariable->getName().empty())
    return false;

  return If->getCond() && isConditionExpressionMergeable(
                              If->getCond(), AllowUserDefinedBoolConversion);
}

// Accept either `if (...) if (...)` or `if (...) { if (...) }` where the
// compound contains exactly one statement.
static const IfStmt *getOnlyNestedIf(const Stmt *Then) {
  if (!Then)
    return nullptr;
  if (const auto *NestedIf = dyn_cast<IfStmt>(Then))
    return NestedIf;

  const auto *const Compound = dyn_cast<CompoundStmt>(Then);
  if (!Compound || Compound->size() != 1)
    return nullptr;

  return dyn_cast<IfStmt>(Compound->body_front());
}

static bool isMergeCandidate(const IfStmt *If, bool AllowInitStorage,
                             bool RequireConstexpr, bool AllowConditionVariable,
                             bool AllowUserDefinedBoolConversion,
                             const LangOptions &LangOpts) {
  assert(If);

  const bool HasAllowedInitStorage = AllowInitStorage || !If->hasInitStorage();
  const bool HasRequiredConstexpr = If->isConstexpr() == RequireConstexpr;
  const bool IsMergeableStructure = If->getThen() && !If->isConsteval() &&
                                    !If->getElse() && HasAllowedInitStorage &&
                                    HasRequiredConstexpr;

  const bool HasMergeableConditionVariable =
      If->hasVarStorage() && AllowConditionVariable && LangOpts.CPlusPlus17 &&
      canRewriteOuterConditionVariable(If, AllowUserDefinedBoolConversion);
  const bool HasMergeableConditionExpression =
      !If->hasVarStorage() && If->getCond() &&
      isConditionExpressionMergeable(If->getCond(),
                                     AllowUserDefinedBoolConversion);

  return IsMergeableStructure &&
         (HasMergeableConditionVariable || HasMergeableConditionExpression);
}

// Statement attributes wrap the `if` in an `AttributedStmt`, so removing nested
// `if` tokens can invalidate attribute placement.
static bool isAttributedIf(const IfStmt *If, ASTContext &Context) {
  assert(If);

  const DynTypedNodeList Parents = Context.getParents(*If);
  return !Parents.empty() && Parents[0].get<AttributedStmt>() != nullptr;
}

static IfChain getMergeChain(const IfStmt *Root, ASTContext &Context,
                             bool AllowUserDefinedBoolConversion) {
  assert(Root);

  IfChain Chain;
  const LangOptions &LangOpts = Context.getLangOpts();
  const bool RequireConstexpr = Root->isConstexpr();
  if (!isMergeCandidate(Root, /*AllowInitStorage=*/true,
                        /*RequireConstexpr=*/RequireConstexpr,
                        /*AllowConditionVariable=*/true,
                        AllowUserDefinedBoolConversion, LangOpts) ||
      isAttributedIf(Root, Context))
    return Chain;

  Chain.push_back(Root);
  const IfStmt *Current = Root;
  while (const IfStmt *const Nested = getOnlyNestedIf(Current->getThen())) {
    if (!isMergeCandidate(Nested, /*AllowInitStorage=*/false,
                          /*RequireConstexpr=*/RequireConstexpr,
                          /*AllowConditionVariable=*/false,
                          AllowUserDefinedBoolConversion, LangOpts) ||
        isAttributedIf(Nested, Context))
      break;

    Chain.push_back(Nested);
    Current = Nested;
  }

  return Chain;
}

static bool isConstantBooleanCondition(const Expr *Condition,
                                       const ASTContext &Context,
                                       bool RequiredValue) {
  if (!Condition || Condition->isValueDependent() ||
      Condition->isInstantiationDependent())
    return false;

  bool EvaluatedValue = false;
  return Condition->EvaluateAsBooleanCondition(EvaluatedValue, Context) &&
         EvaluatedValue == RequiredValue;
}

// Some instantiation-dependent conditions are safe to form even when an
// earlier `if constexpr` condition is not known to be `true`. For example,
// non-type template parameters and `requires` expressions do not depend on a
// discarded branch to avoid hard substitution errors.
static bool isAlwaysFormableDependentConstexprCondition(const Expr *Condition) {
  assert(Condition);

  Condition = Condition->IgnoreParenImpCasts();
  if (isa<CXXBoolLiteralExpr, IntegerLiteral, RequiresExpr>(Condition))
    return true;

  if (const auto *DeclRef = dyn_cast<DeclRefExpr>(Condition))
    return isa<NonTypeTemplateParmDecl>(DeclRef->getDecl());

  if (const auto *Unary = dyn_cast<UnaryOperator>(Condition))
    return isAlwaysFormableDependentConstexprCondition(Unary->getSubExpr());

  if (const auto *Binary = dyn_cast<BinaryOperator>(Condition)) {
    if (Binary->isAssignmentOp() || Binary->getOpcode() == BO_Comma)
      return false;

    return isAlwaysFormableDependentConstexprCondition(Binary->getLHS()) &&
           isAlwaysFormableDependentConstexprCondition(Binary->getRHS());
  }

  return false;
}

static bool
isConstexprChainSemanticallySafe(llvm::ArrayRef<const IfStmt *> Chain,
                                 const ASTContext &Context) {
  if (Chain.empty() || !Chain.front()->isConstexpr())
    return true;

  bool AllPreviousConditionsAreConstantTrue = true;
  for (const IfStmt *If : Chain) {
    const Expr *const Condition = If->getCond();
    if (Condition->isInstantiationDependent() &&
        !AllPreviousConditionsAreConstantTrue &&
        !isAlwaysFormableDependentConstexprCondition(Condition))
      return false;

    AllPreviousConditionsAreConstantTrue =
        AllPreviousConditionsAreConstantTrue &&
        isConstantBooleanCondition(Condition, Context, /*RequiredValue=*/true);
  }

  return true;
}

// A range is unsafe for text edits if it crosses macro expansions or
// preprocessor directives.
template <typename RangeT> static bool isUnsafeRangeSpelling(RangeT Range) {
  return Range.isInvalid() || Range.getBegin().isMacroID() ||
         Range.getEnd().isMacroID();
}

static bool isUnsafeTokenRange(SourceRange Range, const SourceManager &SM,
                               const LangOptions &LangOpts) {
  if (isUnsafeRangeSpelling(Range))
    return true;

  return Lexer::makeFileCharRange(CharSourceRange::getTokenRange(Range), SM,
                                  LangOpts)
             .isInvalid() ||
         utils::lexer::rangeContainsExpansionsOrDirectives(Range, SM, LangOpts);
}

static bool isUnsafeCharRange(CharSourceRange Range, const SourceManager &SM,
                              const LangOptions &LangOpts) {
  if (isUnsafeRangeSpelling(Range))
    return true;

  return Lexer::makeFileCharRange(Range, SM, LangOpts).isInvalid() ||
         utils::lexer::rangeContainsExpansionsOrDirectives(Range.getAsRange(),
                                                           SM, LangOpts);
}

// Validate every range that contributes to the final edit set before offering
// fix-its. If any range is unsafe, keep looking for a diagnosable child chain.
static bool isFixitSafeForChain(llvm::ArrayRef<const IfStmt *> Chain,
                                const SourceManager &SM,
                                const LangOptions &LangOpts) {
  if (Chain.empty())
    return false;

  const IfStmt *const Root = Chain.front();
  const std::optional<CharSourceRange> RootConditionRange =
      Root->hasInitStorage() && !Root->hasVarStorage()
          ? std::optional<CharSourceRange>(CharSourceRange::getTokenRange(
                Root->getCond()->getSourceRange()))
          : getIfConditionRange(Root, SM, LangOpts);
  if (!RootConditionRange ||
      isUnsafeCharRange(*RootConditionRange, SM, LangOpts))
    return false;
  if (!Root->hasVarStorage() &&
      isUnsafeTokenRange(Root->getCond()->getSourceRange(), SM, LangOpts))
    return false;

  if (Root->hasVarStorage()) {
    const DeclStmt *const ConditionVariableDeclStmt =
        Root->getConditionVariableDeclStmt();
    if (!ConditionVariableDeclStmt ||
        isUnsafeTokenRange(ConditionVariableDeclStmt->getSourceRange(), SM,
                           LangOpts)) {
      return false;
    }
  }

  const llvm::ArrayRef<const IfStmt *> ChainRef(Chain);
  return llvm::all_of(
      llvm::zip(ChainRef.drop_back(), ChainRef.drop_front()),
      [&](const auto &ParentAndChild) {
        const auto &[Parent, Child] = ParentAndChild;
        if (isUnsafeTokenRange(Child->getCond()->getSourceRange(), SM,
                               LangOpts))
          return false;

        const CharSourceRange ChildHeaderRange = CharSourceRange::getCharRange(
            Child->getBeginLoc(), Child->getThen()->getBeginLoc());
        if (isUnsafeCharRange(ChildHeaderRange, SM, LangOpts))
          return false;

        const auto *const Wrapper = dyn_cast<CompoundStmt>(Parent->getThen());
        return !Wrapper ||
               (!isUnsafeTokenRange(
                    SourceRange(Wrapper->getLBracLoc(), Wrapper->getLBracLoc()),
                    SM, LangOpts) &&
                !isUnsafeTokenRange(
                    SourceRange(Wrapper->getRBracLoc(), Wrapper->getRBracLoc()),
                    SM, LangOpts));
      });
}

static std::string wrapConditionText(StringRef ConditionText,
                                     bool NeedBoolCast) {
  if (!NeedBoolCast)
    return ConditionText.str();

  std::string Result("static_cast<bool>(");
  Result += ConditionText;
  Result += ')';
  return Result;
}

static std::optional<std::string> getConjunctText(const IfStmt *If,
                                                  const ASTContext &Context,
                                                  bool UseConditionExprText) {
  assert(If);

  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  std::optional<std::string> ConditionText;
  if (UseConditionExprText) {
    const StringRef ConditionExprText =
        tooling::fixit::getText(*If->getCond(), Context);
    if (ConditionExprText.empty())
      return std::nullopt;
    ConditionText = ConditionExprText.str();
  } else {
    ConditionText = getIfConditionText(If, SM, LangOpts);
    if (!ConditionText)
      return std::nullopt;
  }

  return wrapConditionText(*ConditionText,
                           conditionNeedsBoolCast(If->getCond()));
}

static CombinedConditionBuildResult
buildCombinedCondition(llvm::ArrayRef<const IfStmt *> Chain,
                       const ASTContext &Context) {
  if (Chain.empty())
    return {};

  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();
  std::string CombinedCondition;

  for (const auto &[Index, If] : llvm::enumerate(Chain)) {
    const bool IsRoot = Index == 0;
    if (!IsRoot && !hasOnlyPayloadCommentsInNestedHeader(If, SM, LangOpts))
      return {CombinedConditionBuildStatus::UnsupportedCommentPlacement, {}};

    if (IsRoot && If->hasVarStorage()) {
      const VarDecl *const ConditionVariable = If->getConditionVariable();
      if (!ConditionVariable)
        return {};

      const std::optional<std::string> ConditionText =
          getIfConditionText(If, SM, LangOpts);
      if (!ConditionText)
        return {};

      CombinedCondition = *ConditionText;
      CombinedCondition += "; ";
      CombinedCondition += wrapConditionText(
          ConditionVariable->getName(), conditionNeedsBoolCast(If->getCond()));
      continue;
    }

    const std::optional<std::string> ConjunctText =
        getConjunctText(If, Context,
                        /*UseConditionExprText=*/IsRoot &&
                            If->hasInitStorage() && !If->hasVarStorage());
    if (!ConjunctText)
      return {};

    if (!CombinedCondition.empty())
      CombinedCondition += " && ";
    CombinedCondition += '(';
    CombinedCondition += *ConjunctText;
    CombinedCondition += ')';
  }

  return {CombinedConditionBuildStatus::Success, std::move(CombinedCondition)};
}

static std::optional<CharSourceRange>
getConditionReplacementRange(const IfStmt *If, const SourceManager &SM,
                             const LangOptions &LangOpts) {
  assert(If);

  return If->hasInitStorage() && !If->hasVarStorage()
             ? std::optional<CharSourceRange>(CharSourceRange::getTokenRange(
                   If->getCond()->getSourceRange()))
             : getIfConditionRange(If, SM, LangOpts);
}

static WarningType
getWarningType(llvm::ArrayRef<const IfStmt *> Chain, const ASTContext &Context,
               const SourceManager &SM, const LangOptions &LangOpts,
               std::optional<std::string> *CombinedCondition) {
  if (Chain.size() < 2 || !isFixitSafeForChain(Chain, SM, LangOpts))
    return WarningType::None;

  if (!isConstexprChainSemanticallySafe(Chain, Context))
    return WarningType::None;

  const CombinedConditionBuildResult Combined =
      buildCombinedCondition(Chain, Context);
  if (Combined.Status ==
      CombinedConditionBuildStatus::UnsupportedCommentPlacement)
    return WarningType::WarnOnly;
  if (Combined.Status != CombinedConditionBuildStatus::Success)
    return WarningType::None;

  if (CombinedCondition)
    *CombinedCondition = Combined.Text;
  return WarningType::WarnAndFix;
}

static void emitNestedIfNotes(RedundantNestedIfCheck &Check,
                              llvm::ArrayRef<const IfStmt *> Chain) {
  for (const IfStmt *Nested : llvm::drop_begin(Chain))
    Check.diag(Nested->getIfLoc(), NestedIfNote, DiagnosticIDs::Note);
}

static void diagnoseChain(RedundantNestedIfCheck &Check, const IfStmt *If,
                          ASTContext &Context,
                          bool AllowUserDefinedBoolConversion);

static void diagnoseChildChain(RedundantNestedIfCheck &Check,
                               const Stmt *Branch, ASTContext &Context,
                               bool AllowUserDefinedBoolConversion) {
  if (const IfStmt *const Nested = getOnlyNestedIf(Branch))
    diagnoseChain(Check, Nested, Context, AllowUserDefinedBoolConversion);
}

// Match only syntactic chain roots. If a root cannot be diagnosed because it is
// unsafe to rewrite, descend into excluded single-child nested `if` statements
// in both branches and try again there.
static void diagnoseChain(RedundantNestedIfCheck &Check, const IfStmt *If,
                          ASTContext &Context,
                          bool AllowUserDefinedBoolConversion) {
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();
  const IfChain Chain =
      getMergeChain(If, Context, AllowUserDefinedBoolConversion);

  std::optional<std::string> CombinedCondition;
  const WarningType Handling =
      getWarningType(Chain, Context, SM, LangOpts, &CombinedCondition);
  if (Handling == WarningType::None) {
    diagnoseChildChain(Check, If->getThen(), Context,
                       AllowUserDefinedBoolConversion);
    diagnoseChildChain(Check, If->getElse(), Context,
                       AllowUserDefinedBoolConversion);
    return;
  }

  {
    const DiagnosticBuilder Diag = Check.diag(If->getIfLoc(), MergeableIfDiag);
    if (Handling == WarningType::WarnAndFix) {
      const std::optional<CharSourceRange> ConditionRange =
          getConditionReplacementRange(If, SM, LangOpts);
      if (!ConditionRange || !CombinedCondition)
        return;

      Diag << FixItHint::CreateReplacement(*ConditionRange, *CombinedCondition);
      const llvm::ArrayRef<const IfStmt *> ChainRef(Chain);
      for (const auto &[Parent, Child] :
           llvm::zip(ChainRef.drop_back(), ChainRef.drop_front())) {
        Diag << FixItHint::CreateRemoval(CharSourceRange::getCharRange(
            Child->getBeginLoc(), Child->getThen()->getBeginLoc()));

        const auto *const Wrapper = dyn_cast<CompoundStmt>(Parent->getThen());
        if (!Wrapper)
          continue;

        Diag << FixItHint::CreateRemoval(Wrapper->getLBracLoc())
             << FixItHint::CreateRemoval(Wrapper->getRBracLoc());
      }
    }
  }

  emitNestedIfNotes(Check, Chain);
}

RedundantNestedIfCheck::RedundantNestedIfCheck(StringRef Name,
                                               ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      AllowUserDefinedBoolConversion(
          Options.get(AllowUserDefinedBoolConversionStr, false)) {}

void RedundantNestedIfCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, AllowUserDefinedBoolConversionStr,
                AllowUserDefinedBoolConversion);
}

void RedundantNestedIfCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      ifStmt(unless(hasElse(stmt())),
             unless(anyOf(hasParent(ifStmt(unless(hasElse(stmt())))),
                          hasParent(compoundStmt(
                              statementCountIs(1),
                              hasParent(ifStmt(unless(hasElse(stmt())))))))))
          .bind("if"),
      this);
}

void RedundantNestedIfCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *const If = Result.Nodes.getNodeAs<IfStmt>("if");
  assert(If);

  diagnoseChain(*this, If, *Result.Context, AllowUserDefinedBoolConversion);
}

} // namespace clang::tidy::readability
