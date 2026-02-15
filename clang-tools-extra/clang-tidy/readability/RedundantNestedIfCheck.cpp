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
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/FixIt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <string>
#include <vector>

using namespace clang::ast_matchers;

namespace clang::tidy {
template <>
struct OptionEnumMapping<
    readability::RedundantNestedIfCheck::UserDefinedBoolConversionMode> {
  static llvm::ArrayRef<std::pair<
      readability::RedundantNestedIfCheck::UserDefinedBoolConversionMode,
      StringRef>>
  getEnumMapping() {
    using Mode =
        readability::RedundantNestedIfCheck::UserDefinedBoolConversionMode;
    static constexpr std::pair<Mode, StringRef> Mapping[] = {
        {Mode::None, "None"},
        {Mode::WarnOnly, "WarnOnly"},
        {Mode::WarnAndFix, "WarnAndFix"},
    };
    return {Mapping};
  }
};
} // namespace clang::tidy

namespace clang::tidy::readability {

static constexpr llvm::StringLiteral WarnOnDependentConstexprIfStr =
    "WarnOnDependentConstexprIf";
static constexpr llvm::StringLiteral UserDefinedBoolConversionModeStr =
    "UserDefinedBoolConversionMode";

namespace {
enum class ChainHandling {
  None,
  WarnOnly,
  WarnOnlyDependentConstexpr,
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
// on user-defined bool conversion. Keep the check conservative and reject such
// conditions for automatic merging.
static bool containsUserDefinedBoolConversion(const Expr *ExprNode) {
  if (!ExprNode)
    return false;

  if (const auto *Cast = dyn_cast<ImplicitCastExpr>(ExprNode);
      Cast && Cast->getCastKind() == CK_UserDefinedConversion)
    return true;

  return llvm::any_of(ExprNode->children(), [](const Stmt *Child) {
    const auto *ChildExpr = dyn_cast_or_null<Expr>(Child);
    return ChildExpr && containsUserDefinedBoolConversion(ChildExpr);
  });
}

static bool isConditionExpressionSafeToConjoin(
    const Expr *Cond, RedundantNestedIfCheck::UserDefinedBoolConversionMode
                          UserBoolConversionMode) {
  if (!Cond || Cond->isTypeDependent())
    return false;
  const bool HasUserDefinedBoolConversion =
      containsUserDefinedBoolConversion(Cond);
  if (UserBoolConversionMode !=
          RedundantNestedIfCheck::UserDefinedBoolConversionMode::WarnAndFix &&
      HasUserDefinedBoolConversion) {
    return false;
  }
  const Expr *Unwrapped = Cond->IgnoreParenImpCasts();
  if (!Unwrapped)
    return false;
  const QualType CondType = Unwrapped->getType();
  if (CondType.isNull())
    return false;
  if (CondType->isScalarType())
    return true;
  return UserBoolConversionMode ==
         RedundantNestedIfCheck::UserDefinedBoolConversionMode::WarnAndFix;
}

static std::optional<CharSourceRange>
getConditionPayloadRange(const IfStmt *If, const SourceManager &SM,
                         const LangOptions &LangOpts) {
  if (!If)
    return std::nullopt;
  const SourceLocation PayloadBegin =
      Lexer::getLocForEndOfToken(If->getLParenLoc(), 0, SM, LangOpts);
  if (PayloadBegin.isInvalid() || If->getRParenLoc().isInvalid())
    return std::nullopt;

  const CharSourceRange PayloadRange =
      CharSourceRange::getCharRange(PayloadBegin, If->getRParenLoc());
  const CharSourceRange FileRange =
      Lexer::makeFileCharRange(PayloadRange, SM, LangOpts);
  if (FileRange.isInvalid())
    return std::nullopt;
  return FileRange;
}

static std::optional<std::string>
getConditionPayloadText(const IfStmt *If, const SourceManager &SM,
                        const LangOptions &LangOpts) {
  const std::optional<CharSourceRange> PayloadRange =
      getConditionPayloadRange(If, SM, LangOpts);
  if (!PayloadRange)
    return std::nullopt;

  bool Invalid = false;
  const StringRef PayloadText =
      Lexer::getSourceText(*PayloadRange, SM, LangOpts, &Invalid);
  if (Invalid || PayloadText.empty())
    return std::nullopt;
  return PayloadText.str();
}

static std::vector<utils::lexer::CommentToken>
getCommentTokensInRange(CharSourceRange Range, const SourceManager &SM,
                        const LangOptions &LangOpts) {
  std::vector<utils::lexer::CommentToken> Comments;
  if (Range.isInvalid())
    return Comments;

  const CharSourceRange FileRange =
      Lexer::makeFileCharRange(Range, SM, LangOpts);
  if (FileRange.isInvalid())
    return Comments;

  const std::pair<FileID, unsigned> BeginLoc =
      SM.getDecomposedLoc(FileRange.getBegin());
  const std::pair<FileID, unsigned> EndLoc =
      SM.getDecomposedLoc(FileRange.getEnd());
  if (BeginLoc.first != EndLoc.first)
    return Comments;

  bool Invalid = false;
  const StringRef Buffer = SM.getBufferData(BeginLoc.first, &Invalid);
  if (Invalid)
    return Comments;

  const char *LexStart = Buffer.data() + BeginLoc.second;
  Lexer TheLexer(SM.getLocForStartOfFile(BeginLoc.first), LangOpts,
                 Buffer.begin(), LexStart, Buffer.end());
  TheLexer.SetCommentRetentionState(true);

  while (true) {
    Token Tok;
    if (TheLexer.LexFromRawLexer(Tok))
      break;
    if (Tok.is(tok::eof) || Tok.getLocation() == FileRange.getEnd() ||
        SM.isBeforeInTranslationUnit(FileRange.getEnd(), Tok.getLocation())) {
      break;
    }

    if (!Tok.is(tok::comment))
      continue;

    const std::pair<FileID, unsigned> CommentLoc =
        SM.getDecomposedLoc(Tok.getLocation());
    if (CommentLoc.first != BeginLoc.first)
      continue;

    Comments.push_back(utils::lexer::CommentToken{
        Tok.getLocation(),
        StringRef(Buffer.begin() + CommentLoc.second, Tok.getLength()),
    });
  }

  return Comments;
}

static bool locationInCharRange(SourceLocation Loc, CharSourceRange Range,
                                const SourceManager &SM) {
  if (Loc.isInvalid() || Range.isInvalid())
    return false;
  return !SM.isBeforeInTranslationUnit(Loc, Range.getBegin()) &&
         SM.isBeforeInTranslationUnit(Loc, Range.getEnd());
}

// Validate comments in the nested-if header we remove. Comments are fix-safe
// only if they are all inside the condition payload, which is preserved
// verbatim. Any other nested-header comment placement keeps the diagnostic but
// suppresses fix-its.
static bool hasOnlyPayloadCommentsInNestedHeader(const IfStmt *Nested,
                                                 const SourceManager &SM,
                                                 const LangOptions &LangOpts) {
  if (!Nested || !Nested->getThen())
    return false;

  const CharSourceRange HeaderRange = CharSourceRange::getCharRange(
      Nested->getBeginLoc(), Nested->getThen()->getBeginLoc());
  const CharSourceRange HeaderFileRange =
      Lexer::makeFileCharRange(HeaderRange, SM, LangOpts);
  if (HeaderFileRange.isInvalid())
    return false;

  const std::optional<CharSourceRange> PayloadRange =
      getConditionPayloadRange(Nested, SM, LangOpts);
  if (!PayloadRange)
    return false;

  const std::vector<utils::lexer::CommentToken> Comments =
      getCommentTokensInRange(HeaderFileRange, SM, LangOpts);
  return llvm::all_of(Comments, [&](const utils::lexer::CommentToken &Comment) {
    return locationInCharRange(Comment.Loc, *PayloadRange, SM);
  });
}

// Only an outer condition variable can be rewritten safely by moving it into
// an init-statement and using the declared variable as the first conjunct.
static bool canRewriteOuterConditionVariable(
    const IfStmt *If, const LangOptions &LangOpts,
    RedundantNestedIfCheck::UserDefinedBoolConversionMode
        UserBoolConversionMode) {
  if (!If || !If->hasVarStorage() || If->hasInitStorage())
    return false;
  // `if (init; cond)` syntax is available in C++17 and later only.
  if (!LangOpts.CPlusPlus17)
    return false;
  const auto *CondVar = If->getConditionVariable();
  const auto *CondVarDeclStmt = If->getConditionVariableDeclStmt();
  if (!CondVar || !CondVarDeclStmt || !CondVarDeclStmt->isSingleDecl() ||
      CondVar->getName().empty()) {
    return false;
  }
  const QualType VarType = CondVar->getType();
  if (VarType.isNull())
    return false;
  if (UserBoolConversionMode !=
          RedundantNestedIfCheck::UserDefinedBoolConversionMode::WarnAndFix &&
      !VarType->isScalarType()) {
    return false;
  }
  return isConditionExpressionSafeToConjoin(If->getCond(),
                                            UserBoolConversionMode);
}

// Accept either `if (...) if (...)` or `if (...) { if (...) }` where the
// compound contains exactly one statement.
static const IfStmt *getOnlyNestedIf(const Stmt *Then) {
  if (!Then)
    return nullptr;
  if (const auto *NestedIf = dyn_cast<IfStmt>(Then))
    return NestedIf;
  const auto *Compound = dyn_cast<CompoundStmt>(Then);
  if (!Compound || Compound->size() != 1)
    return nullptr;
  return dyn_cast<IfStmt>(Compound->body_front());
}

static bool
isMergeCandidate(const IfStmt *If, bool AllowInitStorage, bool RequireConstexpr,
                 bool AllowConditionVariable, const LangOptions &LangOpts,
                 RedundantNestedIfCheck::UserDefinedBoolConversionMode
                     UserBoolConversionMode) {
  if (!If || !If->getThen())
    return false;
  if (If->isConsteval() || If->getElse())
    return false;
  if (!AllowInitStorage && If->hasInitStorage())
    return false;
  if (If->isConstexpr() != RequireConstexpr)
    return false;
  if (If->hasVarStorage())
    return AllowConditionVariable && canRewriteOuterConditionVariable(
                                         If, LangOpts, UserBoolConversionMode);

  return If->getCond() && isConditionExpressionSafeToConjoin(
                              If->getCond(), UserBoolConversionMode);
}

static bool isMergeShapeCandidate(const IfStmt *If, bool AllowInitStorage,
                                  bool RequireConstexpr,
                                  bool AllowConditionVariable,
                                  const LangOptions &LangOpts) {
  if (!If || !If->getThen())
    return false;
  if (If->isConsteval() || If->getElse())
    return false;
  if (!AllowInitStorage && If->hasInitStorage())
    return false;
  if (If->isConstexpr() != RequireConstexpr)
    return false;
  if (If->hasVarStorage())
    return AllowConditionVariable && LangOpts.CPlusPlus17;
  return If->getCond() != nullptr;
}

// Statement attributes are attached outside of the `if` token range; removing
// nested `if` tokens can make attribute placement invalid, so skip them.
static bool isAttributedIf(const IfStmt *If, ASTContext &Context) {
  if (!If)
    return false;
  const DynTypedNodeList Parents = Context.getParents(*If);
  return !Parents.empty() && Parents[0].get<AttributedStmt>() != nullptr;
}

// Build the maximal top-down chain of mergeable nested if statements.
static llvm::SmallVector<const IfStmt *>
getMergeChain(const IfStmt *Root, ASTContext &Context,
              RedundantNestedIfCheck::UserDefinedBoolConversionMode
                  UserBoolConversionMode) {
  llvm::SmallVector<const IfStmt *> Chain;
  if (!Root)
    return Chain;

  const LangOptions &LangOpts = Context.getLangOpts();
  const bool IsConstexpr = Root->isConstexpr();
  if (!isMergeCandidate(Root, /*AllowInitStorage=*/true, IsConstexpr,
                        /*AllowConditionVariable=*/true, LangOpts,
                        UserBoolConversionMode) ||
      isAttributedIf(Root, Context)) {
    return Chain;
  }

  Chain.push_back(Root);
  const IfStmt *Current = Root;
  while (const IfStmt *Nested = getOnlyNestedIf(Current->getThen())) {
    if (!isMergeCandidate(Nested, /*AllowInitStorage=*/false, IsConstexpr,
                          /*AllowConditionVariable=*/false, LangOpts,
                          UserBoolConversionMode) ||
        isAttributedIf(Nested, Context)) {
      break;
    }
    Chain.push_back(Nested);
    Current = Nested;
  }
  return Chain;
}

// Warn-only mode for chains blocked specifically by user-defined bool
// conversion in the outer condition.
static llvm::SmallVector<const IfStmt *>
getUserDefinedBoolWarnChain(const IfStmt *Root, ASTContext &Context) {
  llvm::SmallVector<const IfStmt *> Chain;
  if (!Root)
    return Chain;

  const LangOptions &LangOpts = Context.getLangOpts();
  const bool IsConstexpr = Root->isConstexpr();
  if (!isMergeShapeCandidate(Root, /*AllowInitStorage=*/true, IsConstexpr,
                             /*AllowConditionVariable=*/true, LangOpts) ||
      isAttributedIf(Root, Context) ||
      !containsUserDefinedBoolConversion(Root->getCond())) {
    return Chain;
  }

  Chain.push_back(Root);
  const IfStmt *Current = Root;
  while (const IfStmt *Nested = getOnlyNestedIf(Current->getThen())) {
    if (!isMergeCandidate(
            Nested, /*AllowInitStorage=*/false, IsConstexpr,
            /*AllowConditionVariable=*/false, LangOpts,
            RedundantNestedIfCheck::UserDefinedBoolConversionMode::None) ||
        isAttributedIf(Nested, Context)) {
      break;
    }
    Chain.push_back(Nested);
    Current = Nested;
  }

  if (Chain.size() < 2)
    Chain.clear();
  return Chain;
}

// Locate the parent `if` that owns this node in its then-branch. This lets us
// suppress duplicate diagnostics when the parent chain is already handled.
static const IfStmt *getParentThenIf(const IfStmt *If, ASTContext &Context) {
  if (!If)
    return nullptr;

  const DynTypedNodeList Parents = Context.getParents(*If);
  if (Parents.empty())
    return nullptr;

  if (const auto *ParentIf = Parents[0].get<IfStmt>()) {
    if (ParentIf->getThen() == If)
      return ParentIf;
    return nullptr;
  }

  const auto *ParentCompound = Parents[0].get<CompoundStmt>();
  if (!ParentCompound || ParentCompound->size() != 1 ||
      ParentCompound->body_front() != If) {
    return nullptr;
  }

  const DynTypedNodeList GrandParents = Context.getParents(*ParentCompound);
  if (GrandParents.empty())
    return nullptr;

  const auto *ParentIf = GrandParents[0].get<IfStmt>();
  if (!ParentIf || ParentIf->getThen() != ParentCompound)
    return nullptr;
  return ParentIf;
}

static bool isConstantBooleanCondition(const Expr *Cond, const ASTContext &Ctx,
                                       bool RequiredValue) {
  if (!Cond || Cond->isValueDependent() || Cond->isInstantiationDependent())
    return false;

  bool Evaluated = false;
  if (!Cond->EvaluateAsBooleanCondition(Evaluated, Ctx))
    return false;
  return Evaluated == RequiredValue;
}

static bool
isConstexprChainSemanticallySafe(llvm::ArrayRef<const IfStmt *> Chain,
                                 const ASTContext &Context) {
  if (Chain.empty() || !Chain.front()->isConstexpr())
    return true;

  const bool OuterIsDependent =
      Chain.front()->getCond()->isInstantiationDependent();

  // Allow outer instantiation-dependence only when every nested condition is a
  // non-dependent constant true expression. This preserves constexpr discard
  // behavior for template branches.
  for (std::size_t Index = 1; Index < Chain.size(); ++Index) {
    const Expr *NestedCond = Chain[Index]->getCond();
    if (NestedCond->isInstantiationDependent())
      return false;
    if (OuterIsDependent &&
        !isConstantBooleanCondition(NestedCond, Context,
                                    /*RequiredValue=*/true)) {
      return false;
    }
  }
  return true;
}

// A range is unsafe for text edits if it crosses macro expansions or
// preprocessor directives.
static bool isUnsafeTokenRange(SourceRange Range, const SourceManager &SM,
                               const LangOptions &LangOpts) {
  if (!Range.isValid())
    return true;
  if (Range.getBegin().isMacroID() || Range.getEnd().isMacroID())
    return true;
  if (Lexer::makeFileCharRange(CharSourceRange::getTokenRange(Range), SM,
                               LangOpts)
          .isInvalid()) {
    return true;
  }
  return utils::lexer::rangeContainsExpansionsOrDirectives(Range, SM, LangOpts);
}

static bool isUnsafeCharRange(CharSourceRange Range, const SourceManager &SM,
                              const LangOptions &LangOpts) {
  if (Range.isInvalid())
    return true;
  if (Range.getBegin().isMacroID() || Range.getEnd().isMacroID())
    return true;
  if (Lexer::makeFileCharRange(Range, SM, LangOpts).isInvalid())
    return true;
  return utils::lexer::rangeContainsExpansionsOrDirectives(Range.getAsRange(),
                                                           SM, LangOpts);
}

// Validate every range that contributes to the final edit set before offering
// fix-its. If any range is unsafe, keep diagnostics but do not rewrite.
static bool isFixitSafeForChain(llvm::ArrayRef<const IfStmt *> Chain,
                                const SourceManager &SM,
                                const LangOptions &LangOpts) {
  if (Chain.empty())
    return false;

  const IfStmt *const Root = Chain.front();
  if (Root->hasInitStorage() && !Root->hasVarStorage()) {
    if (isUnsafeTokenRange(Chain.front()->getCond()->getSourceRange(), SM,
                           LangOpts)) {
      return false;
    }
  } else {
    const std::optional<CharSourceRange> RootConditionRange =
        getConditionPayloadRange(Root, SM, LangOpts);
    if (!RootConditionRange ||
        isUnsafeCharRange(*RootConditionRange, SM, LangOpts)) {
      return false;
    }
  }

  if (Root->hasVarStorage()) {
    const auto *CondVarDeclStmt = Root->getConditionVariableDeclStmt();
    if (!CondVarDeclStmt ||
        isUnsafeTokenRange(CondVarDeclStmt->getSourceRange(), SM, LangOpts)) {
      return false;
    }
  } else if (!Root->hasInitStorage() &&
             isUnsafeTokenRange(Root->getCond()->getSourceRange(), SM,
                                LangOpts)) {
    return false;
  }

  for (std::size_t Index = 1; Index < Chain.size(); ++Index) {
    const IfStmt *const Nested = Chain[Index];
    if (isUnsafeTokenRange(Nested->getCond()->getSourceRange(), SM, LangOpts))
      return false;

    const CharSourceRange NestedHeaderRange = CharSourceRange::getCharRange(
        Nested->getBeginLoc(), Nested->getThen()->getBeginLoc());
    if (isUnsafeCharRange(NestedHeaderRange, SM, LangOpts))
      return false;

    const auto *Wrapper = dyn_cast<CompoundStmt>(Chain[Index - 1]->getThen());
    if (!Wrapper)
      continue;

    if (isUnsafeTokenRange(Wrapper->getSourceRange(), SM, LangOpts) ||
        isUnsafeTokenRange(
            SourceRange(Wrapper->getLBracLoc(), Wrapper->getLBracLoc()), SM,
            LangOpts) ||
        isUnsafeTokenRange(
            SourceRange(Wrapper->getRBracLoc(), Wrapper->getRBracLoc()), SM,
            LangOpts)) {
      return false;
    }
  }

  return true;
}

static CombinedConditionBuildResult
buildCombinedCondition(llvm::ArrayRef<const IfStmt *> Chain,
                       const ASTContext &Context) {
  CombinedConditionBuildResult Result;
  if (Chain.empty())
    return Result;

  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();
  std::string Combined;

  for (std::size_t Index = 0; Index < Chain.size(); ++Index) {
    const IfStmt *If = Chain[Index];
    if (Index == 0 && If->hasVarStorage()) {
      const auto *CondVar = If->getConditionVariable();
      if (!CondVar)
        return Result;

      // Preserve comments and spelling in declaration conditions by using the
      // full payload text between parentheses instead of the AST declaration
      // source range.
      const std::optional<std::string> CondPayloadText =
          getConditionPayloadText(If, SM, LangOpts);
      if (!CondPayloadText)
        return Result;
      Combined += *CondPayloadText;
      Combined += "; ";
      const llvm::StringRef CondVarName = CondVar->getName();
      Combined.append(CondVarName.begin(), CondVarName.end());
      continue;
    }

    std::optional<std::string> CondText;
    if (Index == 0 && If->hasInitStorage()) {
      const llvm::StringRef CondExprText =
          tooling::fixit::getText(*If->getCond(), Context);
      if (CondExprText.empty())
        return Result;
      CondText = CondExprText.str();
    } else {
      CondText = getConditionPayloadText(If, SM, LangOpts);
      if (!CondText)
        return Result;
    }

    // Nested headers are removed by the fix-it. Keep only comments that are
    // semantically local to the removed header and can be preserved safely.
    if (Index > 0 && !hasOnlyPayloadCommentsInNestedHeader(If, SM, LangOpts)) {
      Result.Status = CombinedConditionBuildStatus::UnsupportedCommentPlacement;
      return Result;
    }

    if (!Combined.empty())
      Combined += " && ";
    // Parenthesize each condition to preserve precedence in the rewritten
    // condition regardless of original operator binding.
    Combined += "(";
    Combined += *CondText;
    Combined += ")";
  }
  Result.Status = CombinedConditionBuildStatus::Success;
  Result.Text = std::move(Combined);
  return Result;
}

static std::optional<CharSourceRange>
getConditionReplacementRange(const IfStmt *If, const SourceManager &SM,
                             const LangOptions &LangOpts) {
  if (!If)
    return std::nullopt;
  if (If->hasInitStorage() && !If->hasVarStorage())
    return CharSourceRange::getTokenRange(If->getCond()->getSourceRange());
  return getConditionPayloadRange(If, SM, LangOpts);
}

static ChainHandling
getChainHandling(llvm::ArrayRef<const IfStmt *> Chain,
                 const ASTContext &Context, const SourceManager &SM,
                 const LangOptions &LangOpts, bool WarnOnDependentConstexprIf,
                 std::optional<std::string> *CombinedCondition) {
  // One place for all gating logic: chain shape, edit safety, constexpr
  // semantic safety, and final condition synthesis.
  if (Chain.size() < 2)
    return ChainHandling::None;
  if (!isFixitSafeForChain(Chain, SM, LangOpts))
    return ChainHandling::None;
  if (!isConstexprChainSemanticallySafe(Chain, Context)) {
    return WarnOnDependentConstexprIf
               ? ChainHandling::WarnOnlyDependentConstexpr
               : ChainHandling::None;
  }

  const CombinedConditionBuildResult Combined =
      buildCombinedCondition(Chain, Context);
  if (Combined.Status ==
      CombinedConditionBuildStatus::UnsupportedCommentPlacement)
    return ChainHandling::WarnOnly;
  if (Combined.Status != CombinedConditionBuildStatus::Success)
    return ChainHandling::None;
  if (CombinedCondition)
    *CombinedCondition = Combined.Text;
  return ChainHandling::WarnAndFix;
}

static bool
parentWillHandleThisIf(const IfStmt *If, ASTContext &Context,
                       const SourceManager &SM, const LangOptions &LangOpts,
                       bool WarnOnDependentConstexprIf,
                       RedundantNestedIfCheck::UserDefinedBoolConversionMode
                           UserBoolConversionMode) {
  const IfStmt *const ParentIf = getParentThenIf(If, Context);
  if (!ParentIf)
    return false;

  const auto ParentChain =
      getMergeChain(ParentIf, Context, UserBoolConversionMode);
  if (ParentChain.size() < 2 || ParentChain[1] != If)
    return false;

  return getChainHandling(ParentChain, Context, SM, LangOpts,
                          WarnOnDependentConstexprIf,
                          /*CombinedCondition=*/nullptr) != ChainHandling::None;
}

static bool
parentWillHandleUserDefinedBoolWarning(const IfStmt *If, ASTContext &Context,
                                       const SourceManager &SM,
                                       const LangOptions &LangOpts) {
  const IfStmt *const ParentIf = getParentThenIf(If, Context);
  if (!ParentIf)
    return false;

  const auto ParentChain = getUserDefinedBoolWarnChain(ParentIf, Context);
  if (ParentChain.size() < 2 || ParentChain[1] != If)
    return false;
  return isFixitSafeForChain(ParentChain, SM, LangOpts);
}

RedundantNestedIfCheck::RedundantNestedIfCheck(StringRef Name,
                                               ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context), WarnOnDependentConstexprIf(Options.get(
                                         WarnOnDependentConstexprIfStr, false)),
      UserBoolConversionMode(Options.get(UserDefinedBoolConversionModeStr,
                                         UserDefinedBoolConversionMode::None)) {
}

void RedundantNestedIfCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, WarnOnDependentConstexprIfStr,
                WarnOnDependentConstexprIf);
  Options.store(Opts, UserDefinedBoolConversionModeStr, UserBoolConversionMode);
}

void RedundantNestedIfCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(ifStmt().bind("if"), this);
}

void RedundantNestedIfCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *If = Result.Nodes.getNodeAs<IfStmt>("if");
  if (!If)
    return;

  ASTContext &Context = *Result.Context;
  const SourceManager &SM = *Result.SourceManager;
  const LangOptions &LangOpts = Context.getLangOpts();

  const auto Chain = getMergeChain(If, Context, UserBoolConversionMode);
  if (Chain.size() < 2) {
    if (UserBoolConversionMode != UserDefinedBoolConversionMode::WarnOnly)
      return;

    if (parentWillHandleUserDefinedBoolWarning(If, Context, SM, LangOpts))
      return;

    const auto WarnChain = getUserDefinedBoolWarnChain(If, Context);
    if (WarnChain.size() < 2 || !isFixitSafeForChain(WarnChain, SM, LangOpts))
      return;

    diag(If->getIfLoc(), "nested if statements can be merged");
    return;
  }
  if (parentWillHandleThisIf(If, Context, SM, LangOpts,
                             WarnOnDependentConstexprIf,
                             UserBoolConversionMode)) {
    return;
  }

  std::optional<std::string> CombinedCondition;
  const ChainHandling Handling =
      getChainHandling(Chain, Context, SM, LangOpts, WarnOnDependentConstexprIf,
                       &CombinedCondition);
  if (Handling == ChainHandling::None)
    return;

  if (Handling == ChainHandling::WarnOnlyDependentConstexpr) {
    // Keep this as diagnostic-only: the option explicitly asks for awareness
    // in templates even when rewrite safety cannot be guaranteed.
    diag(If->getIfLoc(),
         "nested instantiation-dependent if constexpr statements can be "
         "merged");
    return;
  }

  if (Handling == ChainHandling::WarnOnly) {
    diag(If->getIfLoc(), "nested if statements can be merged");
    return;
  }

  auto Diag = diag(If->getIfLoc(), "nested if statements can be merged");
  const std::optional<CharSourceRange> CondRange =
      getConditionReplacementRange(If, SM, LangOpts);
  if (!CondRange || !CombinedCondition)
    return;
  Diag << FixItHint::CreateReplacement(*CondRange, *CombinedCondition);

  for (std::size_t Index = 1; Index < Chain.size(); ++Index) {
    const IfStmt *const Nested = Chain[Index];
    Diag << FixItHint::CreateRemoval(CharSourceRange::getCharRange(
        Nested->getBeginLoc(), Nested->getThen()->getBeginLoc()));

    if (const auto *Wrapper =
            dyn_cast<CompoundStmt>(Chain[Index - 1]->getThen())) {
      Diag << FixItHint::CreateRemoval(Wrapper->getLBracLoc())
           << FixItHint::CreateRemoval(Wrapper->getRBracLoc());
    }
  }
}

} // namespace clang::tidy::readability
