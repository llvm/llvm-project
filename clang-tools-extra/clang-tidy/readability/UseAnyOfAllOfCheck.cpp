//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseAnyOfAllOfCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Analysis/Analyses/ExprMutationAnalyzer.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

using namespace clang::ast_matchers;

namespace clang {
namespace {
/// Matches a Stmt whose parent is a CompoundStmt, and which is directly
/// followed by a Stmt matching the inner matcher.
AST_MATCHER_P(Stmt, nextStmt, ast_matchers::internal::Matcher<Stmt>,
              InnerMatcher) {
  const DynTypedNodeList Parents = Finder->getASTContext().getParents(Node);
  if (Parents.size() != 1)
    return false;

  auto *C = Parents[0].get<CompoundStmt>();
  if (!C)
    return false;

  const auto *I = llvm::find(C->body(), &Node);
  assert(I != C->body_end() && "C is parent of Node");
  if (++I == C->body_end())
    return false; // Node is last statement.

  return InnerMatcher.matches(**I, Finder, Builder);
}

static bool returnsBoolLiteral(const ReturnStmt *Ret, bool Value) {
  if (!Ret || !Ret->getRetValue())
    return false;

  const auto *BoolLit =
      dyn_cast<CXXBoolLiteralExpr>(Ret->getRetValue()->IgnoreImplicit());
  return BoolLit && BoolLit->getValue() == Value;
}

static StringRef getExprText(const Expr *E, const SourceManager &SM,
                             const LangOptions &LangOpts) {
  return Lexer::getSourceText(
      CharSourceRange::getTokenRange(E->getSourceRange()), SM, LangOpts);
}

static SourceLocation getStmtEndIncludingSemicolon(const Stmt &S,
                                                   const SourceManager &SM,
                                                   const LangOptions &LangOpts) {
  SourceLocation End = S.getEndLoc();
  if (std::optional<Token> Next = Lexer::findNextToken(End, SM, LangOpts)) {
    if (Next->is(tok::semi))
      return Next->getEndLoc();
  }
  return End;
}

struct FixItInfo {
  llvm::StringRef Algorithm;
  std::string Replacement;
};

} // namespace

namespace tidy::readability {

UseAnyOfAllOfCheck::UseAnyOfAllOfCheck(StringRef Name,
                                       ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Inserter(Options.getLocalOrGlobal("IncludeStyle",
                                        utils::IncludeSorter::IS_LLVM),
               areDiagsSelfContained()) {}

void UseAnyOfAllOfCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", Inserter.getStyle());
}

void UseAnyOfAllOfCheck::registerPPCallbacks(const SourceManager &SM,
                                             Preprocessor *PP,
                                             Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void UseAnyOfAllOfCheck::registerMatchers(MatchFinder *Finder) {
  auto Returns = [](bool V) {
    return returnStmt(hasReturnValue(cxxBoolLiteral(equals(V))));
  };

  auto ReturnsButNotTrue =
      returnStmt(hasReturnValue(unless(cxxBoolLiteral(equals(true)))));
  auto ReturnsButNotFalse =
      returnStmt(hasReturnValue(unless(cxxBoolLiteral(equals(false)))));

  Finder->addMatcher(
      cxxForRangeStmt(
          nextStmt(Returns(false).bind("final_return")),
          hasBody(allOf(hasDescendant(Returns(true)),
                        unless(anyOf(hasDescendant(breakStmt()),
                                     hasDescendant(gotoStmt()),
                                     hasDescendant(ReturnsButNotTrue))))))
          .bind("any_of_loop"),
      this);

  Finder->addMatcher(
      cxxForRangeStmt(
          nextStmt(Returns(true).bind("final_return")),
          hasBody(allOf(hasDescendant(Returns(false)),
                        unless(anyOf(hasDescendant(breakStmt()),
                                     hasDescendant(gotoStmt()),
                                     hasDescendant(ReturnsButNotFalse))))))
          .bind("all_of_loop"),
      this);
}

static bool isViableLoop(const CXXForRangeStmt &S, ASTContext &Context) {
  ExprMutationAnalyzer Mutations(*S.getBody(), Context);
  if (Mutations.isMutated(S.getLoopVariable()))
    return false;
  const auto Matches =
      match(findAll(declRefExpr().bind("decl_ref")), *S.getBody(), Context);

  return llvm::none_of(Matches, [&Mutations](auto &DeclRef) {
    // TODO: allow modifications of loop-local variables
    return Mutations.isMutated(
        DeclRef.template getNodeAs<DeclRefExpr>("decl_ref")->getDecl());
  });
}

static const IfStmt *getSingleIfInLoopBody(const CXXForRangeStmt &Loop) {
  const Stmt *Body = Loop.getBody();
  if (const auto *If = dyn_cast<IfStmt>(Body))
    return If;

  const auto *Compound = dyn_cast<CompoundStmt>(Body);
  if (!Compound || Compound->size() != 1)
    return nullptr;

  return dyn_cast<IfStmt>(Compound->body_front());
}

static std::optional<FixItInfo>
createFixItForLoop(const CXXForRangeStmt &Loop, const ReturnStmt &FinalReturn,
                   bool IfMustReturnTrue, const SourceManager &SM,
                   const LangOptions &LangOpts, bool UseRanges) {
  if (Loop.getBeginLoc().isMacroID() || FinalReturn.getBeginLoc().isMacroID())
    return std::nullopt;

  const IfStmt *If = getSingleIfInLoopBody(Loop);
  if (!If || If->getElse() || If->getInit() || If->getConditionVariable())
    return std::nullopt;

  const ReturnStmt *IfReturn = nullptr;
  if (const auto *ThenCompound = dyn_cast<CompoundStmt>(If->getThen())) {
    if (ThenCompound->size() != 1)
      return std::nullopt;
    IfReturn = dyn_cast<ReturnStmt>(ThenCompound->body_front());
  } else {
    IfReturn = dyn_cast<ReturnStmt>(If->getThen());
  }

  if (!returnsBoolLiteral(IfReturn, IfMustReturnTrue))
    return std::nullopt;

  if (!returnsBoolLiteral(&FinalReturn, !IfMustReturnTrue))
    return std::nullopt;

  const Expr *Container = Loop.getRangeInit();
  const VarDecl *LoopVar = Loop.getLoopVariable();
  const Expr *PredicateExpr = If->getCond();
  llvm::StringRef Algorithm = "any_of";

  if (!IfMustReturnTrue) {
    if (const auto *Negated = dyn_cast<UnaryOperator>(
            PredicateExpr->IgnoreParenImpCasts())) {
      if (Negated->getOpcode() == UO_LNot) {
        Algorithm = "all_of";
        PredicateExpr = Negated->getSubExpr();
      } else {
        Algorithm = "none_of";
      }
    } else {
      Algorithm = "none_of";
    }
  }

  const StringRef ContainerText = getExprText(Container, SM, LangOpts);
  const StringRef PredicateText = getExprText(PredicateExpr, SM, LangOpts);
  const StringRef LoopVarName = LoopVar->getName();

  if (ContainerText.empty() || PredicateText.empty() || LoopVarName.empty())
    return std::nullopt;

  std::string LoopVarType;
  const SourceLocation TypeStart = LoopVar->getBeginLoc();
  const SourceLocation NameStart = LoopVar->getLocation();
  if (TypeStart.isValid() && NameStart.isValid()) {
    LoopVarType =
        Lexer::getSourceText(CharSourceRange::getCharRange(TypeStart, NameStart),
                             SM, LangOpts)
            .rtrim()
            .str();
  }
  if (LoopVarType.empty())
    LoopVarType = LoopVar->getType().getAsString();

  const StringRef Separator =
      (!LoopVarType.empty() &&
       (LoopVarType.back() == '&' || LoopVarType.back() == '*'))
          ? ""
          : " ";

  std::string Replacement;
  llvm::raw_string_ostream OS(Replacement);
  OS << "return std";
  if (UseRanges) {
    OS << "::ranges::" << Algorithm << "(" << ContainerText << ", ["
       << "](" << LoopVarType << Separator << LoopVarName << ") { return "
       << PredicateText << "; });";
  } else {
    OS << "::" << Algorithm << "(std::begin(" << ContainerText
       << "), std::end(" << ContainerText << "), ["
       << "](" << LoopVarType << Separator << LoopVarName << ") { return "
       << PredicateText << "; });";
  }
  OS.flush();

  return FixItInfo{Algorithm, Replacement};
}

void UseAnyOfAllOfCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *FinalReturn = Result.Nodes.getNodeAs<ReturnStmt>("final_return");
  if (!FinalReturn)
    return;

  const bool UseRanges = getLangOpts().CPlusPlus20;

  if (const auto *Loop = Result.Nodes.getNodeAs<CXXForRangeStmt>("any_of_loop")) {
    if (!isViableLoop(*Loop, *Result.Context))
      return;

    if (std::optional<FixItInfo> Fix =
            createFixItForLoop(*Loop, *FinalReturn, true, *Result.SourceManager,
                               Result.Context->getLangOpts(), UseRanges)) {
      auto Diag =
          diag(Loop->getForLoc(), "replace loop by 'std%select{|::ranges}0::%1()'")
          << UseRanges << Fix->Algorithm;
      const SourceLocation ReplaceEnd = getStmtEndIncludingSemicolon(
          *FinalReturn, *Result.SourceManager, Result.Context->getLangOpts());
      Diag << FixItHint::CreateReplacement(
          CharSourceRange::getCharRange(Loop->getBeginLoc(), ReplaceEnd),
          Fix->Replacement);
      if (auto IncludeFixIt = Inserter.createIncludeInsertion(
              Result.SourceManager->getFileID(Loop->getBeginLoc()),
              "<algorithm>"))
        Diag << *IncludeFixIt;
      if (!UseRanges) {
        if (auto IteratorIncludeFixIt = Inserter.createIncludeInsertion(
                Result.SourceManager->getFileID(Loop->getBeginLoc()),
                "<iterator>"))
          Diag << *IteratorIncludeFixIt;
      }
      return;
    }

    diag(Loop->getForLoc(), "replace loop by 'std%select{|::ranges}0::any_of()'")
        << UseRanges;
    return;
  }

  if (const auto *Loop = Result.Nodes.getNodeAs<CXXForRangeStmt>("all_of_loop")) {
    if (!isViableLoop(*Loop, *Result.Context))
      return;

    if (std::optional<FixItInfo> Fix =
            createFixItForLoop(*Loop, *FinalReturn, false, *Result.SourceManager,
                               Result.Context->getLangOpts(), UseRanges)) {
      auto Diag =
          diag(Loop->getForLoc(), "replace loop by 'std%select{|::ranges}0::%1()'")
          << UseRanges << Fix->Algorithm;
      const SourceLocation ReplaceEnd = getStmtEndIncludingSemicolon(
          *FinalReturn, *Result.SourceManager, Result.Context->getLangOpts());
      Diag << FixItHint::CreateReplacement(
          CharSourceRange::getCharRange(Loop->getBeginLoc(), ReplaceEnd),
          Fix->Replacement);
      if (auto IncludeFixIt = Inserter.createIncludeInsertion(
              Result.SourceManager->getFileID(Loop->getBeginLoc()),
              "<algorithm>"))
        Diag << *IncludeFixIt;
      if (!UseRanges) {
        if (auto IteratorIncludeFixIt = Inserter.createIncludeInsertion(
                Result.SourceManager->getFileID(Loop->getBeginLoc()),
                "<iterator>"))
          Diag << *IteratorIncludeFixIt;
      }
      return;
    }

    diag(Loop->getForLoc(), "replace loop by 'std%select{|::ranges}0::all_of()'")
        << UseRanges;
  }
}

} // namespace tidy::readability
} // namespace clang
