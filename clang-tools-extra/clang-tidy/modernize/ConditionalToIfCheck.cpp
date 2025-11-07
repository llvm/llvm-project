#include "ConditionalToIfCheck.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;
using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

namespace {

bool isInMacro(const SourceRange &R, const MatchFinder::MatchResult &Res) {
  return R.getBegin().isMacroID() || R.getEnd().isMacroID();
}

CharSourceRange tokenRange(const SourceRange &R,
                           const MatchFinder::MatchResult &Res) {
  return CharSourceRange::getTokenRange(R);
}

std::string getTokenText(const SourceRange &R,
                         const MatchFinder::MatchResult &Res) {
  return Lexer::getSourceText(tokenRange(R, Res),
                              *Res.SourceManager, Res.Context->getLangOpts())
      .str();
}

const Expr *strip(const Expr *E) { return E ? E->IgnoreParenImpCasts() : E; }

// Find the statement that directly contains E (best-effort).
const Stmt *enclosingStmt(const Expr *E, const MatchFinder::MatchResult &Res) {
  if (!E) return nullptr;
  const Stmt *Cur = E;
  while (true) {
    auto Parents = Res.Context->getParents(*Cur);
    if (Parents.empty()) return Cur;
    if (const Stmt *S = Parents[0].get<Stmt>()) {
      Cur = S;
      // Stop when Cur itself is a statement that could be replaced wholesale.
      if (isa<ExprWithCleanups>(Cur) || isa<ReturnStmt>(Cur) ||
          isa<CompoundStmt>(Cur) || isa<IfStmt>(Cur) ||
          isa<DeclStmt>(Cur) || isa<Expr>(Cur))
        continue;
      return Cur;
    } else {
      return Cur;
    }
  }
}

} // namespace

ConditionalToIfCheck::ConditionalToIfCheck(StringRef Name,
                                           ClangTidyContext *Ctx)
    : ClangTidyCheck(Name, Ctx) {
  StringRef V = Options.get("PreferredForm", "if");
  // Use StringSwitch for broad LLVM version compatibility.
  PreferredForm = llvm::StringSwitch<Preferred>(V)
                      .CaseLower("conditional", Preferred::Conditional)
                      .Default(Preferred::If);
}

void ConditionalToIfCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "PreferredForm",
                PreferredForm == Preferred::If ? "if" : "conditional");
}

void ConditionalToIfCheck::registerMatchers(MatchFinder *Finder) {
  // Only match code written in the main file to avoid noisy headers.
  auto InMain = isExpansionInMainFile();

  if (PreferredForm == Preferred::If) {
    // 1) x = cond ? a : b;
    Finder->addMatcher(
        binaryOperator(InMain, isAssignmentOperator(),
                       hasRHS(conditionalOperator().bind("condop")),
                       hasLHS(expr().bind("assignLHS")))
            .bind("assign"),
        this);

    // 2) return cond ? a : b;
    Finder->addMatcher(
        returnStmt(InMain,
                   hasReturnValue(conditionalOperator().bind("retop")))
            .bind("ret"),
        this);
  } else {
    // Preferred::Conditional

    // 3a) if (cond) x = A; else x = B;
    // (Match a simple assignment in each branch; allow it to be wrapped in
    // an ExprWithCleanups/ExprStmt or a single-statement compound.)
    auto AssignThen =
        binaryOperator(isAssignmentOperator())
            .bind("thenA");
    auto AssignElse =
        binaryOperator(isAssignmentOperator())
            .bind("elseA");

    Finder->addMatcher(
        ifStmt(InMain,
               hasThen(anyOf(
                   hasDescendant(AssignThen),
                   compoundStmt(statementCountIs(1),
                                hasAnySubstatement(hasDescendant(AssignThen))))),
               hasElse(anyOf(
                   hasDescendant(AssignElse),
                   compoundStmt(statementCountIs(1),
                                hasAnySubstatement(hasDescendant(AssignElse))))))
            .bind("ifAssign"),
        this);

    // 3b) if (cond) return A; else return B;
    auto RetThen = returnStmt(hasReturnValue(expr().bind("thenR"))).bind("thenRet");
    auto RetElse = returnStmt(hasReturnValue(expr().bind("elseR"))).bind("elseRet");

    Finder->addMatcher(
        ifStmt(InMain,
               hasThen(anyOf(RetThen,
                             compoundStmt(statementCountIs(1),
                                          hasAnySubstatement(RetThen)))),
               hasElse(anyOf(RetElse,
                             compoundStmt(statementCountIs(1),
                                          hasAnySubstatement(RetElse)))))
            .bind("ifReturn"),
        this);
  }
}

bool ConditionalToIfCheck::locationsAreOK(
    const SourceRange &R, const MatchFinder::MatchResult &Rst) {
  if (R.isInvalid())
    return false;
  if (isInMacro(R, Rst))
    return false;
  if (!Rst.SourceManager->isWrittenInMainFile(R.getBegin()))
    return false;
  return true;
}

std::string ConditionalToIfCheck::getText(
    const SourceRange &R, const MatchFinder::MatchResult &Rst) {
  return getTokenText(R, Rst);
}

bool ConditionalToIfCheck::hasObviousSideEffects(const Expr *E,
                                                 ASTContext &Ctx) {
  if (!E)
    return true;
  E = strip(E);

  // Very conservative: rely on Clang's side-effect query.
  if (E->HasSideEffects(Ctx))
    return true;

  // Additional heuristics for common side-effect nodes.
  if (isa<CallExpr>(E) || isa<CXXConstructExpr>(E) || isa<CXXOperatorCallExpr>(E))
    return true;

  if (const auto *U = dyn_cast<UnaryOperator>(E)) {
    if (U->isIncrementDecrementOp() || U->getOpcode() == UO_AddrOf ||
        U->getOpcode() == UO_Deref)
      return true;
  }
  if (const auto *B = dyn_cast<BinaryOperator>(E)) {
    if (B->isAssignmentOp() || B->getOpcode() == BO_Comma)
      return true;
  }
  return false;
}

void ConditionalToIfCheck::check(const MatchFinder::MatchResult &Res) {
  ASTContext &Ctx = *Res.Context;

  if (PreferredForm == Preferred::If) {
    // Handle: return cond ? a : b;
    if (const auto *Ret = Res.Nodes.getNodeAs<ReturnStmt>("ret")) {
      const auto *CO = Res.Nodes.getNodeAs<ConditionalOperator>("retop");
      if (!CO) return;

      const Expr *Cond = strip(CO->getCond());
      const Expr *TrueE = strip(CO->getTrueExpr());
      const Expr *FalseE = strip(CO->getFalseExpr());

      if (hasObviousSideEffects(Cond, Ctx) ||
          hasObviousSideEffects(TrueE, Ctx) ||
          hasObviousSideEffects(FalseE, Ctx))
        return;

      SourceRange SR = Ret->getSourceRange();
      if (!locationsAreOK(SR, Res))
        return;

      std::string CondS = getText(Cond->getSourceRange(), Res);
      std::string TS = getText(TrueE->getSourceRange(), Res);
      std::string FS = getText(FalseE->getSourceRange(), Res);

      std::string Replacement = "if (" + CondS + ") {\n  return " + TS +
                                ";\n} else {\n  return " + FS + ";\n}";
      diag(Ret->getBeginLoc(),
           "replace simple conditional return with if/else")
          << FixItHint::CreateReplacement(CharSourceRange::getCharRange(SR),
                                          Replacement);
      return;
    }

    // Handle: x = cond ? a : b;
    if (const auto *Assign = Res.Nodes.getNodeAs<BinaryOperator>("assign")) {
      const auto *CO = Res.Nodes.getNodeAs<ConditionalOperator>("condop");
      const auto *LHS = Res.Nodes.getNodeAs<Expr>("assignLHS");
      if (!CO || !LHS) return;

      const Expr *Cond = strip(CO->getCond());
      const Expr *TrueE = strip(CO->getTrueExpr());
      const Expr *FalseE = strip(CO->getFalseExpr());

      if (hasObviousSideEffects(Cond, Ctx) ||
          hasObviousSideEffects(TrueE, Ctx) ||
          hasObviousSideEffects(FalseE, Ctx) ||
          hasObviousSideEffects(LHS, Ctx))
        return;

      const Stmt *Carrier = enclosingStmt(Assign, Res);
      if (!Carrier)
        Carrier = Assign;

      SourceRange SR = Carrier->getSourceRange();
      if (!locationsAreOK(SR, Res))
        return;

      std::string LHSS = getText(LHS->getSourceRange(), Res);
      std::string CondS = getText(Cond->getSourceRange(), Res);
      std::string TS = getText(TrueE->getSourceRange(), Res);
      std::string FS = getText(FalseE->getSourceRange(), Res);

      std::string Replacement = "if (" + CondS + ") {\n  " + LHSS + " = " + TS +
                                ";\n} else {\n  " + LHSS + " = " + FS +
                                ";\n}";
      diag(Carrier->getBeginLoc(),
           "replace simple conditional assignment with if/else")
          << FixItHint::CreateReplacement(CharSourceRange::getCharRange(SR),
                                          Replacement);
      return;
    }
    return;
  }

  // PreferredForm == Conditional

  // if (cond) return A; else return B;
  if (const auto *IfR = Res.Nodes.getNodeAs<IfStmt>("ifReturn")) {
    const Expr *Cond = strip(IfR->getCond());
    const Expr *ThenR = strip(Res.Nodes.getNodeAs<Expr>("thenR"));
    const Expr *ElseR = strip(Res.Nodes.getNodeAs<Expr>("elseR"));
    if (!Cond || !ThenR || !ElseR) return;

    if (hasObviousSideEffects(Cond, Ctx) ||
        hasObviousSideEffects(ThenR, Ctx) ||
        hasObviousSideEffects(ElseR, Ctx))
      return;

    SourceRange SR = IfR->getSourceRange();
    if (!locationsAreOK(SR, Res))
      return;

    std::string CondS = getText(Cond->getSourceRange(), Res);
    std::string TS = getText(ThenR->getSourceRange(), Res);
    std::string FS = getText(ElseR->getSourceRange(), Res);

    std::string Replacement =
        "return (" + CondS + ") ? " + TS + " : " + FS + ";";
    diag(IfR->getBeginLoc(), "replace simple if/else with conditional return")
        << FixItHint::CreateReplacement(CharSourceRange::getCharRange(SR),
                                        Replacement);
    return;
  }

  // if (cond) x = A; else x = B;
  if (const auto *IfA = Res.Nodes.getNodeAs<IfStmt>("ifAssign")) {
    const Expr *Cond = strip(IfA->getCond());
    const auto *ThenA = Res.Nodes.getNodeAs<BinaryOperator>("thenA");
    const auto *ElseA = Res.Nodes.getNodeAs<BinaryOperator>("elseA");
    if (!Cond || !ThenA || !ElseA)
      return;

    const Expr *ThenL = strip(ThenA->getLHS());
    const Expr *ElseL = strip(ElseA->getLHS());
    const Expr *ThenR = strip(ThenA->getRHS());
    const Expr *ElseR = strip(ElseA->getRHS());
    if (!ThenL || !ElseL || !ThenR || !ElseR)
      return;

    // LHS must be textually identical (safe & simple).
    std::string LThen = getText(ThenL->getSourceRange(), Res);
    std::string LElse = getText(ElseL->getSourceRange(), Res);
    if (LThen != LElse)
      return;

    if (hasObviousSideEffects(Cond, Ctx) ||
        hasObviousSideEffects(ThenR, Ctx) ||
        hasObviousSideEffects(ElseR, Ctx) ||
        hasObviousSideEffects(ThenL, Ctx))
      return;

    SourceRange SR = IfA->getSourceRange();
    if (!locationsAreOK(SR, Res))
      return;

    std::string CondS = getText(Cond->getSourceRange(), Res);
    std::string TS = getText(ThenR->getSourceRange(), Res);
    std::string FS = getText(ElseR->getSourceRange(), Res);

    std::string Replacement =
        LThen + " = (" + CondS + ") ? " + TS + " : " + FS + ";";
    diag(IfA->getBeginLoc(),
         "replace simple if/else assignment with conditional expression")
        << FixItHint::CreateReplacement(CharSourceRange::getCharRange(SR),
                                        Replacement);
    return;
  }
}

} // namespace clang::tidy::modernize
