//===--- IfSwitchConversion.cpp -  ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the "convert to switch" refactoring operation.
//
//===----------------------------------------------------------------------===//

#include "RefactoringOperations.h"
#include "SourceLocationUtilities.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"

using namespace clang;
using namespace clang::tooling;

namespace {

class IfSwitchConversionOperation : public RefactoringOperation {
public:
  IfSwitchConversionOperation(const IfStmt *If) : If(If) {}

  const Stmt *getTransformedStmt() const override { return If; }

  llvm::Expected<RefactoringResult> perform(ASTContext &Context, const Preprocessor &ThePreprocessor,
          const RefactoringOptionSet &Options,
          unsigned SelectedCandidateIndex) override;

  const IfStmt *If;
};

class ValidIfBodyVerifier : public RecursiveASTVisitor<ValidIfBodyVerifier> {
  bool CheckBreaks = true;

public:
  bool IsValid = true;

  bool VisitBreakStmt(const BreakStmt *S) {
    if (!CheckBreaks)
      return true;
    IsValid = false;
    return false;
  }
  bool VisitDefaultStmt(const DefaultStmt *S) {
    IsValid = false;
    return false;
  }
  bool VisitCaseStmt(const CaseStmt *S) {
    IsValid = false;
    return false;
  }

// Handle nested loops:

#define TRAVERSE_LOOP(STMT)                                                    \
  bool Traverse##STMT(STMT *S) {                                               \
    bool Prev = CheckBreaks;                                                   \
    CheckBreaks = false;                                                       \
    RecursiveASTVisitor::Traverse##STMT(S);                                    \
    CheckBreaks = Prev;                                                        \
    return true;                                                               \
  }

  TRAVERSE_LOOP(ForStmt)
  TRAVERSE_LOOP(WhileStmt)
  TRAVERSE_LOOP(DoStmt)
  TRAVERSE_LOOP(CXXForRangeStmt)
  TRAVERSE_LOOP(ObjCForCollectionStmt)

#undef TRAVERSE_LOOP

  // Handle switches:

  bool TraverseSwitchStmt(SwitchStmt *S) {
    // Don't visit the body as 'break'/'case'/'default' are all allowed inside
    // switches.
    return true;
  }
};

} // end anonymous namespace

/// Returns true if any of the if statements in the given if construct have
/// conditions that aren't allowed by the "convert to switch" operation.
static bool checkIfsHaveConditionExpression(const IfStmt *If) {
  for (; If; If = dyn_cast_or_null<IfStmt>(If->getElse())) {
    if (If->getConditionVariable() || If->getInit() || !If->getCond())
      return true;
  }
  return false;
}

static Optional<std::pair<const Expr *, const Expr *>>
matchBinOp(const Expr *E, BinaryOperator::Opcode Kind) {
  const auto *BinOp = dyn_cast<BinaryOperator>(E->IgnoreParens());
  if (!BinOp || BinOp->getOpcode() != Kind)
    return None;
  return std::pair<const Expr *, const Expr *>(
      BinOp->getLHS()->IgnoreParenImpCasts(), BinOp->getRHS()->IgnoreParens());
}

typedef llvm::SmallDenseSet<int64_t, 4> RHSValueSet;

/// Returns true if the conditional expression of an 'if' statement allows
/// the "convert to switch" refactoring action.
static bool isConditionValid(const Expr *E, ASTContext &Context,
                             Optional<llvm::FoldingSetNodeID> &MatchedLHSNodeID,
                             RHSValueSet &RHSValues) {
  auto Equals = matchBinOp(E, BO_EQ);
  if (!Equals.hasValue()) {
    auto LogicalOr = matchBinOp(E, BO_LOr);
    if (!LogicalOr.hasValue())
      return false;
    return isConditionValid(LogicalOr.getValue().first, Context,
                            MatchedLHSNodeID, RHSValues) &&
           isConditionValid(LogicalOr.getValue().second, Context,
                            MatchedLHSNodeID, RHSValues);
  }
  const Expr *LHS = Equals.getValue().first;
  const Expr *RHS = Equals.getValue().second;
  if (!LHS->getType()->isIntegralOrEnumerationType() ||
      !RHS->getType()->isIntegralOrEnumerationType())
    return false;

  // RHS must be a constant and unique.
  Expr::EvalResult Result;
  if (!RHS->EvaluateAsInt(Result, Context))
    return false;
  // Only allow constant that fix into 64 bits.
  if (Result.Val.getInt().getMinSignedBits() > 64 ||
      !RHSValues.insert(Result.Val.getInt().getExtValue()).second)
    return false;

  // LHS must be identical to the other LHS expressions.
  llvm::FoldingSetNodeID LHSNodeID;
  LHS->Profile(LHSNodeID, Context, /*Canonical=*/false);
  if (MatchedLHSNodeID.hasValue()) {
    if (MatchedLHSNodeID.getValue() != LHSNodeID)
      return false;
  } else
    MatchedLHSNodeID = std::move(LHSNodeID);
  return true;
}

RefactoringOperationResult clang::tooling::initiateIfSwitchConversionOperation(
    ASTSlice &Slice, ASTContext &Context, SourceLocation Location,
    SourceRange SelectionRange, bool CreateOperation) {
  // FIXME: Add support for selections.
  const auto *If = cast_or_null<IfStmt>(Slice.nearestStmt(Stmt::IfStmtClass));
  if (!If)
    return None;

  // Don't allow if statements without any 'else' or 'else if'.
  if (!If->getElse())
    return None;

  // Don't allow ifs with variable declarations in conditions or C++17
  // initializer statements.
  if (checkIfsHaveConditionExpression(If))
    return None;

  // Find the ranges in which initiation can be performed and verify that the
  // ifs don't have any initialization expressions or condition variables.
  SmallVector<SourceRange, 4> Ranges;
  SourceLocation RangeStart = If->getBeginLoc();
  const IfStmt *CurrentIf = If;
  const SourceManager &SM = Context.getSourceManager();
  while (true) {
    const Stmt *Then = CurrentIf->getThen();
    Ranges.emplace_back(RangeStart,
                        findLastLocationOfSourceConstruct(
                            CurrentIf->getCond()->getEndLoc(), Then, SM));
    const auto *Else = CurrentIf->getElse();
    if (!Else)
      break;
    RangeStart =
        findFirstLocationOfSourceConstruct(CurrentIf->getElseLoc(), Then, SM);
    if (const auto *If = dyn_cast<IfStmt>(Else)) {
      CurrentIf = If;
      continue;
    }
    Ranges.emplace_back(RangeStart, findLastLocationOfSourceConstruct(
                                        CurrentIf->getElseLoc(), Else, SM));
    break;
  }

  if (!isLocationInAnyRange(Location, Ranges, SM))
    return None;

  // Verify that the bodies don't have any 'break'/'default'/'case' statements.
  ValidIfBodyVerifier BodyVerifier;
  BodyVerifier.TraverseStmt(const_cast<IfStmt *>(If));
  if (!BodyVerifier.IsValid)
    return RefactoringOperationResult(
        "if's body contains a 'break'/'default'/'case' statement");

  // FIXME: Use ASTMatchers if possible.
  Optional<llvm::FoldingSetNodeID> MatchedLHSNodeID;
  RHSValueSet RHSValues;
  for (const IfStmt *CurrentIf = If; CurrentIf;
       CurrentIf = dyn_cast_or_null<IfStmt>(CurrentIf->getElse())) {
    if (!isConditionValid(CurrentIf->getCond(), Context, MatchedLHSNodeID,
                          RHSValues))
      return RefactoringOperationResult("unsupported conditional expression");
  }

  RefactoringOperationResult Result;
  Result.Initiated = true;
  if (CreateOperation)
    Result.RefactoringOp.reset(new IfSwitchConversionOperation(If));
  return Result;
}

/// Returns the first LHS expression in the if's condition.
const Expr *getConditionFirstLHS(const Expr *E) {
  auto Equals = matchBinOp(E, BO_EQ);
  if (!Equals.hasValue()) {
    auto LogicalOr = matchBinOp(E, BO_LOr);
    if (!LogicalOr.hasValue())
      return nullptr;
    return getConditionFirstLHS(LogicalOr.getValue().first);
  }
  return Equals.getValue().first;
}

/// Gathers all of the RHS operands of the == expressions in the if's condition.
void gatherCaseValues(const Expr *E,
                      SmallVectorImpl<const Expr *> &CaseValues) {
  auto Equals = matchBinOp(E, BO_EQ);
  if (Equals.hasValue()) {
    CaseValues.push_back(Equals.getValue().second);
    return;
  }
  auto LogicalOr = matchBinOp(E, BO_LOr);
  if (!LogicalOr.hasValue())
    return;
  gatherCaseValues(LogicalOr.getValue().first, CaseValues);
  gatherCaseValues(LogicalOr.getValue().second, CaseValues);
}

/// Return true iff the given body should be terminated with a 'break' statement
/// when used inside of a switch.
static bool isBreakNeeded(const Stmt *Body) {
  const auto *CS = dyn_cast<CompoundStmt>(Body);
  if (!CS)
    return !isa<ReturnStmt>(Body);
  return CS->body_empty() ? true : isBreakNeeded(CS->body_back());
}

/// Returns true if the given statement declares a variable.
static bool isVarDeclaringStatement(const Stmt *S) {
  const auto *DS = dyn_cast<DeclStmt>(S);
  if (!DS)
    return false;
  for (const Decl *D : DS->decls()) {
    if (isa<VarDecl>(D))
      return true;
  }
  return false;
}

/// Return true if the body of an if/else if/else needs to be wrapped in braces
/// when put in a switch.
static bool areBracesNeeded(const Stmt *Body) {
  const auto *CS = dyn_cast<CompoundStmt>(Body);
  if (!CS)
    return isVarDeclaringStatement(Body);
  for (const Stmt *S : CS->body()) {
    if (isVarDeclaringStatement(S))
      return true;
  }
  return false;
}

namespace {

/// Information about the replacement that replaces 'if'/'else' with a 'case' or
/// a 'default'.
struct CasePlacement {
  /// The location of the 'case' or 'default'.
  SourceLocation CaseStartLoc;
  /// True when this 'case' or 'default' statement needs a newline.
  bool NeedsNewLine;
  /// True if this the first 'if' in the source construct.
  bool IsFirstIf;
  /// True if we need to insert a 'break' to terminate the previous body
  /// before the 'case' or 'default'.
  bool IsBreakNeeded;
  /// True if we need to insert a '}' before the case.
  bool ArePreviousBracesNeeded;

  CasePlacement(SourceLocation Loc)
      : CaseStartLoc(Loc), NeedsNewLine(false), IsFirstIf(true),
        IsBreakNeeded(false), ArePreviousBracesNeeded(false) {}

  CasePlacement(const IfStmt *If, const SourceManager &SM,
                bool AreBracesNeeded) {
    CaseStartLoc = SM.getSpellingLoc(isa<CompoundStmt>(If->getThen())
                                         ? If->getThen()->getEndLoc()
                                         : If->getElseLoc());
    SourceLocation BodyEndLoc = findLastNonCompoundLocation(If->getThen());
    NeedsNewLine = BodyEndLoc.isValid()
                       ? areOnSameLine(CaseStartLoc, BodyEndLoc, SM)
                       : false;
    IsFirstIf = false;
    IsBreakNeeded = isBreakNeeded(If->getThen());
    ArePreviousBracesNeeded = AreBracesNeeded;
  }

  std::string getCaseReplacementString(bool IsDefault = false,
                                       bool AreNextBracesNeeded = false) const {
    if (IsFirstIf)
      return ") {\ncase ";
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    if (NeedsNewLine)
      OS << '\n';
    if (IsBreakNeeded)
      OS << "break;\n";
    if (ArePreviousBracesNeeded)
      OS << "}\n";
    OS << (IsDefault ? "default:" : "case ");
    if (IsDefault && AreNextBracesNeeded)
      OS << " {";
    return std::move(OS.str());
  }
};

} // end anonymous namespace

static llvm::Error
addCaseReplacements(const IfStmt *If, const CasePlacement &CaseInfo,
                    bool &AreBracesNeeded,
                    std::vector<RefactoringReplacement> &Replacements,
                    const SourceManager &SM, const LangOptions &LangOpts) {
  SmallVector<const Expr *, 2> CaseValues;
  gatherCaseValues(If->getCond(), CaseValues);
  assert(!CaseValues.empty());
  Replacements.emplace_back(
      SourceRange(CaseInfo.CaseStartLoc,
                  SM.getSpellingLoc(CaseValues[0]->getBeginLoc())),
      CaseInfo.getCaseReplacementString());

  SourceLocation PrevCaseEnd = getPreciseTokenLocEnd(
      SM.getSpellingLoc(CaseValues[0]->getEndLoc()), SM, LangOpts);
  for (const Expr *CaseValue : llvm::makeArrayRef(CaseValues).drop_front()) {
    Replacements.emplace_back(
        SourceRange(PrevCaseEnd, SM.getSpellingLoc(CaseValue->getBeginLoc())),
        StringRef(":\ncase "));
    PrevCaseEnd = getPreciseTokenLocEnd(
        SM.getSpellingLoc(CaseValue->getEndLoc()), SM, LangOpts);
  }

  AreBracesNeeded = areBracesNeeded(If->getThen());
  StringRef ColonReplacement = AreBracesNeeded ? ": {" : ":";
  if (isa<CompoundStmt>(If->getThen())) {
    Replacements.emplace_back(
        SourceRange(
            PrevCaseEnd,
            getPreciseTokenLocEnd(
                SM.getSpellingLoc(If->getThen()->getBeginLoc()), SM, LangOpts)),
        ColonReplacement);
  } else {
    // Find the location of the if's ')'
    SourceLocation End = findClosingParenLocEnd(
        SM.getSpellingLoc(If->getCond()->getEndLoc()), SM, LangOpts);
    if (!End.isValid())
      return llvm::make_error<RefactoringOperationError>(
          "couldn't find the location of ')'");
    Replacements.emplace_back(SourceRange(PrevCaseEnd, End), ColonReplacement);
  }
  return llvm::Error::success();
}

llvm::Expected<RefactoringResult>
IfSwitchConversionOperation::perform(ASTContext &Context,
                                     const Preprocessor &ThePreprocessor,
                                     const RefactoringOptionSet &Options,
                                     unsigned SelectedCandidateIndex) {
  std::vector<RefactoringReplacement> Replacements;
  const SourceManager &SM = Context.getSourceManager();
  const LangOptions &LangOpts = Context.getLangOpts();

  // The first if should be replaced with a 'switch' and the text for first LHS
  // should be preserved.
  const Expr *LHS = getConditionFirstLHS(If->getCond());
  assert(LHS && "Missing == expression");
  Replacements.emplace_back(SourceRange(SM.getSpellingLoc(If->getBeginLoc()),
                                        SM.getSpellingLoc(LHS->getBeginLoc())),
                            StringRef("switch ("));

  bool AreBracesNeeded = false;
  if (auto Error = addCaseReplacements(
          If, CasePlacement(getPreciseTokenLocEnd(
                  SM.getSpellingLoc(LHS->getEndLoc()), SM, LangOpts)),
          AreBracesNeeded, Replacements, SM, LangOpts))
    return std::move(Error);

  // Convert the remaining ifs to 'case' statements.
  const IfStmt *CurrentIf = If;
  while (true) {
    const IfStmt *NextIf = dyn_cast_or_null<IfStmt>(CurrentIf->getElse());
    if (!NextIf)
      break;
    if (auto Error = addCaseReplacements(
            NextIf, CasePlacement(CurrentIf, SM, AreBracesNeeded),
            AreBracesNeeded, Replacements, SM, LangOpts))
      return std::move(Error);
    CurrentIf = NextIf;
  }

  // Convert the 'else' to 'default'
  if (const Stmt *Else = CurrentIf->getElse()) {
    CasePlacement DefaultInfo(CurrentIf, SM, AreBracesNeeded);
    AreBracesNeeded = areBracesNeeded(Else);

    SourceLocation EndLoc = getPreciseTokenLocEnd(
        SM.getSpellingLoc(isa<CompoundStmt>(Else) ? Else->getBeginLoc()
                                                  : CurrentIf->getElseLoc()),
        SM, LangOpts);
    Replacements.emplace_back(SourceRange(DefaultInfo.CaseStartLoc, EndLoc),
                              DefaultInfo.getCaseReplacementString(
                                  /*IsDefault=*/true, AreBracesNeeded));
  }

  // Add the trailing break and one or two '}' if needed.
  const Stmt *LastBody =
      CurrentIf->getElse() ? CurrentIf->getElse() : CurrentIf->getThen();
  bool IsLastBreakNeeded = isBreakNeeded(LastBody);
  SourceLocation TerminatingReplacementLoc;
  std::string TerminatingReplacement;
  llvm::raw_string_ostream OS(TerminatingReplacement);
  if (!isa<CompoundStmt>(LastBody)) {
    TerminatingReplacementLoc = LastBody->getEndLoc();
    // Try to adjust the location in order to preserve any trailing comments on
    // the last line of the last body.
    if (!TerminatingReplacementLoc.isMacroID())
      TerminatingReplacementLoc = getLastLineLocationUnlessItHasOtherTokens(
          TerminatingReplacementLoc, SM, LangOpts);
    if (IsLastBreakNeeded)
      OS << "\nbreak;";
    OS << "\n}";
    if (AreBracesNeeded)
      OS << "\n}";
  } else {
    TerminatingReplacementLoc = LastBody->getEndLoc();
    if (IsLastBreakNeeded)
      OS << "break;\n";
    if (AreBracesNeeded)
      OS << "}\n";
  }

  if (!OS.str().empty()) {
    TerminatingReplacementLoc = SM.getSpellingLoc(TerminatingReplacementLoc);
    Replacements.emplace_back(
        SourceRange(TerminatingReplacementLoc, TerminatingReplacementLoc),
        std::move(OS.str()));
  }

  return std::move(Replacements);
}
