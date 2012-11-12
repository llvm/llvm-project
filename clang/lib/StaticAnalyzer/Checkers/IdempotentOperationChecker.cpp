//==- IdempotentOperationChecker.cpp - Idempotent Operations ----*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a set of path-sensitive checks for idempotent and/or
// tautological operations. Each potential operation is checked along all paths
// to see if every path results in a pointless operation.
//                 +-------------------------------------------+
//                 |Table of idempotent/tautological operations|
//                 +-------------------------------------------+
//+--------------------------------------------------------------------------+
//|Operator | x op x | x op 1 | 1 op x | x op 0 | 0 op x | x op ~0 | ~0 op x |
//+--------------------------------------------------------------------------+
//  +, +=   |        |        |        |   x    |   x    |         |
//  -, -=   |        |        |        |   x    |   -x   |         |
//  *, *=   |        |   x    |   x    |   0    |   0    |         |
//  /, /=   |   1    |   x    |        |  N/A   |   0    |         |
//  &, &=   |   x    |        |        |   0    |   0    |   x     |    x
//  |, |=   |   x    |        |        |   x    |   x    |   ~0    |    ~0
//  ^, ^=   |   0    |        |        |   x    |   x    |         |
//  <<, <<= |        |        |        |   x    |   0    |         |
//  >>, >>= |        |        |        |   x    |   0    |         |
//  ||      |   1    |   1    |   1    |   x    |   x    |   1     |    1
//  &&      |   1    |   x    |   x    |   0    |   0    |   x     |    x
//  =       |   x    |        |        |        |        |         |
//  ==      |   1    |        |        |        |        |         |
//  >=      |   1    |        |        |        |        |         |
//  <=      |   1    |        |        |        |        |         |
//  >       |   0    |        |        |        |        |         |
//  <       |   0    |        |        |        |        |         |
//  !=      |   0    |        |        |        |        |         |
//===----------------------------------------------------------------------===//
//
// Things TODO:
// - Improved error messages
// - Handle mixed assumptions (which assumptions can belong together?)
// - Finer grained false positive control (levels)
// - Handling ~0 values

#include "ClangSACheckers.h"
#include "clang/Analysis/CFGStmtMap.h"
#include "clang/Analysis/Analyses/PseudoConstantAnalysis.h"
#include "clang/Analysis/Analyses/CFGReachabilityAnalysis.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerHelpers.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CoreEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "clang/AST/Stmt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace ento;

namespace {
class IdempotentOperationChecker
  : public Checker<check::PreStmt<BinaryOperator>,
                     check::PostStmt<BinaryOperator>,
                     check::EndAnalysis> {
public:
  void checkPreStmt(const BinaryOperator *B, CheckerContext &C) const;
  void checkPostStmt(const BinaryOperator *B, CheckerContext &C) const;
  void checkEndAnalysis(ExplodedGraph &G, BugReporter &B,ExprEngine &Eng) const;

private:
  // Our assumption about a particular operation.
  enum Assumption { Possible = 0, Impossible, Equal, LHSis1, RHSis1, LHSis0,
      RHSis0 };

  static void UpdateAssumption(Assumption &A, const Assumption &New);

  // False positive reduction methods
  static bool isSelfAssign(const Expr *LHS, const Expr *RHS);
  static bool isUnused(const Expr *E, AnalysisDeclContext *AC);
  static bool isTruncationExtensionAssignment(const Expr *LHS,
                                              const Expr *RHS);
  static bool pathWasCompletelyAnalyzed(AnalysisDeclContext *AC,
                                        const CFGBlock *CB,
                                        const CoreEngine &CE);
  static bool CanVary(const Expr *Ex,
                      AnalysisDeclContext *AC);
  static bool isConstantOrPseudoConstant(const DeclRefExpr *DR,
                                         AnalysisDeclContext *AC);
  static bool containsNonLocalVarDecl(const Stmt *S);

  // Hash table and related data structures
  struct BinaryOperatorData {
    BinaryOperatorData() : assumption(Possible) {}

    Assumption assumption;
    ExplodedNodeSet explodedNodes; // Set of ExplodedNodes that refer to a
                                   // BinaryOperator
  };
  typedef llvm::DenseMap<const BinaryOperator *, BinaryOperatorData>
      AssumptionMap;
  mutable AssumptionMap hash;
  mutable OwningPtr<BugType> BT;
};
}

void IdempotentOperationChecker::checkPreStmt(const BinaryOperator *B,
                                              CheckerContext &C) const {
  // Find or create an entry in the hash for this BinaryOperator instance.
  // If we haven't done a lookup before, it will get default initialized to
  // 'Possible'. At this stage we do not store the ExplodedNode, as it has not
  // been created yet.
  BinaryOperatorData &Data = hash[B];
  Assumption &A = Data.assumption;
  AnalysisDeclContext *AC = C.getCurrentAnalysisDeclContext();

  // If we already have visited this node on a path that does not contain an
  // idempotent operation, return immediately.
  if (A == Impossible)
    return;

  // Retrieve both sides of the operator and determine if they can vary (which
  // may mean this is a false positive.
  const Expr *LHS = B->getLHS();
  const Expr *RHS = B->getRHS();

  // At this stage we can calculate whether each side contains a false positive
  // that applies to all operators. We only need to calculate this the first
  // time.
  bool LHSContainsFalsePositive = false, RHSContainsFalsePositive = false;
  if (A == Possible) {
    // An expression contains a false positive if it can't vary, or if it
    // contains a known false positive VarDecl.
    LHSContainsFalsePositive = !CanVary(LHS, AC)
        || containsNonLocalVarDecl(LHS);
    RHSContainsFalsePositive = !CanVary(RHS, AC)
        || containsNonLocalVarDecl(RHS);
  }

  ProgramStateRef state = C.getState();
  const LocationContext *LCtx = C.getLocationContext();
  SVal LHSVal = state->getSVal(LHS, LCtx);
  SVal RHSVal = state->getSVal(RHS, LCtx);

  // If either value is unknown, we can't be 100% sure of all paths.
  if (LHSVal.isUnknownOrUndef() || RHSVal.isUnknownOrUndef()) {
    A = Impossible;
    return;
  }
  BinaryOperator::Opcode Op = B->getOpcode();

  // Dereference the LHS SVal if this is an assign operation
  switch (Op) {
  default:
    break;

  // Fall through intentional
  case BO_AddAssign:
  case BO_SubAssign:
  case BO_MulAssign:
  case BO_DivAssign:
  case BO_AndAssign:
  case BO_OrAssign:
  case BO_XorAssign:
  case BO_ShlAssign:
  case BO_ShrAssign:
  case BO_Assign:
  // Assign statements have one extra level of indirection
    if (!isa<Loc>(LHSVal)) {
      A = Impossible;
      return;
    }
    LHSVal = state->getSVal(cast<Loc>(LHSVal), LHS->getType());
  }


  // We now check for various cases which result in an idempotent operation.

  // x op x
  switch (Op) {
  default:
    break; // We don't care about any other operators.

  // Fall through intentional
  case BO_Assign:
    // x Assign x can be used to silence unused variable warnings intentionally.
    // If this is a self assignment and the variable is referenced elsewhere,
    // and the assignment is not a truncation or extension, then it is a false
    // positive.
    if (isSelfAssign(LHS, RHS)) {
      if (!isUnused(LHS, AC) && !isTruncationExtensionAssignment(LHS, RHS)) {
        UpdateAssumption(A, Equal);
        return;
      }
      else {
        A = Impossible;
        return;
      }
    }

  case BO_SubAssign:
  case BO_DivAssign:
  case BO_AndAssign:
  case BO_OrAssign:
  case BO_XorAssign:
  case BO_Sub:
  case BO_Div:
  case BO_And:
  case BO_Or:
  case BO_Xor:
  case BO_LOr:
  case BO_LAnd:
  case BO_EQ:
  case BO_NE:
    if (LHSVal != RHSVal || LHSContainsFalsePositive
        || RHSContainsFalsePositive)
      break;
    UpdateAssumption(A, Equal);
    return;
  }

  // x op 1
  switch (Op) {
   default:
     break; // We don't care about any other operators.

   // Fall through intentional
   case BO_MulAssign:
   case BO_DivAssign:
   case BO_Mul:
   case BO_Div:
   case BO_LOr:
   case BO_LAnd:
     if (!RHSVal.isConstant(1) || RHSContainsFalsePositive)
       break;
     UpdateAssumption(A, RHSis1);
     return;
  }

  // 1 op x
  switch (Op) {
  default:
    break; // We don't care about any other operators.

  // Fall through intentional
  case BO_MulAssign:
  case BO_Mul:
  case BO_LOr:
  case BO_LAnd:
    if (!LHSVal.isConstant(1) || LHSContainsFalsePositive)
      break;
    UpdateAssumption(A, LHSis1);
    return;
  }

  // x op 0
  switch (Op) {
  default:
    break; // We don't care about any other operators.

  // Fall through intentional
  case BO_AddAssign:
  case BO_SubAssign:
  case BO_MulAssign:
  case BO_AndAssign:
  case BO_OrAssign:
  case BO_XorAssign:
  case BO_Add:
  case BO_Sub:
  case BO_Mul:
  case BO_And:
  case BO_Or:
  case BO_Xor:
  case BO_Shl:
  case BO_Shr:
  case BO_LOr:
  case BO_LAnd:
    if (!RHSVal.isConstant(0) || RHSContainsFalsePositive)
      break;
    UpdateAssumption(A, RHSis0);
    return;
  }

  // 0 op x
  switch (Op) {
  default:
    break; // We don't care about any other operators.

  // Fall through intentional
  //case BO_AddAssign: // Common false positive
  case BO_SubAssign: // Check only if unsigned
  case BO_MulAssign:
  case BO_DivAssign:
  case BO_AndAssign:
  //case BO_OrAssign: // Common false positive
  //case BO_XorAssign: // Common false positive
  case BO_ShlAssign:
  case BO_ShrAssign:
  case BO_Add:
  case BO_Sub:
  case BO_Mul:
  case BO_Div:
  case BO_And:
  case BO_Or:
  case BO_Xor:
  case BO_Shl:
  case BO_Shr:
  case BO_LOr:
  case BO_LAnd:
    if (!LHSVal.isConstant(0) || LHSContainsFalsePositive)
      break;
    UpdateAssumption(A, LHSis0);
    return;
  }

  // If we get to this point, there has been a valid use of this operation.
  A = Impossible;
}

// At the post visit stage, the predecessor ExplodedNode will be the
// BinaryOperator that was just created. We use this hook to collect the
// ExplodedNode.
void IdempotentOperationChecker::checkPostStmt(const BinaryOperator *B,
                                               CheckerContext &C) const {
  // Add the ExplodedNode we just visited
  BinaryOperatorData &Data = hash[B];

  const Stmt *predStmt 
    = cast<StmtPoint>(C.getPredecessor()->getLocation()).getStmt();
  
  // Ignore implicit calls to setters.
  if (!isa<BinaryOperator>(predStmt))
    return;

  Data.explodedNodes.Add(C.getPredecessor());
}

void IdempotentOperationChecker::checkEndAnalysis(ExplodedGraph &G,
                                                  BugReporter &BR,
                                                  ExprEngine &Eng) const {
  if (!BT)
    BT.reset(new BugType("Idempotent operation", "Dead code"));

  // Iterate over the hash to see if we have any paths with definite
  // idempotent operations.
  for (AssumptionMap::const_iterator i = hash.begin(); i != hash.end(); ++i) {
    // Unpack the hash contents
    const BinaryOperatorData &Data = i->second;
    const Assumption &A = Data.assumption;
    const ExplodedNodeSet &ES = Data.explodedNodes;

    // If there are no nodes accosted with the expression, nothing to report.
    // FIXME: This is possible because the checker does part of processing in
    // checkPreStmt and part in checkPostStmt.
    if (ES.begin() == ES.end())
      continue;

    const BinaryOperator *B = i->first;

    if (A == Impossible)
      continue;

    // If the analyzer did not finish, check to see if we can still emit this
    // warning
    if (Eng.hasWorkRemaining()) {
      // If we can trace back
      AnalysisDeclContext *AC = (*ES.begin())->getLocationContext()
                                         ->getAnalysisDeclContext();
      if (!pathWasCompletelyAnalyzed(AC,
                                     AC->getCFGStmtMap()->getBlock(B),
                                     Eng.getCoreEngine()))
        continue;
    }

    // Select the error message and SourceRanges to report.
    SmallString<128> buf;
    llvm::raw_svector_ostream os(buf);
    bool LHSRelevant = false, RHSRelevant = false;
    switch (A) {
    case Equal:
      LHSRelevant = true;
      RHSRelevant = true;
      if (B->getOpcode() == BO_Assign)
        os << "Assigned value is always the same as the existing value";
      else
        os << "Both operands to '" << B->getOpcodeStr()
           << "' always have the same value";
      break;
    case LHSis1:
      LHSRelevant = true;
      os << "The left operand to '" << B->getOpcodeStr() << "' is always 1";
      break;
    case RHSis1:
      RHSRelevant = true;
      os << "The right operand to '" << B->getOpcodeStr() << "' is always 1";
      break;
    case LHSis0:
      LHSRelevant = true;
      os << "The left operand to '" << B->getOpcodeStr() << "' is always 0";
      break;
    case RHSis0:
      RHSRelevant = true;
      os << "The right operand to '" << B->getOpcodeStr() << "' is always 0";
      break;
    case Possible:
      llvm_unreachable("Operation was never marked with an assumption");
    case Impossible:
      llvm_unreachable(0);
    }

    // Add a report for each ExplodedNode
    for (ExplodedNodeSet::iterator I = ES.begin(), E = ES.end(); I != E; ++I) {
      BugReport *report = new BugReport(*BT, os.str(), *I);

      // Add source ranges and visitor hooks
      if (LHSRelevant) {
        const Expr *LHS = i->first->getLHS();
        report->addRange(LHS->getSourceRange());
        FindLastStoreBRVisitor::registerStatementVarDecls(*report, LHS);
      }
      if (RHSRelevant) {
        const Expr *RHS = i->first->getRHS();
        report->addRange(i->first->getRHS()->getSourceRange());
        FindLastStoreBRVisitor::registerStatementVarDecls(*report, RHS);
      }

      BR.emitReport(report);
    }
  }

  hash.clear();
}

// Updates the current assumption given the new assumption
inline void IdempotentOperationChecker::UpdateAssumption(Assumption &A,
                                                        const Assumption &New) {
// If the assumption is the same, there is nothing to do
  if (A == New)
    return;

  switch (A) {
  // If we don't currently have an assumption, set it
  case Possible:
    A = New;
    return;

  // If we have determined that a valid state happened, ignore the new
  // assumption.
  case Impossible:
    return;

  // Any other case means that we had a different assumption last time. We don't
  // currently support mixing assumptions for diagnostic reasons, so we set
  // our assumption to be impossible.
  default:
    A = Impossible;
    return;
  }
}

// Check for a statement where a variable is self assigned to possibly avoid an
// unused variable warning.
bool IdempotentOperationChecker::isSelfAssign(const Expr *LHS, const Expr *RHS) {
  LHS = LHS->IgnoreParenCasts();
  RHS = RHS->IgnoreParenCasts();

  const DeclRefExpr *LHS_DR = dyn_cast<DeclRefExpr>(LHS);
  if (!LHS_DR)
    return false;

  const VarDecl *VD = dyn_cast<VarDecl>(LHS_DR->getDecl());
  if (!VD)
    return false;

  const DeclRefExpr *RHS_DR = dyn_cast<DeclRefExpr>(RHS);
  if (!RHS_DR)
    return false;

  if (VD != RHS_DR->getDecl())
    return false;

  return true;
}

// Returns true if the Expr points to a VarDecl that is not read anywhere
// outside of self-assignments.
bool IdempotentOperationChecker::isUnused(const Expr *E,
                                          AnalysisDeclContext *AC) {
  if (!E)
    return false;

  const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(E->IgnoreParenCasts());
  if (!DR)
    return false;

  const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl());
  if (!VD)
    return false;

  if (AC->getPseudoConstantAnalysis()->wasReferenced(VD))
    return false;

  return true;
}

// Check for self casts truncating/extending a variable
bool IdempotentOperationChecker::isTruncationExtensionAssignment(
                                                              const Expr *LHS,
                                                              const Expr *RHS) {

  const DeclRefExpr *LHS_DR = dyn_cast<DeclRefExpr>(LHS->IgnoreParenCasts());
  if (!LHS_DR)
    return false;

  const VarDecl *VD = dyn_cast<VarDecl>(LHS_DR->getDecl());
  if (!VD)
    return false;

  const DeclRefExpr *RHS_DR = dyn_cast<DeclRefExpr>(RHS->IgnoreParenCasts());
  if (!RHS_DR)
    return false;

  if (VD != RHS_DR->getDecl())
     return false;

  return dyn_cast<DeclRefExpr>(RHS->IgnoreParenLValueCasts()) == NULL;
}

// Returns false if a path to this block was not completely analyzed, or true
// otherwise.
bool
IdempotentOperationChecker::pathWasCompletelyAnalyzed(AnalysisDeclContext *AC,
                                                      const CFGBlock *CB,
                                                      const CoreEngine &CE) {

  CFGReverseBlockReachabilityAnalysis *CRA = AC->getCFGReachablityAnalysis();
  
  // Test for reachability from any aborted blocks to this block
  typedef CoreEngine::BlocksExhausted::const_iterator ExhaustedIterator;
  for (ExhaustedIterator I = CE.blocks_exhausted_begin(),
      E = CE.blocks_exhausted_end(); I != E; ++I) {
    const BlockEdge &BE =  I->first;

    // The destination block on the BlockEdge is the first block that was not
    // analyzed. If we can reach this block from the aborted block, then this
    // block was not completely analyzed.
    //
    // Also explicitly check if the current block is the destination block.
    // While technically reachable, it means we aborted the analysis on
    // a path that included that block.
    const CFGBlock *destBlock = BE.getDst();
    if (destBlock == CB || CRA->isReachable(destBlock, CB))
      return false;
  }

  // Test for reachability from blocks we just gave up on.
  typedef CoreEngine::BlocksAborted::const_iterator AbortedIterator;
  for (AbortedIterator I = CE.blocks_aborted_begin(),
       E = CE.blocks_aborted_end(); I != E; ++I) {
    const CFGBlock *destBlock = I->first;
    if (destBlock == CB || CRA->isReachable(destBlock, CB))
      return false;
  }
  
  // For the items still on the worklist, see if they are in blocks that
  // can eventually reach 'CB'.
  class VisitWL : public WorkList::Visitor {
    const CFGStmtMap *CBM;
    const CFGBlock *TargetBlock;
    CFGReverseBlockReachabilityAnalysis &CRA;
  public:
    VisitWL(const CFGStmtMap *cbm, const CFGBlock *targetBlock,
            CFGReverseBlockReachabilityAnalysis &cra)
      : CBM(cbm), TargetBlock(targetBlock), CRA(cra) {}
    virtual bool visit(const WorkListUnit &U) {
      ProgramPoint P = U.getNode()->getLocation();
      const CFGBlock *B = 0;
      if (StmtPoint *SP = dyn_cast<StmtPoint>(&P)) {
        B = CBM->getBlock(SP->getStmt());
      }
      else if (BlockEdge *BE = dyn_cast<BlockEdge>(&P)) {
        B = BE->getDst();
      }
      else if (BlockEntrance *BEnt = dyn_cast<BlockEntrance>(&P)) {
        B = BEnt->getBlock();
      }
      else if (BlockExit *BExit = dyn_cast<BlockExit>(&P)) {
        B = BExit->getBlock();
      }
      if (!B)
        return true;
      
      return B == TargetBlock || CRA.isReachable(B, TargetBlock);
    }
  };
  VisitWL visitWL(AC->getCFGStmtMap(), CB, *CRA);
  // Were there any items in the worklist that could potentially reach
  // this block?
  if (CE.getWorkList()->visitItemsInWorkList(visitWL))
    return false;

  // Verify that this block is reachable from the entry block
  if (!CRA->isReachable(&AC->getCFG()->getEntry(), CB))
    return false;

  // If we get to this point, there is no connection to the entry block or an
  // aborted block. This path is unreachable and we can report the error.
  return true;
}

// Recursive function that determines whether an expression contains any element
// that varies. This could be due to a compile-time constant like sizeof. An
// expression may also involve a variable that behaves like a constant. The
// function returns true if the expression varies, and false otherwise.
bool IdempotentOperationChecker::CanVary(const Expr *Ex,
                                         AnalysisDeclContext *AC) {
  // Parentheses and casts are irrelevant here
  Ex = Ex->IgnoreParenCasts();

  if (Ex->getLocStart().isMacroID())
    return false;

  switch (Ex->getStmtClass()) {
  // Trivially true cases
  case Stmt::ArraySubscriptExprClass:
  case Stmt::MemberExprClass:
  case Stmt::StmtExprClass:
  case Stmt::CallExprClass:
  case Stmt::VAArgExprClass:
  case Stmt::ShuffleVectorExprClass:
    return true;
  default:
    return true;

  // Trivially false cases
  case Stmt::IntegerLiteralClass:
  case Stmt::CharacterLiteralClass:
  case Stmt::FloatingLiteralClass:
  case Stmt::PredefinedExprClass:
  case Stmt::ImaginaryLiteralClass:
  case Stmt::StringLiteralClass:
  case Stmt::OffsetOfExprClass:
  case Stmt::CompoundLiteralExprClass:
  case Stmt::AddrLabelExprClass:
  case Stmt::BinaryTypeTraitExprClass:
  case Stmt::GNUNullExprClass:
  case Stmt::InitListExprClass:
  case Stmt::DesignatedInitExprClass:
  case Stmt::BlockExprClass:
    return false;

  // Cases requiring custom logic
  case Stmt::UnaryExprOrTypeTraitExprClass: {
    const UnaryExprOrTypeTraitExpr *SE = 
                       cast<const UnaryExprOrTypeTraitExpr>(Ex);
    if (SE->getKind() != UETT_SizeOf)
      return false;
    return SE->getTypeOfArgument()->isVariableArrayType();
  }
  case Stmt::DeclRefExprClass:
    // Check for constants/pseudoconstants
    return !isConstantOrPseudoConstant(cast<DeclRefExpr>(Ex), AC);

  // The next cases require recursion for subexpressions
  case Stmt::BinaryOperatorClass: {
    const BinaryOperator *B = cast<const BinaryOperator>(Ex);

    // Exclude cases involving pointer arithmetic.  These are usually
    // false positives.
    if (B->getOpcode() == BO_Sub || B->getOpcode() == BO_Add)
      if (B->getLHS()->getType()->getAs<PointerType>())
        return false;

    return CanVary(B->getRHS(), AC)
        || CanVary(B->getLHS(), AC);
   }
  case Stmt::UnaryOperatorClass: {
    const UnaryOperator *U = cast<const UnaryOperator>(Ex);
    // Handle trivial case first
    switch (U->getOpcode()) {
    case UO_Extension:
      return false;
    default:
      return CanVary(U->getSubExpr(), AC);
    }
  }
  case Stmt::ChooseExprClass:
    return CanVary(cast<const ChooseExpr>(Ex)->getChosenSubExpr(
        AC->getASTContext()), AC);
  case Stmt::ConditionalOperatorClass:
  case Stmt::BinaryConditionalOperatorClass:
    return CanVary(cast<AbstractConditionalOperator>(Ex)->getCond(), AC);
  }
}

// Returns true if a DeclRefExpr is or behaves like a constant.
bool IdempotentOperationChecker::isConstantOrPseudoConstant(
                                                          const DeclRefExpr *DR,
                                                          AnalysisDeclContext *AC) {
  // Check if the type of the Decl is const-qualified
  if (DR->getType().isConstQualified())
    return true;

  // Check for an enum
  if (isa<EnumConstantDecl>(DR->getDecl()))
    return true;

  const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl());
  if (!VD)
    return true;

  // Check if the Decl behaves like a constant. This check also takes care of
  // static variables, which can only change between function calls if they are
  // modified in the AST.
  PseudoConstantAnalysis *PCA = AC->getPseudoConstantAnalysis();
  if (PCA->isPseudoConstant(VD))
    return true;

  return false;
}

// Recursively find any substatements containing VarDecl's with storage other
// than local
bool IdempotentOperationChecker::containsNonLocalVarDecl(const Stmt *S) {
  const DeclRefExpr *DR = dyn_cast<DeclRefExpr>(S);

  if (DR)
    if (const VarDecl *VD = dyn_cast<VarDecl>(DR->getDecl()))
      if (!VD->hasLocalStorage())
        return true;

  for (Stmt::const_child_iterator I = S->child_begin(); I != S->child_end();
      ++I)
    if (const Stmt *child = *I)
      if (containsNonLocalVarDecl(child))
        return true;

  return false;
}


void ento::registerIdempotentOperationChecker(CheckerManager &mgr) {
  mgr.registerChecker<IdempotentOperationChecker>();
}
