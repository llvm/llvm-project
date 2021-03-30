//===- BugReporterVisitors.cpp - Helpers for reporting bugs ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a set of BugReporter "visitors" which can be used to
//  enhance the diagnostics reported for a bug.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/BugReporter/BugReporterVisitors.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Analysis/Analyses/Dominators.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/CFGStmtMap.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Analysis/ProgramPoint.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/StaticAnalyzer/Core/AnalyzerOptions.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExplodedGraph.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState_Fwd.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SMTConv.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SValBuilder.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <deque>
#include <memory>
#include <string>
#include <utility>

using namespace clang;
using namespace ento;

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

static const Expr *peelOffPointerArithmetic(const BinaryOperator *B) {
  if (B->isAdditiveOp() && B->getType()->isPointerType()) {
    if (B->getLHS()->getType()->isPointerType()) {
      return B->getLHS();
    } else if (B->getRHS()->getType()->isPointerType()) {
      return B->getRHS();
    }
  }
  return nullptr;
}

/// Given that expression S represents a pointer that would be dereferenced,
/// try to find a sub-expression from which the pointer came from.
/// This is used for tracking down origins of a null or undefined value:
/// "this is null because that is null because that is null" etc.
/// We wipe away field and element offsets because they merely add offsets.
/// We also wipe away all casts except lvalue-to-rvalue casts, because the
/// latter represent an actual pointer dereference; however, we remove
/// the final lvalue-to-rvalue cast before returning from this function
/// because it demonstrates more clearly from where the pointer rvalue was
/// loaded. Examples:
///   x->y.z      ==>  x (lvalue)
///   foo()->y.z  ==>  foo() (rvalue)
const Expr *bugreporter::getDerefExpr(const Stmt *S) {
  const auto *E = dyn_cast<Expr>(S);
  if (!E)
    return nullptr;

  while (true) {
    if (const auto *CE = dyn_cast<CastExpr>(E)) {
      if (CE->getCastKind() == CK_LValueToRValue) {
        // This cast represents the load we're looking for.
        break;
      }
      E = CE->getSubExpr();
    } else if (const auto *B = dyn_cast<BinaryOperator>(E)) {
      // Pointer arithmetic: '*(x + 2)' -> 'x') etc.
      if (const Expr *Inner = peelOffPointerArithmetic(B)) {
        E = Inner;
      } else {
        // Probably more arithmetic can be pattern-matched here,
        // but for now give up.
        break;
      }
    } else if (const auto *U = dyn_cast<UnaryOperator>(E)) {
      if (U->getOpcode() == UO_Deref || U->getOpcode() == UO_AddrOf ||
          (U->isIncrementDecrementOp() && U->getType()->isPointerType())) {
        // Operators '*' and '&' don't actually mean anything.
        // We look at casts instead.
        E = U->getSubExpr();
      } else {
        // Probably more arithmetic can be pattern-matched here,
        // but for now give up.
        break;
      }
    }
    // Pattern match for a few useful cases: a[0], p->f, *p etc.
    else if (const auto *ME = dyn_cast<MemberExpr>(E)) {
      E = ME->getBase();
    } else if (const auto *IvarRef = dyn_cast<ObjCIvarRefExpr>(E)) {
      E = IvarRef->getBase();
    } else if (const auto *AE = dyn_cast<ArraySubscriptExpr>(E)) {
      E = AE->getBase();
    } else if (const auto *PE = dyn_cast<ParenExpr>(E)) {
      E = PE->getSubExpr();
    } else if (const auto *FE = dyn_cast<FullExpr>(E)) {
      E = FE->getSubExpr();
    } else {
      // Other arbitrary stuff.
      break;
    }
  }

  // Special case: remove the final lvalue-to-rvalue cast, but do not recurse
  // deeper into the sub-expression. This way we return the lvalue from which
  // our pointer rvalue was loaded.
  if (const auto *CE = dyn_cast<ImplicitCastExpr>(E))
    if (CE->getCastKind() == CK_LValueToRValue)
      E = CE->getSubExpr();

  return E;
}

/// Comparing internal representations of symbolic values (via
/// SVal::operator==()) is a valid way to check if the value was updated,
/// unless it's a LazyCompoundVal that may have a different internal
/// representation every time it is loaded from the state. In this function we
/// do an approximate comparison for lazy compound values, checking that they
/// are the immediate snapshots of the tracked region's bindings within the
/// node's respective states but not really checking that these snapshots
/// actually contain the same set of bindings.
static bool hasVisibleUpdate(const ExplodedNode *LeftNode, SVal LeftVal,
                             const ExplodedNode *RightNode, SVal RightVal) {
  if (LeftVal == RightVal)
    return true;

  const auto LLCV = LeftVal.getAs<nonloc::LazyCompoundVal>();
  if (!LLCV)
    return false;

  const auto RLCV = RightVal.getAs<nonloc::LazyCompoundVal>();
  if (!RLCV)
    return false;

  return LLCV->getRegion() == RLCV->getRegion() &&
    LLCV->getStore() == LeftNode->getState()->getStore() &&
    RLCV->getStore() == RightNode->getState()->getStore();
}

static Optional<SVal> getSValForVar(const Expr *CondVarExpr,
                                    const ExplodedNode *N) {
  ProgramStateRef State = N->getState();
  const LocationContext *LCtx = N->getLocationContext();

  assert(CondVarExpr);
  CondVarExpr = CondVarExpr->IgnoreImpCasts();

  // The declaration of the value may rely on a pointer so take its l-value.
  // FIXME: As seen in VisitCommonDeclRefExpr, sometimes DeclRefExpr may
  // evaluate to a FieldRegion when it refers to a declaration of a lambda
  // capture variable. We most likely need to duplicate that logic here.
  if (const auto *DRE = dyn_cast<DeclRefExpr>(CondVarExpr))
    if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()))
      return State->getSVal(State->getLValue(VD, LCtx));

  if (const auto *ME = dyn_cast<MemberExpr>(CondVarExpr))
    if (const auto *FD = dyn_cast<FieldDecl>(ME->getMemberDecl()))
      if (auto FieldL = State->getSVal(ME, LCtx).getAs<Loc>())
        return State->getRawSVal(*FieldL, FD->getType());

  return None;
}

static Optional<const llvm::APSInt *>
getConcreteIntegerValue(const Expr *CondVarExpr, const ExplodedNode *N) {

  if (Optional<SVal> V = getSValForVar(CondVarExpr, N))
    if (auto CI = V->getAs<nonloc::ConcreteInt>())
      return &CI->getValue();
  return None;
}

static bool isVarAnInterestingCondition(const Expr *CondVarExpr,
                                        const ExplodedNode *N,
                                        const PathSensitiveBugReport *B) {
  // Even if this condition is marked as interesting, it isn't *that*
  // interesting if it didn't happen in a nested stackframe, the user could just
  // follow the arrows.
  if (!B->getErrorNode()->getStackFrame()->isParentOf(N->getStackFrame()))
    return false;

  if (Optional<SVal> V = getSValForVar(CondVarExpr, N))
    if (Optional<bugreporter::TrackingKind> K = B->getInterestingnessKind(*V))
      return *K == bugreporter::TrackingKind::Condition;

  return false;
}

static bool isInterestingExpr(const Expr *E, const ExplodedNode *N,
                              const PathSensitiveBugReport *B) {
  if (Optional<SVal> V = getSValForVar(E, N))
    return B->getInterestingnessKind(*V).hasValue();
  return false;
}

/// \return name of the macro inside the location \p Loc.
static StringRef getMacroName(SourceLocation Loc,
    BugReporterContext &BRC) {
  return Lexer::getImmediateMacroName(
      Loc,
      BRC.getSourceManager(),
      BRC.getASTContext().getLangOpts());
}

/// \return Whether given spelling location corresponds to an expansion
/// of a function-like macro.
static bool isFunctionMacroExpansion(SourceLocation Loc,
                                const SourceManager &SM) {
  if (!Loc.isMacroID())
    return false;
  while (SM.isMacroArgExpansion(Loc))
    Loc = SM.getImmediateExpansionRange(Loc).getBegin();
  std::pair<FileID, unsigned> TLInfo = SM.getDecomposedLoc(Loc);
  SrcMgr::SLocEntry SE = SM.getSLocEntry(TLInfo.first);
  const SrcMgr::ExpansionInfo &EInfo = SE.getExpansion();
  return EInfo.isFunctionMacroExpansion();
}

/// \return Whether \c RegionOfInterest was modified at \p N,
/// where \p ValueAfter is \c RegionOfInterest's value at the end of the
/// stack frame.
static bool wasRegionOfInterestModifiedAt(const SubRegion *RegionOfInterest,
                                          const ExplodedNode *N,
                                          SVal ValueAfter) {
  ProgramStateRef State = N->getState();
  ProgramStateManager &Mgr = N->getState()->getStateManager();

  if (!N->getLocationAs<PostStore>() && !N->getLocationAs<PostInitializer>() &&
      !N->getLocationAs<PostStmt>())
    return false;

  // Writing into region of interest.
  if (auto PS = N->getLocationAs<PostStmt>())
    if (auto *BO = PS->getStmtAs<BinaryOperator>())
      if (BO->isAssignmentOp() && RegionOfInterest->isSubRegionOf(
                                      N->getSVal(BO->getLHS()).getAsRegion()))
        return true;

  // SVal after the state is possibly different.
  SVal ValueAtN = N->getState()->getSVal(RegionOfInterest);
  if (!Mgr.getSValBuilder()
           .areEqual(State, ValueAtN, ValueAfter)
           .isConstrainedTrue() &&
      (!ValueAtN.isUndef() || !ValueAfter.isUndef()))
    return true;

  return false;
}

//===----------------------------------------------------------------------===//
// Implementation of BugReporterVisitor.
//===----------------------------------------------------------------------===//

PathDiagnosticPieceRef BugReporterVisitor::getEndPath(BugReporterContext &,
                                                      const ExplodedNode *,
                                                      PathSensitiveBugReport &) {
  return nullptr;
}

void BugReporterVisitor::finalizeVisitor(BugReporterContext &,
                                         const ExplodedNode *,
                                         PathSensitiveBugReport &) {}

PathDiagnosticPieceRef
BugReporterVisitor::getDefaultEndPath(const BugReporterContext &BRC,
                                      const ExplodedNode *EndPathNode,
                                      const PathSensitiveBugReport &BR) {
  PathDiagnosticLocation L = BR.getLocation();
  const auto &Ranges = BR.getRanges();

  // Only add the statement itself as a range if we didn't specify any
  // special ranges for this report.
  auto P = std::make_shared<PathDiagnosticEventPiece>(
      L, BR.getDescription(), Ranges.begin() == Ranges.end());
  for (SourceRange Range : Ranges)
    P->addRange(Range);

  return P;
}

//===----------------------------------------------------------------------===//
// Implementation of NoStoreFuncVisitor.
//===----------------------------------------------------------------------===//

namespace {

/// Put a diagnostic on return statement of all inlined functions
/// for which  the region of interest \p RegionOfInterest was passed into,
/// but not written inside, and it has caused an undefined read or a null
/// pointer dereference outside.
class NoStoreFuncVisitor final : public BugReporterVisitor {
  const SubRegion *RegionOfInterest;
  MemRegionManager &MmrMgr;
  const SourceManager &SM;
  const PrintingPolicy &PP;
  bugreporter::TrackingKind TKind;

  /// Recursion limit for dereferencing fields when looking for the
  /// region of interest.
  /// The limit of two indicates that we will dereference fields only once.
  static const unsigned DEREFERENCE_LIMIT = 2;

  /// Frames writing into \c RegionOfInterest.
  /// This visitor generates a note only if a function does not write into
  /// a region of interest. This information is not immediately available
  /// by looking at the node associated with the exit from the function
  /// (usually the return statement). To avoid recomputing the same information
  /// many times (going up the path for each node and checking whether the
  /// region was written into) we instead lazily compute the
  /// stack frames along the path which write into the region of interest.
  llvm::SmallPtrSet<const StackFrameContext *, 32> FramesModifyingRegion;
  llvm::SmallPtrSet<const StackFrameContext *, 32> FramesModifyingCalculated;

  using RegionVector = SmallVector<const MemRegion *, 5>;

public:
  NoStoreFuncVisitor(const SubRegion *R, bugreporter::TrackingKind TKind)
      : RegionOfInterest(R), MmrMgr(R->getMemRegionManager()),
        SM(MmrMgr.getContext().getSourceManager()),
        PP(MmrMgr.getContext().getPrintingPolicy()), TKind(TKind) {}

  void Profile(llvm::FoldingSetNodeID &ID) const override {
    static int Tag = 0;
    ID.AddPointer(&Tag);
    ID.AddPointer(RegionOfInterest);
  }

  void *getTag() const {
    static int Tag = 0;
    return static_cast<void *>(&Tag);
  }

  PathDiagnosticPieceRef VisitNode(const ExplodedNode *N,
                                   BugReporterContext &BR,
                                   PathSensitiveBugReport &R) override;

private:
  /// Attempts to find the region of interest in a given record decl,
  /// by either following the base classes or fields.
  /// Dereferences fields up to a given recursion limit.
  /// Note that \p Vec is passed by value, leading to quadratic copying cost,
  /// but it's OK in practice since its length is limited to DEREFERENCE_LIMIT.
  /// \return A chain fields leading to the region of interest or None.
  const Optional<RegionVector>
  findRegionOfInterestInRecord(const RecordDecl *RD, ProgramStateRef State,
                               const MemRegion *R, const RegionVector &Vec = {},
                               int depth = 0);

  /// Check and lazily calculate whether the region of interest is
  /// modified in the stack frame to which \p N belongs.
  /// The calculation is cached in FramesModifyingRegion.
  bool isRegionOfInterestModifiedInFrame(const ExplodedNode *N) {
    const LocationContext *Ctx = N->getLocationContext();
    const StackFrameContext *SCtx = Ctx->getStackFrame();
    if (!FramesModifyingCalculated.count(SCtx))
      findModifyingFrames(N);
    return FramesModifyingRegion.count(SCtx);
  }

  /// Write to \c FramesModifyingRegion all stack frames along
  /// the path in the current stack frame which modify \c RegionOfInterest.
  void findModifyingFrames(const ExplodedNode *N);

  /// Consume the information on the no-store stack frame in order to
  /// either emit a note or suppress the report enirely.
  /// \return Diagnostics piece for region not modified in the current function,
  /// if it decides to emit one.
  PathDiagnosticPieceRef
  maybeEmitNote(PathSensitiveBugReport &R, const CallEvent &Call,
                const ExplodedNode *N, const RegionVector &FieldChain,
                const MemRegion *MatchedRegion, StringRef FirstElement,
                bool FirstIsReferenceType, unsigned IndirectionLevel);

  /// Pretty-print region \p MatchedRegion to \p os.
  /// \return Whether printing succeeded.
  bool prettyPrintRegionName(StringRef FirstElement, bool FirstIsReferenceType,
                             const MemRegion *MatchedRegion,
                             const RegionVector &FieldChain,
                             int IndirectionLevel,
                             llvm::raw_svector_ostream &os);

  /// Print first item in the chain, return new separator.
  static StringRef prettyPrintFirstElement(StringRef FirstElement,
                                           bool MoreItemsExpected,
                                           int IndirectionLevel,
                                           llvm::raw_svector_ostream &os);
};

} // end of anonymous namespace

/// \return Whether the method declaration \p Parent
/// syntactically has a binary operation writing into the ivar \p Ivar.
static bool potentiallyWritesIntoIvar(const Decl *Parent,
                                      const ObjCIvarDecl *Ivar) {
  using namespace ast_matchers;
  const char *IvarBind = "Ivar";
  if (!Parent || !Parent->hasBody())
    return false;
  StatementMatcher WriteIntoIvarM = binaryOperator(
      hasOperatorName("="),
      hasLHS(ignoringParenImpCasts(
          objcIvarRefExpr(hasDeclaration(equalsNode(Ivar))).bind(IvarBind))));
  StatementMatcher ParentM = stmt(hasDescendant(WriteIntoIvarM));
  auto Matches = match(ParentM, *Parent->getBody(), Parent->getASTContext());
  for (BoundNodes &Match : Matches) {
    auto IvarRef = Match.getNodeAs<ObjCIvarRefExpr>(IvarBind);
    if (IvarRef->isFreeIvar())
      return true;

    const Expr *Base = IvarRef->getBase();
    if (const auto *ICE = dyn_cast<ImplicitCastExpr>(Base))
      Base = ICE->getSubExpr();

    if (const auto *DRE = dyn_cast<DeclRefExpr>(Base))
      if (const auto *ID = dyn_cast<ImplicitParamDecl>(DRE->getDecl()))
        if (ID->getParameterKind() == ImplicitParamDecl::ObjCSelf)
          return true;

    return false;
  }
  return false;
}

/// Get parameters associated with runtime definition in order
/// to get the correct parameter name.
static ArrayRef<ParmVarDecl *> getCallParameters(CallEventRef<> Call) {
  // Use runtime definition, if available.
  RuntimeDefinition RD = Call->getRuntimeDefinition();
  if (const auto *FD = dyn_cast_or_null<FunctionDecl>(RD.getDecl()))
    return FD->parameters();
  if (const auto *MD = dyn_cast_or_null<ObjCMethodDecl>(RD.getDecl()))
    return MD->parameters();

  return Call->parameters();
}

/// \return whether \p Ty points to a const type, or is a const reference.
static bool isPointerToConst(QualType Ty) {
  return !Ty->getPointeeType().isNull() &&
         Ty->getPointeeType().getCanonicalType().isConstQualified();
}

/// Attempts to find the region of interest in a given CXX decl,
/// by either following the base classes or fields.
/// Dereferences fields up to a given recursion limit.
/// Note that \p Vec is passed by value, leading to quadratic copying cost,
/// but it's OK in practice since its length is limited to DEREFERENCE_LIMIT.
/// \return A chain fields leading to the region of interest or None.
const Optional<NoStoreFuncVisitor::RegionVector>
NoStoreFuncVisitor::findRegionOfInterestInRecord(
    const RecordDecl *RD, ProgramStateRef State, const MemRegion *R,
    const NoStoreFuncVisitor::RegionVector &Vec /* = {} */,
    int depth /* = 0 */) {

  if (depth == DEREFERENCE_LIMIT) // Limit the recursion depth.
    return None;

  if (const auto *RDX = dyn_cast<CXXRecordDecl>(RD))
    if (!RDX->hasDefinition())
      return None;

  // Recursively examine the base classes.
  // Note that following base classes does not increase the recursion depth.
  if (const auto *RDX = dyn_cast<CXXRecordDecl>(RD))
    for (const auto &II : RDX->bases())
      if (const RecordDecl *RRD = II.getType()->getAsRecordDecl())
        if (Optional<RegionVector> Out =
                findRegionOfInterestInRecord(RRD, State, R, Vec, depth))
          return Out;

  for (const FieldDecl *I : RD->fields()) {
    QualType FT = I->getType();
    const FieldRegion *FR = MmrMgr.getFieldRegion(I, cast<SubRegion>(R));
    const SVal V = State->getSVal(FR);
    const MemRegion *VR = V.getAsRegion();

    RegionVector VecF = Vec;
    VecF.push_back(FR);

    if (RegionOfInterest == VR)
      return VecF;

    if (const RecordDecl *RRD = FT->getAsRecordDecl())
      if (auto Out =
              findRegionOfInterestInRecord(RRD, State, FR, VecF, depth + 1))
        return Out;

    QualType PT = FT->getPointeeType();
    if (PT.isNull() || PT->isVoidType() || !VR)
      continue;

    if (const RecordDecl *RRD = PT->getAsRecordDecl())
      if (Optional<RegionVector> Out =
              findRegionOfInterestInRecord(RRD, State, VR, VecF, depth + 1))
        return Out;
  }

  return None;
}

PathDiagnosticPieceRef
NoStoreFuncVisitor::VisitNode(const ExplodedNode *N, BugReporterContext &BR,
                              PathSensitiveBugReport &R) {

  const LocationContext *Ctx = N->getLocationContext();
  const StackFrameContext *SCtx = Ctx->getStackFrame();
  ProgramStateRef State = N->getState();
  auto CallExitLoc = N->getLocationAs<CallExitBegin>();

  // No diagnostic if region was modified inside the frame.
  if (!CallExitLoc || isRegionOfInterestModifiedInFrame(N))
    return nullptr;

  CallEventRef<> Call =
      BR.getStateManager().getCallEventManager().getCaller(SCtx, State);

  // Region of interest corresponds to an IVar, exiting a method
  // which could have written into that IVar, but did not.
  if (const auto *MC = dyn_cast<ObjCMethodCall>(Call)) {
    if (const auto *IvarR = dyn_cast<ObjCIvarRegion>(RegionOfInterest)) {
      const MemRegion *SelfRegion = MC->getReceiverSVal().getAsRegion();
      if (RegionOfInterest->isSubRegionOf(SelfRegion) &&
          potentiallyWritesIntoIvar(Call->getRuntimeDefinition().getDecl(),
                                    IvarR->getDecl()))
        return maybeEmitNote(R, *Call, N, {}, SelfRegion, "self",
                             /*FirstIsReferenceType=*/false, 1);
    }
  }

  if (const auto *CCall = dyn_cast<CXXConstructorCall>(Call)) {
    const MemRegion *ThisR = CCall->getCXXThisVal().getAsRegion();
    if (RegionOfInterest->isSubRegionOf(ThisR) &&
        !CCall->getDecl()->isImplicit())
      return maybeEmitNote(R, *Call, N, {}, ThisR, "this",
                           /*FirstIsReferenceType=*/false, 1);

    // Do not generate diagnostics for not modified parameters in
    // constructors.
    return nullptr;
  }

  ArrayRef<ParmVarDecl *> parameters = getCallParameters(Call);
  for (unsigned I = 0; I < Call->getNumArgs() && I < parameters.size(); ++I) {
    const ParmVarDecl *PVD = parameters[I];
    SVal V = Call->getArgSVal(I);
    bool ParamIsReferenceType = PVD->getType()->isReferenceType();
    std::string ParamName = PVD->getNameAsString();

    int IndirectionLevel = 1;
    QualType T = PVD->getType();
    while (const MemRegion *MR = V.getAsRegion()) {
      if (RegionOfInterest->isSubRegionOf(MR) && !isPointerToConst(T))
        return maybeEmitNote(R, *Call, N, {}, MR, ParamName,
                             ParamIsReferenceType, IndirectionLevel);

      QualType PT = T->getPointeeType();
      if (PT.isNull() || PT->isVoidType())
        break;

      if (const RecordDecl *RD = PT->getAsRecordDecl())
        if (Optional<RegionVector> P =
                findRegionOfInterestInRecord(RD, State, MR))
          return maybeEmitNote(R, *Call, N, *P, RegionOfInterest, ParamName,
                               ParamIsReferenceType, IndirectionLevel);

      V = State->getSVal(MR, PT);
      T = PT;
      IndirectionLevel++;
    }
  }

  return nullptr;
}

void NoStoreFuncVisitor::findModifyingFrames(const ExplodedNode *N) {
  assert(N->getLocationAs<CallExitBegin>());
  ProgramStateRef LastReturnState = N->getState();
  SVal ValueAtReturn = LastReturnState->getSVal(RegionOfInterest);
  const LocationContext *Ctx = N->getLocationContext();
  const StackFrameContext *OriginalSCtx = Ctx->getStackFrame();

  do {
    ProgramStateRef State = N->getState();
    auto CallExitLoc = N->getLocationAs<CallExitBegin>();
    if (CallExitLoc) {
      LastReturnState = State;
      ValueAtReturn = LastReturnState->getSVal(RegionOfInterest);
    }

    FramesModifyingCalculated.insert(N->getLocationContext()->getStackFrame());

    if (wasRegionOfInterestModifiedAt(RegionOfInterest, N, ValueAtReturn)) {
      const StackFrameContext *SCtx = N->getStackFrame();
      while (!SCtx->inTopFrame()) {
        auto p = FramesModifyingRegion.insert(SCtx);
        if (!p.second)
          break; // Frame and all its parents already inserted.
        SCtx = SCtx->getParent()->getStackFrame();
      }
    }

    // Stop calculation at the call to the current function.
    if (auto CE = N->getLocationAs<CallEnter>())
      if (CE->getCalleeContext() == OriginalSCtx)
        break;

    N = N->getFirstPred();
  } while (N);
}

static llvm::StringLiteral WillBeUsedForACondition =
    ", which participates in a condition later";

PathDiagnosticPieceRef NoStoreFuncVisitor::maybeEmitNote(
    PathSensitiveBugReport &R, const CallEvent &Call, const ExplodedNode *N,
    const RegionVector &FieldChain, const MemRegion *MatchedRegion,
    StringRef FirstElement, bool FirstIsReferenceType,
    unsigned IndirectionLevel) {
  // Optimistically suppress uninitialized value bugs that result
  // from system headers having a chance to initialize the value
  // but failing to do so. It's too unlikely a system header's fault.
  // It's much more likely a situation in which the function has a failure
  // mode that the user decided not to check. If we want to hunt such
  // omitted checks, we should provide an explicit function-specific note
  // describing the precondition under which the function isn't supposed to
  // initialize its out-parameter, and additionally check that such
  // precondition can actually be fulfilled on the current path.
  if (Call.isInSystemHeader()) {
    // We make an exception for system header functions that have no branches.
    // Such functions unconditionally fail to initialize the variable.
    // If they call other functions that have more paths within them,
    // this suppression would still apply when we visit these inner functions.
    // One common example of a standard function that doesn't ever initialize
    // its out parameter is operator placement new; it's up to the follow-up
    // constructor (if any) to initialize the memory.
    if (!N->getStackFrame()->getCFG()->isLinear())
      R.markInvalid(getTag(), nullptr);
    return nullptr;
  }

  PathDiagnosticLocation L =
      PathDiagnosticLocation::create(N->getLocation(), SM);

  // For now this shouldn't trigger, but once it does (as we add more
  // functions to the body farm), we'll need to decide if these reports
  // are worth suppressing as well.
  if (!L.hasValidLocation())
    return nullptr;

  SmallString<256> sbuf;
  llvm::raw_svector_ostream os(sbuf);
  os << "Returning without writing to '";

  // Do not generate the note if failed to pretty-print.
  if (!prettyPrintRegionName(FirstElement, FirstIsReferenceType, MatchedRegion,
                             FieldChain, IndirectionLevel, os))
    return nullptr;

  os << "'";
  if (TKind == bugreporter::TrackingKind::Condition)
    os << WillBeUsedForACondition;
  return std::make_shared<PathDiagnosticEventPiece>(L, os.str());
}

bool NoStoreFuncVisitor::prettyPrintRegionName(StringRef FirstElement,
                                               bool FirstIsReferenceType,
                                               const MemRegion *MatchedRegion,
                                               const RegionVector &FieldChain,
                                               int IndirectionLevel,
                                               llvm::raw_svector_ostream &os) {

  if (FirstIsReferenceType)
    IndirectionLevel--;

  RegionVector RegionSequence;

  // Add the regions in the reverse order, then reverse the resulting array.
  assert(RegionOfInterest->isSubRegionOf(MatchedRegion));
  const MemRegion *R = RegionOfInterest;
  while (R != MatchedRegion) {
    RegionSequence.push_back(R);
    R = cast<SubRegion>(R)->getSuperRegion();
  }
  std::reverse(RegionSequence.begin(), RegionSequence.end());
  RegionSequence.append(FieldChain.begin(), FieldChain.end());

  StringRef Sep;
  for (const MemRegion *R : RegionSequence) {

    // Just keep going up to the base region.
    // Element regions may appear due to casts.
    if (isa<CXXBaseObjectRegion>(R) || isa<CXXTempObjectRegion>(R))
      continue;

    if (Sep.empty())
      Sep = prettyPrintFirstElement(FirstElement,
                                    /*MoreItemsExpected=*/true,
                                    IndirectionLevel, os);

    os << Sep;

    // Can only reasonably pretty-print DeclRegions.
    if (!isa<DeclRegion>(R))
      return false;

    const auto *DR = cast<DeclRegion>(R);
    Sep = DR->getValueType()->isAnyPointerType() ? "->" : ".";
    DR->getDecl()->getDeclName().print(os, PP);
  }

  if (Sep.empty())
    prettyPrintFirstElement(FirstElement,
                            /*MoreItemsExpected=*/false, IndirectionLevel, os);
  return true;
}

StringRef NoStoreFuncVisitor::prettyPrintFirstElement(
    StringRef FirstElement, bool MoreItemsExpected, int IndirectionLevel,
    llvm::raw_svector_ostream &os) {
  StringRef Out = ".";

  if (IndirectionLevel > 0 && MoreItemsExpected) {
    IndirectionLevel--;
    Out = "->";
  }

  if (IndirectionLevel > 0 && MoreItemsExpected)
    os << "(";

  for (int i = 0; i < IndirectionLevel; i++)
    os << "*";
  os << FirstElement;

  if (IndirectionLevel > 0 && MoreItemsExpected)
    os << ")";

  return Out;
}

//===----------------------------------------------------------------------===//
// Implementation of MacroNullReturnSuppressionVisitor.
//===----------------------------------------------------------------------===//

namespace {

/// Suppress null-pointer-dereference bugs where dereferenced null was returned
/// the macro.
class MacroNullReturnSuppressionVisitor final : public BugReporterVisitor {
  const SubRegion *RegionOfInterest;
  const SVal ValueAtDereference;

  // Do not invalidate the reports where the value was modified
  // after it got assigned to from the macro.
  bool WasModified = false;

public:
  MacroNullReturnSuppressionVisitor(const SubRegion *R, const SVal V)
      : RegionOfInterest(R), ValueAtDereference(V) {}

  PathDiagnosticPieceRef VisitNode(const ExplodedNode *N,
                                   BugReporterContext &BRC,
                                   PathSensitiveBugReport &BR) override {
    if (WasModified)
      return nullptr;

    auto BugPoint = BR.getErrorNode()->getLocation().getAs<StmtPoint>();
    if (!BugPoint)
      return nullptr;

    const SourceManager &SMgr = BRC.getSourceManager();
    if (auto Loc = matchAssignment(N)) {
      if (isFunctionMacroExpansion(*Loc, SMgr)) {
        std::string MacroName = std::string(getMacroName(*Loc, BRC));
        SourceLocation BugLoc = BugPoint->getStmt()->getBeginLoc();
        if (!BugLoc.isMacroID() || getMacroName(BugLoc, BRC) != MacroName)
          BR.markInvalid(getTag(), MacroName.c_str());
      }
    }

    if (wasRegionOfInterestModifiedAt(RegionOfInterest, N, ValueAtDereference))
      WasModified = true;

    return nullptr;
  }

  static void addMacroVisitorIfNecessary(
        const ExplodedNode *N, const MemRegion *R,
        bool EnableNullFPSuppression, PathSensitiveBugReport &BR,
        const SVal V) {
    AnalyzerOptions &Options = N->getState()->getAnalysisManager().options;
    if (EnableNullFPSuppression &&
        Options.ShouldSuppressNullReturnPaths && V.getAs<Loc>())
      BR.addVisitor(std::make_unique<MacroNullReturnSuppressionVisitor>(
              R->getAs<SubRegion>(), V));
  }

  void* getTag() const {
    static int Tag = 0;
    return static_cast<void *>(&Tag);
  }

  void Profile(llvm::FoldingSetNodeID &ID) const override {
    ID.AddPointer(getTag());
  }

private:
  /// \return Source location of right hand side of an assignment
  /// into \c RegionOfInterest, empty optional if none found.
  Optional<SourceLocation> matchAssignment(const ExplodedNode *N) {
    const Stmt *S = N->getStmtForDiagnostics();
    ProgramStateRef State = N->getState();
    auto *LCtx = N->getLocationContext();
    if (!S)
      return None;

    if (const auto *DS = dyn_cast<DeclStmt>(S)) {
      if (const auto *VD = dyn_cast<VarDecl>(DS->getSingleDecl()))
        if (const Expr *RHS = VD->getInit())
          if (RegionOfInterest->isSubRegionOf(
                  State->getLValue(VD, LCtx).getAsRegion()))
            return RHS->getBeginLoc();
    } else if (const auto *BO = dyn_cast<BinaryOperator>(S)) {
      const MemRegion *R = N->getSVal(BO->getLHS()).getAsRegion();
      const Expr *RHS = BO->getRHS();
      if (BO->isAssignmentOp() && RegionOfInterest->isSubRegionOf(R)) {
        return RHS->getBeginLoc();
      }
    }
    return None;
  }
};

} // end of anonymous namespace

namespace {

/// Emits an extra note at the return statement of an interesting stack frame.
///
/// The returned value is marked as an interesting value, and if it's null,
/// adds a visitor to track where it became null.
///
/// This visitor is intended to be used when another visitor discovers that an
/// interesting value comes from an inlined function call.
class ReturnVisitor : public BugReporterVisitor {
  const StackFrameContext *CalleeSFC;
  enum {
    Initial,
    MaybeUnsuppress,
    Satisfied
  } Mode = Initial;

  bool EnableNullFPSuppression;
  bool ShouldInvalidate = true;
  AnalyzerOptions& Options;
  bugreporter::TrackingKind TKind;

public:
  ReturnVisitor(const StackFrameContext *Frame, bool Suppressed,
                AnalyzerOptions &Options, bugreporter::TrackingKind TKind)
      : CalleeSFC(Frame), EnableNullFPSuppression(Suppressed),
        Options(Options), TKind(TKind) {}

  static void *getTag() {
    static int Tag = 0;
    return static_cast<void *>(&Tag);
  }

  void Profile(llvm::FoldingSetNodeID &ID) const override {
    ID.AddPointer(ReturnVisitor::getTag());
    ID.AddPointer(CalleeSFC);
    ID.AddBoolean(EnableNullFPSuppression);
  }

  /// Adds a ReturnVisitor if the given statement represents a call that was
  /// inlined.
  ///
  /// This will search back through the ExplodedGraph, starting from the given
  /// node, looking for when the given statement was processed. If it turns out
  /// the statement is a call that was inlined, we add the visitor to the
  /// bug report, so it can print a note later.
  static void addVisitorIfNecessary(const ExplodedNode *Node, const Stmt *S,
                                    PathSensitiveBugReport &BR,
                                    bool InEnableNullFPSuppression,
                                    bugreporter::TrackingKind TKind) {
    if (!CallEvent::isCallStmt(S))
      return;

    // First, find when we processed the statement.
    // If we work with a 'CXXNewExpr' that is going to be purged away before
    // its call take place. We would catch that purge in the last condition
    // as a 'StmtPoint' so we have to bypass it.
    const bool BypassCXXNewExprEval = isa<CXXNewExpr>(S);

    // This is moving forward when we enter into another context.
    const StackFrameContext *CurrentSFC = Node->getStackFrame();

    do {
      // If that is satisfied we found our statement as an inlined call.
      if (Optional<CallExitEnd> CEE = Node->getLocationAs<CallExitEnd>())
        if (CEE->getCalleeContext()->getCallSite() == S)
          break;

      // Try to move forward to the end of the call-chain.
      Node = Node->getFirstPred();
      if (!Node)
        break;

      const StackFrameContext *PredSFC = Node->getStackFrame();

      // If that is satisfied we found our statement.
      // FIXME: This code currently bypasses the call site for the
      //        conservatively evaluated allocator.
      if (!BypassCXXNewExprEval)
        if (Optional<StmtPoint> SP = Node->getLocationAs<StmtPoint>())
          // See if we do not enter into another context.
          if (SP->getStmt() == S && CurrentSFC == PredSFC)
            break;

      CurrentSFC = PredSFC;
    } while (Node->getStackFrame() == CurrentSFC);

    // Next, step over any post-statement checks.
    while (Node && Node->getLocation().getAs<PostStmt>())
      Node = Node->getFirstPred();
    if (!Node)
      return;

    // Finally, see if we inlined the call.
    Optional<CallExitEnd> CEE = Node->getLocationAs<CallExitEnd>();
    if (!CEE)
      return;

    const StackFrameContext *CalleeContext = CEE->getCalleeContext();
    if (CalleeContext->getCallSite() != S)
      return;

    // Check the return value.
    ProgramStateRef State = Node->getState();
    SVal RetVal = Node->getSVal(S);

    // Handle cases where a reference is returned and then immediately used.
    if (cast<Expr>(S)->isGLValue())
      if (Optional<Loc> LValue = RetVal.getAs<Loc>())
        RetVal = State->getSVal(*LValue);

    // See if the return value is NULL. If so, suppress the report.
    AnalyzerOptions &Options = State->getAnalysisManager().options;

    bool EnableNullFPSuppression = false;
    if (InEnableNullFPSuppression &&
        Options.ShouldSuppressNullReturnPaths)
      if (Optional<Loc> RetLoc = RetVal.getAs<Loc>())
        EnableNullFPSuppression = State->isNull(*RetLoc).isConstrainedTrue();

    BR.addVisitor(std::make_unique<ReturnVisitor>(CalleeContext,
                                                   EnableNullFPSuppression,
                                                   Options, TKind));
  }

  PathDiagnosticPieceRef visitNodeInitial(const ExplodedNode *N,
                                          BugReporterContext &BRC,
                                          PathSensitiveBugReport &BR) {
    // Only print a message at the interesting return statement.
    if (N->getLocationContext() != CalleeSFC)
      return nullptr;

    Optional<StmtPoint> SP = N->getLocationAs<StmtPoint>();
    if (!SP)
      return nullptr;

    const auto *Ret = dyn_cast<ReturnStmt>(SP->getStmt());
    if (!Ret)
      return nullptr;

    // Okay, we're at the right return statement, but do we have the return
    // value available?
    ProgramStateRef State = N->getState();
    SVal V = State->getSVal(Ret, CalleeSFC);
    if (V.isUnknownOrUndef())
      return nullptr;

    // Don't print any more notes after this one.
    Mode = Satisfied;

    const Expr *RetE = Ret->getRetValue();
    assert(RetE && "Tracking a return value for a void function");

    // Handle cases where a reference is returned and then immediately used.
    Optional<Loc> LValue;
    if (RetE->isGLValue()) {
      if ((LValue = V.getAs<Loc>())) {
        SVal RValue = State->getRawSVal(*LValue, RetE->getType());
        if (RValue.getAs<DefinedSVal>())
          V = RValue;
      }
    }

    // Ignore aggregate rvalues.
    if (V.getAs<nonloc::LazyCompoundVal>() ||
        V.getAs<nonloc::CompoundVal>())
      return nullptr;

    RetE = RetE->IgnoreParenCasts();

    // Let's track the return value.
    bugreporter::trackExpressionValue(
        N, RetE, BR, TKind, EnableNullFPSuppression);

    // Build an appropriate message based on the return value.
    SmallString<64> Msg;
    llvm::raw_svector_ostream Out(Msg);

    bool WouldEventBeMeaningless = false;

    if (State->isNull(V).isConstrainedTrue()) {
      if (V.getAs<Loc>()) {

        // If we have counter-suppression enabled, make sure we keep visiting
        // future nodes. We want to emit a path note as well, in case
        // the report is resurrected as valid later on.
        if (EnableNullFPSuppression &&
            Options.ShouldAvoidSuppressingNullArgumentPaths)
          Mode = MaybeUnsuppress;

        if (RetE->getType()->isObjCObjectPointerType()) {
          Out << "Returning nil";
        } else {
          Out << "Returning null pointer";
        }
      } else {
        Out << "Returning zero";
      }

    } else {
      if (auto CI = V.getAs<nonloc::ConcreteInt>()) {
        Out << "Returning the value " << CI->getValue();
      } else {
        // There is nothing interesting about returning a value, when it is
        // plain value without any constraints, and the function is guaranteed
        // to return that every time. We could use CFG::isLinear() here, but
        // constexpr branches are obvious to the compiler, not necesserily to
        // the programmer.
        if (N->getCFG().size() == 3)
          WouldEventBeMeaningless = true;

        if (V.getAs<Loc>())
          Out << "Returning pointer";
        else
          Out << "Returning value";
      }
    }

    if (LValue) {
      if (const MemRegion *MR = LValue->getAsRegion()) {
        if (MR->canPrintPretty()) {
          Out << " (reference to ";
          MR->printPretty(Out);
          Out << ")";
        }
      }
    } else {
      // FIXME: We should have a more generalized location printing mechanism.
      if (const auto *DR = dyn_cast<DeclRefExpr>(RetE))
        if (const auto *DD = dyn_cast<DeclaratorDecl>(DR->getDecl()))
          Out << " (loaded from '" << *DD << "')";
    }

    PathDiagnosticLocation L(Ret, BRC.getSourceManager(), CalleeSFC);
    if (!L.isValid() || !L.asLocation().isValid())
      return nullptr;

    if (TKind == bugreporter::TrackingKind::Condition)
      Out << WillBeUsedForACondition;

    auto EventPiece = std::make_shared<PathDiagnosticEventPiece>(L, Out.str());

    // If we determined that the note is meaningless, make it prunable, and
    // don't mark the stackframe interesting.
    if (WouldEventBeMeaningless)
      EventPiece->setPrunable(true);
    else
      BR.markInteresting(CalleeSFC);

    return EventPiece;
  }

  PathDiagnosticPieceRef visitNodeMaybeUnsuppress(const ExplodedNode *N,
                                                  BugReporterContext &BRC,
                                                  PathSensitiveBugReport &BR) {
    assert(Options.ShouldAvoidSuppressingNullArgumentPaths);

    // Are we at the entry node for this call?
    Optional<CallEnter> CE = N->getLocationAs<CallEnter>();
    if (!CE)
      return nullptr;

    if (CE->getCalleeContext() != CalleeSFC)
      return nullptr;

    Mode = Satisfied;

    // Don't automatically suppress a report if one of the arguments is
    // known to be a null pointer. Instead, start tracking /that/ null
    // value back to its origin.
    ProgramStateManager &StateMgr = BRC.getStateManager();
    CallEventManager &CallMgr = StateMgr.getCallEventManager();

    ProgramStateRef State = N->getState();
    CallEventRef<> Call = CallMgr.getCaller(CalleeSFC, State);
    for (unsigned I = 0, E = Call->getNumArgs(); I != E; ++I) {
      Optional<Loc> ArgV = Call->getArgSVal(I).getAs<Loc>();
      if (!ArgV)
        continue;

      const Expr *ArgE = Call->getArgExpr(I);
      if (!ArgE)
        continue;

      // Is it possible for this argument to be non-null?
      if (!State->isNull(*ArgV).isConstrainedTrue())
        continue;

      if (trackExpressionValue(N, ArgE, BR, TKind, EnableNullFPSuppression))
        ShouldInvalidate = false;

      // If we /can't/ track the null pointer, we should err on the side of
      // false negatives, and continue towards marking this report invalid.
      // (We will still look at the other arguments, though.)
    }

    return nullptr;
  }

  PathDiagnosticPieceRef VisitNode(const ExplodedNode *N,
                                   BugReporterContext &BRC,
                                   PathSensitiveBugReport &BR) override {
    switch (Mode) {
    case Initial:
      return visitNodeInitial(N, BRC, BR);
    case MaybeUnsuppress:
      return visitNodeMaybeUnsuppress(N, BRC, BR);
    case Satisfied:
      return nullptr;
    }

    llvm_unreachable("Invalid visit mode!");
  }

  void finalizeVisitor(BugReporterContext &, const ExplodedNode *,
                       PathSensitiveBugReport &BR) override {
    if (EnableNullFPSuppression && ShouldInvalidate)
      BR.markInvalid(ReturnVisitor::getTag(), CalleeSFC);
  }
};

} // end of anonymous namespace

//===----------------------------------------------------------------------===//
// Implementation of FindLastStoreBRVisitor.
//===----------------------------------------------------------------------===//

void FindLastStoreBRVisitor::Profile(llvm::FoldingSetNodeID &ID) const {
  static int tag = 0;
  ID.AddPointer(&tag);
  ID.AddPointer(R);
  ID.Add(V);
  ID.AddInteger(static_cast<int>(TKind));
  ID.AddBoolean(EnableNullFPSuppression);
}

/// Returns true if \p N represents the DeclStmt declaring and initializing
/// \p VR.
static bool isInitializationOfVar(const ExplodedNode *N, const VarRegion *VR) {
  Optional<PostStmt> P = N->getLocationAs<PostStmt>();
  if (!P)
    return false;

  const DeclStmt *DS = P->getStmtAs<DeclStmt>();
  if (!DS)
    return false;

  if (DS->getSingleDecl() != VR->getDecl())
    return false;

  const MemSpaceRegion *VarSpace = VR->getMemorySpace();
  const auto *FrameSpace = dyn_cast<StackSpaceRegion>(VarSpace);
  if (!FrameSpace) {
    // If we ever directly evaluate global DeclStmts, this assertion will be
    // invalid, but this still seems preferable to silently accepting an
    // initialization that may be for a path-sensitive variable.
    assert(VR->getDecl()->isStaticLocal() && "non-static stackless VarRegion");
    return true;
  }

  assert(VR->getDecl()->hasLocalStorage());
  const LocationContext *LCtx = N->getLocationContext();
  return FrameSpace->getStackFrame() == LCtx->getStackFrame();
}

/// Show diagnostics for initializing or declaring a region \p R with a bad value.
static void showBRDiagnostics(const char *action, llvm::raw_svector_ostream &os,
                              const MemRegion *R, SVal V, const DeclStmt *DS) {
  if (R->canPrintPretty()) {
    R->printPretty(os);
    os << " ";
  }

  if (V.getAs<loc::ConcreteInt>()) {
    bool b = false;
    if (R->isBoundable()) {
      if (const auto *TR = dyn_cast<TypedValueRegion>(R)) {
        if (TR->getValueType()->isObjCObjectPointerType()) {
          os << action << "nil";
          b = true;
        }
      }
    }
    if (!b)
      os << action << "a null pointer value";

  } else if (auto CVal = V.getAs<nonloc::ConcreteInt>()) {
    os << action << CVal->getValue();
  } else if (DS) {
    if (V.isUndef()) {
      if (isa<VarRegion>(R)) {
        const auto *VD = cast<VarDecl>(DS->getSingleDecl());
        if (VD->getInit()) {
          os << (R->canPrintPretty() ? "initialized" : "Initializing")
            << " to a garbage value";
        } else {
          os << (R->canPrintPretty() ? "declared" : "Declaring")
            << " without an initial value";
        }
      }
    } else {
      os << (R->canPrintPretty() ? "initialized" : "Initialized")
        << " here";
    }
  }
}

/// Display diagnostics for passing bad region as a parameter.
static void showBRParamDiagnostics(llvm::raw_svector_ostream& os,
    const VarRegion *VR,
    SVal V) {
  const auto *Param = cast<ParmVarDecl>(VR->getDecl());

  os << "Passing ";

  if (V.getAs<loc::ConcreteInt>()) {
    if (Param->getType()->isObjCObjectPointerType())
      os << "nil object reference";
    else
      os << "null pointer value";
  } else if (V.isUndef()) {
    os << "uninitialized value";
  } else if (auto CI = V.getAs<nonloc::ConcreteInt>()) {
    os << "the value " << CI->getValue();
  } else {
    os << "value";
  }

  // Printed parameter indexes are 1-based, not 0-based.
  unsigned Idx = Param->getFunctionScopeIndex() + 1;
  os << " via " << Idx << llvm::getOrdinalSuffix(Idx) << " parameter";
  if (VR->canPrintPretty()) {
    os << " ";
    VR->printPretty(os);
  }
}

/// Show default diagnostics for storing bad region.
static void showBRDefaultDiagnostics(llvm::raw_svector_ostream &os,
                                     const MemRegion *R, SVal V) {
  if (V.getAs<loc::ConcreteInt>()) {
    bool b = false;
    if (R->isBoundable()) {
      if (const auto *TR = dyn_cast<TypedValueRegion>(R)) {
        if (TR->getValueType()->isObjCObjectPointerType()) {
          os << "nil object reference stored";
          b = true;
        }
      }
    }
    if (!b) {
      if (R->canPrintPretty())
        os << "Null pointer value stored";
      else
        os << "Storing null pointer value";
    }

  } else if (V.isUndef()) {
    if (R->canPrintPretty())
      os << "Uninitialized value stored";
    else
      os << "Storing uninitialized value";

  } else if (auto CV = V.getAs<nonloc::ConcreteInt>()) {
    if (R->canPrintPretty())
      os << "The value " << CV->getValue() << " is assigned";
    else
      os << "Assigning " << CV->getValue();

  } else {
    if (R->canPrintPretty())
      os << "Value assigned";
    else
      os << "Assigning value";
  }

  if (R->canPrintPretty()) {
    os << " to ";
    R->printPretty(os);
  }
}

PathDiagnosticPieceRef
FindLastStoreBRVisitor::VisitNode(const ExplodedNode *Succ,
                                  BugReporterContext &BRC,
                                  PathSensitiveBugReport &BR) {
  if (Satisfied)
    return nullptr;

  const ExplodedNode *StoreSite = nullptr;
  const ExplodedNode *Pred = Succ->getFirstPred();
  const Expr *InitE = nullptr;
  bool IsParam = false;

  // First see if we reached the declaration of the region.
  if (const auto *VR = dyn_cast<VarRegion>(R)) {
    if (isInitializationOfVar(Pred, VR)) {
      StoreSite = Pred;
      InitE = VR->getDecl()->getInit();
    }
  }

  // If this is a post initializer expression, initializing the region, we
  // should track the initializer expression.
  if (Optional<PostInitializer> PIP = Pred->getLocationAs<PostInitializer>()) {
    const MemRegion *FieldReg = (const MemRegion *)PIP->getLocationValue();
    if (FieldReg == R) {
      StoreSite = Pred;
      InitE = PIP->getInitializer()->getInit();
    }
  }

  // Otherwise, see if this is the store site:
  // (1) Succ has this binding and Pred does not, i.e. this is
  //     where the binding first occurred.
  // (2) Succ has this binding and is a PostStore node for this region, i.e.
  //     the same binding was re-assigned here.
  if (!StoreSite) {
    if (Succ->getState()->getSVal(R) != V)
      return nullptr;

    if (hasVisibleUpdate(Pred, Pred->getState()->getSVal(R), Succ, V)) {
      Optional<PostStore> PS = Succ->getLocationAs<PostStore>();
      if (!PS || PS->getLocationValue() != R)
        return nullptr;
    }

    StoreSite = Succ;

    // If this is an assignment expression, we can track the value
    // being assigned.
    if (Optional<PostStmt> P = Succ->getLocationAs<PostStmt>())
      if (const BinaryOperator *BO = P->getStmtAs<BinaryOperator>())
        if (BO->isAssignmentOp())
          InitE = BO->getRHS();

    // If this is a call entry, the variable should be a parameter.
    // FIXME: Handle CXXThisRegion as well. (This is not a priority because
    // 'this' should never be NULL, but this visitor isn't just for NULL and
    // UndefinedVal.)
    if (Optional<CallEnter> CE = Succ->getLocationAs<CallEnter>()) {
      if (const auto *VR = dyn_cast<VarRegion>(R)) {

        if (const auto *Param = dyn_cast<ParmVarDecl>(VR->getDecl())) {
          ProgramStateManager &StateMgr = BRC.getStateManager();
          CallEventManager &CallMgr = StateMgr.getCallEventManager();

          CallEventRef<> Call = CallMgr.getCaller(CE->getCalleeContext(),
                                                  Succ->getState());
          InitE = Call->getArgExpr(Param->getFunctionScopeIndex());
        } else {
          // Handle Objective-C 'self'.
          assert(isa<ImplicitParamDecl>(VR->getDecl()));
          InitE = cast<ObjCMessageExpr>(CE->getCalleeContext()->getCallSite())
                      ->getInstanceReceiver()->IgnoreParenCasts();
        }
        IsParam = true;
      }
    }

    // If this is a CXXTempObjectRegion, the Expr responsible for its creation
    // is wrapped inside of it.
    if (const auto *TmpR = dyn_cast<CXXTempObjectRegion>(R))
      InitE = TmpR->getExpr();
  }

  if (!StoreSite)
    return nullptr;

  Satisfied = true;

  // If we have an expression that provided the value, try to track where it
  // came from.
  if (InitE) {
    if (!IsParam)
      InitE = InitE->IgnoreParenCasts();

    bugreporter::trackExpressionValue(
        StoreSite, InitE, BR, TKind, EnableNullFPSuppression);
  }

  if (TKind == TrackingKind::Condition &&
      !OriginSFC->isParentOf(StoreSite->getStackFrame()))
    return nullptr;

  // Okay, we've found the binding. Emit an appropriate message.
  SmallString<256> sbuf;
  llvm::raw_svector_ostream os(sbuf);

  if (Optional<PostStmt> PS = StoreSite->getLocationAs<PostStmt>()) {
    const Stmt *S = PS->getStmt();
    const char *action = nullptr;
    const auto *DS = dyn_cast<DeclStmt>(S);
    const auto *VR = dyn_cast<VarRegion>(R);

    if (DS) {
      action = R->canPrintPretty() ? "initialized to " :
                                     "Initializing to ";
    } else if (isa<BlockExpr>(S)) {
      action = R->canPrintPretty() ? "captured by block as " :
                                     "Captured by block as ";
      if (VR) {
        // See if we can get the BlockVarRegion.
        ProgramStateRef State = StoreSite->getState();
        SVal V = StoreSite->getSVal(S);
        if (const auto *BDR =
              dyn_cast_or_null<BlockDataRegion>(V.getAsRegion())) {
          if (const VarRegion *OriginalR = BDR->getOriginalRegion(VR)) {
            if (auto KV = State->getSVal(OriginalR).getAs<KnownSVal>())
              BR.addVisitor(std::make_unique<FindLastStoreBRVisitor>(
                  *KV, OriginalR, EnableNullFPSuppression, TKind, OriginSFC));
          }
        }
      }
    }
    if (action)
      showBRDiagnostics(action, os, R, V, DS);

  } else if (StoreSite->getLocation().getAs<CallEnter>()) {
    if (const auto *VR = dyn_cast<VarRegion>(R))
      showBRParamDiagnostics(os, VR, V);
  }

  if (os.str().empty())
    showBRDefaultDiagnostics(os, R, V);

  if (TKind == bugreporter::TrackingKind::Condition)
    os << WillBeUsedForACondition;

  // Construct a new PathDiagnosticPiece.
  ProgramPoint P = StoreSite->getLocation();
  PathDiagnosticLocation L;
  if (P.getAs<CallEnter>() && InitE)
    L = PathDiagnosticLocation(InitE, BRC.getSourceManager(),
                               P.getLocationContext());

  if (!L.isValid() || !L.asLocation().isValid())
    L = PathDiagnosticLocation::create(P, BRC.getSourceManager());

  if (!L.isValid() || !L.asLocation().isValid())
    return nullptr;

  return std::make_shared<PathDiagnosticEventPiece>(L, os.str());
}

//===----------------------------------------------------------------------===//
// Implementation of TrackConstraintBRVisitor.
//===----------------------------------------------------------------------===//

void TrackConstraintBRVisitor::Profile(llvm::FoldingSetNodeID &ID) const {
  static int tag = 0;
  ID.AddPointer(&tag);
  ID.AddBoolean(Assumption);
  ID.Add(Constraint);
}

/// Return the tag associated with this visitor.  This tag will be used
/// to make all PathDiagnosticPieces created by this visitor.
const char *TrackConstraintBRVisitor::getTag() {
  return "TrackConstraintBRVisitor";
}

bool TrackConstraintBRVisitor::isUnderconstrained(const ExplodedNode *N) const {
  if (IsZeroCheck)
    return N->getState()->isNull(Constraint).isUnderconstrained();
  return (bool)N->getState()->assume(Constraint, !Assumption);
}

PathDiagnosticPieceRef TrackConstraintBRVisitor::VisitNode(
    const ExplodedNode *N, BugReporterContext &BRC, PathSensitiveBugReport &) {
  const ExplodedNode *PrevN = N->getFirstPred();
  if (IsSatisfied)
    return nullptr;

  // Start tracking after we see the first state in which the value is
  // constrained.
  if (!IsTrackingTurnedOn)
    if (!isUnderconstrained(N))
      IsTrackingTurnedOn = true;
  if (!IsTrackingTurnedOn)
    return nullptr;

  // Check if in the previous state it was feasible for this constraint
  // to *not* be true.
  if (isUnderconstrained(PrevN)) {
    IsSatisfied = true;

    // As a sanity check, make sure that the negation of the constraint
    // was infeasible in the current state.  If it is feasible, we somehow
    // missed the transition point.
    assert(!isUnderconstrained(N));

    // We found the transition point for the constraint.  We now need to
    // pretty-print the constraint. (work-in-progress)
    SmallString<64> sbuf;
    llvm::raw_svector_ostream os(sbuf);

    if (Constraint.getAs<Loc>()) {
      os << "Assuming pointer value is ";
      os << (Assumption ? "non-null" : "null");
    }

    if (os.str().empty())
      return nullptr;

    // Construct a new PathDiagnosticPiece.
    ProgramPoint P = N->getLocation();
    PathDiagnosticLocation L =
      PathDiagnosticLocation::create(P, BRC.getSourceManager());
    if (!L.isValid())
      return nullptr;

    auto X = std::make_shared<PathDiagnosticEventPiece>(L, os.str());
    X->setTag(getTag());
    return std::move(X);
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Implementation of SuppressInlineDefensiveChecksVisitor.
//===----------------------------------------------------------------------===//

SuppressInlineDefensiveChecksVisitor::
SuppressInlineDefensiveChecksVisitor(DefinedSVal Value, const ExplodedNode *N)
    : V(Value) {
  // Check if the visitor is disabled.
  AnalyzerOptions &Options = N->getState()->getAnalysisManager().options;
  if (!Options.ShouldSuppressInlinedDefensiveChecks)
    IsSatisfied = true;
}

void SuppressInlineDefensiveChecksVisitor::Profile(
    llvm::FoldingSetNodeID &ID) const {
  static int id = 0;
  ID.AddPointer(&id);
  ID.Add(V);
}

const char *SuppressInlineDefensiveChecksVisitor::getTag() {
  return "IDCVisitor";
}

PathDiagnosticPieceRef
SuppressInlineDefensiveChecksVisitor::VisitNode(const ExplodedNode *Succ,
                                                BugReporterContext &BRC,
                                                PathSensitiveBugReport &BR) {
  const ExplodedNode *Pred = Succ->getFirstPred();
  if (IsSatisfied)
    return nullptr;

  // Start tracking after we see the first state in which the value is null.
  if (!IsTrackingTurnedOn)
    if (Succ->getState()->isNull(V).isConstrainedTrue())
      IsTrackingTurnedOn = true;
  if (!IsTrackingTurnedOn)
    return nullptr;

  // Check if in the previous state it was feasible for this value
  // to *not* be null.
  if (!Pred->getState()->isNull(V).isConstrainedTrue() &&
      Succ->getState()->isNull(V).isConstrainedTrue()) {
    IsSatisfied = true;

    // Check if this is inlined defensive checks.
    const LocationContext *CurLC = Succ->getLocationContext();
    const LocationContext *ReportLC = BR.getErrorNode()->getLocationContext();
    if (CurLC != ReportLC && !CurLC->isParentOf(ReportLC)) {
      BR.markInvalid("Suppress IDC", CurLC);
      return nullptr;
    }

    // Treat defensive checks in function-like macros as if they were an inlined
    // defensive check. If the bug location is not in a macro and the
    // terminator for the current location is in a macro then suppress the
    // warning.
    auto BugPoint = BR.getErrorNode()->getLocation().getAs<StmtPoint>();

    if (!BugPoint)
      return nullptr;

    ProgramPoint CurPoint = Succ->getLocation();
    const Stmt *CurTerminatorStmt = nullptr;
    if (auto BE = CurPoint.getAs<BlockEdge>()) {
      CurTerminatorStmt = BE->getSrc()->getTerminator().getStmt();
    } else if (auto SP = CurPoint.getAs<StmtPoint>()) {
      const Stmt *CurStmt = SP->getStmt();
      if (!CurStmt->getBeginLoc().isMacroID())
        return nullptr;

      CFGStmtMap *Map = CurLC->getAnalysisDeclContext()->getCFGStmtMap();
      CurTerminatorStmt = Map->getBlock(CurStmt)->getTerminatorStmt();
    } else {
      return nullptr;
    }

    if (!CurTerminatorStmt)
      return nullptr;

    SourceLocation TerminatorLoc = CurTerminatorStmt->getBeginLoc();
    if (TerminatorLoc.isMacroID()) {
      SourceLocation BugLoc = BugPoint->getStmt()->getBeginLoc();

      // Suppress reports unless we are in that same macro.
      if (!BugLoc.isMacroID() ||
          getMacroName(BugLoc, BRC) != getMacroName(TerminatorLoc, BRC)) {
        BR.markInvalid("Suppress Macro IDC", CurLC);
      }
      return nullptr;
    }
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// TrackControlDependencyCondBRVisitor.
//===----------------------------------------------------------------------===//

namespace {
/// Tracks the expressions that are a control dependency of the node that was
/// supplied to the constructor.
/// For example:
///
///   cond = 1;
///   if (cond)
///     10 / 0;
///
/// An error is emitted at line 3. This visitor realizes that the branch
/// on line 2 is a control dependency of line 3, and tracks it's condition via
/// trackExpressionValue().
class TrackControlDependencyCondBRVisitor final : public BugReporterVisitor {
  const ExplodedNode *Origin;
  ControlDependencyCalculator ControlDeps;
  llvm::SmallSet<const CFGBlock *, 32> VisitedBlocks;

public:
  TrackControlDependencyCondBRVisitor(const ExplodedNode *O)
  : Origin(O), ControlDeps(&O->getCFG()) {}

  void Profile(llvm::FoldingSetNodeID &ID) const override {
    static int x = 0;
    ID.AddPointer(&x);
  }

  PathDiagnosticPieceRef VisitNode(const ExplodedNode *N,
                                   BugReporterContext &BRC,
                                   PathSensitiveBugReport &BR) override;
};
} // end of anonymous namespace

static std::shared_ptr<PathDiagnosticEventPiece>
constructDebugPieceForTrackedCondition(const Expr *Cond,
                                       const ExplodedNode *N,
                                       BugReporterContext &BRC) {

  if (BRC.getAnalyzerOptions().AnalysisDiagOpt == PD_NONE ||
      !BRC.getAnalyzerOptions().ShouldTrackConditionsDebug)
    return nullptr;

  std::string ConditionText = std::string(Lexer::getSourceText(
      CharSourceRange::getTokenRange(Cond->getSourceRange()),
      BRC.getSourceManager(), BRC.getASTContext().getLangOpts()));

  return std::make_shared<PathDiagnosticEventPiece>(
      PathDiagnosticLocation::createBegin(
          Cond, BRC.getSourceManager(), N->getLocationContext()),
          (Twine() + "Tracking condition '" + ConditionText + "'").str());
}

static bool isAssertlikeBlock(const CFGBlock *B, ASTContext &Context) {
  if (B->succ_size() != 2)
    return false;

  const CFGBlock *Then = B->succ_begin()->getReachableBlock();
  const CFGBlock *Else = (B->succ_begin() + 1)->getReachableBlock();

  if (!Then || !Else)
    return false;

  if (Then->isInevitablySinking() != Else->isInevitablySinking())
    return true;

  // For the following condition the following CFG would be built:
  //
  //                          ------------->
  //                         /              \
  //                       [B1] -> [B2] -> [B3] -> [sink]
  // assert(A && B || C);            \       \
  //                                  -----------> [go on with the execution]
  //
  // It so happens that CFGBlock::getTerminatorCondition returns 'A' for block
  // B1, 'A && B' for B2, and 'A && B || C' for B3. Let's check whether we
  // reached the end of the condition!
  if (const Stmt *ElseCond = Else->getTerminatorCondition())
    if (const auto *BinOp = dyn_cast<BinaryOperator>(ElseCond))
      if (BinOp->isLogicalOp())
        return isAssertlikeBlock(Else, Context);

  return false;
}

PathDiagnosticPieceRef
TrackControlDependencyCondBRVisitor::VisitNode(const ExplodedNode *N,
                                               BugReporterContext &BRC,
                                               PathSensitiveBugReport &BR) {
  // We can only reason about control dependencies within the same stack frame.
  if (Origin->getStackFrame() != N->getStackFrame())
    return nullptr;

  CFGBlock *NB = const_cast<CFGBlock *>(N->getCFGBlock());

  // Skip if we already inspected this block.
  if (!VisitedBlocks.insert(NB).second)
    return nullptr;

  CFGBlock *OriginB = const_cast<CFGBlock *>(Origin->getCFGBlock());

  // TODO: Cache CFGBlocks for each ExplodedNode.
  if (!OriginB || !NB)
    return nullptr;

  if (isAssertlikeBlock(NB, BRC.getASTContext()))
    return nullptr;

  if (ControlDeps.isControlDependent(OriginB, NB)) {
    // We don't really want to explain for range loops. Evidence suggests that
    // the only thing that leads to is the addition of calls to operator!=.
    if (llvm::isa_and_nonnull<CXXForRangeStmt>(NB->getTerminatorStmt()))
      return nullptr;

    if (const Expr *Condition = NB->getLastCondition()) {
      // Keeping track of the already tracked conditions on a visitor level
      // isn't sufficient, because a new visitor is created for each tracked
      // expression, hence the BugReport level set.
      if (BR.addTrackedCondition(N)) {
        bugreporter::trackExpressionValue(
            N, Condition, BR, bugreporter::TrackingKind::Condition,
            /*EnableNullFPSuppression=*/false);
        return constructDebugPieceForTrackedCondition(Condition, N, BRC);
      }
    }
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Implementation of trackExpressionValue.
//===----------------------------------------------------------------------===//

static const MemRegion *getLocationRegionIfReference(const Expr *E,
                                                     const ExplodedNode *N) {
  if (const auto *DR = dyn_cast<DeclRefExpr>(E)) {
    if (const auto *VD = dyn_cast<VarDecl>(DR->getDecl())) {
      if (!VD->getType()->isReferenceType())
        return nullptr;
      ProgramStateManager &StateMgr = N->getState()->getStateManager();
      MemRegionManager &MRMgr = StateMgr.getRegionManager();
      return MRMgr.getVarRegion(VD, N->getLocationContext());
    }
  }

  // FIXME: This does not handle other kinds of null references,
  // for example, references from FieldRegions:
  //   struct Wrapper { int &ref; };
  //   Wrapper w = { *(int *)0 };
  //   w.ref = 1;

  return nullptr;
}

/// \return A subexpression of @c Ex which represents the
/// expression-of-interest.
static const Expr *peelOffOuterExpr(const Expr *Ex,
                                    const ExplodedNode *N) {
  Ex = Ex->IgnoreParenCasts();
  if (const auto *FE = dyn_cast<FullExpr>(Ex))
    return peelOffOuterExpr(FE->getSubExpr(), N);
  if (const auto *OVE = dyn_cast<OpaqueValueExpr>(Ex))
    return peelOffOuterExpr(OVE->getSourceExpr(), N);
  if (const auto *POE = dyn_cast<PseudoObjectExpr>(Ex)) {
    const auto *PropRef = dyn_cast<ObjCPropertyRefExpr>(POE->getSyntacticForm());
    if (PropRef && PropRef->isMessagingGetter()) {
      const Expr *GetterMessageSend =
          POE->getSemanticExpr(POE->getNumSemanticExprs() - 1);
      assert(isa<ObjCMessageExpr>(GetterMessageSend->IgnoreParenCasts()));
      return peelOffOuterExpr(GetterMessageSend, N);
    }
  }

  // Peel off the ternary operator.
  if (const auto *CO = dyn_cast<ConditionalOperator>(Ex)) {
    // Find a node where the branching occurred and find out which branch
    // we took (true/false) by looking at the ExplodedGraph.
    const ExplodedNode *NI = N;
    do {
      ProgramPoint ProgPoint = NI->getLocation();
      if (Optional<BlockEdge> BE = ProgPoint.getAs<BlockEdge>()) {
        const CFGBlock *srcBlk = BE->getSrc();
        if (const Stmt *term = srcBlk->getTerminatorStmt()) {
          if (term == CO) {
            bool TookTrueBranch = (*(srcBlk->succ_begin()) == BE->getDst());
            if (TookTrueBranch)
              return peelOffOuterExpr(CO->getTrueExpr(), N);
            else
              return peelOffOuterExpr(CO->getFalseExpr(), N);
          }
        }
      }
      NI = NI->getFirstPred();
    } while (NI);
  }

  if (auto *BO = dyn_cast<BinaryOperator>(Ex))
    if (const Expr *SubEx = peelOffPointerArithmetic(BO))
      return peelOffOuterExpr(SubEx, N);

  if (auto *UO = dyn_cast<UnaryOperator>(Ex)) {
    if (UO->getOpcode() == UO_LNot)
      return peelOffOuterExpr(UO->getSubExpr(), N);

    // FIXME: There's a hack in our Store implementation that always computes
    // field offsets around null pointers as if they are always equal to 0.
    // The idea here is to report accesses to fields as null dereferences
    // even though the pointer value that's being dereferenced is actually
    // the offset of the field rather than exactly 0.
    // See the FIXME in StoreManager's getLValueFieldOrIvar() method.
    // This code interacts heavily with this hack; otherwise the value
    // would not be null at all for most fields, so we'd be unable to track it.
    if (UO->getOpcode() == UO_AddrOf && UO->getSubExpr()->isLValue())
      if (const Expr *DerefEx = bugreporter::getDerefExpr(UO->getSubExpr()))
        return peelOffOuterExpr(DerefEx, N);
  }

  return Ex;
}

/// Find the ExplodedNode where the lvalue (the value of 'Ex')
/// was computed.
static const ExplodedNode* findNodeForExpression(const ExplodedNode *N,
                                                 const Expr *Inner) {
  while (N) {
    if (N->getStmtForDiagnostics() == Inner)
      return N;
    N = N->getFirstPred();
  }
  return N;
}

/// Attempts to add visitors to track an RValue expression back to its point of
/// origin. Works similarly to trackExpressionValue, but accepts only RValues.
static void trackRValueExpression(const ExplodedNode *InputNode, const Expr *E,
                                  PathSensitiveBugReport &report,
                                  bugreporter::TrackingKind TKind,
                                  bool EnableNullFPSuppression) {
  assert(E->isRValue() && "The expression is not an rvalue!");
  const ExplodedNode *RVNode = findNodeForExpression(InputNode, E);
  if (!RVNode)
    return;
  ProgramStateRef RVState = RVNode->getState();
  SVal V = RVState->getSValAsScalarOrLoc(E, RVNode->getLocationContext());
  const auto *BO = dyn_cast<BinaryOperator>(E);
  if (!BO)
    return;
  if (!V.isZeroConstant())
    return;
  if (!BO->isMultiplicativeOp())
    return;

  SVal RHSV = RVState->getSVal(BO->getRHS(), RVNode->getLocationContext());
  SVal LHSV = RVState->getSVal(BO->getLHS(), RVNode->getLocationContext());

  // Track both LHS and RHS of a multiplication.
  if (BO->getOpcode() == BO_Mul) {
    if (LHSV.isZeroConstant())
      trackExpressionValue(InputNode, BO->getLHS(), report, TKind,
                           EnableNullFPSuppression);
    if (RHSV.isZeroConstant())
      trackExpressionValue(InputNode, BO->getRHS(), report, TKind,
                           EnableNullFPSuppression);
  } else { // Track only the LHS of a division or a modulo.
    if (LHSV.isZeroConstant())
      trackExpressionValue(InputNode, BO->getLHS(), report, TKind,
                           EnableNullFPSuppression);
  }
}

bool bugreporter::trackExpressionValue(const ExplodedNode *InputNode,
                                       const Expr *E,
                                       PathSensitiveBugReport &report,
                                       bugreporter::TrackingKind TKind,
                                       bool EnableNullFPSuppression) {

  if (!E || !InputNode)
    return false;

  const Expr *Inner = peelOffOuterExpr(E, InputNode);
  const ExplodedNode *LVNode = findNodeForExpression(InputNode, Inner);
  if (!LVNode)
    return false;

  ProgramStateRef LVState = LVNode->getState();
  const StackFrameContext *SFC = LVNode->getStackFrame();

  // We only track expressions if we believe that they are important. Chances
  // are good that control dependencies to the tracking point are also important
  // because of this, let's explain why we believe control reached this point.
  // TODO: Shouldn't we track control dependencies of every bug location, rather
  // than only tracked expressions?
  if (LVState->getAnalysisManager().getAnalyzerOptions().ShouldTrackConditions)
    report.addVisitor(std::make_unique<TrackControlDependencyCondBRVisitor>(
          InputNode));

  // The message send could be nil due to the receiver being nil.
  // At this point in the path, the receiver should be live since we are at the
  // message send expr. If it is nil, start tracking it.
  if (const Expr *Receiver = NilReceiverBRVisitor::getNilReceiver(Inner, LVNode))
    trackExpressionValue(
        LVNode, Receiver, report, TKind, EnableNullFPSuppression);

  // Track the index if this is an array subscript.
  if (const auto *Arr = dyn_cast<ArraySubscriptExpr>(Inner))
    trackExpressionValue(
        LVNode, Arr->getIdx(), report, TKind, /*EnableNullFPSuppression*/false);

  // See if the expression we're interested refers to a variable.
  // If so, we can track both its contents and constraints on its value.
  if (ExplodedGraph::isInterestingLValueExpr(Inner)) {
    SVal LVal = LVNode->getSVal(Inner);

    const MemRegion *RR = getLocationRegionIfReference(Inner, LVNode);
    bool LVIsNull = LVState->isNull(LVal).isConstrainedTrue();

    // If this is a C++ reference to a null pointer, we are tracking the
    // pointer. In addition, we should find the store at which the reference
    // got initialized.
    if (RR && !LVIsNull)
      if (auto KV = LVal.getAs<KnownSVal>())
        report.addVisitor(std::make_unique<FindLastStoreBRVisitor>(
            *KV, RR, EnableNullFPSuppression, TKind, SFC));

    // In case of C++ references, we want to differentiate between a null
    // reference and reference to null pointer.
    // If the LVal is null, check if we are dealing with null reference.
    // For those, we want to track the location of the reference.
    const MemRegion *R = (RR && LVIsNull) ? RR :
        LVNode->getSVal(Inner).getAsRegion();

    if (R) {

      // Mark both the variable region and its contents as interesting.
      SVal V = LVState->getRawSVal(loc::MemRegionVal(R));
      report.addVisitor(
          std::make_unique<NoStoreFuncVisitor>(cast<SubRegion>(R), TKind));

      MacroNullReturnSuppressionVisitor::addMacroVisitorIfNecessary(
          LVNode, R, EnableNullFPSuppression, report, V);

      report.markInteresting(V, TKind);
      report.addVisitor(std::make_unique<UndefOrNullArgVisitor>(R));

      // If the contents are symbolic and null, find out when they became null.
      if (V.getAsLocSymbol(/*IncludeBaseRegions=*/true))
        if (LVState->isNull(V).isConstrainedTrue())
          report.addVisitor(std::make_unique<TrackConstraintBRVisitor>(
              V.castAs<DefinedSVal>(), false));

      // Add visitor, which will suppress inline defensive checks.
      if (auto DV = V.getAs<DefinedSVal>())
        if (!DV->isZeroConstant() && EnableNullFPSuppression) {
          // Note that LVNode may be too late (i.e., too far from the InputNode)
          // because the lvalue may have been computed before the inlined call
          // was evaluated. InputNode may as well be too early here, because
          // the symbol is already dead; this, however, is fine because we can
          // still find the node in which it collapsed to null previously.
          report.addVisitor(
              std::make_unique<SuppressInlineDefensiveChecksVisitor>(
                  *DV, InputNode));
        }

      if (auto KV = V.getAs<KnownSVal>())
        report.addVisitor(std::make_unique<FindLastStoreBRVisitor>(
            *KV, R, EnableNullFPSuppression, TKind, SFC));
      return true;
    }
  }

  // If the expression is not an "lvalue expression", we can still
  // track the constraints on its contents.
  SVal V = LVState->getSValAsScalarOrLoc(Inner, LVNode->getLocationContext());

  ReturnVisitor::addVisitorIfNecessary(
    LVNode, Inner, report, EnableNullFPSuppression, TKind);

  // Is it a symbolic value?
  if (auto L = V.getAs<loc::MemRegionVal>()) {
    // FIXME: this is a hack for fixing a later crash when attempting to
    // dereference a void* pointer.
    // We should not try to dereference pointers at all when we don't care
    // what is written inside the pointer.
    bool CanDereference = true;
    if (const auto *SR = L->getRegionAs<SymbolicRegion>()) {
      if (SR->getSymbol()->getType()->getPointeeType()->isVoidType())
        CanDereference = false;
    } else if (L->getRegionAs<AllocaRegion>())
      CanDereference = false;

    // At this point we are dealing with the region's LValue.
    // However, if the rvalue is a symbolic region, we should track it as well.
    // Try to use the correct type when looking up the value.
    SVal RVal;
    if (ExplodedGraph::isInterestingLValueExpr(Inner))
      RVal = LVState->getRawSVal(L.getValue(), Inner->getType());
    else if (CanDereference)
      RVal = LVState->getSVal(L->getRegion());

    if (CanDereference) {
      report.addVisitor(
          std::make_unique<UndefOrNullArgVisitor>(L->getRegion()));

      if (auto KV = RVal.getAs<KnownSVal>())
        report.addVisitor(std::make_unique<FindLastStoreBRVisitor>(
            *KV, L->getRegion(), EnableNullFPSuppression, TKind, SFC));
    }

    const MemRegion *RegionRVal = RVal.getAsRegion();
    if (RegionRVal && isa<SymbolicRegion>(RegionRVal)) {
      report.markInteresting(RegionRVal, TKind);
      report.addVisitor(std::make_unique<TrackConstraintBRVisitor>(
            loc::MemRegionVal(RegionRVal), /*assumption=*/false));
    }
  }

  if (Inner->isRValue())
    trackRValueExpression(LVNode, Inner, report, TKind,
                          EnableNullFPSuppression);

  return true;
}

//===----------------------------------------------------------------------===//
// Implementation of NulReceiverBRVisitor.
//===----------------------------------------------------------------------===//

const Expr *NilReceiverBRVisitor::getNilReceiver(const Stmt *S,
                                                 const ExplodedNode *N) {
  const auto *ME = dyn_cast<ObjCMessageExpr>(S);
  if (!ME)
    return nullptr;
  if (const Expr *Receiver = ME->getInstanceReceiver()) {
    ProgramStateRef state = N->getState();
    SVal V = N->getSVal(Receiver);
    if (state->isNull(V).isConstrainedTrue())
      return Receiver;
  }
  return nullptr;
}

PathDiagnosticPieceRef
NilReceiverBRVisitor::VisitNode(const ExplodedNode *N, BugReporterContext &BRC,
                                PathSensitiveBugReport &BR) {
  Optional<PreStmt> P = N->getLocationAs<PreStmt>();
  if (!P)
    return nullptr;

  const Stmt *S = P->getStmt();
  const Expr *Receiver = getNilReceiver(S, N);
  if (!Receiver)
    return nullptr;

  llvm::SmallString<256> Buf;
  llvm::raw_svector_ostream OS(Buf);

  if (const auto *ME = dyn_cast<ObjCMessageExpr>(S)) {
    OS << "'";
    ME->getSelector().print(OS);
    OS << "' not called";
  }
  else {
    OS << "No method is called";
  }
  OS << " because the receiver is nil";

  // The receiver was nil, and hence the method was skipped.
  // Register a BugReporterVisitor to issue a message telling us how
  // the receiver was null.
  bugreporter::trackExpressionValue(
      N, Receiver, BR, bugreporter::TrackingKind::Thorough,
      /*EnableNullFPSuppression*/ false);
  // Issue a message saying that the method was skipped.
  PathDiagnosticLocation L(Receiver, BRC.getSourceManager(),
                                     N->getLocationContext());
  return std::make_shared<PathDiagnosticEventPiece>(L, OS.str());
}

//===----------------------------------------------------------------------===//
// Visitor that tries to report interesting diagnostics from conditions.
//===----------------------------------------------------------------------===//

/// Return the tag associated with this visitor.  This tag will be used
/// to make all PathDiagnosticPieces created by this visitor.
const char *ConditionBRVisitor::getTag() { return "ConditionBRVisitor"; }

PathDiagnosticPieceRef
ConditionBRVisitor::VisitNode(const ExplodedNode *N, BugReporterContext &BRC,
                              PathSensitiveBugReport &BR) {
  auto piece = VisitNodeImpl(N, BRC, BR);
  if (piece) {
    piece->setTag(getTag());
    if (auto *ev = dyn_cast<PathDiagnosticEventPiece>(piece.get()))
      ev->setPrunable(true, /* override */ false);
  }
  return piece;
}

PathDiagnosticPieceRef
ConditionBRVisitor::VisitNodeImpl(const ExplodedNode *N,
                                  BugReporterContext &BRC,
                                  PathSensitiveBugReport &BR) {
  ProgramPoint ProgPoint = N->getLocation();
  const std::pair<const ProgramPointTag *, const ProgramPointTag *> &Tags =
      ExprEngine::geteagerlyAssumeBinOpBifurcationTags();

  // If an assumption was made on a branch, it should be caught
  // here by looking at the state transition.
  if (Optional<BlockEdge> BE = ProgPoint.getAs<BlockEdge>()) {
    const CFGBlock *SrcBlock = BE->getSrc();
    if (const Stmt *Term = SrcBlock->getTerminatorStmt()) {
      // If the tag of the previous node is 'Eagerly Assume...' the current
      // 'BlockEdge' has the same constraint information. We do not want to
      // report the value as it is just an assumption on the predecessor node
      // which will be caught in the next VisitNode() iteration as a 'PostStmt'.
      const ProgramPointTag *PreviousNodeTag =
          N->getFirstPred()->getLocation().getTag();
      if (PreviousNodeTag == Tags.first || PreviousNodeTag == Tags.second)
        return nullptr;

      return VisitTerminator(Term, N, SrcBlock, BE->getDst(), BR, BRC);
    }
    return nullptr;
  }

  if (Optional<PostStmt> PS = ProgPoint.getAs<PostStmt>()) {
    const ProgramPointTag *CurrentNodeTag = PS->getTag();
    if (CurrentNodeTag != Tags.first && CurrentNodeTag != Tags.second)
      return nullptr;

    bool TookTrue = CurrentNodeTag == Tags.first;
    return VisitTrueTest(cast<Expr>(PS->getStmt()), BRC, BR, N, TookTrue);
  }

  return nullptr;
}

PathDiagnosticPieceRef ConditionBRVisitor::VisitTerminator(
    const Stmt *Term, const ExplodedNode *N, const CFGBlock *srcBlk,
    const CFGBlock *dstBlk, PathSensitiveBugReport &R,
    BugReporterContext &BRC) {
  const Expr *Cond = nullptr;

  // In the code below, Term is a CFG terminator and Cond is a branch condition
  // expression upon which the decision is made on this terminator.
  //
  // For example, in "if (x == 0)", the "if (x == 0)" statement is a terminator,
  // and "x == 0" is the respective condition.
  //
  // Another example: in "if (x && y)", we've got two terminators and two
  // conditions due to short-circuit nature of operator "&&":
  // 1. The "if (x && y)" statement is a terminator,
  //    and "y" is the respective condition.
  // 2. Also "x && ..." is another terminator,
  //    and "x" is its condition.

  switch (Term->getStmtClass()) {
  // FIXME: Stmt::SwitchStmtClass is worth handling, however it is a bit
  // more tricky because there are more than two branches to account for.
  default:
    return nullptr;
  case Stmt::IfStmtClass:
    Cond = cast<IfStmt>(Term)->getCond();
    break;
  case Stmt::ConditionalOperatorClass:
    Cond = cast<ConditionalOperator>(Term)->getCond();
    break;
  case Stmt::BinaryOperatorClass:
    // When we encounter a logical operator (&& or ||) as a CFG terminator,
    // then the condition is actually its LHS; otherwise, we'd encounter
    // the parent, such as if-statement, as a terminator.
    const auto *BO = cast<BinaryOperator>(Term);
    assert(BO->isLogicalOp() &&
           "CFG terminator is not a short-circuit operator!");
    Cond = BO->getLHS();
    break;
  }

  Cond = Cond->IgnoreParens();

  // However, when we encounter a logical operator as a branch condition,
  // then the condition is actually its RHS, because LHS would be
  // the condition for the logical operator terminator.
  while (const auto *InnerBO = dyn_cast<BinaryOperator>(Cond)) {
    if (!InnerBO->isLogicalOp())
      break;
    Cond = InnerBO->getRHS()->IgnoreParens();
  }

  assert(Cond);
  assert(srcBlk->succ_size() == 2);
  const bool TookTrue = *(srcBlk->succ_begin()) == dstBlk;
  return VisitTrueTest(Cond, BRC, R, N, TookTrue);
}

PathDiagnosticPieceRef
ConditionBRVisitor::VisitTrueTest(const Expr *Cond, BugReporterContext &BRC,
                                  PathSensitiveBugReport &R,
                                  const ExplodedNode *N, bool TookTrue) {
  ProgramStateRef CurrentState = N->getState();
  ProgramStateRef PrevState = N->getFirstPred()->getState();
  const LocationContext *LCtx = N->getLocationContext();

  // If the constraint information is changed between the current and the
  // previous program state we assuming the newly seen constraint information.
  // If we cannot evaluate the condition (and the constraints are the same)
  // the analyzer has no information about the value and just assuming it.
  bool IsAssuming =
      !BRC.getStateManager().haveEqualConstraints(CurrentState, PrevState) ||
      CurrentState->getSVal(Cond, LCtx).isUnknownOrUndef();

  // These will be modified in code below, but we need to preserve the original
  //  values in case we want to throw the generic message.
  const Expr *CondTmp = Cond;
  bool TookTrueTmp = TookTrue;

  while (true) {
    CondTmp = CondTmp->IgnoreParenCasts();
    switch (CondTmp->getStmtClass()) {
      default:
        break;
      case Stmt::BinaryOperatorClass:
        if (auto P = VisitTrueTest(Cond, cast<BinaryOperator>(CondTmp),
                                   BRC, R, N, TookTrueTmp, IsAssuming))
          return P;
        break;
      case Stmt::DeclRefExprClass:
        if (auto P = VisitTrueTest(Cond, cast<DeclRefExpr>(CondTmp),
                                   BRC, R, N, TookTrueTmp, IsAssuming))
          return P;
        break;
      case Stmt::MemberExprClass:
        if (auto P = VisitTrueTest(Cond, cast<MemberExpr>(CondTmp),
                                   BRC, R, N, TookTrueTmp, IsAssuming))
          return P;
        break;
      case Stmt::UnaryOperatorClass: {
        const auto *UO = cast<UnaryOperator>(CondTmp);
        if (UO->getOpcode() == UO_LNot) {
          TookTrueTmp = !TookTrueTmp;
          CondTmp = UO->getSubExpr();
          continue;
        }
        break;
      }
    }
    break;
  }

  // Condition too complex to explain? Just say something so that the user
  // knew we've made some path decision at this point.
  // If it is too complex and we know the evaluation of the condition do not
  // repeat the note from 'BugReporter.cpp'
  if (!IsAssuming)
    return nullptr;

  PathDiagnosticLocation Loc(Cond, BRC.getSourceManager(), LCtx);
  if (!Loc.isValid() || !Loc.asLocation().isValid())
    return nullptr;

  return std::make_shared<PathDiagnosticEventPiece>(
      Loc, TookTrue ? GenericTrueMessage : GenericFalseMessage);
}

bool ConditionBRVisitor::patternMatch(const Expr *Ex,
                                      const Expr *ParentEx,
                                      raw_ostream &Out,
                                      BugReporterContext &BRC,
                                      PathSensitiveBugReport &report,
                                      const ExplodedNode *N,
                                      Optional<bool> &prunable,
                                      bool IsSameFieldName) {
  const Expr *OriginalExpr = Ex;
  Ex = Ex->IgnoreParenCasts();

  if (isa<GNUNullExpr>(Ex) || isa<ObjCBoolLiteralExpr>(Ex) ||
      isa<CXXBoolLiteralExpr>(Ex) || isa<IntegerLiteral>(Ex) ||
      isa<FloatingLiteral>(Ex)) {
    // Use heuristics to determine if the expression is a macro
    // expanding to a literal and if so, use the macro's name.
    SourceLocation BeginLoc = OriginalExpr->getBeginLoc();
    SourceLocation EndLoc = OriginalExpr->getEndLoc();
    if (BeginLoc.isMacroID() && EndLoc.isMacroID()) {
      const SourceManager &SM = BRC.getSourceManager();
      const LangOptions &LO = BRC.getASTContext().getLangOpts();
      if (Lexer::isAtStartOfMacroExpansion(BeginLoc, SM, LO) &&
          Lexer::isAtEndOfMacroExpansion(EndLoc, SM, LO)) {
        CharSourceRange R = Lexer::getAsCharRange({BeginLoc, EndLoc}, SM, LO);
        Out << Lexer::getSourceText(R, SM, LO);
        return false;
      }
    }
  }

  if (const auto *DR = dyn_cast<DeclRefExpr>(Ex)) {
    const bool quotes = isa<VarDecl>(DR->getDecl());
    if (quotes) {
      Out << '\'';
      const LocationContext *LCtx = N->getLocationContext();
      const ProgramState *state = N->getState().get();
      if (const MemRegion *R = state->getLValue(cast<VarDecl>(DR->getDecl()),
                                                LCtx).getAsRegion()) {
        if (report.isInteresting(R))
          prunable = false;
        else {
          const ProgramState *state = N->getState().get();
          SVal V = state->getSVal(R);
          if (report.isInteresting(V))
            prunable = false;
        }
      }
    }
    Out << DR->getDecl()->getDeclName().getAsString();
    if (quotes)
      Out << '\'';
    return quotes;
  }

  if (const auto *IL = dyn_cast<IntegerLiteral>(Ex)) {
    QualType OriginalTy = OriginalExpr->getType();
    if (OriginalTy->isPointerType()) {
      if (IL->getValue() == 0) {
        Out << "null";
        return false;
      }
    }
    else if (OriginalTy->isObjCObjectPointerType()) {
      if (IL->getValue() == 0) {
        Out << "nil";
        return false;
      }
    }

    Out << IL->getValue();
    return false;
  }

  if (const auto *ME = dyn_cast<MemberExpr>(Ex)) {
    if (!IsSameFieldName)
      Out << "field '" << ME->getMemberDecl()->getName() << '\'';
    else
      Out << '\''
          << Lexer::getSourceText(
                 CharSourceRange::getTokenRange(Ex->getSourceRange()),
                 BRC.getSourceManager(), BRC.getASTContext().getLangOpts(), 0)
          << '\'';
  }

  return false;
}

PathDiagnosticPieceRef ConditionBRVisitor::VisitTrueTest(
    const Expr *Cond, const BinaryOperator *BExpr, BugReporterContext &BRC,
    PathSensitiveBugReport &R, const ExplodedNode *N, bool TookTrue,
    bool IsAssuming) {
  bool shouldInvert = false;
  Optional<bool> shouldPrune;

  // Check if the field name of the MemberExprs is ambiguous. Example:
  // " 'a.d' is equal to 'h.d' " in 'test/Analysis/null-deref-path-notes.cpp'.
  bool IsSameFieldName = false;
  const auto *LhsME = dyn_cast<MemberExpr>(BExpr->getLHS()->IgnoreParenCasts());
  const auto *RhsME = dyn_cast<MemberExpr>(BExpr->getRHS()->IgnoreParenCasts());

  if (LhsME && RhsME)
    IsSameFieldName =
        LhsME->getMemberDecl()->getName() == RhsME->getMemberDecl()->getName();

  SmallString<128> LhsString, RhsString;
  {
    llvm::raw_svector_ostream OutLHS(LhsString), OutRHS(RhsString);
    const bool isVarLHS = patternMatch(BExpr->getLHS(), BExpr, OutLHS, BRC, R,
                                       N, shouldPrune, IsSameFieldName);
    const bool isVarRHS = patternMatch(BExpr->getRHS(), BExpr, OutRHS, BRC, R,
                                       N, shouldPrune, IsSameFieldName);

    shouldInvert = !isVarLHS && isVarRHS;
  }

  BinaryOperator::Opcode Op = BExpr->getOpcode();

  if (BinaryOperator::isAssignmentOp(Op)) {
    // For assignment operators, all that we care about is that the LHS
    // evaluates to "true" or "false".
    return VisitConditionVariable(LhsString, BExpr->getLHS(), BRC, R, N,
                                  TookTrue);
  }

  // For non-assignment operations, we require that we can understand
  // both the LHS and RHS.
  if (LhsString.empty() || RhsString.empty() ||
      !BinaryOperator::isComparisonOp(Op) || Op == BO_Cmp)
    return nullptr;

  // Should we invert the strings if the LHS is not a variable name?
  SmallString<256> buf;
  llvm::raw_svector_ostream Out(buf);
  Out << (IsAssuming ? "Assuming " : "")
      << (shouldInvert ? RhsString : LhsString) << " is ";

  // Do we need to invert the opcode?
  if (shouldInvert)
    switch (Op) {
      default: break;
      case BO_LT: Op = BO_GT; break;
      case BO_GT: Op = BO_LT; break;
      case BO_LE: Op = BO_GE; break;
      case BO_GE: Op = BO_LE; break;
    }

  if (!TookTrue)
    switch (Op) {
      case BO_EQ: Op = BO_NE; break;
      case BO_NE: Op = BO_EQ; break;
      case BO_LT: Op = BO_GE; break;
      case BO_GT: Op = BO_LE; break;
      case BO_LE: Op = BO_GT; break;
      case BO_GE: Op = BO_LT; break;
      default:
        return nullptr;
    }

  switch (Op) {
    case BO_EQ:
      Out << "equal to ";
      break;
    case BO_NE:
      Out << "not equal to ";
      break;
    default:
      Out << BinaryOperator::getOpcodeStr(Op) << ' ';
      break;
  }

  Out << (shouldInvert ? LhsString : RhsString);
  const LocationContext *LCtx = N->getLocationContext();
  const SourceManager &SM = BRC.getSourceManager();

  if (isVarAnInterestingCondition(BExpr->getLHS(), N, &R) ||
      isVarAnInterestingCondition(BExpr->getRHS(), N, &R))
    Out << WillBeUsedForACondition;

  // Convert 'field ...' to 'Field ...' if it is a MemberExpr.
  std::string Message = std::string(Out.str());
  Message[0] = toupper(Message[0]);

  // If we know the value create a pop-up note to the value part of 'BExpr'.
  if (!IsAssuming) {
    PathDiagnosticLocation Loc;
    if (!shouldInvert) {
      if (LhsME && LhsME->getMemberLoc().isValid())
        Loc = PathDiagnosticLocation(LhsME->getMemberLoc(), SM);
      else
        Loc = PathDiagnosticLocation(BExpr->getLHS(), SM, LCtx);
    } else {
      if (RhsME && RhsME->getMemberLoc().isValid())
        Loc = PathDiagnosticLocation(RhsME->getMemberLoc(), SM);
      else
        Loc = PathDiagnosticLocation(BExpr->getRHS(), SM, LCtx);
    }

    return std::make_shared<PathDiagnosticPopUpPiece>(Loc, Message);
  }

  PathDiagnosticLocation Loc(Cond, SM, LCtx);
  auto event = std::make_shared<PathDiagnosticEventPiece>(Loc, Message);
  if (shouldPrune.hasValue())
    event->setPrunable(shouldPrune.getValue());
  return event;
}

PathDiagnosticPieceRef ConditionBRVisitor::VisitConditionVariable(
    StringRef LhsString, const Expr *CondVarExpr, BugReporterContext &BRC,
    PathSensitiveBugReport &report, const ExplodedNode *N, bool TookTrue) {
  // FIXME: If there's already a constraint tracker for this variable,
  // we shouldn't emit anything here (c.f. the double note in
  // test/Analysis/inlining/path-notes.c)
  SmallString<256> buf;
  llvm::raw_svector_ostream Out(buf);
  Out << "Assuming " << LhsString << " is ";

  if (!printValue(CondVarExpr, Out, N, TookTrue, /*IsAssuming=*/true))
    return nullptr;

  const LocationContext *LCtx = N->getLocationContext();
  PathDiagnosticLocation Loc(CondVarExpr, BRC.getSourceManager(), LCtx);

  if (isVarAnInterestingCondition(CondVarExpr, N, &report))
    Out << WillBeUsedForACondition;

  auto event = std::make_shared<PathDiagnosticEventPiece>(Loc, Out.str());

  if (isInterestingExpr(CondVarExpr, N, &report))
    event->setPrunable(false);

  return event;
}

PathDiagnosticPieceRef ConditionBRVisitor::VisitTrueTest(
    const Expr *Cond, const DeclRefExpr *DRE, BugReporterContext &BRC,
    PathSensitiveBugReport &report, const ExplodedNode *N, bool TookTrue,
    bool IsAssuming) {
  const auto *VD = dyn_cast<VarDecl>(DRE->getDecl());
  if (!VD)
    return nullptr;

  SmallString<256> Buf;
  llvm::raw_svector_ostream Out(Buf);

  Out << (IsAssuming ? "Assuming '" : "'") << VD->getDeclName() << "' is ";

  if (!printValue(DRE, Out, N, TookTrue, IsAssuming))
    return nullptr;

  const LocationContext *LCtx = N->getLocationContext();

  if (isVarAnInterestingCondition(DRE, N, &report))
    Out << WillBeUsedForACondition;

  // If we know the value create a pop-up note to the 'DRE'.
  if (!IsAssuming) {
    PathDiagnosticLocation Loc(DRE, BRC.getSourceManager(), LCtx);
    return std::make_shared<PathDiagnosticPopUpPiece>(Loc, Out.str());
  }

  PathDiagnosticLocation Loc(Cond, BRC.getSourceManager(), LCtx);
  auto event = std::make_shared<PathDiagnosticEventPiece>(Loc, Out.str());

  if (isInterestingExpr(DRE, N, &report))
    event->setPrunable(false);

  return std::move(event);
}

PathDiagnosticPieceRef ConditionBRVisitor::VisitTrueTest(
    const Expr *Cond, const MemberExpr *ME, BugReporterContext &BRC,
    PathSensitiveBugReport &report, const ExplodedNode *N, bool TookTrue,
    bool IsAssuming) {
  SmallString<256> Buf;
  llvm::raw_svector_ostream Out(Buf);

  Out << (IsAssuming ? "Assuming field '" : "Field '")
      << ME->getMemberDecl()->getName() << "' is ";

  if (!printValue(ME, Out, N, TookTrue, IsAssuming))
    return nullptr;

  const LocationContext *LCtx = N->getLocationContext();
  PathDiagnosticLocation Loc;

  // If we know the value create a pop-up note to the member of the MemberExpr.
  if (!IsAssuming && ME->getMemberLoc().isValid())
    Loc = PathDiagnosticLocation(ME->getMemberLoc(), BRC.getSourceManager());
  else
    Loc = PathDiagnosticLocation(Cond, BRC.getSourceManager(), LCtx);

  if (!Loc.isValid() || !Loc.asLocation().isValid())
    return nullptr;

  if (isVarAnInterestingCondition(ME, N, &report))
    Out << WillBeUsedForACondition;

  // If we know the value create a pop-up note.
  if (!IsAssuming)
    return std::make_shared<PathDiagnosticPopUpPiece>(Loc, Out.str());

  auto event = std::make_shared<PathDiagnosticEventPiece>(Loc, Out.str());
  if (isInterestingExpr(ME, N, &report))
    event->setPrunable(false);
  return event;
}

bool ConditionBRVisitor::printValue(const Expr *CondVarExpr, raw_ostream &Out,
                                    const ExplodedNode *N, bool TookTrue,
                                    bool IsAssuming) {
  QualType Ty = CondVarExpr->getType();

  if (Ty->isPointerType()) {
    Out << (TookTrue ? "non-null" : "null");
    return true;
  }

  if (Ty->isObjCObjectPointerType()) {
    Out << (TookTrue ? "non-nil" : "nil");
    return true;
  }

  if (!Ty->isIntegralOrEnumerationType())
    return false;

  Optional<const llvm::APSInt *> IntValue;
  if (!IsAssuming)
    IntValue = getConcreteIntegerValue(CondVarExpr, N);

  if (IsAssuming || !IntValue.hasValue()) {
    if (Ty->isBooleanType())
      Out << (TookTrue ? "true" : "false");
    else
      Out << (TookTrue ? "not equal to 0" : "0");
  } else {
    if (Ty->isBooleanType())
      Out << (IntValue.getValue()->getBoolValue() ? "true" : "false");
    else
      Out << *IntValue.getValue();
  }

  return true;
}

constexpr llvm::StringLiteral ConditionBRVisitor::GenericTrueMessage;
constexpr llvm::StringLiteral ConditionBRVisitor::GenericFalseMessage;

bool ConditionBRVisitor::isPieceMessageGeneric(
    const PathDiagnosticPiece *Piece) {
  return Piece->getString() == GenericTrueMessage ||
         Piece->getString() == GenericFalseMessage;
}

//===----------------------------------------------------------------------===//
// Implementation of LikelyFalsePositiveSuppressionBRVisitor.
//===----------------------------------------------------------------------===//

void LikelyFalsePositiveSuppressionBRVisitor::finalizeVisitor(
    BugReporterContext &BRC, const ExplodedNode *N,
    PathSensitiveBugReport &BR) {
  // Here we suppress false positives coming from system headers. This list is
  // based on known issues.
  const AnalyzerOptions &Options = BRC.getAnalyzerOptions();
  const Decl *D = N->getLocationContext()->getDecl();

  if (AnalysisDeclContext::isInStdNamespace(D)) {
    // Skip reports within the 'std' namespace. Although these can sometimes be
    // the user's fault, we currently don't report them very well, and
    // Note that this will not help for any other data structure libraries, like
    // TR1, Boost, or llvm/ADT.
    if (Options.ShouldSuppressFromCXXStandardLibrary) {
      BR.markInvalid(getTag(), nullptr);
      return;
    } else {
      // If the complete 'std' suppression is not enabled, suppress reports
      // from the 'std' namespace that are known to produce false positives.

      // The analyzer issues a false use-after-free when std::list::pop_front
      // or std::list::pop_back are called multiple times because we cannot
      // reason about the internal invariants of the data structure.
      if (const auto *MD = dyn_cast<CXXMethodDecl>(D)) {
        const CXXRecordDecl *CD = MD->getParent();
        if (CD->getName() == "list") {
          BR.markInvalid(getTag(), nullptr);
          return;
        }
      }

      // The analyzer issues a false positive when the constructor of
      // std::__independent_bits_engine from algorithms is used.
      if (const auto *MD = dyn_cast<CXXConstructorDecl>(D)) {
        const CXXRecordDecl *CD = MD->getParent();
        if (CD->getName() == "__independent_bits_engine") {
          BR.markInvalid(getTag(), nullptr);
          return;
        }
      }

      for (const LocationContext *LCtx = N->getLocationContext(); LCtx;
           LCtx = LCtx->getParent()) {
        const auto *MD = dyn_cast<CXXMethodDecl>(LCtx->getDecl());
        if (!MD)
          continue;

        const CXXRecordDecl *CD = MD->getParent();
        // The analyzer issues a false positive on
        //   std::basic_string<uint8_t> v; v.push_back(1);
        // and
        //   std::u16string s; s += u'a';
        // because we cannot reason about the internal invariants of the
        // data structure.
        if (CD->getName() == "basic_string") {
          BR.markInvalid(getTag(), nullptr);
          return;
        }

        // The analyzer issues a false positive on
        //    std::shared_ptr<int> p(new int(1)); p = nullptr;
        // because it does not reason properly about temporary destructors.
        if (CD->getName() == "shared_ptr") {
          BR.markInvalid(getTag(), nullptr);
          return;
        }
      }
    }
  }

  // Skip reports within the sys/queue.h macros as we do not have the ability to
  // reason about data structure shapes.
  const SourceManager &SM = BRC.getSourceManager();
  FullSourceLoc Loc = BR.getLocation().asLocation();
  while (Loc.isMacroID()) {
    Loc = Loc.getSpellingLoc();
    if (SM.getFilename(Loc).endswith("sys/queue.h")) {
      BR.markInvalid(getTag(), nullptr);
      return;
    }
  }
}

//===----------------------------------------------------------------------===//
// Implementation of UndefOrNullArgVisitor.
//===----------------------------------------------------------------------===//

PathDiagnosticPieceRef
UndefOrNullArgVisitor::VisitNode(const ExplodedNode *N, BugReporterContext &BRC,
                                 PathSensitiveBugReport &BR) {
  ProgramStateRef State = N->getState();
  ProgramPoint ProgLoc = N->getLocation();

  // We are only interested in visiting CallEnter nodes.
  Optional<CallEnter> CEnter = ProgLoc.getAs<CallEnter>();
  if (!CEnter)
    return nullptr;

  // Check if one of the arguments is the region the visitor is tracking.
  CallEventManager &CEMgr = BRC.getStateManager().getCallEventManager();
  CallEventRef<> Call = CEMgr.getCaller(CEnter->getCalleeContext(), State);
  unsigned Idx = 0;
  ArrayRef<ParmVarDecl *> parms = Call->parameters();

  for (const auto ParamDecl : parms) {
    const MemRegion *ArgReg = Call->getArgSVal(Idx).getAsRegion();
    ++Idx;

    // Are we tracking the argument or its subregion?
    if ( !ArgReg || !R->isSubRegionOf(ArgReg->StripCasts()))
      continue;

    // Check the function parameter type.
    assert(ParamDecl && "Formal parameter has no decl?");
    QualType T = ParamDecl->getType();

    if (!(T->isAnyPointerType() || T->isReferenceType())) {
      // Function can only change the value passed in by address.
      continue;
    }

    // If it is a const pointer value, the function does not intend to
    // change the value.
    if (T->getPointeeType().isConstQualified())
      continue;

    // Mark the call site (LocationContext) as interesting if the value of the
    // argument is undefined or '0'/'NULL'.
    SVal BoundVal = State->getSVal(R);
    if (BoundVal.isUndef() || BoundVal.isZeroConstant()) {
      BR.markInteresting(CEnter->getCalleeContext());
      return nullptr;
    }
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Implementation of FalsePositiveRefutationBRVisitor.
//===----------------------------------------------------------------------===//

FalsePositiveRefutationBRVisitor::FalsePositiveRefutationBRVisitor()
    : Constraints(ConstraintMap::Factory().getEmptyMap()) {}

void FalsePositiveRefutationBRVisitor::finalizeVisitor(
    BugReporterContext &BRC, const ExplodedNode *EndPathNode,
    PathSensitiveBugReport &BR) {
  // Collect new constraints
  addConstraints(EndPathNode, /*OverwriteConstraintsOnExistingSyms=*/true);

  // Create a refutation manager
  llvm::SMTSolverRef RefutationSolver = llvm::CreateZ3Solver();
  ASTContext &Ctx = BRC.getASTContext();

  // Add constraints to the solver
  for (const auto &I : Constraints) {
    const SymbolRef Sym = I.first;
    auto RangeIt = I.second.begin();

    llvm::SMTExprRef SMTConstraints = SMTConv::getRangeExpr(
        RefutationSolver, Ctx, Sym, RangeIt->From(), RangeIt->To(),
        /*InRange=*/true);
    while ((++RangeIt) != I.second.end()) {
      SMTConstraints = RefutationSolver->mkOr(
          SMTConstraints, SMTConv::getRangeExpr(RefutationSolver, Ctx, Sym,
                                                RangeIt->From(), RangeIt->To(),
                                                /*InRange=*/true));
    }

    RefutationSolver->addConstraint(SMTConstraints);
  }

  // And check for satisfiability
  Optional<bool> IsSAT = RefutationSolver->check();
  if (!IsSAT.hasValue())
    return;

  if (!IsSAT.getValue())
    BR.markInvalid("Infeasible constraints", EndPathNode->getLocationContext());
}

void FalsePositiveRefutationBRVisitor::addConstraints(
    const ExplodedNode *N, bool OverwriteConstraintsOnExistingSyms) {
  // Collect new constraints
  ConstraintMap NewCs = getConstraintMap(N->getState());
  ConstraintMap::Factory &CF = N->getState()->get_context<ConstraintMap>();

  // Add constraints if we don't have them yet
  for (auto const &C : NewCs) {
    const SymbolRef &Sym = C.first;
    if (!Constraints.contains(Sym)) {
      // This symbol is new, just add the constraint.
      Constraints = CF.add(Constraints, Sym, C.second);
    } else if (OverwriteConstraintsOnExistingSyms) {
      // Overwrite the associated constraint of the Symbol.
      Constraints = CF.remove(Constraints, Sym);
      Constraints = CF.add(Constraints, Sym, C.second);
    }
  }
}

PathDiagnosticPieceRef FalsePositiveRefutationBRVisitor::VisitNode(
    const ExplodedNode *N, BugReporterContext &, PathSensitiveBugReport &) {
  addConstraints(N, /*OverwriteConstraintsOnExistingSyms=*/false);
  return nullptr;
}

void FalsePositiveRefutationBRVisitor::Profile(
    llvm::FoldingSetNodeID &ID) const {
  static int Tag = 0;
  ID.AddPointer(&Tag);
}

//===----------------------------------------------------------------------===//
// Implementation of TagVisitor.
//===----------------------------------------------------------------------===//

int NoteTag::Kind = 0;

void TagVisitor::Profile(llvm::FoldingSetNodeID &ID) const {
  static int Tag = 0;
  ID.AddPointer(&Tag);
}

PathDiagnosticPieceRef TagVisitor::VisitNode(const ExplodedNode *N,
                                             BugReporterContext &BRC,
                                             PathSensitiveBugReport &R) {
  ProgramPoint PP = N->getLocation();
  const NoteTag *T = dyn_cast_or_null<NoteTag>(PP.getTag());
  if (!T)
    return nullptr;

  if (Optional<std::string> Msg = T->generateMessage(BRC, R)) {
    PathDiagnosticLocation Loc =
        PathDiagnosticLocation::create(PP, BRC.getSourceManager());
    auto Piece = std::make_shared<PathDiagnosticEventPiece>(Loc, *Msg);
    Piece->setPrunable(T->isPrunable());
    return Piece;
  }

  return nullptr;
}
