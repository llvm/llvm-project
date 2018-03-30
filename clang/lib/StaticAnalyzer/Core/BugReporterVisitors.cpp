//===- BugReporterVisitors.cpp - Helpers for reporting bugs ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/CFGStmtMap.h"
#include "clang/Analysis/ProgramPoint.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/StaticAnalyzer/Core/AnalyzerOptions.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/PathDiagnostic.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExplodedGraph.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState_Fwd.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SValBuilder.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SubEngine.h"
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

bool bugreporter::isDeclRefExprToReference(const Expr *E) {
  if (const auto *DRE = dyn_cast<DeclRefExpr>(E))
    return DRE->getDecl()->getType()->isReferenceType();
  return false;
}

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

const Stmt *bugreporter::GetDenomExpr(const ExplodedNode *N) {
  const Stmt *S = N->getLocationAs<PreStmt>()->getStmt();
  if (const auto *BE = dyn_cast<BinaryOperator>(S))
    return BE->getRHS();
  return nullptr;
}

const Stmt *bugreporter::GetRetValExpr(const ExplodedNode *N) {
  const Stmt *S = N->getLocationAs<PostStmt>()->getStmt();
  if (const auto *RS = dyn_cast<ReturnStmt>(S))
    return RS->getRetValue();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Definitions for bug reporter visitors.
//===----------------------------------------------------------------------===//

std::unique_ptr<PathDiagnosticPiece>
BugReporterVisitor::getEndPath(BugReporterContext &BRC,
                               const ExplodedNode *EndPathNode, BugReport &BR) {
  return nullptr;
}

std::unique_ptr<PathDiagnosticPiece> BugReporterVisitor::getDefaultEndPath(
    BugReporterContext &BRC, const ExplodedNode *EndPathNode, BugReport &BR) {
  PathDiagnosticLocation L =
    PathDiagnosticLocation::createEndOfPath(EndPathNode,BRC.getSourceManager());

  const auto &Ranges = BR.getRanges();

  // Only add the statement itself as a range if we didn't specify any
  // special ranges for this report.
  auto P = llvm::make_unique<PathDiagnosticEventPiece>(
      L, BR.getDescription(), Ranges.begin() == Ranges.end());
  for (SourceRange Range : Ranges)
    P->addRange(Range);

  return std::move(P);
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
    Loc = SM.getImmediateExpansionRange(Loc).first;
  std::pair<FileID, unsigned> TLInfo = SM.getDecomposedLoc(Loc);
  SrcMgr::SLocEntry SE = SM.getSLocEntry(TLInfo.first);
  const SrcMgr::ExpansionInfo &EInfo = SE.getExpansion();
  return EInfo.isFunctionMacroExpansion();
}

namespace {

/// Put a diagnostic on return statement of all inlined functions
/// for which  the region of interest \p RegionOfInterest was passed into,
/// but not written inside, and it has caused an undefined read or a null
/// pointer dereference outside.
class NoStoreFuncVisitor final
    : public BugReporterVisitorImpl<NoStoreFuncVisitor> {
  const SubRegion *RegionOfInterest;
  static constexpr const char *DiagnosticsMsg =
      "Returning without writing to '";

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

public:
  NoStoreFuncVisitor(const SubRegion *R) : RegionOfInterest(R) {}

  void Profile(llvm::FoldingSetNodeID &ID) const override {
    static int Tag = 0;
    ID.AddPointer(&Tag);
  }

  std::shared_ptr<PathDiagnosticPiece> VisitNode(const ExplodedNode *N,
                                                 const ExplodedNode *PrevN,
                                                 BugReporterContext &BRC,
                                                 BugReport &BR) override {

    const LocationContext *Ctx = N->getLocationContext();
    const StackFrameContext *SCtx = Ctx->getCurrentStackFrame();
    ProgramStateRef State = N->getState();
    auto CallExitLoc = N->getLocationAs<CallExitBegin>();

    // No diagnostic if region was modified inside the frame.
    if (!CallExitLoc)
      return nullptr;

    CallEventRef<> Call =
        BRC.getStateManager().getCallEventManager().getCaller(SCtx, State);

    const PrintingPolicy &PP = BRC.getASTContext().getPrintingPolicy();
    const SourceManager &SM = BRC.getSourceManager();
    if (const auto *CCall = dyn_cast<CXXConstructorCall>(Call)) {
      const MemRegion *ThisRegion = CCall->getCXXThisVal().getAsRegion();
      if (RegionOfInterest->isSubRegionOf(ThisRegion)
          && !CCall->getDecl()->isImplicit()
          && !isRegionOfInterestModifiedInFrame(N))
        return notModifiedInConstructorDiagnostics(Ctx, SM, PP, *CallExitLoc,
                                                   CCall, ThisRegion);
    }

    ArrayRef<ParmVarDecl *> parameters = getCallParameters(Call);
    for (unsigned I = 0, E = Call->getNumArgs(); I != E; ++I) {
      const ParmVarDecl *PVD = parameters[I];
      SVal S = Call->getArgSVal(I);
      unsigned IndirectionLevel = 1;
      QualType T = PVD->getType();
      while (const MemRegion *R = S.getAsRegion()) {
        if (RegionOfInterest->isSubRegionOf(R)
            && !isPointerToConst(PVD->getType())) {

          if (isRegionOfInterestModifiedInFrame(N))
            return nullptr;

          return notModifiedDiagnostics(
              Ctx, SM, PP, *CallExitLoc, Call, PVD, R, IndirectionLevel);
        }
        QualType PT = T->getPointeeType();
        if (PT.isNull() || PT->isVoidType()) break;
        S = State->getSVal(R, PT);
        T = PT;
        IndirectionLevel++;
      }
    }

    return nullptr;
  }

private:
  /// Check and lazily calculate whether the region of interest is
  /// modified in the stack frame to which \p N belongs.
  /// The calculation is cached in FramesModifyingRegion.
  bool isRegionOfInterestModifiedInFrame(const ExplodedNode *N) {
    const LocationContext *Ctx = N->getLocationContext();
    const StackFrameContext *SCtx = Ctx->getCurrentStackFrame();
    if (!FramesModifyingCalculated.count(SCtx))
      findModifyingFrames(N);
    return FramesModifyingRegion.count(SCtx);
  }


  /// Write to \c FramesModifyingRegion all stack frames along
  /// the path in the current stack frame which modify \c RegionOfInterest.
  void findModifyingFrames(const ExplodedNode *N) {
    assert(N->getLocationAs<CallExitBegin>());
    ProgramStateRef LastReturnState = N->getState();
    SVal ValueAtReturn = LastReturnState->getSVal(RegionOfInterest);
    const LocationContext *Ctx = N->getLocationContext();
    const StackFrameContext *OriginalSCtx = Ctx->getCurrentStackFrame();

    do {
      ProgramStateRef State = N->getState();
      auto CallExitLoc = N->getLocationAs<CallExitBegin>();
      if (CallExitLoc) {
        LastReturnState = State;
        ValueAtReturn = LastReturnState->getSVal(RegionOfInterest);
      }

      FramesModifyingCalculated.insert(
        N->getLocationContext()->getCurrentStackFrame());

      if (wasRegionOfInterestModifiedAt(N, LastReturnState, ValueAtReturn)) {
        const StackFrameContext *SCtx =
            N->getLocationContext()->getCurrentStackFrame();
        while (!SCtx->inTopFrame()) {
          auto p = FramesModifyingRegion.insert(SCtx);
          if (!p.second)
            break; // Frame and all its parents already inserted.
          SCtx = SCtx->getParent()->getCurrentStackFrame();
        }
      }

      // Stop calculation at the call to the current function.
      if (auto CE = N->getLocationAs<CallEnter>())
        if (CE->getCalleeContext() == OriginalSCtx)
          break;

      N = N->getFirstPred();
    } while (N);
  }

  /// \return Whether \c RegionOfInterest was modified at \p N,
  /// where \p ReturnState is a state associated with the return
  /// from the current frame.
  bool wasRegionOfInterestModifiedAt(const ExplodedNode *N,
                                     ProgramStateRef ReturnState,
                                     SVal ValueAtReturn) {
    if (!N->getLocationAs<PostStore>()
        && !N->getLocationAs<PostInitializer>()
        && !N->getLocationAs<PostStmt>())
      return false;

    // Writing into region of interest.
    if (auto PS = N->getLocationAs<PostStmt>())
      if (auto *BO = PS->getStmtAs<BinaryOperator>())
        if (BO->isAssignmentOp() && RegionOfInterest->isSubRegionOf(
                                        N->getSVal(BO->getLHS()).getAsRegion()))
          return true;

    // SVal after the state is possibly different.
    SVal ValueAtN = N->getState()->getSVal(RegionOfInterest);
    if (!ReturnState->areEqual(ValueAtN, ValueAtReturn).isConstrainedTrue() &&
        (!ValueAtN.isUndef() || !ValueAtReturn.isUndef()))
      return true;

    return false;
  }

  /// Get parameters associated with runtime definition in order
  /// to get the correct parameter name.
  ArrayRef<ParmVarDecl *> getCallParameters(CallEventRef<> Call) {
    // Use runtime definition, if available.
    RuntimeDefinition RD = Call->getRuntimeDefinition();
    if (const auto *FD = dyn_cast_or_null<FunctionDecl>(RD.getDecl()))
      return FD->parameters();

    return Call->parameters();
  }

  /// \return whether \p Ty points to a const type, or is a const reference.
  bool isPointerToConst(QualType Ty) {
    return !Ty->getPointeeType().isNull() &&
           Ty->getPointeeType().getCanonicalType().isConstQualified();
  }

  std::shared_ptr<PathDiagnosticPiece> notModifiedInConstructorDiagnostics(
      const LocationContext *Ctx,
      const SourceManager &SM,
      const PrintingPolicy &PP,
      CallExitBegin &CallExitLoc,
      const CXXConstructorCall *Call,
      const MemRegion *ArgRegion) {
    SmallString<256> sbuf;
    llvm::raw_svector_ostream os(sbuf);
    os << DiagnosticsMsg;
    bool out = prettyPrintRegionName(
        "this", "->", /*IsReference=*/true,
        /*IndirectionLevel=*/1, ArgRegion, os, PP);

    // Return nothing if we have failed to pretty-print.
    if (!out)
      return nullptr;

    os << "'";
    PathDiagnosticLocation L =
        getPathDiagnosticLocation(nullptr, SM, Ctx, Call);
    return std::make_shared<PathDiagnosticEventPiece>(L, os.str());
  }

  /// \p IndirectionLevel How many times \c ArgRegion has to be dereferenced
  /// before we get to the super region of \c RegionOfInterest
  std::shared_ptr<PathDiagnosticPiece>
  notModifiedDiagnostics(const LocationContext *Ctx,
                         const SourceManager &SM,
                         const PrintingPolicy &PP,
                         CallExitBegin &CallExitLoc,
                         CallEventRef<> Call,
                         const ParmVarDecl *PVD,
                         const MemRegion *ArgRegion,
                         unsigned IndirectionLevel) {
    PathDiagnosticLocation L = getPathDiagnosticLocation(
        CallExitLoc.getReturnStmt(), SM, Ctx, Call);
    SmallString<256> sbuf;
    llvm::raw_svector_ostream os(sbuf);
    os << DiagnosticsMsg;
    bool IsReference = PVD->getType()->isReferenceType();
    const char *Sep = IsReference && IndirectionLevel == 1 ? "." : "->";
    bool Success = prettyPrintRegionName(
        PVD->getQualifiedNameAsString().c_str(),
        Sep, IsReference, IndirectionLevel, ArgRegion, os, PP);

    // Print the parameter name if the pretty-printing has failed.
    if (!Success)
      PVD->printQualifiedName(os);
    os << "'";
    return std::make_shared<PathDiagnosticEventPiece>(L, os.str());
  }

  /// \return a path diagnostic location for the optionally
  /// present return statement \p RS.
  PathDiagnosticLocation getPathDiagnosticLocation(const ReturnStmt *RS,
                                                   const SourceManager &SM,
                                                   const LocationContext *Ctx,
                                                   CallEventRef<> Call) {
    if (RS)
      return PathDiagnosticLocation::createBegin(RS, SM, Ctx);
    return PathDiagnosticLocation(
        Call->getRuntimeDefinition().getDecl()->getSourceRange().getEnd(), SM);
  }

  /// Pretty-print region \p ArgRegion starting from parent to \p os.
  /// \return whether printing has succeeded
  bool prettyPrintRegionName(const char *TopRegionName,
                             const char *Sep,
                             bool IsReference,
                             int IndirectionLevel,
                             const MemRegion *ArgRegion,
                             llvm::raw_svector_ostream &os,
                             const PrintingPolicy &PP) {
    SmallVector<const MemRegion *, 5> Subregions;
    const MemRegion *R = RegionOfInterest;
    while (R != ArgRegion) {
      if (!(isa<FieldRegion>(R) || isa<CXXBaseObjectRegion>(R)))
        return false; // Pattern-matching failed.
      Subregions.push_back(R);
      R = dyn_cast<SubRegion>(R)->getSuperRegion();
    }
    bool IndirectReference = !Subregions.empty();

    if (IndirectReference)
      IndirectionLevel--; // Due to "->" symbol.

    if (IsReference)
      IndirectionLevel--; // Due to reference semantics.

    bool ShouldSurround = IndirectReference && IndirectionLevel > 0;

    if (ShouldSurround)
      os << "(";
    for (int i = 0; i < IndirectionLevel; i++)
      os << "*";
    os << TopRegionName;
    if (ShouldSurround)
      os << ")";

    for (auto I = Subregions.rbegin(), E = Subregions.rend(); I != E; ++I) {
      if (const auto *FR = dyn_cast<FieldRegion>(*I)) {
        os << Sep;
        FR->getDecl()->getDeclName().print(os, PP);
        Sep = ".";
      } else if (isa<CXXBaseObjectRegion>(*I)) {
        continue; // Just keep going up to the base region.
      } else {
        llvm_unreachable("Previous check has missed an unexpected region");
      }
    }
    return true;
  }
};

class MacroNullReturnSuppressionVisitor final
    : public BugReporterVisitorImpl<MacroNullReturnSuppressionVisitor> {
  const SubRegion *RegionOfInterest;

public:
  MacroNullReturnSuppressionVisitor(const SubRegion *R) : RegionOfInterest(R) {}

  static void *getTag() {
    static int Tag = 0;
    return static_cast<void *>(&Tag);
  }

  void Profile(llvm::FoldingSetNodeID &ID) const override {
    ID.AddPointer(getTag());
  }

  std::shared_ptr<PathDiagnosticPiece> VisitNode(const ExplodedNode *N,
                                                 const ExplodedNode *PrevN,
                                                 BugReporterContext &BRC,
                                                 BugReport &BR) override {
    auto BugPoint = BR.getErrorNode()->getLocation().getAs<StmtPoint>();
    if (!BugPoint)
      return nullptr;

    const SourceManager &SMgr = BRC.getSourceManager();
    if (auto Loc = matchAssignment(N, BRC)) {
      if (isFunctionMacroExpansion(*Loc, SMgr)) {
        std::string MacroName = getMacroName(*Loc, BRC);
        SourceLocation BugLoc = BugPoint->getStmt()->getLocStart();
        if (!BugLoc.isMacroID() || getMacroName(BugLoc, BRC) != MacroName)
          BR.markInvalid(getTag(), MacroName.c_str());
      }
    }
    return nullptr;
  }

  static void addMacroVisitorIfNecessary(
        const ExplodedNode *N, const MemRegion *R,
        bool EnableNullFPSuppression, BugReport &BR,
        const SVal V) {
    AnalyzerOptions &Options = N->getState()->getStateManager()
        .getOwningEngine()->getAnalysisManager().options;
    if (EnableNullFPSuppression && Options.shouldSuppressNullReturnPaths()
          && V.getAs<Loc>())
      BR.addVisitor(llvm::make_unique<MacroNullReturnSuppressionVisitor>(
              R->getAs<SubRegion>()));
  }

private:
  /// \return Source location of right hand side of an assignment
  /// into \c RegionOfInterest, empty optional if none found.
  Optional<SourceLocation> matchAssignment(const ExplodedNode *N,
                                           BugReporterContext &BRC) {
    const Stmt *S = PathDiagnosticLocation::getStmt(N);
    ProgramStateRef State = N->getState();
    auto *LCtx = N->getLocationContext();
    if (!S)
      return None;

    if (const auto *DS = dyn_cast<DeclStmt>(S)) {
      if (const auto *VD = dyn_cast<VarDecl>(DS->getSingleDecl()))
        if (const Expr *RHS = VD->getInit())
          if (RegionOfInterest->isSubRegionOf(
                  State->getLValue(VD, LCtx).getAsRegion()))
            return RHS->getLocStart();
    } else if (const auto *BO = dyn_cast<BinaryOperator>(S)) {
      const MemRegion *R = N->getSVal(BO->getLHS()).getAsRegion();
      const Expr *RHS = BO->getRHS();
      if (BO->isAssignmentOp() && RegionOfInterest->isSubRegionOf(R)) {
        return RHS->getLocStart();
      }
    }
    return None;
  }
};

/// Emits an extra note at the return statement of an interesting stack frame.
///
/// The returned value is marked as an interesting value, and if it's null,
/// adds a visitor to track where it became null.
///
/// This visitor is intended to be used when another visitor discovers that an
/// interesting value comes from an inlined function call.
class ReturnVisitor : public BugReporterVisitorImpl<ReturnVisitor> {
  const StackFrameContext *StackFrame;
  enum {
    Initial,
    MaybeUnsuppress,
    Satisfied
  } Mode = Initial;

  bool EnableNullFPSuppression;

public:
  ReturnVisitor(const StackFrameContext *Frame, bool Suppressed)
      : StackFrame(Frame), EnableNullFPSuppression(Suppressed) {}

  static void *getTag() {
    static int Tag = 0;
    return static_cast<void *>(&Tag);
  }

  void Profile(llvm::FoldingSetNodeID &ID) const override {
    ID.AddPointer(ReturnVisitor::getTag());
    ID.AddPointer(StackFrame);
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
                                    BugReport &BR,
                                    bool InEnableNullFPSuppression) {
    if (!CallEvent::isCallStmt(S))
      return;

    // First, find when we processed the statement.
    do {
      if (Optional<CallExitEnd> CEE = Node->getLocationAs<CallExitEnd>())
        if (CEE->getCalleeContext()->getCallSite() == S)
          break;
      if (Optional<StmtPoint> SP = Node->getLocationAs<StmtPoint>())
        if (SP->getStmt() == S)
          break;

      Node = Node->getFirstPred();
    } while (Node);

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
    SubEngine *Eng = State->getStateManager().getOwningEngine();
    assert(Eng && "Cannot file a bug report without an owning engine");
    AnalyzerOptions &Options = Eng->getAnalysisManager().options;

    bool EnableNullFPSuppression = false;
    if (InEnableNullFPSuppression && Options.shouldSuppressNullReturnPaths())
      if (Optional<Loc> RetLoc = RetVal.getAs<Loc>())
        EnableNullFPSuppression = State->isNull(*RetLoc).isConstrainedTrue();

    BR.markInteresting(CalleeContext);
    BR.addVisitor(llvm::make_unique<ReturnVisitor>(CalleeContext,
                                                   EnableNullFPSuppression));
  }

  /// Returns true if any counter-suppression heuristics are enabled for
  /// ReturnVisitor.
  static bool hasCounterSuppression(AnalyzerOptions &Options) {
    return Options.shouldAvoidSuppressingNullArgumentPaths();
  }

  std::shared_ptr<PathDiagnosticPiece>
  visitNodeInitial(const ExplodedNode *N, const ExplodedNode *PrevN,
                   BugReporterContext &BRC, BugReport &BR) {
    // Only print a message at the interesting return statement.
    if (N->getLocationContext() != StackFrame)
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
    SVal V = State->getSVal(Ret, StackFrame);
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

    // If we can't prove the return value is 0, just mark it interesting, and
    // make sure to track it into any further inner functions.
    if (!State->isNull(V).isConstrainedTrue()) {
      BR.markInteresting(V);
      ReturnVisitor::addVisitorIfNecessary(N, RetE, BR,
                                           EnableNullFPSuppression);
      return nullptr;
    }

    // If we're returning 0, we should track where that 0 came from.
    bugreporter::trackNullOrUndefValue(N, RetE, BR, /*IsArg*/ false,
                                       EnableNullFPSuppression);

    // Build an appropriate message based on the return value.
    SmallString<64> Msg;
    llvm::raw_svector_ostream Out(Msg);

    if (V.getAs<Loc>()) {
      // If we have counter-suppression enabled, make sure we keep visiting
      // future nodes. We want to emit a path note as well, in case
      // the report is resurrected as valid later on.
      ExprEngine &Eng = BRC.getBugReporter().getEngine();
      AnalyzerOptions &Options = Eng.getAnalysisManager().options;
      if (EnableNullFPSuppression && hasCounterSuppression(Options))
        Mode = MaybeUnsuppress;

      if (RetE->getType()->isObjCObjectPointerType())
        Out << "Returning nil";
      else
        Out << "Returning null pointer";
    } else {
      Out << "Returning zero";
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

    PathDiagnosticLocation L(Ret, BRC.getSourceManager(), StackFrame);
    if (!L.isValid() || !L.asLocation().isValid())
      return nullptr;

    return std::make_shared<PathDiagnosticEventPiece>(L, Out.str());
  }

  std::shared_ptr<PathDiagnosticPiece>
  visitNodeMaybeUnsuppress(const ExplodedNode *N, const ExplodedNode *PrevN,
                           BugReporterContext &BRC, BugReport &BR) {
#ifndef NDEBUG
    ExprEngine &Eng = BRC.getBugReporter().getEngine();
    AnalyzerOptions &Options = Eng.getAnalysisManager().options;
    assert(hasCounterSuppression(Options));
#endif

    // Are we at the entry node for this call?
    Optional<CallEnter> CE = N->getLocationAs<CallEnter>();
    if (!CE)
      return nullptr;

    if (CE->getCalleeContext() != StackFrame)
      return nullptr;

    Mode = Satisfied;

    // Don't automatically suppress a report if one of the arguments is
    // known to be a null pointer. Instead, start tracking /that/ null
    // value back to its origin.
    ProgramStateManager &StateMgr = BRC.getStateManager();
    CallEventManager &CallMgr = StateMgr.getCallEventManager();

    ProgramStateRef State = N->getState();
    CallEventRef<> Call = CallMgr.getCaller(StackFrame, State);
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

      if (bugreporter::trackNullOrUndefValue(N, ArgE, BR, /*IsArg=*/true,
                                             EnableNullFPSuppression))
        BR.removeInvalidation(ReturnVisitor::getTag(), StackFrame);

      // If we /can't/ track the null pointer, we should err on the side of
      // false negatives, and continue towards marking this report invalid.
      // (We will still look at the other arguments, though.)
    }

    return nullptr;
  }

  std::shared_ptr<PathDiagnosticPiece> VisitNode(const ExplodedNode *N,
                                                 const ExplodedNode *PrevN,
                                                 BugReporterContext &BRC,
                                                 BugReport &BR) override {
    switch (Mode) {
    case Initial:
      return visitNodeInitial(N, PrevN, BRC, BR);
    case MaybeUnsuppress:
      return visitNodeMaybeUnsuppress(N, PrevN, BRC, BR);
    case Satisfied:
      return nullptr;
    }

    llvm_unreachable("Invalid visit mode!");
  }

  std::unique_ptr<PathDiagnosticPiece> getEndPath(BugReporterContext &BRC,
                                                  const ExplodedNode *N,
                                                  BugReport &BR) override {
    if (EnableNullFPSuppression)
      BR.markInvalid(ReturnVisitor::getTag(), StackFrame);
    return nullptr;
  }
};

} // namespace

void FindLastStoreBRVisitor::Profile(llvm::FoldingSetNodeID &ID) const {
  static int tag = 0;
  ID.AddPointer(&tag);
  ID.AddPointer(R);
  ID.Add(V);
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
  return FrameSpace->getStackFrame() == LCtx->getCurrentStackFrame();
}

/// Show diagnostics for initializing or declaring a region \p R with a bad value.
void showBRDiagnostics(const char *action,
    llvm::raw_svector_ostream& os,
    const MemRegion *R,
    SVal V,
    const DeclStmt *DS) {
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
static void showBRDefaultDiagnostics(llvm::raw_svector_ostream& os,
    const MemRegion *R,
    SVal V) {
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

std::shared_ptr<PathDiagnosticPiece>
FindLastStoreBRVisitor::VisitNode(const ExplodedNode *Succ,
                                  const ExplodedNode *Pred,
                                  BugReporterContext &BRC, BugReport &BR) {
  if (Satisfied)
    return nullptr;

  const ExplodedNode *StoreSite = nullptr;
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
    if (FieldReg && FieldReg == R) {
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

    if (Pred->getState()->getSVal(R) == V) {
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
        const auto *Param = cast<ParmVarDecl>(VR->getDecl());

        ProgramStateManager &StateMgr = BRC.getStateManager();
        CallEventManager &CallMgr = StateMgr.getCallEventManager();

        CallEventRef<> Call = CallMgr.getCaller(CE->getCalleeContext(),
                                                Succ->getState());
        InitE = Call->getArgExpr(Param->getFunctionScopeIndex());
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
    if (V.isUndef() ||
        V.getAs<loc::ConcreteInt>() || V.getAs<nonloc::ConcreteInt>()) {
      if (!IsParam)
        InitE = InitE->IgnoreParenCasts();
      bugreporter::trackNullOrUndefValue(StoreSite, InitE, BR, IsParam,
                                         EnableNullFPSuppression);
    } else {
      ReturnVisitor::addVisitorIfNecessary(StoreSite, InitE->IgnoreParenCasts(),
                                           BR, EnableNullFPSuppression);
    }
  }

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
            if (Optional<KnownSVal> KV =
                State->getSVal(OriginalR).getAs<KnownSVal>())
              BR.addVisitor(llvm::make_unique<FindLastStoreBRVisitor>(
                  *KV, OriginalR, EnableNullFPSuppression));
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

std::shared_ptr<PathDiagnosticPiece>
TrackConstraintBRVisitor::VisitNode(const ExplodedNode *N,
                                    const ExplodedNode *PrevN,
                                    BugReporterContext &BRC, BugReport &BR) {
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

SuppressInlineDefensiveChecksVisitor::
SuppressInlineDefensiveChecksVisitor(DefinedSVal Value, const ExplodedNode *N)
    : V(Value) {
  // Check if the visitor is disabled.
  SubEngine *Eng = N->getState()->getStateManager().getOwningEngine();
  assert(Eng && "Cannot file a bug report without an owning engine");
  AnalyzerOptions &Options = Eng->getAnalysisManager().options;
  if (!Options.shouldSuppressInlinedDefensiveChecks())
    IsSatisfied = true;

  assert(N->getState()->isNull(V).isConstrainedTrue() &&
         "The visitor only tracks the cases where V is constrained to 0");
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

std::shared_ptr<PathDiagnosticPiece>
SuppressInlineDefensiveChecksVisitor::VisitNode(const ExplodedNode *Succ,
                                                const ExplodedNode *Pred,
                                                BugReporterContext &BRC,
                                                BugReport &BR) {
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
  if (!Pred->getState()->isNull(V).isConstrainedTrue()) {
    IsSatisfied = true;

    assert(Succ->getState()->isNull(V).isConstrainedTrue());

    // Check if this is inlined defensive checks.
    const LocationContext *CurLC =Succ->getLocationContext();
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
      if (!CurStmt->getLocStart().isMacroID())
        return nullptr;

      CFGStmtMap *Map = CurLC->getAnalysisDeclContext()->getCFGStmtMap();
      CurTerminatorStmt = Map->getBlock(CurStmt)->getTerminator();
    } else {
      return nullptr;
    }

    if (!CurTerminatorStmt)
      return nullptr;

    SourceLocation TerminatorLoc = CurTerminatorStmt->getLocStart();
    if (TerminatorLoc.isMacroID()) {
      SourceLocation BugLoc = BugPoint->getStmt()->getLocStart();

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

static const Expr *peelOffOuterExpr(const Expr *Ex,
                                    const ExplodedNode *N) {
  Ex = Ex->IgnoreParenCasts();
  if (const auto *EWC = dyn_cast<ExprWithCleanups>(Ex))
    return peelOffOuterExpr(EWC->getSubExpr(), N);
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
        if (const Stmt *term = srcBlk->getTerminator()) {
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

  return Ex;
}

/// Walk through nodes until we get one that matches the statement exactly.
/// Alternately, if we hit a known lvalue for the statement, we know we've
/// gone too far (though we can likely track the lvalue better anyway).
static const ExplodedNode* findNodeForStatement(const ExplodedNode *N,
                                                const Stmt *S,
                                                const Expr *Inner) {
  do {
    const ProgramPoint &pp = N->getLocation();
    if (auto ps = pp.getAs<StmtPoint>()) {
      if (ps->getStmt() == S || ps->getStmt() == Inner)
        break;
    } else if (auto CEE = pp.getAs<CallExitEnd>()) {
      if (CEE->getCalleeContext()->getCallSite() == S ||
          CEE->getCalleeContext()->getCallSite() == Inner)
        break;
    }
    N = N->getFirstPred();
  } while (N);
  return N;
}

/// Find the ExplodedNode where the lvalue (the value of 'Ex')
/// was computed.
static const ExplodedNode* findNodeForExpression(const ExplodedNode *N,
    const Expr *Inner) {
  while (N) {
    if (auto P = N->getLocation().getAs<PostStmt>()) {
      if (P->getStmt() == Inner)
        break;
    }
    N = N->getFirstPred();
  }
  assert(N && "Unable to find the lvalue node.");
  return N;
}

/// Performing operator `&' on an lvalue expression is essentially a no-op.
/// Then, if we are taking addresses of fields or elements, these are also
/// unlikely to matter.
static const Expr* peelOfOuterAddrOf(const Expr* Ex) {
  Ex = Ex->IgnoreParenCasts();

  // FIXME: There's a hack in our Store implementation that always computes
  // field offsets around null pointers as if they are always equal to 0.
  // The idea here is to report accesses to fields as null dereferences
  // even though the pointer value that's being dereferenced is actually
  // the offset of the field rather than exactly 0.
  // See the FIXME in StoreManager's getLValueFieldOrIvar() method.
  // This code interacts heavily with this hack; otherwise the value
  // would not be null at all for most fields, so we'd be unable to track it.
  if (const auto *Op = dyn_cast<UnaryOperator>(Ex))
    if (Op->getOpcode() == UO_AddrOf && Op->getSubExpr()->isLValue())
      if (const Expr *DerefEx = bugreporter::getDerefExpr(Op->getSubExpr()))
        return DerefEx;
  return Ex;
}

bool bugreporter::trackNullOrUndefValue(const ExplodedNode *N,
                                        const Stmt *S,
                                        BugReport &report, bool IsArg,
                                        bool EnableNullFPSuppression) {
  if (!S || !N)
    return false;

  if (const auto *Ex = dyn_cast<Expr>(S))
    S = peelOffOuterExpr(Ex, N);

  const Expr *Inner = nullptr;
  if (const auto *Ex = dyn_cast<Expr>(S)) {
    Ex = peelOfOuterAddrOf(Ex);
    Ex = Ex->IgnoreParenCasts();

    if (Ex && (ExplodedGraph::isInterestingLValueExpr(Ex)
          || CallEvent::isCallStmt(Ex)))
      Inner = Ex;
  }

  if (IsArg && !Inner) {
    assert(N->getLocation().getAs<CallEnter>() && "Tracking arg but not at call");
  } else {
    N = findNodeForStatement(N, S, Inner);
    if (!N)
      return false;
  }

  ProgramStateRef state = N->getState();

  // The message send could be nil due to the receiver being nil.
  // At this point in the path, the receiver should be live since we are at the
  // message send expr. If it is nil, start tracking it.
  if (const Expr *Receiver = NilReceiverBRVisitor::getNilReceiver(S, N))
    trackNullOrUndefValue(N, Receiver, report, /* IsArg=*/ false,
        EnableNullFPSuppression);

  // See if the expression we're interested refers to a variable.
  // If so, we can track both its contents and constraints on its value.
  if (Inner && ExplodedGraph::isInterestingLValueExpr(Inner)) {
    const ExplodedNode *LVNode = findNodeForExpression(N, Inner);
    ProgramStateRef LVState = LVNode->getState();
    SVal LVal = LVNode->getSVal(Inner);

    const MemRegion *RR = getLocationRegionIfReference(Inner, N);
    bool LVIsNull = LVState->isNull(LVal).isConstrainedTrue();

    // If this is a C++ reference to a null pointer, we are tracking the
    // pointer. In addition, we should find the store at which the reference
    // got initialized.
    if (RR && !LVIsNull) {
      if (auto KV = LVal.getAs<KnownSVal>())
        report.addVisitor(llvm::make_unique<FindLastStoreBRVisitor>(
              *KV, RR, EnableNullFPSuppression));
    }

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
          llvm::make_unique<NoStoreFuncVisitor>(cast<SubRegion>(R)));

      MacroNullReturnSuppressionVisitor::addMacroVisitorIfNecessary(
          N, R, EnableNullFPSuppression, report, V);

      report.markInteresting(R);
      report.markInteresting(V);
      report.addVisitor(llvm::make_unique<UndefOrNullArgVisitor>(R));

      // If the contents are symbolic, find out when they became null.
      if (V.getAsLocSymbol(/*IncludeBaseRegions*/ true))
        report.addVisitor(llvm::make_unique<TrackConstraintBRVisitor>(
              V.castAs<DefinedSVal>(), false));

      // Add visitor, which will suppress inline defensive checks.
      if (auto DV = V.getAs<DefinedSVal>()) {
        if (!DV->isZeroConstant() && LVState->isNull(*DV).isConstrainedTrue() &&
            EnableNullFPSuppression) {
          report.addVisitor(
              llvm::make_unique<SuppressInlineDefensiveChecksVisitor>(*DV,
                LVNode));
        }
      }

      if (auto KV = V.getAs<KnownSVal>())
        report.addVisitor(llvm::make_unique<FindLastStoreBRVisitor>(
              *KV, R, EnableNullFPSuppression));
      return true;
    }
  }

  // If the expression is not an "lvalue expression", we can still
  // track the constraints on its contents.
  SVal V = state->getSValAsScalarOrLoc(S, N->getLocationContext());

  // If the value came from an inlined function call, we should at least make
  // sure that function isn't pruned in our output.
  if (const auto *E = dyn_cast<Expr>(S))
    S = E->IgnoreParenCasts();

  ReturnVisitor::addVisitorIfNecessary(N, S, report, EnableNullFPSuppression);

  // Uncomment this to find cases where we aren't properly getting the
  // base value that was dereferenced.
  // assert(!V.isUnknownOrUndef());
  // Is it a symbolic value?
  if (auto L = V.getAs<loc::MemRegionVal>()) {
    report.addVisitor(llvm::make_unique<UndefOrNullArgVisitor>(L->getRegion()));

    // At this point we are dealing with the region's LValue.
    // However, if the rvalue is a symbolic region, we should track it as well.
    // Try to use the correct type when looking up the value.
    SVal RVal;
    if (const auto *E = dyn_cast<Expr>(S))
      RVal = state->getRawSVal(L.getValue(), E->getType());
    else
      RVal = state->getSVal(L->getRegion());

    if (auto KV = RVal.getAs<KnownSVal>())
      report.addVisitor(llvm::make_unique<FindLastStoreBRVisitor>(
            *KV, L->getRegion(), EnableNullFPSuppression));

    const MemRegion *RegionRVal = RVal.getAsRegion();
    if (RegionRVal && isa<SymbolicRegion>(RegionRVal)) {
      report.markInteresting(RegionRVal);
      report.addVisitor(llvm::make_unique<TrackConstraintBRVisitor>(
            loc::MemRegionVal(RegionRVal), false));
    }
  }
  return true;
}

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

std::shared_ptr<PathDiagnosticPiece>
NilReceiverBRVisitor::VisitNode(const ExplodedNode *N,
                                const ExplodedNode *PrevN,
                                BugReporterContext &BRC, BugReport &BR) {
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
  bugreporter::trackNullOrUndefValue(N, Receiver, BR, /*IsArg*/ false,
                                     /*EnableNullFPSuppression*/ false);
  // Issue a message saying that the method was skipped.
  PathDiagnosticLocation L(Receiver, BRC.getSourceManager(),
                                     N->getLocationContext());
  return std::make_shared<PathDiagnosticEventPiece>(L, OS.str());
}

// Registers every VarDecl inside a Stmt with a last store visitor.
void FindLastStoreBRVisitor::registerStatementVarDecls(BugReport &BR,
                                                const Stmt *S,
                                                bool EnableNullFPSuppression) {
  const ExplodedNode *N = BR.getErrorNode();
  std::deque<const Stmt *> WorkList;
  WorkList.push_back(S);

  while (!WorkList.empty()) {
    const Stmt *Head = WorkList.front();
    WorkList.pop_front();

    ProgramStateManager &StateMgr = N->getState()->getStateManager();

    if (const auto *DR = dyn_cast<DeclRefExpr>(Head)) {
      if (const auto *VD = dyn_cast<VarDecl>(DR->getDecl())) {
        const VarRegion *R =
        StateMgr.getRegionManager().getVarRegion(VD, N->getLocationContext());

        // What did we load?
        SVal V = N->getSVal(S);

        if (V.getAs<loc::ConcreteInt>() || V.getAs<nonloc::ConcreteInt>()) {
          // Register a new visitor with the BugReport.
          BR.addVisitor(llvm::make_unique<FindLastStoreBRVisitor>(
              V.castAs<KnownSVal>(), R, EnableNullFPSuppression));
        }
      }
    }

    for (const Stmt *SubStmt : Head->children())
      WorkList.push_back(SubStmt);
  }
}

//===----------------------------------------------------------------------===//
// Visitor that tries to report interesting diagnostics from conditions.
//===----------------------------------------------------------------------===//

/// Return the tag associated with this visitor.  This tag will be used
/// to make all PathDiagnosticPieces created by this visitor.
const char *ConditionBRVisitor::getTag() {
  return "ConditionBRVisitor";
}

std::shared_ptr<PathDiagnosticPiece>
ConditionBRVisitor::VisitNode(const ExplodedNode *N, const ExplodedNode *Prev,
                              BugReporterContext &BRC, BugReport &BR) {
  auto piece = VisitNodeImpl(N, Prev, BRC, BR);
  if (piece) {
    piece->setTag(getTag());
    if (auto *ev = dyn_cast<PathDiagnosticEventPiece>(piece.get()))
      ev->setPrunable(true, /* override */ false);
  }
  return piece;
}

std::shared_ptr<PathDiagnosticPiece>
ConditionBRVisitor::VisitNodeImpl(const ExplodedNode *N,
                                  const ExplodedNode *Prev,
                                  BugReporterContext &BRC, BugReport &BR) {
  ProgramPoint progPoint = N->getLocation();
  ProgramStateRef CurrentState = N->getState();
  ProgramStateRef PrevState = Prev->getState();

  // Compare the GDMs of the state, because that is where constraints
  // are managed.  Note that ensure that we only look at nodes that
  // were generated by the analyzer engine proper, not checkers.
  if (CurrentState->getGDM().getRoot() ==
      PrevState->getGDM().getRoot())
    return nullptr;

  // If an assumption was made on a branch, it should be caught
  // here by looking at the state transition.
  if (Optional<BlockEdge> BE = progPoint.getAs<BlockEdge>()) {
    const CFGBlock *srcBlk = BE->getSrc();
    if (const Stmt *term = srcBlk->getTerminator())
      return VisitTerminator(term, N, srcBlk, BE->getDst(), BR, BRC);
    return nullptr;
  }

  if (Optional<PostStmt> PS = progPoint.getAs<PostStmt>()) {
    // FIXME: Assuming that BugReporter is a GRBugReporter is a layering
    // violation.
    const std::pair<const ProgramPointTag *, const ProgramPointTag *> &tags =
      cast<GRBugReporter>(BRC.getBugReporter()).
        getEngine().geteagerlyAssumeBinOpBifurcationTags();

    const ProgramPointTag *tag = PS->getTag();
    if (tag == tags.first)
      return VisitTrueTest(cast<Expr>(PS->getStmt()), true,
                           BRC, BR, N);
    if (tag == tags.second)
      return VisitTrueTest(cast<Expr>(PS->getStmt()), false,
                           BRC, BR, N);

    return nullptr;
  }

  return nullptr;
}

std::shared_ptr<PathDiagnosticPiece> ConditionBRVisitor::VisitTerminator(
    const Stmt *Term, const ExplodedNode *N, const CFGBlock *srcBlk,
    const CFGBlock *dstBlk, BugReport &R, BugReporterContext &BRC) {
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
  const bool tookTrue = *(srcBlk->succ_begin()) == dstBlk;
  return VisitTrueTest(Cond, tookTrue, BRC, R, N);
}

std::shared_ptr<PathDiagnosticPiece>
ConditionBRVisitor::VisitTrueTest(const Expr *Cond, bool tookTrue,
                                  BugReporterContext &BRC, BugReport &R,
                                  const ExplodedNode *N) {
  // These will be modified in code below, but we need to preserve the original
  //  values in case we want to throw the generic message.
  const Expr *CondTmp = Cond;
  bool tookTrueTmp = tookTrue;

  while (true) {
    CondTmp = CondTmp->IgnoreParenCasts();
    switch (CondTmp->getStmtClass()) {
      default:
        break;
      case Stmt::BinaryOperatorClass:
        if (auto P = VisitTrueTest(Cond, cast<BinaryOperator>(CondTmp),
                                   tookTrueTmp, BRC, R, N))
          return P;
        break;
      case Stmt::DeclRefExprClass:
        if (auto P = VisitTrueTest(Cond, cast<DeclRefExpr>(CondTmp),
                                   tookTrueTmp, BRC, R, N))
          return P;
        break;
      case Stmt::UnaryOperatorClass: {
        const auto *UO = cast<UnaryOperator>(CondTmp);
        if (UO->getOpcode() == UO_LNot) {
          tookTrueTmp = !tookTrueTmp;
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
  const LocationContext *LCtx = N->getLocationContext();
  PathDiagnosticLocation Loc(Cond, BRC.getSourceManager(), LCtx);
  if (!Loc.isValid() || !Loc.asLocation().isValid())
    return nullptr;

  return std::make_shared<PathDiagnosticEventPiece>(
      Loc, tookTrue ? GenericTrueMessage : GenericFalseMessage);
}

bool ConditionBRVisitor::patternMatch(const Expr *Ex,
                                      const Expr *ParentEx,
                                      raw_ostream &Out,
                                      BugReporterContext &BRC,
                                      BugReport &report,
                                      const ExplodedNode *N,
                                      Optional<bool> &prunable) {
  const Expr *OriginalExpr = Ex;
  Ex = Ex->IgnoreParenCasts();

  // Use heuristics to determine if Ex is a macro expending to a literal and
  // if so, use the macro's name.
  SourceLocation LocStart = Ex->getLocStart();
  SourceLocation LocEnd = Ex->getLocEnd();
  if (LocStart.isMacroID() && LocEnd.isMacroID() &&
      (isa<GNUNullExpr>(Ex) ||
       isa<ObjCBoolLiteralExpr>(Ex) ||
       isa<CXXBoolLiteralExpr>(Ex) ||
       isa<IntegerLiteral>(Ex) ||
       isa<FloatingLiteral>(Ex))) {
    StringRef StartName = Lexer::getImmediateMacroNameForDiagnostics(LocStart,
      BRC.getSourceManager(), BRC.getASTContext().getLangOpts());
    StringRef EndName = Lexer::getImmediateMacroNameForDiagnostics(LocEnd,
      BRC.getSourceManager(), BRC.getASTContext().getLangOpts());
    bool beginAndEndAreTheSameMacro = StartName.equals(EndName);

    bool partOfParentMacro = false;
    if (ParentEx->getLocStart().isMacroID()) {
      StringRef PName = Lexer::getImmediateMacroNameForDiagnostics(
        ParentEx->getLocStart(), BRC.getSourceManager(),
        BRC.getASTContext().getLangOpts());
      partOfParentMacro = PName.equals(StartName);
    }

    if (beginAndEndAreTheSameMacro && !partOfParentMacro ) {
      // Get the location of the macro name as written by the caller.
      SourceLocation Loc = LocStart;
      while (LocStart.isMacroID()) {
        Loc = LocStart;
        LocStart = BRC.getSourceManager().getImmediateMacroCallerLoc(LocStart);
      }
      StringRef MacroName = Lexer::getImmediateMacroNameForDiagnostics(
        Loc, BRC.getSourceManager(), BRC.getASTContext().getLangOpts());

      // Return the macro name.
      Out << MacroName;
      return false;
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

  return false;
}

std::shared_ptr<PathDiagnosticPiece>
ConditionBRVisitor::VisitTrueTest(const Expr *Cond, const BinaryOperator *BExpr,
                                  const bool tookTrue, BugReporterContext &BRC,
                                  BugReport &R, const ExplodedNode *N) {
  bool shouldInvert = false;
  Optional<bool> shouldPrune;

  SmallString<128> LhsString, RhsString;
  {
    llvm::raw_svector_ostream OutLHS(LhsString), OutRHS(RhsString);
    const bool isVarLHS = patternMatch(BExpr->getLHS(), BExpr, OutLHS,
                                       BRC, R, N, shouldPrune);
    const bool isVarRHS = patternMatch(BExpr->getRHS(), BExpr, OutRHS,
                                       BRC, R, N, shouldPrune);

    shouldInvert = !isVarLHS && isVarRHS;
  }

  BinaryOperator::Opcode Op = BExpr->getOpcode();

  if (BinaryOperator::isAssignmentOp(Op)) {
    // For assignment operators, all that we care about is that the LHS
    // evaluates to "true" or "false".
    return VisitConditionVariable(LhsString, BExpr->getLHS(), tookTrue,
                                  BRC, R, N);
  }

  // For non-assignment operations, we require that we can understand
  // both the LHS and RHS.
  if (LhsString.empty() || RhsString.empty() ||
      !BinaryOperator::isComparisonOp(Op) || Op == BO_Cmp)
    return nullptr;

  // Should we invert the strings if the LHS is not a variable name?
  SmallString<256> buf;
  llvm::raw_svector_ostream Out(buf);
  Out << "Assuming " << (shouldInvert ? RhsString : LhsString) << " is ";

  // Do we need to invert the opcode?
  if (shouldInvert)
    switch (Op) {
      default: break;
      case BO_LT: Op = BO_GT; break;
      case BO_GT: Op = BO_LT; break;
      case BO_LE: Op = BO_GE; break;
      case BO_GE: Op = BO_LE; break;
    }

  if (!tookTrue)
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
  PathDiagnosticLocation Loc(Cond, BRC.getSourceManager(), LCtx);
  auto event = std::make_shared<PathDiagnosticEventPiece>(Loc, Out.str());
  if (shouldPrune.hasValue())
    event->setPrunable(shouldPrune.getValue());
  return event;
}

std::shared_ptr<PathDiagnosticPiece> ConditionBRVisitor::VisitConditionVariable(
    StringRef LhsString, const Expr *CondVarExpr, const bool tookTrue,
    BugReporterContext &BRC, BugReport &report, const ExplodedNode *N) {
  // FIXME: If there's already a constraint tracker for this variable,
  // we shouldn't emit anything here (c.f. the double note in
  // test/Analysis/inlining/path-notes.c)
  SmallString<256> buf;
  llvm::raw_svector_ostream Out(buf);
  Out << "Assuming " << LhsString << " is ";

  QualType Ty = CondVarExpr->getType();

  if (Ty->isPointerType())
    Out << (tookTrue ? "not null" : "null");
  else if (Ty->isObjCObjectPointerType())
    Out << (tookTrue ? "not nil" : "nil");
  else if (Ty->isBooleanType())
    Out << (tookTrue ? "true" : "false");
  else if (Ty->isIntegralOrEnumerationType())
    Out << (tookTrue ? "non-zero" : "zero");
  else
    return nullptr;

  const LocationContext *LCtx = N->getLocationContext();
  PathDiagnosticLocation Loc(CondVarExpr, BRC.getSourceManager(), LCtx);
  auto event = std::make_shared<PathDiagnosticEventPiece>(Loc, Out.str());

  if (const auto *DR = dyn_cast<DeclRefExpr>(CondVarExpr)) {
    if (const auto *VD = dyn_cast<VarDecl>(DR->getDecl())) {
      const ProgramState *state = N->getState().get();
      if (const MemRegion *R = state->getLValue(VD, LCtx).getAsRegion()) {
        if (report.isInteresting(R))
          event->setPrunable(false);
      }
    }
  }

  return event;
}

std::shared_ptr<PathDiagnosticPiece>
ConditionBRVisitor::VisitTrueTest(const Expr *Cond, const DeclRefExpr *DR,
                                  const bool tookTrue, BugReporterContext &BRC,
                                  BugReport &report, const ExplodedNode *N) {
  const auto *VD = dyn_cast<VarDecl>(DR->getDecl());
  if (!VD)
    return nullptr;

  SmallString<256> Buf;
  llvm::raw_svector_ostream Out(Buf);

  Out << "Assuming '" << VD->getDeclName() << "' is ";

  QualType VDTy = VD->getType();

  if (VDTy->isPointerType())
    Out << (tookTrue ? "non-null" : "null");
  else if (VDTy->isObjCObjectPointerType())
    Out << (tookTrue ? "non-nil" : "nil");
  else if (VDTy->isScalarType())
    Out << (tookTrue ? "not equal to 0" : "0");
  else
    return nullptr;

  const LocationContext *LCtx = N->getLocationContext();
  PathDiagnosticLocation Loc(Cond, BRC.getSourceManager(), LCtx);
  auto event = std::make_shared<PathDiagnosticEventPiece>(Loc, Out.str());

  const ProgramState *state = N->getState().get();
  if (const MemRegion *R = state->getLValue(VD, LCtx).getAsRegion()) {
    if (report.isInteresting(R))
      event->setPrunable(false);
    else {
      SVal V = state->getSVal(R);
      if (report.isInteresting(V))
        event->setPrunable(false);
    }
  }
  return std::move(event);
}

const char *const ConditionBRVisitor::GenericTrueMessage =
    "Assuming the condition is true";
const char *const ConditionBRVisitor::GenericFalseMessage =
    "Assuming the condition is false";

bool ConditionBRVisitor::isPieceMessageGeneric(
    const PathDiagnosticPiece *Piece) {
  return Piece->getString() == GenericTrueMessage ||
         Piece->getString() == GenericFalseMessage;
}

std::unique_ptr<PathDiagnosticPiece>
LikelyFalsePositiveSuppressionBRVisitor::getEndPath(BugReporterContext &BRC,
                                                    const ExplodedNode *N,
                                                    BugReport &BR) {
  // Here we suppress false positives coming from system headers. This list is
  // based on known issues.
  ExprEngine &Eng = BRC.getBugReporter().getEngine();
  AnalyzerOptions &Options = Eng.getAnalysisManager().options;
  const Decl *D = N->getLocationContext()->getDecl();

  if (AnalysisDeclContext::isInStdNamespace(D)) {
    // Skip reports within the 'std' namespace. Although these can sometimes be
    // the user's fault, we currently don't report them very well, and
    // Note that this will not help for any other data structure libraries, like
    // TR1, Boost, or llvm/ADT.
    if (Options.shouldSuppressFromCXXStandardLibrary()) {
      BR.markInvalid(getTag(), nullptr);
      return nullptr;
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
          return nullptr;
        }
      }

      // The analyzer issues a false positive when the constructor of
      // std::__independent_bits_engine from algorithms is used.
      if (const auto *MD = dyn_cast<CXXConstructorDecl>(D)) {
        const CXXRecordDecl *CD = MD->getParent();
        if (CD->getName() == "__independent_bits_engine") {
          BR.markInvalid(getTag(), nullptr);
          return nullptr;
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
          return nullptr;
        }

        // The analyzer issues a false positive on
        //    std::shared_ptr<int> p(new int(1)); p = nullptr;
        // because it does not reason properly about temporary destructors.
        if (CD->getName() == "shared_ptr") {
          BR.markInvalid(getTag(), nullptr);
          return nullptr;
        }
      }
    }
  }

  // Skip reports within the sys/queue.h macros as we do not have the ability to
  // reason about data structure shapes.
  SourceManager &SM = BRC.getSourceManager();
  FullSourceLoc Loc = BR.getLocation(SM).asLocation();
  while (Loc.isMacroID()) {
    Loc = Loc.getSpellingLoc();
    if (SM.getFilename(Loc).endswith("sys/queue.h")) {
      BR.markInvalid(getTag(), nullptr);
      return nullptr;
    }
  }

  return nullptr;
}

std::shared_ptr<PathDiagnosticPiece>
UndefOrNullArgVisitor::VisitNode(const ExplodedNode *N,
                                 const ExplodedNode *PrevN,
                                 BugReporterContext &BRC, BugReport &BR) {
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

std::shared_ptr<PathDiagnosticPiece>
CXXSelfAssignmentBRVisitor::VisitNode(const ExplodedNode *Succ,
                                      const ExplodedNode *Pred,
                                      BugReporterContext &BRC, BugReport &BR) {
  if (Satisfied)
    return nullptr;

  const auto Edge = Succ->getLocation().getAs<BlockEdge>();
  if (!Edge.hasValue())
    return nullptr;

  auto Tag = Edge->getTag();
  if (!Tag)
    return nullptr;

  if (Tag->getTagDescription() != "cplusplus.SelfAssignment")
    return nullptr;

  Satisfied = true;

  const auto *Met =
      dyn_cast<CXXMethodDecl>(Succ->getCodeDecl().getAsFunction());
  assert(Met && "Not a C++ method.");
  assert((Met->isCopyAssignmentOperator() || Met->isMoveAssignmentOperator()) &&
         "Not a copy/move assignment operator.");

  const auto *LCtx = Edge->getLocationContext();

  const auto &State = Succ->getState();
  auto &SVB = State->getStateManager().getSValBuilder();

  const auto Param =
      State->getSVal(State->getRegion(Met->getParamDecl(0), LCtx));
  const auto This =
      State->getSVal(SVB.getCXXThis(Met, LCtx->getCurrentStackFrame()));

  auto L = PathDiagnosticLocation::create(Met, BRC.getSourceManager());

  if (!L.isValid() || !L.asLocation().isValid())
    return nullptr;

  SmallString<256> Buf;
  llvm::raw_svector_ostream Out(Buf);

  Out << "Assuming " << Met->getParamDecl(0)->getName() <<
    ((Param == This) ? " == " : " != ") << "*this";

  auto Piece = std::make_shared<PathDiagnosticEventPiece>(L, Out.str());
  Piece->addRange(Met->getSourceRange());

  return std::move(Piece);
}
