//=== UnsequencedAccessChecker.cpp - Unsequenced access checker ---*- C++ -*--//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defined a checker for finding unsequenced accesses on the same
// memory region.
//
//===----------------------------------------------------------------------===//


#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

#include "clang/AST/ParentMapContext.h"

#include <iostream>

using namespace clang;
using namespace ento;

namespace {

using StmtVec = llvm::SmallVector<const Stmt*>;

class AccessStmt {
    llvm::PointerIntPair<const Stmt*, 1, bool> StmtAndIsStore;
    const StackFrameContext* StackFrame;
    friend bool operator==(const AccessStmt& A, const AccessStmt& B);
public:
    AccessStmt(const Stmt* Stmt, bool isStore,
               const StackFrameContext* StackFrame)
        : StmtAndIsStore(Stmt, isStore), StackFrame(StackFrame) {}

    bool isStore() const { return StmtAndIsStore.getInt(); }
    const Stmt* getStmt() const { return StmtAndIsStore.getPointer(); }
    const StackFrameContext* getFrame() const { return StackFrame; }

    void Profile(llvm::FoldingSetNodeID &ID) const {
        ID.AddBoolean(isStore());
        ID.AddPointer(getStmt());
        ID.AddPointer(getFrame());
    }
};

bool operator==(const AccessStmt& A, const AccessStmt& B) {
    return A.StmtAndIsStore == B.StmtAndIsStore &&
           A.StackFrame == B.StackFrame;
}

bool operator<(const AccessStmt& A, const AccessStmt& B) {
    return A.isStore() < B.isStore() || 
        (A.isStore() == B.isStore() && (A.getStmt() < B.getStmt() || 
            (A.getStmt() == B.getStmt() && A.getFrame() < B.getFrame())));
}

class UnsequencedAccessChecker
    : public Checker<check::Location> {

    static StmtVec getAllASTAncestors(const Stmt* CurrentS,
                                        const StackFrameContext* CurrentFrame,
                                        ParentMapContext& ParentMap);
    static int findCommonAncestorIdx(const StmtVec& A, const StmtVec& B);

    bool checkAncestors(int Common, AccessStmt A, const StmtVec& AVec,
                        AccessStmt B, const StmtVec& BVec,
                        const LangOptions& Opts) const;

    void reportBug(CheckerContext& C, const Stmt* Common, AccessStmt A,
                   AccessStmt B) const;

    ProgramStateRef resetState(ProgramStateRef S) const;

    BugType BT{this, "Unsequenced-access", categories::LogicError};

public:
    void checkLocation(SVal Location, bool isLoad, const Stmt *S,
                       CheckerContext &C) const;

    // Checker Options
    bool PrintDebugLog;
};
} // namespace

REGISTER_SET_FACTORY_WITH_PROGRAMSTATE(AccessStmtSet, AccessStmt);
REGISTER_MAP_WITH_PROGRAMSTATE(Accesses, const MemRegion*, AccessStmtSet);

StmtVec UnsequencedAccessChecker::getAllASTAncestors(
            const Stmt* CurrentS, const StackFrameContext* CurrentFrame,
            ParentMapContext& ParentMap) {

    StmtVec Ancestors;

    while (CurrentS) {

        DynTypedNodeList PL = ParentMap.getParents(*CurrentS);
        if (PL.size() != 1)
            break;

        const Stmt* ParentS = PL[0].get<Stmt>();

        if (!ParentS) {
            if (CurrentFrame->inTopFrame())
                break;

            ParentS = CurrentFrame->getCallSite();
            CurrentFrame = CurrentFrame->getParent()->getStackFrame();
        }

        Ancestors.push_back(CurrentS);

        CurrentS = ParentS;
    }

    Ancestors.push_back(CurrentS);

    return Ancestors;
}

static void print_ancestors(const StmtVec& ancestors) {
    for (const Stmt* S : ancestors) {
        std::cout << S->getStmtClassName() << " " << S << std::endl;
    }
}

ProgramStateRef UnsequencedAccessChecker::resetState(ProgramStateRef S) const {
    if (PrintDebugLog) {
        std::cout << "Reset State\n\n";
    }
    auto& MapFactory = S->get_context<Accesses>();
    return S->set<Accesses>(MapFactory.getEmptyMap());
}

void UnsequencedAccessChecker::checkLocation(
        SVal Location, bool isLoad, const Stmt *S, CheckerContext &C) const {

    ASTContext& ASTCtx = C.getASTContext();
    ParentMapContext& ParentMap = ASTCtx.getParentMapContext();
    ProgramStateRef State = C.getState();
    AccessStmtSet::Factory& SetFactory = State->get_context<AccessStmtSet>();

    const MemRegion* Region = Location.getAsRegion();
    AccessStmt Current(S, !isLoad, C.getStackFrame());
    StmtVec Ancestors = getAllASTAncestors(S, C.getStackFrame(), ParentMap);

    if (PrintDebugLog) {
        SourceLocation Loc = S->getBeginLoc();
        int Line = C.getSourceManager().getSpellingLineNumber(Loc);
        std::cout << (isLoad ? "Load" : "Store") << " on line " << Line
                  << std::endl;
        print_ancestors(Ancestors);
        std::cout << std::endl;
    }

    if (State->contains<Accesses>(Region)) {

        const AccessStmtSet* Accs = State->get<Accesses>(Region);

        for (AccessStmt AS : *Accs) {
            if (AS.isStore() || !isLoad) {
                StmtVec CurrentAncestors =
                    getAllASTAncestors(AS.getStmt(), AS.getFrame(), ParentMap);

                int CommonIdx =
                    findCommonAncestorIdx(Ancestors, CurrentAncestors);

                if (CommonIdx == -1)
                    continue;

                if(checkAncestors(CommonIdx, AS, Ancestors, Current,
                                  CurrentAncestors, C.getLangOpts())) {
                    const Stmt* CommonS =
                        Ancestors[Ancestors.size() - CommonIdx];
                    reportBug(C, CommonS, AS, Current);
                }

            }
        }
        AccessStmtSet Updated =
            SetFactory.add(*Accs, AccessStmt(S, !isLoad, C.getStackFrame()));

        C.addTransition(State->set<Accesses>(Region, Updated));
    } else {
        AccessStmtSet NewSet = SetFactory.add(
                SetFactory.getEmptySet(),
                AccessStmt(S, !isLoad, C.getStackFrame()));

        ProgramStateRef LocAdded = State->set<Accesses>(Region, NewSet);
        C.addTransition(LocAdded);
    }
}

int UnsequencedAccessChecker::findCommonAncestorIdx(const StmtVec& A,
                                                    const StmtVec& B) {
    unsigned I = 1;
    while (I <= std::min(A.size(), B.size())) {
        if (A[A.size() - I] != B[B.size() - I]) {
            break;
        }
        I++;
    }

    if (I == 1)
        return -1;

    return (I - 1);
}

bool UnsequencedAccessChecker::checkAncestors(
        int CommonIdx, AccessStmt A, const StmtVec& AVec, AccessStmt B,
        const StmtVec& BVec, const LangOptions& Opts) const {

    const Stmt* CommonS = AVec[AVec.size() - CommonIdx];

    if (const auto* BO = dyn_cast<BinaryOperator>(CommonS)) {
        if (BO->isLogicalOp() || BO->isCommaOp() || BO->isPtrMemOp())
            return false;

        if (Opts.CPlusPlus17 && BO->isShiftOp()) {
            return false;
        }

        if (BO->isAssignmentOp()) {
            const Stmt* Accessed = BO->getLHS()->IgnoreParens();
            if (Accessed == A.getStmt() && !B.isStore())
                return false;

            if (Accessed == B.getStmt() && !A.isStore())
                return false;
        }

        return true;
    }

    if (const auto* CE = dyn_cast<CallExpr>(CommonS)) {
        if (A.getStmt() == CommonS || B.getStmt() == CommonS) {
            // i.e. CommonS is the first element of one or both vectors.
            return false;
        }

        int MinSize = std::min(AVec.size(), BVec.size());
        assert(CommonIdx < MinSize);

        bool AFound = false;
        bool BFound = false;

        const Stmt* AFirstChild = AVec[AVec.size() - CommonIdx - 1];
        const Stmt* BFirstChild = BVec[BVec.size() - CommonIdx - 1];

        if (!Opts.CPlusPlus17 && AFirstChild == CE->getCallee())
            AFound = true;

        if (!Opts.CPlusPlus17 && BFirstChild == CE->getCallee())
            BFound = true;

        for (const Stmt* Arg : CE->arguments()) {
            if (Arg == AFirstChild)
                AFound = true;

            if (Arg == BFirstChild)
                BFound = true;

            if (AFound && BFound)
                return true;
        }

        return false;
    }

    return false;
}

void UnsequencedAccessChecker::reportBug(CheckerContext& C, const Stmt* Common,
                                         AccessStmt A, AccessStmt B) const {
    if (ExplodedNode* N = C.generateNonFatalErrorNode()) {
        PathDiagnosticLocation CommonLoc(Common, C.getSourceManager(),
                                         C.getLocationContext());

        PathDiagnosticLocation ALoc(A.getStmt(), C.getSourceManager(),
                                    C.getLocationContext());

        PathDiagnosticLocation BLoc(B.getStmt(), C.getSourceManager(),
                                    C.getLocationContext());

        bool bothStore = A.isStore() && B.isStore();

        auto BR = std::make_unique<PathSensitiveBugReport>(BT,
                bothStore ? "unsequenced writes to variable" 
                          : "unsequenced write to and read from variable", N,
                CommonLoc, nullptr);

        BR->addNote(A.isStore() ? "variable is written to here" 
                                : "variable is read from here", ALoc);

        BR->addNote(B.isStore() ? "variable is written to here" 
                                : "variable is read from here", BLoc);

        C.emitReport(std::move(BR));
    }
}

void ento::registerUnsequencedAccessChecker(CheckerManager &mgr) {
  auto* Checker = mgr.registerChecker<UnsequencedAccessChecker>();
  const AnalyzerOptions &Opts = mgr.getAnalyzerOptions();
  Checker->PrintDebugLog = Opts.getCheckerBooleanOption(Checker,
                                                        "PrintDebugLog");
}

bool ento::shouldRegisterUnsequencedAccessChecker(const CheckerManager &mgr) {
  return !mgr.getLangOpts().ObjC;
}
