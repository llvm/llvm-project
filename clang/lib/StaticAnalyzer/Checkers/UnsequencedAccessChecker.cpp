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
} // namespace

REGISTER_SET_FACTORY_WITH_PROGRAMSTATE(AccessStmtSet, AccessStmt);
REGISTER_MAP_WITH_PROGRAMSTATE(Loads, const MemRegion*, AccessStmtSet);
REGISTER_MAP_WITH_PROGRAMSTATE(Stores, const MemRegion*, AccessStmtSet);

// Keep these around to make sure we don't walk too far up the stack too often.
// These are also useful for state resets.
REGISTER_TRAIT_WITH_PROGRAMSTATE(HighestUnseqFrame, const StackFrameContext*);
REGISTER_TRAIT_WITH_PROGRAMSTATE(HighestUnseqStmt, const Stmt*);

namespace {

class UnsequencedAccessChecker
    : public Checker<check::Location> {

    static StmtVec getAllASTAncestors(
            const Stmt* CurrentS, const StackFrameContext* CurrentFrame,
            ParentMapContext& ParentMap, const Stmt* HighestS,
            const StackFrameContext* HighestFrame,
            bool* DidHitHighest = nullptr);
    static int findCommonAncestorIdx(const StmtVec& A, const StmtVec& B);

    void checkAgainstSet(const AccessStmtSet* Set, AccessStmt CurrentAccess,
                         const StmtVec& Ancestors, const Stmt* HighestStmt,
                         const StackFrameContext* HighestFrame,
                         CheckerContext& C, bool* FoundSame) const;

    bool checkAncestors(int Common, AccessStmt A, const StmtVec& AVec,
                        AccessStmt B, const StmtVec& BVec,
                        const LangOptions& Opts) const;

    std::pair<const Stmt*, const StackFrameContext*>
    findHighestUnsequencedStmt(const Stmt* CurrentS,
                               const StackFrameContext* CurrentFrame,
                               const LangOptions& Opts, 
                               ParentMapContext& ParentMap) const;

    bool isUnsequencedStmt(const Stmt* S, const LangOptions& Opts) const;

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

StmtVec UnsequencedAccessChecker::getAllASTAncestors(
            const Stmt* CurrentS, const StackFrameContext* CurrentFrame,
            ParentMapContext& ParentMap, const Stmt* HighestS,
            const StackFrameContext* HighestFrame, bool* DidHitHighest) {

    assert(CurrentS != nullptr);
    assert(CurrentFrame != nullptr);

    StmtVec Ancestors;

    while (true) {

        if (CurrentS == HighestS && CurrentFrame == HighestFrame) {
            if (DidHitHighest != nullptr)
                *DidHitHighest = true;
            break;
        }

        DynTypedNodeList PL = ParentMap.getParents(*CurrentS);
        const Stmt* ParentS;

        if (PL.size() == 1) {
            // This could also yield a nullptr
            ParentS = PL[0].get<Stmt>();
        } else {
            ParentS = nullptr;
        }

        if (!ParentS) {
            // A call can originate from a declaration and therefore the call
            // site may be null. Treat that as top level for now.
            if (CurrentFrame->inTopFrame() || 
                    CurrentFrame->getCallSite() == nullptr) {

                if (DidHitHighest != nullptr)
                    *DidHitHighest = false;
                break;
            }

            ParentS = CurrentFrame->getCallSite();
            CurrentFrame = CurrentFrame->getParent()->getStackFrame();
        }
        assert(ParentS != nullptr);

        Ancestors.push_back(CurrentS);

        CurrentS = ParentS;
    }

    Ancestors.push_back(CurrentS);

    return Ancestors;
}

std::pair<const Stmt*, const StackFrameContext*>
UnsequencedAccessChecker::findHighestUnsequencedStmt(
        const Stmt* CurrentS, const StackFrameContext* CurrentFrame,
        const LangOptions& Opts, ParentMapContext& ParentMap) const {

    assert(CurrentS != nullptr);
    assert(CurrentFrame != nullptr);

    const Stmt* HighestUnseqS = CurrentS;
    const StackFrameContext* HighestUnseqFrame = CurrentFrame;

    while (true) {

        if (isUnsequencedStmt(CurrentS, Opts)) {
            HighestUnseqS = CurrentS;
            HighestUnseqFrame = CurrentFrame;
        }

        DynTypedNodeList PL = ParentMap.getParents(*CurrentS);
        const Stmt* ParentS;

        if (PL.size() == 1) {
            // This could also yield a nullptr
            ParentS = PL[0].get<Stmt>();
        } else {
            ParentS = nullptr;
        }

        if (!ParentS) {
            // A call can originate from a declaration and therefore the call
            // site may be null. Treat that as top level for now.
            if (CurrentFrame->inTopFrame() || 
                    CurrentFrame->getCallSite() == nullptr)
                break;

            ParentS = CurrentFrame->getCallSite();
            CurrentFrame = CurrentFrame->getParent()->getStackFrame();
        }

        assert(ParentS != nullptr);
        CurrentS = ParentS;
    }

    return std::make_pair(HighestUnseqS, HighestUnseqFrame);
}

bool UnsequencedAccessChecker::isUnsequencedStmt(
        const Stmt* S, const LangOptions& Opts) const {

    if (isa<CallExpr>(S))
        return true;

    if (const auto* BO = dyn_cast<BinaryOperator>(S)) {
        if (BO->isLogicalOp() || BO->isCommaOp() || BO->isPtrMemOp())
            return false;

        if (Opts.CPlusPlus17 && BO->isShiftOp())
            return false;
        
        return true;
    }

    return false;
}

static void print_ancestors(const StmtVec& ancestors) {
    for (const Stmt* S : ancestors) {
        std::cout << S->getStmtClassName() << " " << S << std::endl;
    }
}

static int get_stmt_line_num(const Stmt* S, CheckerContext& C) {
    SourceLocation Loc = S->getBeginLoc();
    int Line = C.getSourceManager().getSpellingLineNumber(Loc);
    return Line;
}

ProgramStateRef UnsequencedAccessChecker::resetState(ProgramStateRef S) const {
    if (PrintDebugLog)
        std::cout << "Reset State\n\n";

    auto& MapFactory = S->get_context<Loads>();
    return S->set<Loads>(MapFactory.getEmptyMap())
            ->set<Stores>(MapFactory.getEmptyMap())
            ->set<HighestUnseqStmt>(nullptr)
            ->set<HighestUnseqFrame>(nullptr);
}

void UnsequencedAccessChecker::checkLocation(
        SVal Location, bool isLoad, const Stmt *S, CheckerContext &C) const {

    ASTContext& ASTCtx = C.getASTContext();
    ParentMapContext& ParentMap = ASTCtx.getParentMapContext();
    ProgramStateRef State = C.getState();
    AccessStmtSet::Factory& SetFactory = State->get_context<AccessStmtSet>();

    const MemRegion* Region = Location.getAsRegion();
    AccessStmt CurrentAccess(S, !isLoad, C.getStackFrame());

    const Stmt* HighestStmt = State->get<HighestUnseqStmt>();
    const StackFrameContext* HighestFrame = State->get<HighestUnseqFrame>();

    bool DidHitHighest = false;

    StmtVec Ancestors = getAllASTAncestors(S, C.getStackFrame(), ParentMap,
                                           HighestStmt, HighestFrame,
                                           &DidHitHighest);

    if (!DidHitHighest) {
        std::pair<const Stmt*, const StackFrameContext*> Highest =
            findHighestUnsequencedStmt(S, C.getStackFrame(), C.getLangOpts(),
                                       ParentMap);
        HighestStmt = Highest.first;
        HighestFrame = Highest.second;

        // This could be optimized.
        Ancestors = getAllASTAncestors(S, C.getStackFrame(), ParentMap,
                                       HighestStmt, HighestFrame);

        State = resetState(State);
        State = State->set<HighestUnseqStmt>(HighestStmt)
                     ->set<HighestUnseqFrame>(HighestFrame);
    }

    if (PrintDebugLog) {
        std::cout << (isLoad ? "Load" : "Store") << " on line "
                  << get_stmt_line_num(S, C) << std::endl;
        print_ancestors(Ancestors);
        std::cout << std::endl;
    }

    bool FoundSameLoad = false;
    bool FoundSameStore = false;

    if (!isLoad && State->contains<Loads>(Region))
        checkAgainstSet(State->get<Loads>(Region), CurrentAccess, Ancestors,
                        HighestStmt, HighestFrame, C, &FoundSameLoad);

    if (State->contains<Stores>(Region))
        checkAgainstSet(State->get<Stores>(Region), CurrentAccess, Ancestors,
                        HighestStmt, HighestFrame, C, &FoundSameStore);

    if (isLoad && !FoundSameLoad && !FoundSameStore) {

        AccessStmtSet CurrSet = State->contains<Loads>(Region) 
            ? *State->get<Loads>(Region) : SetFactory.getEmptySet();

        AccessStmtSet Updated = SetFactory.add(CurrSet, CurrentAccess);

        C.addTransition(State->set<Loads>(Region, Updated));

    } else if (PrintDebugLog && isLoad) {
        std::cout << "Not adding load: " << CurrentAccess.getStmt()
                  << std::endl << std::endl;
    }

    if (!isLoad && !FoundSameStore) {

        if (FoundSameLoad) {
            if (PrintDebugLog) {
                std::cout << "Removing load: " << CurrentAccess.getStmt()
                          << std::endl << std::endl;
            }

            AccessStmt LoadAccess(CurrentAccess.getStmt(), /*isStore*/false,
                                  CurrentAccess.getFrame());
            AccessStmtSet LoadRemoved =
                SetFactory.remove(*State->get<Loads>(Region), LoadAccess);
            State = State->set<Loads>(Region, LoadRemoved);
        }

        AccessStmtSet CurrSet = State->contains<Stores>(Region) 
            ? *State->get<Stores>(Region) : SetFactory.getEmptySet();

        AccessStmtSet Updated = SetFactory.add(CurrSet, CurrentAccess);

        C.addTransition(State->set<Stores>(Region, Updated));

    } else if (PrintDebugLog && !isLoad) {
        std::cout << "Not adding store: " << CurrentAccess.getStmt()
                  << std::endl << std::endl;
    }
}

void UnsequencedAccessChecker::checkAgainstSet(
        const AccessStmtSet* Set, AccessStmt CurrentAccess,
        const StmtVec& Ancestors, const Stmt* HighestStmt,
        const StackFrameContext* HighestFrame, CheckerContext& C,
        bool* FoundSame) const {

    ASTContext& ASTCtx = C.getASTContext();
    ParentMapContext& ParentMap = ASTCtx.getParentMapContext();

    for (AccessStmt OtherAccess : *Set) {

        if (CurrentAccess.getStmt() == OtherAccess.getStmt() &&
            CurrentAccess.getFrame() == OtherAccess.getFrame()) {
            *FoundSame = true;
            if (PrintDebugLog)
                std::cout << "Found same, continuing.\n";
            continue;
        }

        StmtVec CurrentAncestors =
            getAllASTAncestors(OtherAccess.getStmt(), OtherAccess.getFrame(),
                               ParentMap, HighestStmt, HighestFrame);

        int CommonIdx = findCommonAncestorIdx(Ancestors, CurrentAncestors);
        if (CommonIdx == -1)
            continue;

        if(checkAncestors(CommonIdx, OtherAccess, Ancestors, CurrentAccess,
                          CurrentAncestors, C.getLangOpts())) {

            const Stmt* CommonS = Ancestors[Ancestors.size() - CommonIdx];

            if (PrintDebugLog) {
                std::cout << "Common found: " << CommonS << " "
                          << CommonS->getStmtClassName() << " Line " 
                          << get_stmt_line_num(CommonS, C) << std::endl
                          << std::endl;
            }

            reportBug(C, CommonS, OtherAccess, CurrentAccess);
        }
    }
}

int UnsequencedAccessChecker::findCommonAncestorIdx(const StmtVec& A,
                                                    const StmtVec& B) {
    unsigned I = 1;
    while (I <= std::min(A.size(), B.size())) {
        if (A[A.size() - I] != B[B.size() - I])
            break;
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

    if (const auto* AS = dyn_cast<ArraySubscriptExpr>(CommonS)) {
        if (Opts.CPlusPlus17)
            return false;

        if (A.getStmt() == AS || B.getStmt() == AS)
            return false;

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
    PathDiagnosticLocation CommonLoc(Common, C.getSourceManager(),
                                     C.getLocationContext());

    PathDiagnosticLocation ALoc(A.getStmt(), C.getSourceManager(),
                                C.getLocationContext());

    PathDiagnosticLocation BLoc(B.getStmt(), C.getSourceManager(),
                                C.getLocationContext());

    bool bothStore = A.isStore() && B.isStore();

    auto BR = std::make_unique<BasicBugReport>(BT,
            bothStore ? "unsequenced writes to variable" 
                      : "unsequenced write to and read from variable",
            CommonLoc);

    BR->addNote(A.isStore() ? "variable is written to here" 
                            : "variable is read from here", ALoc);

    BR->addNote(B.isStore() ? "variable is written to here" 
                            : "variable is read from here", BLoc);

    C.emitReport(std::move(BR));
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
