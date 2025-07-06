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

using StmtVec = llvm::SmallVector<const Stmt*>;

namespace {
class UnsequencedAccessChecker
    : public Checker<check::Location> {

    static StmtVec getAllASTAncestors(const Stmt* CurrentS,
                                      const StackFrameContext* CurrentFrame,
                                      ParentMapContext& ParentMap);
    static const Stmt* findCommonAncestor(const StmtVec& A, const StmtVec& B);

    void reportBug(CheckerContext& C, const Stmt* A, const Stmt* B) const;

    BugType BT{this, "Unsequenced-access", categories::LogicError};

public:
    void checkLocation(SVal Location, bool isLoad, const Stmt *S,
                       CheckerContext &C) const;
};

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
REGISTER_MAP_WITH_PROGRAMSTATE(Accesses, const MemRegion*, AccessStmtSet);

StmtVec UnsequencedAccessChecker::getAllASTAncestors(
            const Stmt* CurrentS, const StackFrameContext* CurrentFrame,
            ParentMapContext& ParentMap) {

    StmtVec Ancestors;

    while (CurrentS) {
        Ancestors.push_back(CurrentS);

        DynTypedNodeList PL = ParentMap.getParents(*CurrentS);
        if (PL.size() != 1)
            break;

        CurrentS = PL[0].get<Stmt>();

        if (!CurrentS) {
            if (CurrentFrame->inTopFrame())
                break;

            CurrentS = CurrentFrame->getCallSite();
            CurrentFrame = CurrentFrame->getParent()->getStackFrame();
        }
    }

    return Ancestors;
}

void UnsequencedAccessChecker::checkLocation(
        SVal Location, bool isLoad, const Stmt *S, CheckerContext &C) const {

    ASTContext& ASTCtx = C.getASTContext();
    ParentMapContext& ParentMap = ASTCtx.getParentMapContext();
    ProgramStateRef State = C.getState();
    AccessStmtSet::Factory& SetFactory = State->get_context<AccessStmtSet>();
    auto& MapFactory = State->get_context<Accesses>();

    const MemRegion* Region = Location.getAsRegion();
    StmtVec Ancestors = getAllASTAncestors(S, C.getStackFrame(), ParentMap);

    if (State->contains<Accesses>(Region)) {

        const AccessStmtSet* Accs = State->get<Accesses>(Region);

        for (AccessStmt AS : *Accs) {
            if (AS.isStore() || !isLoad) {
                StmtVec OtherAncestors = 
                    getAllASTAncestors(AS.getStmt(), AS.getFrame(), ParentMap);
                const Stmt* Common = 
                    findCommonAncestor(Ancestors, OtherAncestors);

                if (!Common) {
                    State = State->set<Accesses>(MapFactory.getEmptyMap());
                    break;
                }

                if (const auto* BO = dyn_cast<BinaryOperator>(Common)) {
                    if (BO->isAdditiveOp()) {
                        reportBug(C, S, AS.getStmt());
                    }
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

const Stmt* UnsequencedAccessChecker::findCommonAncestor(const StmtVec& A,
                                                         const StmtVec& B) {
    unsigned I = 1;
    while (I <= std::min(A.size(), B.size())) {
        if (A[A.size() - I] != B[B.size() - I]) {
            break;
        }
        I++;
    }

    if (I == 1)
        return nullptr;

    return A[A.size() - (I - 1)];
}

void UnsequencedAccessChecker::reportBug(CheckerContext& C, const Stmt* A,
                                         const Stmt* B) const {
    if (ExplodedNode* N = C.generateErrorNode()) {
        auto BR = std::make_unique<PathSensitiveBugReport>(BT, "hello", N);
        C.emitReport(std::move(BR));
    }
}

void ento::registerUnsequencedAccessChecker(CheckerManager &mgr) {
  mgr.registerChecker<UnsequencedAccessChecker>();
}

bool ento::shouldRegisterUnsequencedAccessChecker(const CheckerManager &mgr) {
  return true;
}
