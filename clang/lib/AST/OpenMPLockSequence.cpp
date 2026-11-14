//===- OpenMPLockSequence.cpp - OpenMP lock sequences in a block ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/OpenMPLockSequence.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/AST/StmtOpenMP.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include <optional>

using namespace clang;

bool clang::isOMPLockRoutine(const Decl *D) {
  const auto *FD = dyn_cast_or_null<FunctionDecl>(D);
  if (!FD || !FD->isExternC() || !FD->getIdentifier())
    return false;
  return llvm::StringSwitch<bool>(FD->getName())
      .Cases({"omp_set_lock", "omp_unset_lock", "omp_test_lock",
              "omp_set_nest_lock", "omp_unset_nest_lock", "omp_test_nest_lock"},
             true)
      .Default(false);
}

bool clang::isOMPSetLockCall(const CallExpr *CE) {
  const FunctionDecl *FD = CE->getDirectCallee();
  return isOMPLockRoutine(FD) && FD->getName() == "omp_set_lock";
}

/// If \p S is the statement `Name(Arg);` calling the OpenMP lock routine
/// \p Name, return the call.
static const CallExpr *getLockCall(const Stmt *S, StringRef Name) {
  const auto *E = dyn_cast<Expr>(S);
  if (!E)
    return nullptr;
  const auto *CE = dyn_cast<CallExpr>(E->IgnoreImplicit()->IgnoreParens());
  if (!CE || CE->getNumArgs() != 1)
    return nullptr;
  const FunctionDecl *FD = CE->getDirectCallee();
  if (!isOMPLockRoutine(FD) || FD->getName() != Name)
    return nullptr;
  return CE;
}

/// If \p S is the statement `Name(Arg);` calling the OpenMP lock routine
/// \p Name, return Arg.
static const Expr *getLockCallArg(const Stmt *S, StringRef Name) {
  const CallExpr *CE = getLockCall(S, Name);
  return CE ? CE->getArg(0)->IgnoreParenImpCasts() : nullptr;
}

/// Checks that control can enter \p S only at its start and leave it only at
/// its end, and that \p S contains no OpenMP directive and makes no use of an
/// OpenMP lock routine. Returns why not, if it does not. \p InLoop and
/// \p InSwitch say whether an enclosing loop or switch statement is itself part
/// of the sequence being checked.
static std::optional<OMPLockSequenceRejection>
checkSerializable(const Stmt *S, bool InLoop, bool InSwitch) {
  using R = OMPLockSequenceRejection;
  if (!S)
    return std::nullopt;
  if (isa<OMPExecutableDirective>(S))
    return R::Directive;
  if (isa<ReturnStmt, GotoStmt, IndirectGotoStmt, LabelStmt, CoreturnStmt,
          CoroutineSuspendExpr, CXXTryStmt, SEHTryStmt, SEHLeaveStmt,
          ObjCAtTryStmt, ObjCAtThrowStmt>(S))
    return R::Escape;
  if (const auto *Asm = dyn_cast<GCCAsmStmt>(S); Asm && Asm->isAsmGoto())
    return R::Escape;
  // A lock taken or released conditionally, or other than in a pair at the top
  // level of the sequence, could be held from one thread's turn to the next.
  if (const auto *DRE = dyn_cast<DeclRefExpr>(S);
      DRE && isOMPLockRoutine(DRE->getDecl()))
    return R::OtherLockUse;
  if (isa<BreakStmt>(S))
    return InLoop || InSwitch ? std::nullopt : std::optional(R::Escape);
  if (isa<ContinueStmt>(S))
    return InLoop ? std::nullopt : std::optional(R::Escape);
  if (isa<SwitchCase>(S) && !InSwitch)
    return R::Escape;
  // The bodies of lambdas and blocks belong to other functions.
  if (isa<LambdaExpr, BlockExpr>(S))
    return std::nullopt;
  bool IsLoop = isa<ForStmt, WhileStmt, DoStmt, CXXForRangeStmt>(S);
  bool IsSwitch = isa<SwitchStmt>(S);
  for (const Stmt *Child : S->children())
    if (auto Why =
            checkSerializable(Child, InLoop || IsLoop, InSwitch || IsSwitch))
      return Why;
  return std::nullopt;
}

OMPLockSequence clang::matchOMPLockSequence(ArrayRef<Stmt *> Stmts,
                                            const ASTContext &Ctx) {
  using R = OMPLockSequenceRejection;
  OMPLockSequence Result;
  if (Stmts.empty())
    return Result;

  // Code generation cannot put a label inside the loop that runs the threads
  // through the sequence one at a time, since control would enter the loop
  // through it.
  const Stmt *First = Stmts.front();
  if (isa<LabelStmt, SwitchCase>(First)) {
    const Stmt *Sub = First;
    while (const auto *L = dyn_cast<LabelStmt>(Sub))
      Sub = L->getSubStmt();
    while (const auto *C = dyn_cast<SwitchCase>(Sub))
      Sub = C->getSubStmt();
    if ((Result.SetCall = getLockCall(Sub, "omp_set_lock")))
      Result.Rejection = R::Labelled;
    return Result;
  }

  Result.SetCall = getLockCall(First, "omp_set_lock");
  if (!Result.SetCall)
    return Result;
  auto GetLockID =
      [&](const Expr *Lock) -> std::optional<llvm::FoldingSetNodeID> {
    if (Lock->HasSideEffects(Ctx))
      return std::nullopt;
    llvm::FoldingSetNodeID ID;
    Lock->Profile(ID, Ctx, /*Canonical=*/true);
    return ID;
  };
  auto Reject = [&](R Why) {
    Result.Rejection = Why;
    return Result;
  };

  // The locks set in the sequence and not yet unset, innermost last.
  SmallVector<llvm::FoldingSetNodeID, 2> Held;
  auto ID = GetLockID(Result.SetCall->getArg(0)->IgnoreParenImpCasts());
  if (!ID)
    return Reject(R::SideEffects);
  Held.push_back(*ID);
  for (unsigned I = 1, E = Stmts.size(); I != E; ++I) {
    const Stmt *S = Stmts[I];
    if (const Expr *Set = getLockCallArg(S, "omp_set_lock")) {
      auto ID = GetLockID(Set);
      if (!ID)
        return Reject(R::SideEffects);
      if (llvm::is_contained(Held, *ID))
        return Reject(R::OtherLockUse);
      Held.push_back(*ID);
      continue;
    }
    if (const Expr *Unset = getLockCallArg(S, "omp_unset_lock")) {
      auto ID = GetLockID(Unset);
      if (!ID)
        return Reject(R::SideEffects);
      if (*ID != Held.back())
        return Reject(R::OtherLockUse);
      Held.pop_back();
      if (Held.empty()) {
        Result.Length = I + 1;
        return Result;
      }
      continue;
    }
    // A declaration would remain in scope after the sequence ends.
    if (isa<DeclStmt>(S))
      return Reject(R::Declaration);
    if (auto Why = checkSerializable(S, /*InLoop=*/false, /*InSwitch=*/false))
      return Reject(*Why);
  }
  return Reject(R::NoUnset);
}
