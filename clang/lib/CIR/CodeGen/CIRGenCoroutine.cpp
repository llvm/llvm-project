//===----- CGCoroutine.cpp - Emit CIR Code for C++ coroutines -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of coroutines.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtVisitor.h"
#include "llvm/ADT/ScopeExit.h"

using namespace clang;
using namespace cir;

struct cir::CGCoroData {
  // What is the current await expression kind and how many
  // await/yield expressions were encountered so far.
  // These are used to generate pretty labels for await expressions in LLVM IR.
  // AwaitKind CurrentAwaitKind = AwaitKind::Init;
  // unsigned AwaitNum = 0;
  // unsigned YieldNum = 0;

  // How many co_return statements are in the coroutine. Used to decide whether
  // we need to add co_return; equivalent at the end of the user authored body.
  unsigned CoreturnCount = 0;

  // A branch to this block is emitted when coroutine needs to suspend.
  // llvm::BasicBlock *SuspendBB = nullptr;

  // The promise type's 'unhandled_exception' handler, if it defines one.
  Stmt *ExceptionHandler = nullptr;

  // A temporary i1 alloca that stores whether 'await_resume' threw an
  // exception. If it did, 'true' is stored in this variable, and the coroutine
  // body must be skipped. If the promise type does not define an exception
  // handler, this is null.
  // llvm::Value *ResumeEHVar = nullptr;

  // Stores the jump destination just before the coroutine memory is freed.
  // This is the destination that every suspend point jumps to for the cleanup
  // branch.
  // CodeGenFunction::JumpDest CleanupJD;

  // Stores the jump destination just before the final suspend. The co_return
  // statements jumps to this point after calling return_xxx promise member.
  // CodeGenFunction::JumpDest FinalJD;

  // Stores the llvm.coro.id emitted in the function so that we can supply it
  // as the first argument to coro.begin, coro.alloc and coro.free intrinsics.
  // Note: llvm.coro.id returns a token that cannot be directly expressed in a
  // builtin.
  // llvm::CallInst *CoroId = nullptr;

  // Stores the llvm.coro.begin emitted in the function so that we can replace
  // all coro.frame intrinsics with direct SSA value of coro.begin that returns
  // the address of the coroutine frame of the current coroutine.
  // llvm::CallInst *CoroBegin = nullptr;

  // Stores the last emitted coro.free for the deallocate expressions, we use it
  // to wrap dealloc code with if(auto mem = coro.free) dealloc(mem).
  // llvm::CallInst *LastCoroFree = nullptr;

  // If coro.id came from the builtin, remember the expression to give better
  // diagnostic. If CoroIdExpr is nullptr, the coro.id was created by
  // EmitCoroutineBody.
  CallExpr const *CoroIdExpr = nullptr;
};

// Defining these here allows to keep CGCoroData private to this file.
CIRGenFunction::CGCoroInfo::CGCoroInfo() {}
CIRGenFunction::CGCoroInfo::~CGCoroInfo() {}

static void createCoroData(CIRGenFunction &CGF,
                           CIRGenFunction::CGCoroInfo &CurCoro) {
  if (CurCoro.Data) {
    // if (CurCoro.Data->CoroIdExpr)
    //   CGF.CGM.Error(CoroIdExpr->getBeginLoc(),
    //                 "only one __builtin_coro_id can be used in a function");
    // else if (CoroIdExpr)
    //   CGF.CGM.Error(CoroIdExpr->getBeginLoc(),
    //                 "__builtin_coro_id shall not be used in a C++
    //                 coroutine");
    // else
    llvm_unreachable("EmitCoroutineBodyStatement called twice?");

    return;
  }

  CurCoro.Data = std::unique_ptr<CGCoroData>(new CGCoroData);
  // CurCoro.Data->CoroId = CoroId;
  // CurCoro.Data->CoroIdExpr = CoroIdExpr;
}

namespace {
// FIXME: both GetParamRef and ParamReferenceReplacerRAII are good template
// candidates to be shared among LLVM / CIR codegen.

// Hunts for the parameter reference in the parameter copy/move declaration.
struct GetParamRef : public StmtVisitor<GetParamRef> {
public:
  DeclRefExpr *Expr = nullptr;
  GetParamRef() {}
  void VisitDeclRefExpr(DeclRefExpr *E) {
    assert(Expr == nullptr && "multilple declref in param move");
    Expr = E;
  }
  void VisitStmt(Stmt *S) {
    for (auto *C : S->children()) {
      if (C)
        Visit(C);
    }
  }
};

// This class replaces references to parameters to their copies by changing
// the addresses in CGF.LocalDeclMap and restoring back the original values in
// its destructor.
struct ParamReferenceReplacerRAII {
  CIRGenFunction::DeclMapTy SavedLocals;
  CIRGenFunction::DeclMapTy &LocalDeclMap;

  ParamReferenceReplacerRAII(CIRGenFunction::DeclMapTy &LocalDeclMap)
      : LocalDeclMap(LocalDeclMap) {}

  void addCopy(DeclStmt const *PM) {
    // Figure out what param it refers to.

    assert(PM->isSingleDecl());
    VarDecl const *VD = static_cast<VarDecl const *>(PM->getSingleDecl());
    Expr const *InitExpr = VD->getInit();
    GetParamRef Visitor;
    Visitor.Visit(const_cast<Expr *>(InitExpr));
    assert(Visitor.Expr);
    DeclRefExpr *DREOrig = Visitor.Expr;
    auto *PD = DREOrig->getDecl();

    auto it = LocalDeclMap.find(PD);
    assert(it != LocalDeclMap.end() && "parameter is not found");
    SavedLocals.insert({PD, it->second});

    auto copyIt = LocalDeclMap.find(VD);
    assert(copyIt != LocalDeclMap.end() && "parameter copy is not found");
    it->second = copyIt->getSecond();
  }

  ~ParamReferenceReplacerRAII() {
    for (auto &&SavedLocal : SavedLocals) {
      LocalDeclMap.insert({SavedLocal.first, SavedLocal.second});
    }
  }
};
} // namespace

// Emit coroutine intrinsic and patch up arguments of the token type.
RValue CIRGenFunction::buildCoroutineIntrinsic(const CallExpr *E,
                                               unsigned int IID) {
  llvm_unreachable("NYI");
}

static mlir::LogicalResult buildBodyAndFallthrough(CIRGenFunction &CGF,
                                                   const CoroutineBodyStmt &S,
                                                   Stmt *Body) {
  if (CGF.buildStmt(Body, /*useCurrentScope=*/true).failed())
    return mlir::failure();
  // From LLVM codegen:
  // const bool CanFallthrough = CGF.Builder.GetInsertBlock();
  if (S.getFallthroughHandler()) {
    llvm_unreachable("NYI");
    // if (Stmt *OnFallthrough = S.getFallthroughHandler())
    //   CGF.buildStmt(OnFallthrough, /*useCurrentScope=*/true);
  }
  return mlir::success();
}

mlir::LogicalResult
CIRGenFunction::buildCoroutineBody(const CoroutineBodyStmt &S) {
  // This is very different from LLVM codegen as the current intent is to
  // not expand too much of it here and leave it to dialect codegen.
  // In the LLVM world, this is where we create calls to coro.id,
  // coro.alloc and coro.begin.
  createCoroData(*this, CurCoro);

  // Handle allocation failure if 'ReturnStmtOnAllocFailure' was provided.
  if (auto *RetOnAllocFailure = S.getReturnStmtOnAllocFailure())
    llvm_unreachable("NYI");

  {
    // FIXME: create a new scope to copy out the params?
    // LLVM create scope cleanups here, but might be due to the use
    // of many basic blocks?
    assert(!UnimplementedFeature::generateDebugInfo() && "NYI");
    ParamReferenceReplacerRAII ParamReplacer(LocalDeclMap);

    // Create mapping between parameters and copy-params for coroutine
    // function.
    llvm::ArrayRef<const Stmt *> ParamMoves = S.getParamMoves();
    assert((ParamMoves.size() == 0 || (ParamMoves.size() == FnArgs.size())) &&
           "ParamMoves and FnArgs should be the same size for coroutine "
           "function");
    // For zipping the arg map into debug info.
    assert(!UnimplementedFeature::generateDebugInfo() && "NYI");

    // Create parameter copies. We do it before creating a promise, since an
    // evolution of coroutine TS may allow promise constructor to observe
    // parameter copies.
    for (auto *PM : S.getParamMoves()) {
      if (buildStmt(PM, /*useCurrentScope=*/true).failed())
        return mlir::failure();
      ParamReplacer.addCopy(cast<DeclStmt>(PM));
    }

    if (buildStmt(S.getPromiseDeclStmt(), /*useCurrentScope=*/true).failed())
      return mlir::failure();

    // Address promiseAddr = GetAddrOfLocalVar(S.getPromiseDecl());
    // auto *PromiseAddrVoidPtr =
    //     new llvm::BitCastInst(promiseAddr.getPointer(), VoidPtrTy, "",
    //     CoroId);
    // // Update CoroId to refer to the promise. We could not do it earlier
    // // because promise local variable was not emitted yet.
    // CoroId->setArgOperand(1, PromiseAddrVoidPtr);

    // ReturnValue should be valid as long as the coroutine's return type
    // is not void. The assertion could help us to reduce the check later.
    assert(ReturnValue.isValid() == (bool)S.getReturnStmt());
    // Now we have the promise, initialize the GRO.
    // We need to emit `get_return_object` first. According to:
    // [dcl.fct.def.coroutine]p7
    // The call to get_return_­object is sequenced before the call to
    // initial_suspend and is invoked at most once.
    //
    // So we couldn't emit return value when we emit return statment,
    // otherwise the call to get_return_object wouldn't be in front
    // of initial_suspend.
    if (ReturnValue.isValid()) {
      buildAnyExprToMem(S.getReturnValue(), ReturnValue,
                        S.getReturnValue()->getType().getQualifiers(),
                        /*IsInit*/ true);
    }

    // EHStack.pushCleanup<CallCoroEnd>(EHCleanup);

    // CurCoro.Data->CurrentAwaitKind = AwaitKind::Init;
    // CurCoro.Data->ExceptionHandler = S.getExceptionHandler();
    if (buildStmt(S.getInitSuspendStmt(), /*useCurrentScope=*/true).failed())
      return mlir::failure();
    // CurCoro.Data->FinalJD = getJumpDestInCurrentScope(FinalBB);

    // CurCoro.Data->CurrentAwaitKind = AwaitKind::Normal;

    if (S.getExceptionHandler()) {
      llvm_unreachable("NYI");
    } else {
      if (buildBodyAndFallthrough(*this, S, S.getBody()).failed())
        return mlir::failure();
    }

    // See if we need to generate final suspend.
    // const bool CanFallthrough = Builder.GetInsertBlock();
    // FIXME: LLVM tracks fallthrough by checking the insertion
    // point is valid, we can probably do better.
    const bool CanFallthrough = false;
    const bool HasCoreturns = CurCoro.Data->CoreturnCount > 0;
    if (CanFallthrough || HasCoreturns) {
      // CurCoro.Data->CurrentAwaitKind = AwaitKind::Final;
      if (buildStmt(S.getFinalSuspendStmt(), /*useCurrentScope=*/true).failed())
        return mlir::failure();
    }
  }
  return mlir::success();
}