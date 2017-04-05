//===----- CGCoroutine.cpp - Emit LLVM Code for C++ coroutines ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of coroutines.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "llvm/ADT/ScopeExit.h"
#include "clang/AST/StmtCXX.h"

using namespace clang;
using namespace CodeGen;

using llvm::Value;
using llvm::BasicBlock;

namespace {
enum class AwaitKind { Init, Normal, Yield, Final };
static constexpr llvm::StringLiteral AwaitKindStr[] = {"init", "await", "yield",
                                                       "final"};
}

struct clang::CodeGen::CGCoroData {
  // What is the current await expression kind and how many
  // await/yield expressions were encountered so far.
  // These are used to generate pretty labels for await expressions in LLVM IR.
  AwaitKind CurrentAwaitKind = AwaitKind::Init;
  unsigned AwaitNum = 0;
  unsigned YieldNum = 0;

  // How many co_return statements are in the coroutine. Used to decide whether
  // we need to add co_return; equivalent at the end of the user authored body.
  unsigned CoreturnCount = 0;

  // A branch to this block is emitted when coroutine needs to suspend.
  llvm::BasicBlock *SuspendBB = nullptr;

  // Stores the jump destination just before the coroutine memory is freed.
  // This is the destination that every suspend point jumps to for the cleanup
  // branch.
  CodeGenFunction::JumpDest CleanupJD;

  // Stores the jump destination just before the final suspend. The co_return
  // statements jumps to this point after calling return_xxx promise member.
  CodeGenFunction::JumpDest FinalJD;

  // Stores the llvm.coro.id emitted in the function so that we can supply it
  // as the first argument to coro.begin, coro.alloc and coro.free intrinsics.
  // Note: llvm.coro.id returns a token that cannot be directly expressed in a
  // builtin.
  llvm::CallInst *CoroId = nullptr;

  // If coro.id came from the builtin, remember the expression to give better
  // diagnostic. If CoroIdExpr is nullptr, the coro.id was created by
  // EmitCoroutineBody.
  CallExpr const *CoroIdExpr = nullptr;
};

// Defining these here allows to keep CGCoroData private to this file.
clang::CodeGen::CodeGenFunction::CGCoroInfo::CGCoroInfo() {}
CodeGenFunction::CGCoroInfo::~CGCoroInfo() {}

static void createCoroData(CodeGenFunction &CGF,
                           CodeGenFunction::CGCoroInfo &CurCoro,
                           llvm::CallInst *CoroId,
                           CallExpr const *CoroIdExpr = nullptr) {
  if (CurCoro.Data) {
    if (CurCoro.Data->CoroIdExpr)
      CGF.CGM.Error(CoroIdExpr->getLocStart(),
                    "only one __builtin_coro_id can be used in a function");
    else if (CoroIdExpr)
      CGF.CGM.Error(CoroIdExpr->getLocStart(),
                    "__builtin_coro_id shall not be used in a C++ coroutine");
    else
      llvm_unreachable("EmitCoroutineBodyStatement called twice?");

    return;
  }

  CurCoro.Data = std::unique_ptr<CGCoroData>(new CGCoroData);
  CurCoro.Data->CoroId = CoroId;
  CurCoro.Data->CoroIdExpr = CoroIdExpr;
}

// Synthesize a pretty name for a suspend point.
static SmallString<32> buildSuspendPrefixStr(CGCoroData &Coro, AwaitKind Kind) {
  unsigned No = 0;
  switch (Kind) {
  case AwaitKind::Init:
  case AwaitKind::Final:
    break;
  case AwaitKind::Normal:
    No = ++Coro.AwaitNum;
    break;
  case AwaitKind::Yield:
    No = ++Coro.YieldNum;
    break;
  }
  SmallString<32> Prefix(AwaitKindStr[static_cast<unsigned>(Kind)]);
  if (No > 1) {
    Twine(No).toVector(Prefix);
  }
  return Prefix;
}

// Emit suspend expression which roughly looks like:
//
//   auto && x = CommonExpr();
//   if (!x.await_ready()) {
//      llvm_coro_save();
//      x.await_suspend(...);     (*)
//      llvm_coro_suspend(); (**)
//   }
//   x.await_resume();
//
// where the result of the entire expression is the result of x.await_resume()
//
//   (*) If x.await_suspend return type is bool, it allows to veto a suspend:
//      if (x.await_suspend(...))
//        llvm_coro_suspend();
//
//  (**) llvm_coro_suspend() encodes three possible continuations as
//       a switch instruction:
//
//  %where-to = call i8 @llvm.coro.suspend(...)
//  switch i8 %where-to, label %coro.ret [ ; jump to epilogue to suspend
//    i8 0, label %yield.ready   ; go here when resumed
//    i8 1, label %yield.cleanup ; go here when destroyed
//  ]
//
//  See llvm's docs/Coroutines.rst for more details.
//
static RValue emitSuspendExpression(CodeGenFunction &CGF, CGCoroData &Coro,
                                    CoroutineSuspendExpr const &S,
                                    AwaitKind Kind, AggValueSlot aggSlot,
                                    bool ignoreResult) {
  auto *E = S.getCommonExpr();
  auto Binder =
      CodeGenFunction::OpaqueValueMappingData::bind(CGF, S.getOpaqueValue(), E);
  auto UnbindOnExit = llvm::make_scope_exit([&] { Binder.unbind(CGF); });

  auto Prefix = buildSuspendPrefixStr(Coro, Kind);
  BasicBlock *ReadyBlock = CGF.createBasicBlock(Prefix + Twine(".ready"));
  BasicBlock *SuspendBlock = CGF.createBasicBlock(Prefix + Twine(".suspend"));
  BasicBlock *CleanupBlock = CGF.createBasicBlock(Prefix + Twine(".cleanup"));

  // If expression is ready, no need to suspend.
  CGF.EmitBranchOnBoolExpr(S.getReadyExpr(), ReadyBlock, SuspendBlock, 0);

  // Otherwise, emit suspend logic.
  CGF.EmitBlock(SuspendBlock);

  auto &Builder = CGF.Builder;
  llvm::Function *CoroSave = CGF.CGM.getIntrinsic(llvm::Intrinsic::coro_save);
  auto *NullPtr = llvm::ConstantPointerNull::get(CGF.CGM.Int8PtrTy);
  auto *SaveCall = Builder.CreateCall(CoroSave, {NullPtr});

  auto *SuspendRet = CGF.EmitScalarExpr(S.getSuspendExpr());
  if (SuspendRet != nullptr) {
    // Veto suspension if requested by bool returning await_suspend.
    assert(SuspendRet->getType()->isIntegerTy(1) &&
           "Sema should have already checked that it is void or bool");
    BasicBlock *RealSuspendBlock =
        CGF.createBasicBlock(Prefix + Twine(".suspend.bool"));
    CGF.Builder.CreateCondBr(SuspendRet, RealSuspendBlock, ReadyBlock);
    SuspendBlock = RealSuspendBlock;
    CGF.EmitBlock(RealSuspendBlock);
  }

  // Emit the suspend point.
  const bool IsFinalSuspend = (Kind == AwaitKind::Final);
  llvm::Function *CoroSuspend =
      CGF.CGM.getIntrinsic(llvm::Intrinsic::coro_suspend);
  auto *SuspendResult = Builder.CreateCall(
      CoroSuspend, {SaveCall, Builder.getInt1(IsFinalSuspend)});

  // Create a switch capturing three possible continuations.
  auto *Switch = Builder.CreateSwitch(SuspendResult, Coro.SuspendBB, 2);
  Switch->addCase(Builder.getInt8(0), ReadyBlock);
  Switch->addCase(Builder.getInt8(1), CleanupBlock);

  // Emit cleanup for this suspend point.
  CGF.EmitBlock(CleanupBlock);
  CGF.EmitBranchThroughCleanup(Coro.CleanupJD);

  // Emit await_resume expression.
  CGF.EmitBlock(ReadyBlock);
  return CGF.EmitAnyExpr(S.getResumeExpr(), aggSlot, ignoreResult);
}

RValue CodeGenFunction::EmitCoawaitExpr(const CoawaitExpr &E,
                                        AggValueSlot aggSlot,
                                        bool ignoreResult) {
  return emitSuspendExpression(*this, *CurCoro.Data, E,
                               CurCoro.Data->CurrentAwaitKind, aggSlot,
                               ignoreResult);
}
RValue CodeGenFunction::EmitCoyieldExpr(const CoyieldExpr &E,
                                        AggValueSlot aggSlot,
                                        bool ignoreResult) {
  return emitSuspendExpression(*this, *CurCoro.Data, E, AwaitKind::Yield,
                               aggSlot, ignoreResult);
}

void CodeGenFunction::EmitCoreturnStmt(CoreturnStmt const &S) {
  ++CurCoro.Data->CoreturnCount;
  EmitStmt(S.getPromiseCall());
  EmitBranchThroughCleanup(CurCoro.Data->FinalJD);
}

// For WinEH exception representation backend need to know what funclet coro.end
// belongs to. That information is passed in a funclet bundle.
static SmallVector<llvm::OperandBundleDef, 1>
getBundlesForCoroEnd(CodeGenFunction &CGF) {
  SmallVector<llvm::OperandBundleDef, 1> BundleList;

  if (llvm::Instruction *EHPad = CGF.CurrentFuncletPad)
    BundleList.emplace_back("funclet", EHPad);

  return BundleList;
}

namespace {
// We will insert coro.end to cut any of the destructors for objects that
// do not need to be destroyed once the coroutine is resumed.
// See llvm/docs/Coroutines.rst for more details about coro.end.
struct CallCoroEnd final : public EHScopeStack::Cleanup {
  void Emit(CodeGenFunction &CGF, Flags flags) override {
    auto &CGM = CGF.CGM;
    auto *NullPtr = llvm::ConstantPointerNull::get(CGF.Int8PtrTy);
    llvm::Function *CoroEndFn = CGM.getIntrinsic(llvm::Intrinsic::coro_end);
    // See if we have a funclet bundle to associate coro.end with. (WinEH)
    auto Bundles = getBundlesForCoroEnd(CGF);
    auto *CoroEnd = CGF.Builder.CreateCall(
        CoroEndFn, {NullPtr, CGF.Builder.getTrue()}, Bundles);
    if (Bundles.empty()) {
      // Otherwise, (landingpad model), create a conditional branch that leads
      // either to a cleanup block or a block with EH resume instruction.
      auto *ResumeBB = CGF.getEHResumeBlock(/*cleanup=*/true);
      auto *CleanupContBB = CGF.createBasicBlock("cleanup.cont");
      CGF.Builder.CreateCondBr(CoroEnd, ResumeBB, CleanupContBB);
      CGF.EmitBlock(CleanupContBB);
    }
  }
};
}

namespace {
// Make sure to call coro.delete on scope exit.
struct CallCoroDelete final : public EHScopeStack::Cleanup {
  Stmt *Deallocate;

  // TODO: Wrap deallocate in if(coro.free(...)) Deallocate.
  void Emit(CodeGenFunction &CGF, Flags) override {
    // Note: That deallocation will be emitted twice: once for a normal exit and
    // once for exceptional exit. This usage is safe because Deallocate does not
    // contain any declarations. The SubStmtBuilder::makeNewAndDeleteExpr()
    // builds a single call to a deallocation function which is safe to emit
    // multiple times.
    CGF.EmitStmt(Deallocate);
  }
  explicit CallCoroDelete(Stmt *DeallocStmt) : Deallocate(DeallocStmt) {}
};
}

void CodeGenFunction::EmitCoroutineBody(const CoroutineBodyStmt &S) {
  auto *NullPtr = llvm::ConstantPointerNull::get(Builder.getInt8PtrTy());
  auto &TI = CGM.getContext().getTargetInfo();
  unsigned NewAlign = TI.getNewAlign() / TI.getCharWidth();

  auto *FinalBB = createBasicBlock("coro.final");
  auto *RetBB = createBasicBlock("coro.ret");

  auto *CoroId = Builder.CreateCall(
      CGM.getIntrinsic(llvm::Intrinsic::coro_id),
      {Builder.getInt32(NewAlign), NullPtr, NullPtr, NullPtr});
  createCoroData(*this, CurCoro, CoroId);
  CurCoro.Data->SuspendBB = RetBB;

  auto *AllocateCall = EmitScalarExpr(S.getAllocate());

  // Handle allocation failure if 'ReturnStmtOnAllocFailure' was provided.
  if (auto *RetOnAllocFailure = S.getReturnStmtOnAllocFailure()) {
    auto *RetOnFailureBB = createBasicBlock("coro.ret.on.failure");
    auto *InitBB = createBasicBlock("coro.init");

    // See if allocation was successful.
    auto *NullPtr = llvm::ConstantPointerNull::get(Int8PtrTy);
    auto *Cond = Builder.CreateICmpNE(AllocateCall, NullPtr);
    Builder.CreateCondBr(Cond, InitBB, RetOnFailureBB);

    // If not, return OnAllocFailure object.
    EmitBlock(RetOnFailureBB);
    EmitStmt(RetOnAllocFailure);

    EmitBlock(InitBB);
  }

  CurCoro.Data->CleanupJD = getJumpDestInCurrentScope(RetBB);
  {
    CodeGenFunction::RunCleanupsScope ResumeScope(*this);
    EHStack.pushCleanup<CallCoroDelete>(NormalAndEHCleanup, S.getDeallocate());

    EmitStmt(S.getPromiseDeclStmt());

    EHStack.pushCleanup<CallCoroEnd>(EHCleanup);

    CurCoro.Data->FinalJD = getJumpDestInCurrentScope(FinalBB);

    // FIXME: Emit initial suspend and more before the body.

    CurCoro.Data->CurrentAwaitKind = AwaitKind::Normal;
    EmitStmt(S.getBody());

    // See if we need to generate final suspend.
    const bool CanFallthrough = Builder.GetInsertBlock();
    const bool HasCoreturns = CurCoro.Data->CoreturnCount > 0;
    if (CanFallthrough || HasCoreturns) {
      EmitBlock(FinalBB);
      // FIXME: Emit final suspend.
    }
  }

  EmitBlock(RetBB);
  llvm::Function *CoroEnd = CGM.getIntrinsic(llvm::Intrinsic::coro_end);
  Builder.CreateCall(CoroEnd, {NullPtr, Builder.getFalse()});

  // FIXME: Emit return for the coroutine return object.
}

// Emit coroutine intrinsic and patch up arguments of the token type.
RValue CodeGenFunction::EmitCoroutineIntrinsic(const CallExpr *E,
                                               unsigned int IID) {
  SmallVector<llvm::Value *, 8> Args;
  switch (IID) {
  default:
    break;
  // The following three intrinsics take a token parameter referring to a token
  // returned by earlier call to @llvm.coro.id. Since we cannot represent it in
  // builtins, we patch it up here.
  case llvm::Intrinsic::coro_alloc:
  case llvm::Intrinsic::coro_begin:
  case llvm::Intrinsic::coro_free: {
    if (CurCoro.Data && CurCoro.Data->CoroId) {
      Args.push_back(CurCoro.Data->CoroId);
      break;
    }
    CGM.Error(E->getLocStart(), "this builtin expect that __builtin_coro_id has"
                                " been used earlier in this function");
    // Fallthrough to the next case to add TokenNone as the first argument.
  }
  // @llvm.coro.suspend takes a token parameter. Add token 'none' as the first
  // argument.
  case llvm::Intrinsic::coro_suspend:
    Args.push_back(llvm::ConstantTokenNone::get(getLLVMContext()));
    break;
  }
  for (auto &Arg : E->arguments())
    Args.push_back(EmitScalarExpr(Arg));

  llvm::Value *F = CGM.getIntrinsic(IID);
  llvm::CallInst *Call = Builder.CreateCall(F, Args);

  // If we see @llvm.coro.id remember it in the CoroData. We will update
  // coro.alloc, coro.begin and coro.free intrinsics to refer to it.
  if (IID == llvm::Intrinsic::coro_id) {
    createCoroData(*this, CurCoro, Call, E);
  }
  return RValue::get(Call);
}
