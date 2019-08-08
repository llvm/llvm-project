/**
 ***************************************************************************
 * Copyright (c) 2017, Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2010. Los Alamos National Security, LLC. This software was
 *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of Los Alamos National Security, LLC, Los
 *      Alamos National Laboratory, LANL, the U.S. Government, nor the
 *      names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 *
 ***************************************************************************/

#include "CodeGenFunction.h"
using namespace clang;
using namespace CodeGen;

namespace {

  // A helper function used to extract the lamba expression 
  // used in Kokkos.  Note that the expression is sometimes 
  // wrapped in other AST expressions (e.g. casts). 
  static const LambdaExpr* ExtractLambdaExpr(const Expr *E) 
  {
    if (auto me = dyn_cast<MaterializeTemporaryExpr>(E)) {
      E = me->GetTemporaryExpr();
    }

    if (const CastExpr* c = dyn_cast<CastExpr>(E)) {
      E = c->getSubExpr();
    }

    if (const CXXBindTemporaryExpr *c = dyn_cast<CXXBindTemporaryExpr>(E)) {
      E = c->getSubExpr();
    }

    return dyn_cast<LambdaExpr>(E);
  }

  /// \brief Cleanup to ensure parent stack frame is synced.
  struct RethrowCleanup : public EHScopeStack::Cleanup {
    llvm::BasicBlock *InvokeDest;
  public:
    RethrowCleanup(llvm::BasicBlock *InvokeDest = nullptr)
      : InvokeDest(InvokeDest) {}
    virtual ~RethrowCleanup() {}
    void Emit(CodeGenFunction &CGF, Flags F) {
      llvm::BasicBlock *DetRethrowBlock = CGF.createBasicBlock("det.rethrow");
      if (InvokeDest)
        CGF.Builder.CreateInvoke(
          CGF.CGM.getIntrinsic(llvm::Intrinsic::detached_rethrow),
          DetRethrowBlock, InvokeDest);
      else
        CGF.Builder.CreateBr(DetRethrowBlock);
      CGF.EmitBlock(DetRethrowBlock);
    }
  };

  // Helper routine copied from CodeGenFunction.cpp
  static void EmitIfUsed(CodeGenFunction &CGF, llvm::BasicBlock *BB) {
    if (!BB) return;
    if (!BB->use_empty())
      return CGF.CurFn->getBasicBlockList().push_back(BB);
    delete BB;
  }

}

llvm::Instruction *CodeGenFunction::EmitSyncRegionStart() {
  // Start the sync region.  To ensure the syncregion.start call dominates all
  // uses of the generated token, we insert this call at the alloca insertion
  // point.
  llvm::Instruction *SRStart = llvm::CallInst::Create(
      CGM.getIntrinsic(llvm::Intrinsic::syncregion_start),
      "syncreg", AllocaInsertPt);
  return SRStart;
}

void CodeGenFunction::EmitKokkosConstruct(const CallExpr *CE) {
  using namespace std;

  assert(CE != 0 && "CodeGenFunction::EmitKokkosConstruct: null callexpr passed!");

  const FunctionDecl *Func = CE->getDirectCallee();
  assert(Func != 0 && "Kokkos construct doesn't have a function declaration!");

  // FIXME: This is repeated code from CGExpr... 
  if (Func->getQualifiedNameAsString() == "Kokkos::parallel_for") {
    EmitKokkosParallelFor(CE);
  } else if (Func->getQualifiedNameAsString() == "Kokkos::parallel_reduce") {
    EmitKokkosParallelReduce(CE);
  } else {
    assert(false && "unsupported Kokkos construct!");
  }
}


void CodeGenFunction::EmitKokkosParallelFor(const CallExpr *CE) {
    
  ArrayRef<const Attr *> ForallAttrs; // FIXME: This is unused... 

  // A Kokkos parallel-for is a lambda expression construct -- our 
  // first step is to extract the lambda, which we will then be 
  // transformed into a parallel-for loop body when we have 
  // completed lowering. 
  const LambdaExpr *LE = ExtractLambdaExpr(CE->getArg(1));
  assert(LE && "EmitKokkosParallelFor -- unable to extract lambda!");  
  
  JumpDest LoopExit = getJumpDestInCurrentScope("kokkos.forall.end");
  PushSyncRegion();
  llvm::Instruction *SRStart = EmitSyncRegionStart();
  CurSyncRegion->setSyncRegionStart(SRStart);
  
  LexicalScope ForallScope(*this, CE->getSourceRange());

  // Transform the kokkos lambda expression into a loop.  

  // The first step is to extract the argument to the lamba and convert it 
  // into the loop iterator...  
  // 
  // FIXME: We assume some aspects of the kokkos parallel for construct here
  //        that need to be carefully consdiered...  in particular, 
  //   - the iterator can be assigned a value of zero. 
  //   - the details of what is captured in the lambda seems to be mostly 
  //     ignored... 
  const CXXMethodDecl *MD = LE->getCallOperator();
  assert(MD && "EmitKokkosParallelFor() -- bad method decl!");

  const ParmVarDecl *LoopVar = MD->getParamDecl(0);
  assert(LoopVar && "EmitKokkosParallelFor() -- bad loop variable!");
  EmitVarDecl(*LoopVar);
  Address Addr = GetAddrOfLocalVar(LoopVar);
  llvm::Value *Zero = llvm::ConstantInt::get(ConvertType(LoopVar->getType()), 0);
  Builder.CreateStore(Zero, Addr);

  // Next, work towards determining the end of the loop range.
  llvm::Value *LoopEnd   = EmitScalarExpr(CE->getArg(0));
  llvm::Type  *LoopVarTy = ConvertType(LoopVar->getType());
  unsigned NBits  = LoopEnd->getType()->getPrimitiveSizeInBits();
  unsigned LVBits = LoopVarTy->getPrimitiveSizeInBits();
  // We may need to truncate/extend the range to get it to match 
  // the type of loop variable. 
  if (NBits > LVBits) {
    LoopEnd = Builder.CreateTrunc(LoopEnd, LoopVarTy);
  } else if (NBits < LVBits) {
    LoopEnd = Builder.CreateZExt(LoopEnd, LoopVarTy);
  } else {
    // bit count matches, nothing to do... 
  }

  JumpDest Continue = getJumpDestInCurrentScope("kokkos.forall.cond");
  llvm::BasicBlock *CondBlock = Continue.getBlock();
  EmitBlock(CondBlock);

  // FIXME: Need to get attributes for spawning strategy from 
  // code versus this hard-coded route... 
  LoopStack.setSpawnStrategy(LoopAttributes::DAC);
  const SourceRange &R = CE->getSourceRange();
  LoopStack.push(CondBlock, CGM.getContext(), ForallAttrs,
                 SourceLocToDebugLoc(R.getBegin()),
                 SourceLocToDebugLoc(R.getEnd()));
  JumpDest Preattach = getJumpDestInCurrentScope("kokkos.forall.preattach");
  Continue = getJumpDestInCurrentScope("kokkos.forall.inc");

  // Store the blocks to use for break and continue. 
  // 
  // FIXME?: Why is the code below BreakContinue(Preattach, Preattach)
  // versus BreakContinue(Preattach, Continue)?  
  BreakContinueStack.push_back(BreakContinue(Preattach, Preattach));

  // Create a clean up scope for the condition variable. 
  LexicalScope ConditionalScope(*this, R);

  // Save the old alloca insertion point. 
  llvm::AssertingVH<llvm::Instruction> OldAllocaInsertPt = AllocaInsertPt;
  // Save the old exception handling state. 
  llvm::BasicBlock *OldEHResumeBlock  = EHResumeBlock;
  llvm::Value      *OldExceptionSlot  = ExceptionSlot;
  llvm::AllocaInst *OldEHSelectorSlot = EHSelectorSlot;

  llvm::BasicBlock *SyncContinueBlock = createBasicBlock("kokkos.end.continue");
  bool madeSync = false;


  llvm::BasicBlock  *DetachBlock;
  llvm::BasicBlock  *ForallBodyEntry;
  llvm::BasicBlock  *ForallBody;

  {
    llvm::BasicBlock *ExitBlock = LoopExit.getBlock();
    // If there is any cleanup between here and the loop-exit scope
    // we need to create a block to stage the loop exit. 
    if (ForallScope.requiresCleanups()) {
      ExitBlock = createBasicBlock("kokkos.cond.cleanup");
    }

    // As long as the conditional is true we continue looping... 
    DetachBlock = createBasicBlock("kokkos.forall.detach");
    // Emit extra entry block for the detached body, this ensures 
    // that the detached block has only one predecessor. 
    ForallBodyEntry = createBasicBlock("kokkos.forall.body.entry");
    ForallBody      = createBasicBlock("kokkos.forall.body");

    llvm::Value *LoopVal     = Builder.CreateLoad(Addr);
    llvm::Value *BoolCondVal = Builder.CreateICmpULT(LoopVal, LoopEnd);
    Builder.CreateCondBr(BoolCondVal, DetachBlock, ExitBlock);

    if (ExitBlock != LoopExit.getBlock()) {
      EmitBlock(ExitBlock);
      Builder.CreateSync(SyncContinueBlock, SRStart);
      EmitBlock(SyncContinueBlock);
      PopSyncRegion();
      madeSync = true;
      EmitBranchThroughCleanup(LoopExit);
    }

    EmitBlock(DetachBlock);
    Builder.CreateDetach(ForallBodyEntry, Continue.getBlock(), SRStart);

    // Create a new alloca insertion point.
    llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);
    AllocaInsertPt = new llvm::BitCastInst(Undef, Int32Ty, 
       "", ForallBodyEntry);
    // Set up nested exception handling state. 
    EHResumeBlock  = nullptr;
    ExceptionSlot  = nullptr;
    EHSelectorSlot = nullptr;
    EmitBlock(ForallBodyEntry);
  }

  // Create a scope for the loop-variable cleanup.
  RunCleanupsScope DetachCleanupScope(*this);
  EHStack.pushCleanup<RethrowCleanup>(EHCleanup);

  Builder.CreateBr(ForallBody);
  EmitBlock(ForallBody);
  incrementProfileCounter(CE);

  {
    // Create a separate cleanup scope for the forall body
    // (in case it is not a compound statement).
    RunCleanupsScope BodyScope(*this);

    // Emit the lambda expression as the body of the forall 
    // loop.  Given this is a lambda it may have special wrapped 
    // AST for handling captured variables -- to address this we 
    // have to flag it so we handle it as a special case... 
    InKokkosConstruct = true;
    EmitStmt(LE->getBody());
    InKokkosConstruct = false;
    Builder.CreateBr(Preattach.getBlock());
  }
  
  {
    EmitBlock(Preattach.getBlock());
    DetachCleanupScope.ForceCleanup();
    Builder.CreateReattach(Continue.getBlock(), SRStart);
  }

    {
    llvm::Instruction *Ptr = AllocaInsertPt;
    AllocaInsertPt = OldAllocaInsertPt;
    Ptr->eraseFromParent();

    // Restore the exception handling state. 
    EmitIfUsed(*this, EHResumeBlock);
    EHResumeBlock  = OldEHResumeBlock;
    ExceptionSlot  = OldExceptionSlot;
    EHSelectorSlot = OldEHSelectorSlot;
  }

  // Emit the increment next. 
  EmitBlock(Continue.getBlock());

  // Emit the loop variable increment. 
  llvm::Value *IncVal = Builder.CreateLoad(Addr);
  llvm::Value *One    = llvm::ConstantInt::get(ConvertType(LoopVar->getType()), 1);
  IncVal = Builder.CreateAdd(IncVal, One);
  Builder.CreateStore(IncVal, Addr);
  BreakContinueStack.pop_back();
  ConditionalScope.ForceCleanup();

  EmitStopPoint(CE);
  EmitBranch(CondBlock);
  ForallScope.ForceCleanup();
  LoopStack.pop();

  // Emit the fall-through block. 
  EmitBlock(LoopExit.getBlock(), true);
  if (!madeSync) {
    Builder.CreateSync(SyncContinueBlock, SRStart);
    EmitBlock(SyncContinueBlock);
    PopSyncRegion();
  }
}

void CodeGenFunction::EmitKokkosParallelReduce(const CallExpr *CE) {
  assert(false && "kokkos reductions currently not supported!");
}

