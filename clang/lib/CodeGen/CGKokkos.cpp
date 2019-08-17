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
#include <cstdio>
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "CodeGenFunction.h"


using namespace clang;
using namespace CodeGen;

namespace {

  /// Just a simple wrapper around some calls to prune down the 
  /// expressions we walk when doing codegen for kokkos constructs. 
  /// there are other options available within the Expr class to 
  /// get down to the fundamental types so in the future some tweaks
  /// to this implementation might be beneficial. 
  static const Expr *SimplifyExpr(const Expr *E) {
    return E->IgnoreImplicit()->IgnoreImpCasts();
  }

  /// \brief Extract the various expressions from the parallel_for. 
  ///
  /// Extract the various components from the CallExpr that
  /// corresponds to a kokkos parallel_for construct.  Given 
  /// that the construct has already passed through parsing and 
  /// sema we take some liberties here in sorting out the details
  /// of the construct -- in some cases we do not yet support all
  /// variants that are allowed in kokkos.  In particular, 
  ///
  ///    - functors are not supported. 
  ///    - named constructs are extracted but not yet handed off to
  ///      kokkos' underlying profiling/debugging infrastructure 
  ///      hooks. 
  ///    - there are likely still some holes in types that are legal 
  ///      constructs but not yet captured by the code below; these
  ///      cases should result in an 'llvm_unreachable' assertion. 
  ///   

  // FIXME -- functor support needs some thought... Seems unlikely 
  // that we can support all general cases (e.g., code in separate 
  // compilation units). 
  static void ExtractParallelForComponents(const CallExpr* CE,
					   std::string &CN, 
					   const Expr *& BE, 
					   const LambdaExpr *& LE)
  {
    // Currently supported kokkos constructs for parallel_for come
    // in two forms: 
    // 
    //   1. parallel_for(N, lambda_expr...);
    //
    //   2. parallel_for("name", N, lambda_expr...);
    // 

    // The first form contains "N", which represents the upper bounds
    // of the parallel_for execution (shorthand BE is used here for
    // bounds expr).  It can be a literal type, or an binary
    // expression.  The second parameter is the lamba expression (LE).
    //
    // The second form adds a string as the first parameter that is
    // used to name the construct (currently extracted and returned in
    // CN). The details of how this is utilized is up to the details
    // of emiting the parallel_for construct (i.e. it is not used here
    // and is simply returned to the caller).

    unsigned int curArgIndex = 0;

    // If the call expr starts with a "construct name" (we will assume
    // form #2 of parallel_for as shown above). We will extract the
    // actual string from the argument and return it to the caller via
    // the passed in 'CN' string.
    //
    // FIXME: All kokkos examples we have seen use a literal string
    // value here.  We are essentailly hard-coded to deal with this
    // form -- the code is not very robust in a situation where this
    // is not the case and ripe for a bug if encountered.
    const Expr *SE = SimplifyExpr(CE->getArg(curArgIndex));
    if (SE->getStmtClass() == Expr::CXXConstructExprClass) {
      const CXXConstructExpr *CXXCE = dyn_cast<CXXConstructExpr>(SE);
      SE = CXXCE->getArg(0)->IgnoreImplicit();
      if (SE->getStmtClass() == Expr::StringLiteralClass) {
	CN = dyn_cast<StringLiteral>(SE)->getString().str();
	curArgIndex++;
	SE = SimplifyExpr(CE->getArg(curArgIndex));
      } 
    } 

    // The next (or first if a string literal is not provided)
    // argument is the bounds (trip count) for the parallel_for.
    // It can take several forms depending on types of a literal 
    // value or even as a (binary) expression.
    //
    // FIXME: There are likely some missing cases here that could trip
    // us up at some point.  We have been through the parsing and sema
    // checks but we are likely susceptible to some C/C++ type
    // promotion/conversion rules.  Do we need to ponder our use of
    // "SimplifyExpr()" so it doesn't bite us in terms of disabling an
    // expected type converstion).
    if (SE->getStmtClass() == Expr::IntegerLiteralClass) {
      BE = SE;
      curArgIndex++;
      SE = SimplifyExpr(CE->getArg(curArgIndex));
    } else if (SE->getStmtClass() == Expr::BinaryOperatorClass) {
      BE = SE;
      curArgIndex++;
      SE = SimplifyExpr(CE->getArg(curArgIndex));
    } else if (SE->getStmtClass() == Expr::DeclRefExprClass) {
      BE = SE;
      curArgIndex++;
      SE = SimplifyExpr(CE->getArg(curArgIndex));
    } else {
      BE = nullptr;
      LE = nullptr;
      return;
    }

    if (SE->getStmtClass() == Expr::LambdaExprClass) {
      LE = dyn_cast<LambdaExpr>(SE);
    } else {
      LE = nullptr;
      return;
    }
  }


  // FIXME: This should probably be moved out of the kokkos-centric implementation.  
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

  // FIXME: This should probably be moved out of the kokkos-centric implementation.  
  // Helper routine copied from CodeGenFunction.cpp
  static void EmitIfUsed(CodeGenFunction &CGF, llvm::BasicBlock *BB) {
    if (!BB) return;
    if (!BB->use_empty())
      return CGF.CurFn->getBasicBlockList().push_back(BB);
    delete BB;
  }
}

// FIXME: This should probably be moved out of the kokkos-centric implementation.  
llvm::Instruction *CodeGenFunction::EmitSyncRegionStart() {
  // Start the sync region.  To ensure the syncregion.start call dominates all
  // uses of the generated token, we insert this call at the alloca insertion
  // point.
  llvm::Instruction *SRStart = llvm::CallInst::Create(
				CGM.getIntrinsic(llvm::Intrinsic::syncregion_start),
				"syncreg", AllocaInsertPt);
  return SRStart;
}

/// \brief Emit a kokkos-centric construct. 
///
/// This is our high-level entry point for lowering kokkos constructs
/// into the parallel-ir.  See each of the individual emit routines
/// for details on what is supported within each type of construct.
/// 
// FIXME: Right now this routine is somewhat redundant with the code in 
// CGExpr...  Could stand some refactoring to clean it up. One particular 
// feature that would be nice is a way to backtrack and fall through to 
// default C++ codegen if we hit a particular case that is currently not 
// supported. 
bool CodeGenFunction::EmitKokkosConstruct(const CallExpr *CE) {
  assert(CE != 0 && "CodeGenFunction::EmitKokkosConstruct: null callexpr passed!");

  const FunctionDecl *Func = CE->getDirectCallee();
  assert(Func != 0 && "Kokkos construct doesn't have a function declaration!");

  // FIXME: This is repeated code from CGExpr... 
  if (Func->getQualifiedNameAsString() == "Kokkos::parallel_for") {
    return EmitKokkosParallelFor(CE);
  } else if (Func->getQualifiedNameAsString() == "Kokkos::parallel_reduce") {
    return EmitKokkosParallelReduce(CE);
  } else {
    llvm_unreachable("unsupported kokkos construct encountered");
  }
}


// 
// FIXME: Need to add attributes back into the mix (perhaps more for the compiler 
// side of the house vs. the application folks at this point in time). 
//
// FIXME: As discussed above it would be nice to find a way to fallback to the 
// standard C++ codegen if we hit an unhandled construct/feature. 
// 
bool CodeGenFunction::EmitKokkosParallelFor(const CallExpr *CE) {
    
  ArrayRef<const Attr *> ForallAttrs; // FIXME: This is unused... 

  std::string      PFName; 
  const Expr       *BE = nullptr; // "bounds" expression
  const LambdaExpr *LE = nullptr; // the lambda  

  ExtractParallelForComponents(CE, PFName, BE, LE);
  
  if (LE == nullptr) { 
    DiagnosticsEngine &Diags = CGM.getDiags();
    Diags.Report(CE->getExprLoc(), diag::warn_kokkos_no_functor);
    return false;
  }

  if (BE == nullptr) {
    DiagnosticsEngine &Diags = CGM.getDiags();
    Diags.Report(CE->getExprLoc(), diag::warn_kokkos_unknown_bounds_expr);
    return false;
  }

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
  llvm::Value *LoopEnd = nullptr;

  if (BE->getStmtClass() == Expr::BinaryOperatorClass) {
    RValue RV = EmitAnyExprToTemp(BE);
    LoopEnd = RV.getScalarVal();
  } else { 
    LoopEnd = EmitScalarExpr(BE);
  }

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

  return true;
}

bool CodeGenFunction::EmitKokkosParallelReduce(const CallExpr *CE) {
  assert(false && "kokkos reductions currently not supported!");
  return false;
}

