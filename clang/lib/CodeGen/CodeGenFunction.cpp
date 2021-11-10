//===--- CodeGenFunction.cpp - Emit LLVM Code from ASTs for a Function ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This coordinates the per-function state used while generating code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CGBlocks.h"
#include "CGCUDARuntime.h"
#include "CGCXXABI.h"
#include "CGCleanup.h"
#include "CGDebugInfo.h"
#include "CGOpenMPRuntime.h"
#include "CodeGenModule.h"
#include "CodeGenPGO.h"
#include "TargetInfo.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTLambda.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtObjC.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/CodeGenOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/FPEnv.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/CRC.h"
#include "llvm/Transforms/Scalar/LowerExpectIntrinsic.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

using namespace clang;
using namespace CodeGen;

/// shouldEmitLifetimeMarkers - Decide whether we need emit the life-time
/// markers.
static bool shouldEmitLifetimeMarkers(const CodeGenOptions &CGOpts,
                                      const LangOptions &LangOpts) {
  if (CGOpts.DisableLifetimeMarkers)
    return false;

  // Sanitizers may use markers.
  if (CGOpts.SanitizeAddressUseAfterScope ||
      LangOpts.Sanitize.has(SanitizerKind::HWAddress) ||
      LangOpts.Sanitize.has(SanitizerKind::Memory))
    return true;

  // For now, only in optimized builds.
  return CGOpts.OptimizationLevel != 0;
}

CodeGenFunction::CodeGenFunction(CodeGenModule &cgm, bool suppressNewContext)
    : CodeGenTypeCache(cgm), CGM(cgm), Target(cgm.getTarget()),
      Builder(cgm, cgm.getModule().getContext(), llvm::ConstantFolder(),
              CGBuilderInserterTy(this)),
      SanOpts(CGM.getLangOpts().Sanitize), CurFPFeatures(CGM.getLangOpts()),
      DebugInfo(CGM.getModuleDebugInfo()), PGO(cgm),
      ShouldEmitLifetimeMarkers(
          shouldEmitLifetimeMarkers(CGM.getCodeGenOpts(), CGM.getLangOpts())) {
  if (!suppressNewContext)
    CGM.getCXXABI().getMangleContext().startNewFunction();
  EHStack.setCGF(this);

  SetFastMathFlags(CurFPFeatures);
}

CodeGenFunction::~CodeGenFunction() {
  assert(LifetimeExtendedCleanupStack.empty() && "failed to emit a cleanup");

  if (getLangOpts().OpenMP && CurFn)
    CGM.getOpenMPRuntime().functionFinished(*this);

  // If we have an OpenMPIRBuilder we want to finalize functions (incl.
  // outlining etc) at some point. Doing it once the function codegen is done
  // seems to be a reasonable spot. We do it here, as opposed to the deletion
  // time of the CodeGenModule, because we have to ensure the IR has not yet
  // been "emitted" to the outside, thus, modifications are still sensible.
  if (CGM.getLangOpts().OpenMPIRBuilder && CurFn)
    CGM.getOpenMPRuntime().getOMPBuilder().finalize(CurFn);
}

// Map the LangOption for exception behavior into
// the corresponding enum in the IR.
llvm::fp::ExceptionBehavior
clang::ToConstrainedExceptMD(LangOptions::FPExceptionModeKind Kind) {

  switch (Kind) {
  case LangOptions::FPE_Ignore:  return llvm::fp::ebIgnore;
  case LangOptions::FPE_MayTrap: return llvm::fp::ebMayTrap;
  case LangOptions::FPE_Strict:  return llvm::fp::ebStrict;
  }
  llvm_unreachable("Unsupported FP Exception Behavior");
}

void CodeGenFunction::SetFastMathFlags(FPOptions FPFeatures) {
  llvm::FastMathFlags FMF;
  FMF.setAllowReassoc(FPFeatures.getAllowFPReassociate());
  FMF.setNoNaNs(FPFeatures.getNoHonorNaNs());
  FMF.setNoInfs(FPFeatures.getNoHonorInfs());
  FMF.setNoSignedZeros(FPFeatures.getNoSignedZero());
  FMF.setAllowReciprocal(FPFeatures.getAllowReciprocal());
  FMF.setApproxFunc(FPFeatures.getAllowApproxFunc());
  FMF.setAllowContract(FPFeatures.allowFPContractAcrossStatement());
  Builder.setFastMathFlags(FMF);
}

CodeGenFunction::CGFPOptionsRAII::CGFPOptionsRAII(CodeGenFunction &CGF,
                                                  const Expr *E)
    : CGF(CGF) {
  ConstructorHelper(E->getFPFeaturesInEffect(CGF.getLangOpts()));
}

CodeGenFunction::CGFPOptionsRAII::CGFPOptionsRAII(CodeGenFunction &CGF,
                                                  FPOptions FPFeatures)
    : CGF(CGF) {
  ConstructorHelper(FPFeatures);
}

void CodeGenFunction::CGFPOptionsRAII::ConstructorHelper(FPOptions FPFeatures) {
  OldFPFeatures = CGF.CurFPFeatures;
  CGF.CurFPFeatures = FPFeatures;

  OldExcept = CGF.Builder.getDefaultConstrainedExcept();
  OldRounding = CGF.Builder.getDefaultConstrainedRounding();

  if (OldFPFeatures == FPFeatures)
    return;

  FMFGuard.emplace(CGF.Builder);

  llvm::RoundingMode NewRoundingBehavior =
      static_cast<llvm::RoundingMode>(FPFeatures.getRoundingMode());
  CGF.Builder.setDefaultConstrainedRounding(NewRoundingBehavior);
  auto NewExceptionBehavior =
      ToConstrainedExceptMD(static_cast<LangOptions::FPExceptionModeKind>(
          FPFeatures.getFPExceptionMode()));
  CGF.Builder.setDefaultConstrainedExcept(NewExceptionBehavior);

  CGF.SetFastMathFlags(FPFeatures);

  assert((CGF.CurFuncDecl == nullptr || CGF.Builder.getIsFPConstrained() ||
          isa<CXXConstructorDecl>(CGF.CurFuncDecl) ||
          isa<CXXDestructorDecl>(CGF.CurFuncDecl) ||
          (NewExceptionBehavior == llvm::fp::ebIgnore &&
           NewRoundingBehavior == llvm::RoundingMode::NearestTiesToEven)) &&
         "FPConstrained should be enabled on entire function");

  auto mergeFnAttrValue = [&](StringRef Name, bool Value) {
    auto OldValue =
        CGF.CurFn->getFnAttribute(Name).getValueAsBool();
    auto NewValue = OldValue & Value;
    if (OldValue != NewValue)
      CGF.CurFn->addFnAttr(Name, llvm::toStringRef(NewValue));
  };
  mergeFnAttrValue("no-infs-fp-math", FPFeatures.getNoHonorInfs());
  mergeFnAttrValue("no-nans-fp-math", FPFeatures.getNoHonorNaNs());
  mergeFnAttrValue("no-signed-zeros-fp-math", FPFeatures.getNoSignedZero());
  mergeFnAttrValue("unsafe-fp-math", FPFeatures.getAllowFPReassociate() &&
                                         FPFeatures.getAllowReciprocal() &&
                                         FPFeatures.getAllowApproxFunc() &&
                                         FPFeatures.getNoSignedZero());
}

CodeGenFunction::CGFPOptionsRAII::~CGFPOptionsRAII() {
  CGF.CurFPFeatures = OldFPFeatures;
  CGF.Builder.setDefaultConstrainedExcept(OldExcept);
  CGF.Builder.setDefaultConstrainedRounding(OldRounding);
}

LValue CodeGenFunction::MakeNaturalAlignAddrLValue(llvm::Value *V, QualType T) {
  LValueBaseInfo BaseInfo;
  TBAAAccessInfo TBAAInfo;
  CharUnits Alignment = CGM.getNaturalTypeAlignment(T, &BaseInfo, &TBAAInfo);
  return LValue::MakeAddr(Address(V, Alignment), T, getContext(), BaseInfo,
                          TBAAInfo);
}

/// Given a value of type T* that may not be to a complete object,
/// construct an l-value with the natural pointee alignment of T.
LValue
CodeGenFunction::MakeNaturalAlignPointeeAddrLValue(llvm::Value *V, QualType T) {
  LValueBaseInfo BaseInfo;
  TBAAAccessInfo TBAAInfo;
  CharUnits Align = CGM.getNaturalTypeAlignment(T, &BaseInfo, &TBAAInfo,
                                                /* forPointeeType= */ true);
  return MakeAddrLValue(Address(V, Align), T, BaseInfo, TBAAInfo);
}


llvm::Type *CodeGenFunction::ConvertTypeForMem(QualType T) {
  return CGM.getTypes().ConvertTypeForMem(T);
}

llvm::Type *CodeGenFunction::ConvertType(QualType T) {
  return CGM.getTypes().ConvertType(T);
}

TypeEvaluationKind CodeGenFunction::getEvaluationKind(QualType type) {
  type = type.getCanonicalType();
  while (true) {
    switch (type->getTypeClass()) {
#define TYPE(name, parent)
#define ABSTRACT_TYPE(name, parent)
#define NON_CANONICAL_TYPE(name, parent) case Type::name:
#define DEPENDENT_TYPE(name, parent) case Type::name:
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(name, parent) case Type::name:
#include "clang/AST/TypeNodes.inc"
      llvm_unreachable("non-canonical or dependent type in IR-generation");

    case Type::Auto:
    case Type::DeducedTemplateSpecialization:
      llvm_unreachable("undeduced type in IR-generation");

    // Various scalar types.
    case Type::Builtin:
    case Type::Pointer:
    case Type::BlockPointer:
    case Type::LValueReference:
    case Type::RValueReference:
    case Type::MemberPointer:
    case Type::Vector:
    case Type::ExtVector:
    case Type::ConstantMatrix:
    case Type::FunctionProto:
    case Type::FunctionNoProto:
    case Type::Enum:
    case Type::ObjCObjectPointer:
    case Type::Pipe:
    case Type::ExtInt:
      return TEK_Scalar;

    // Complexes.
    case Type::Complex:
      return TEK_Complex;

    // Arrays, records, and Objective-C objects.
    case Type::ConstantArray:
    case Type::IncompleteArray:
    case Type::VariableArray:
    case Type::Record:
    case Type::ObjCObject:
    case Type::ObjCInterface:
      return TEK_Aggregate;

    // We operate on atomic values according to their underlying type.
    case Type::Atomic:
      type = cast<AtomicType>(type)->getValueType();
      continue;
    }
    llvm_unreachable("unknown type kind!");
  }
}

llvm::DebugLoc CodeGenFunction::EmitReturnBlock() {
  // For cleanliness, we try to avoid emitting the return block for
  // simple cases.
  llvm::BasicBlock *CurBB = Builder.GetInsertBlock();

  if (CurBB) {
    assert(!CurBB->getTerminator() && "Unexpected terminated block.");

    // We have a valid insert point, reuse it if it is empty or there are no
    // explicit jumps to the return block.
    if (CurBB->empty() || ReturnBlock.getBlock()->use_empty()) {
      ReturnBlock.getBlock()->replaceAllUsesWith(CurBB);
      delete ReturnBlock.getBlock();
      ReturnBlock = JumpDest();
    } else
      EmitBlock(ReturnBlock.getBlock());
    return llvm::DebugLoc();
  }

  // Otherwise, if the return block is the target of a single direct
  // branch then we can just put the code in that block instead. This
  // cleans up functions which started with a unified return block.
  if (ReturnBlock.getBlock()->hasOneUse()) {
    llvm::BranchInst *BI =
      dyn_cast<llvm::BranchInst>(*ReturnBlock.getBlock()->user_begin());
    if (BI && BI->isUnconditional() &&
        BI->getSuccessor(0) == ReturnBlock.getBlock()) {
      // Record/return the DebugLoc of the simple 'return' expression to be used
      // later by the actual 'ret' instruction.
      llvm::DebugLoc Loc = BI->getDebugLoc();
      Builder.SetInsertPoint(BI->getParent());
      BI->eraseFromParent();
      delete ReturnBlock.getBlock();
      ReturnBlock = JumpDest();
      return Loc;
    }
  }

  // FIXME: We are at an unreachable point, there is no reason to emit the block
  // unless it has uses. However, we still need a place to put the debug
  // region.end for now.

  EmitBlock(ReturnBlock.getBlock());
  return llvm::DebugLoc();
}

static void EmitIfUsed(CodeGenFunction &CGF, llvm::BasicBlock *BB) {
  if (!BB) return;
  if (!BB->use_empty())
    return CGF.CurFn->getBasicBlockList().push_back(BB);
  delete BB;
}

void CodeGenFunction::FinishFunction(SourceLocation EndLoc) {
  assert(BreakContinueStack.empty() &&
         "mismatched push/pop in break/continue stack!");

  bool OnlySimpleReturnStmts = NumSimpleReturnExprs > 0
    && NumSimpleReturnExprs == NumReturnExprs
    && ReturnBlock.getBlock()->use_empty();
  // Usually the return expression is evaluated before the cleanup
  // code.  If the function contains only a simple return statement,
  // such as a constant, the location before the cleanup code becomes
  // the last useful breakpoint in the function, because the simple
  // return expression will be evaluated after the cleanup code. To be
  // safe, set the debug location for cleanup code to the location of
  // the return statement.  Otherwise the cleanup code should be at the
  // end of the function's lexical scope.
  //
  // If there are multiple branches to the return block, the branch
  // instructions will get the location of the return statements and
  // all will be fine.
  if (CGDebugInfo *DI = getDebugInfo()) {
    if (OnlySimpleReturnStmts)
      DI->EmitLocation(Builder, LastStopPoint);
    else
      DI->EmitLocation(Builder, EndLoc);
  }

  // Pop any cleanups that might have been associated with the
  // parameters.  Do this in whatever block we're currently in; it's
  // important to do this before we enter the return block or return
  // edges will be *really* confused.
  bool HasCleanups = EHStack.stable_begin() != PrologueCleanupDepth;
  bool HasOnlyLifetimeMarkers =
      HasCleanups && EHStack.containsOnlyLifetimeMarkers(PrologueCleanupDepth);
  bool EmitRetDbgLoc = !HasCleanups || HasOnlyLifetimeMarkers;
  if (HasCleanups) {
    // Make sure the line table doesn't jump back into the body for
    // the ret after it's been at EndLoc.
    Optional<ApplyDebugLocation> AL;
    if (CGDebugInfo *DI = getDebugInfo()) {
      if (OnlySimpleReturnStmts)
        DI->EmitLocation(Builder, EndLoc);
      else
        // We may not have a valid end location. Try to apply it anyway, and
        // fall back to an artificial location if needed.
        AL = ApplyDebugLocation::CreateDefaultArtificial(*this, EndLoc);
    }

    PopCleanupBlocks(PrologueCleanupDepth);
  }

  // Emit function epilog (to return).
  llvm::DebugLoc Loc = EmitReturnBlock();

  if (ShouldInstrumentFunction()) {
    if (CGM.getCodeGenOpts().InstrumentFunctions)
      CurFn->addFnAttr("instrument-function-exit", "__cyg_profile_func_exit");
    if (CGM.getCodeGenOpts().InstrumentFunctionsAfterInlining)
      CurFn->addFnAttr("instrument-function-exit-inlined",
                       "__cyg_profile_func_exit");
  }

  if (ShouldSkipSanitizerInstrumentation())
    CurFn->addFnAttr(llvm::Attribute::DisableSanitizerInstrumentation);

  // Emit debug descriptor for function end.
  if (CGDebugInfo *DI = getDebugInfo())
    DI->EmitFunctionEnd(Builder, CurFn);

  // Reset the debug location to that of the simple 'return' expression, if any
  // rather than that of the end of the function's scope '}'.
  ApplyDebugLocation AL(*this, Loc);
  EmitFunctionEpilog(*CurFnInfo, EmitRetDbgLoc, EndLoc);
  EmitEndEHSpec(CurCodeDecl);

  assert(EHStack.empty() &&
         "did not remove all scopes from cleanup stack!");

  // If someone did an indirect goto, emit the indirect goto block at the end of
  // the function.
  if (IndirectBranch) {
    EmitBlock(IndirectBranch->getParent());
    Builder.ClearInsertionPoint();
  }

  // If some of our locals escaped, insert a call to llvm.localescape in the
  // entry block.
  if (!EscapedLocals.empty()) {
    // Invert the map from local to index into a simple vector. There should be
    // no holes.
    SmallVector<llvm::Value *, 4> EscapeArgs;
    EscapeArgs.resize(EscapedLocals.size());
    for (auto &Pair : EscapedLocals)
      EscapeArgs[Pair.second] = Pair.first;
    llvm::Function *FrameEscapeFn = llvm::Intrinsic::getDeclaration(
        &CGM.getModule(), llvm::Intrinsic::localescape);
    CGBuilderTy(*this, AllocaInsertPt).CreateCall(FrameEscapeFn, EscapeArgs);
  }

  // Remove the AllocaInsertPt instruction, which is just a convenience for us.
  llvm::Instruction *Ptr = AllocaInsertPt;
  AllocaInsertPt = nullptr;
  Ptr->eraseFromParent();

  // PostAllocaInsertPt, if created, was lazily created when it was required,
  // remove it now since it was just created for our own convenience.
  if (PostAllocaInsertPt) {
    llvm::Instruction *PostPtr = PostAllocaInsertPt;
    PostAllocaInsertPt = nullptr;
    PostPtr->eraseFromParent();
  }

  // If someone took the address of a label but never did an indirect goto, we
  // made a zero entry PHI node, which is illegal, zap it now.
  if (IndirectBranch) {
    llvm::PHINode *PN = cast<llvm::PHINode>(IndirectBranch->getAddress());
    if (PN->getNumIncomingValues() == 0) {
      PN->replaceAllUsesWith(llvm::UndefValue::get(PN->getType()));
      PN->eraseFromParent();
    }
  }

  EmitIfUsed(*this, EHResumeBlock);
  EmitIfUsed(*this, TerminateLandingPad);
  EmitIfUsed(*this, TerminateHandler);
  EmitIfUsed(*this, UnreachableBlock);

  for (const auto &FuncletAndParent : TerminateFunclets)
    EmitIfUsed(*this, FuncletAndParent.second);

  if (CGM.getCodeGenOpts().EmitDeclMetadata)
    EmitDeclMetadata();

  for (const auto &R : DeferredReplacements) {
    if (llvm::Value *Old = R.first) {
      Old->replaceAllUsesWith(R.second);
      cast<llvm::Instruction>(Old)->eraseFromParent();
    }
  }
  DeferredReplacements.clear();

  // Eliminate CleanupDestSlot alloca by replacing it with SSA values and
  // PHIs if the current function is a coroutine. We don't do it for all
  // functions as it may result in slight increase in numbers of instructions
  // if compiled with no optimizations. We do it for coroutine as the lifetime
  // of CleanupDestSlot alloca make correct coroutine frame building very
  // difficult.
  if (NormalCleanupDest.isValid() && isCoroutine()) {
    llvm::DominatorTree DT(*CurFn);
    llvm::PromoteMemToReg(
        cast<llvm::AllocaInst>(NormalCleanupDest.getPointer()), DT);
    NormalCleanupDest = Address::invalid();
  }

  // Scan function arguments for vector width.
  for (llvm::Argument &A : CurFn->args())
    if (auto *VT = dyn_cast<llvm::VectorType>(A.getType()))
      LargestVectorWidth =
          std::max((uint64_t)LargestVectorWidth,
                   VT->getPrimitiveSizeInBits().getKnownMinSize());

  // Update vector width based on return type.
  if (auto *VT = dyn_cast<llvm::VectorType>(CurFn->getReturnType()))
    LargestVectorWidth =
        std::max((uint64_t)LargestVectorWidth,
                 VT->getPrimitiveSizeInBits().getKnownMinSize());

  // Add the required-vector-width attribute. This contains the max width from:
  // 1. min-vector-width attribute used in the source program.
  // 2. Any builtins used that have a vector width specified.
  // 3. Values passed in and out of inline assembly.
  // 4. Width of vector arguments and return types for this function.
  // 5. Width of vector aguments and return types for functions called by this
  //    function.
  CurFn->addFnAttr("min-legal-vector-width", llvm::utostr(LargestVectorWidth));

  // Add vscale_range attribute if appropriate.
  Optional<std::pair<unsigned, unsigned>> VScaleRange =
      getContext().getTargetInfo().getVScaleRange(getLangOpts());
  if (VScaleRange) {
    CurFn->addFnAttr(llvm::Attribute::getWithVScaleRangeArgs(
        getLLVMContext(), VScaleRange.getValue().first,
        VScaleRange.getValue().second));
  }

  // If we generated an unreachable return block, delete it now.
  if (ReturnBlock.isValid() && ReturnBlock.getBlock()->use_empty()) {
    Builder.ClearInsertionPoint();
    ReturnBlock.getBlock()->eraseFromParent();
  }
  if (ReturnValue.isValid()) {
    auto *RetAlloca = dyn_cast<llvm::AllocaInst>(ReturnValue.getPointer());
    if (RetAlloca && RetAlloca->use_empty()) {
      RetAlloca->eraseFromParent();
      ReturnValue = Address::invalid();
    }
  }
}

/// ShouldInstrumentFunction - Return true if the current function should be
/// instrumented with __cyg_profile_func_* calls
bool CodeGenFunction::ShouldInstrumentFunction() {
  if (!CGM.getCodeGenOpts().InstrumentFunctions &&
      !CGM.getCodeGenOpts().InstrumentFunctionsAfterInlining &&
      !CGM.getCodeGenOpts().InstrumentFunctionEntryBare)
    return false;
  if (!CurFuncDecl || CurFuncDecl->hasAttr<NoInstrumentFunctionAttr>())
    return false;
  return true;
}

bool CodeGenFunction::ShouldSkipSanitizerInstrumentation() {
  if (!CurFuncDecl)
    return false;
  return CurFuncDecl->hasAttr<DisableSanitizerInstrumentationAttr>();
}

/// ShouldXRayInstrument - Return true if the current function should be
/// instrumented with XRay nop sleds.
bool CodeGenFunction::ShouldXRayInstrumentFunction() const {
  return CGM.getCodeGenOpts().XRayInstrumentFunctions;
}

/// AlwaysEmitXRayCustomEvents - Return true if we should emit IR for calls to
/// the __xray_customevent(...) builtin calls, when doing XRay instrumentation.
bool CodeGenFunction::AlwaysEmitXRayCustomEvents() const {
  return CGM.getCodeGenOpts().XRayInstrumentFunctions &&
         (CGM.getCodeGenOpts().XRayAlwaysEmitCustomEvents ||
          CGM.getCodeGenOpts().XRayInstrumentationBundle.Mask ==
              XRayInstrKind::Custom);
}

bool CodeGenFunction::AlwaysEmitXRayTypedEvents() const {
  return CGM.getCodeGenOpts().XRayInstrumentFunctions &&
         (CGM.getCodeGenOpts().XRayAlwaysEmitTypedEvents ||
          CGM.getCodeGenOpts().XRayInstrumentationBundle.Mask ==
              XRayInstrKind::Typed);
}

llvm::Constant *
CodeGenFunction::EncodeAddrForUseInPrologue(llvm::Function *F,
                                            llvm::Constant *Addr) {
  // Addresses stored in prologue data can't require run-time fixups and must
  // be PC-relative. Run-time fixups are undesirable because they necessitate
  // writable text segments, which are unsafe. And absolute addresses are
  // undesirable because they break PIE mode.

  // Add a layer of indirection through a private global. Taking its address
  // won't result in a run-time fixup, even if Addr has linkonce_odr linkage.
  auto *GV = new llvm::GlobalVariable(CGM.getModule(), Addr->getType(),
                                      /*isConstant=*/true,
                                      llvm::GlobalValue::PrivateLinkage, Addr);

  // Create a PC-relative address.
  auto *GOTAsInt = llvm::ConstantExpr::getPtrToInt(GV, IntPtrTy);
  auto *FuncAsInt = llvm::ConstantExpr::getPtrToInt(F, IntPtrTy);
  auto *PCRelAsInt = llvm::ConstantExpr::getSub(GOTAsInt, FuncAsInt);
  return (IntPtrTy == Int32Ty)
             ? PCRelAsInt
             : llvm::ConstantExpr::getTrunc(PCRelAsInt, Int32Ty);
}

llvm::Value *
CodeGenFunction::DecodeAddrUsedInPrologue(llvm::Value *F,
                                          llvm::Value *EncodedAddr) {
  // Reconstruct the address of the global.
  auto *PCRelAsInt = Builder.CreateSExt(EncodedAddr, IntPtrTy);
  auto *FuncAsInt = Builder.CreatePtrToInt(F, IntPtrTy, "func_addr.int");
  auto *GOTAsInt = Builder.CreateAdd(PCRelAsInt, FuncAsInt, "global_addr.int");
  auto *GOTAddr = Builder.CreateIntToPtr(GOTAsInt, Int8PtrPtrTy, "global_addr");

  // Load the original pointer through the global.
  return Builder.CreateLoad(Address(GOTAddr, getPointerAlign()),
                            "decoded_addr");
}

void CodeGenFunction::EmitOpenCLKernelMetadata(const FunctionDecl *FD,
                                               llvm::Function *Fn)
{
  if (!FD->hasAttr<OpenCLKernelAttr>())
    return;

  llvm::LLVMContext &Context = getLLVMContext();

  CGM.GenOpenCLArgMetadata(Fn, FD, this);

  if (const VecTypeHintAttr *A = FD->getAttr<VecTypeHintAttr>()) {
    QualType HintQTy = A->getTypeHint();
    const ExtVectorType *HintEltQTy = HintQTy->getAs<ExtVectorType>();
    bool IsSignedInteger =
        HintQTy->isSignedIntegerType() ||
        (HintEltQTy && HintEltQTy->getElementType()->isSignedIntegerType());
    llvm::Metadata *AttrMDArgs[] = {
        llvm::ConstantAsMetadata::get(llvm::UndefValue::get(
            CGM.getTypes().ConvertType(A->getTypeHint()))),
        llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
            llvm::IntegerType::get(Context, 32),
            llvm::APInt(32, (uint64_t)(IsSignedInteger ? 1 : 0))))};
    Fn->setMetadata("vec_type_hint", llvm::MDNode::get(Context, AttrMDArgs));
  }

  if (const WorkGroupSizeHintAttr *A = FD->getAttr<WorkGroupSizeHintAttr>()) {
    llvm::Metadata *AttrMDArgs[] = {
        llvm::ConstantAsMetadata::get(Builder.getInt32(A->getXDim())),
        llvm::ConstantAsMetadata::get(Builder.getInt32(A->getYDim())),
        llvm::ConstantAsMetadata::get(Builder.getInt32(A->getZDim()))};
    Fn->setMetadata("work_group_size_hint", llvm::MDNode::get(Context, AttrMDArgs));
  }

  if (const ReqdWorkGroupSizeAttr *A = FD->getAttr<ReqdWorkGroupSizeAttr>()) {
    llvm::Metadata *AttrMDArgs[] = {
        llvm::ConstantAsMetadata::get(Builder.getInt32(A->getXDim())),
        llvm::ConstantAsMetadata::get(Builder.getInt32(A->getYDim())),
        llvm::ConstantAsMetadata::get(Builder.getInt32(A->getZDim()))};
    Fn->setMetadata("reqd_work_group_size", llvm::MDNode::get(Context, AttrMDArgs));
  }

  if (const OpenCLIntelReqdSubGroupSizeAttr *A =
          FD->getAttr<OpenCLIntelReqdSubGroupSizeAttr>()) {
    llvm::Metadata *AttrMDArgs[] = {
        llvm::ConstantAsMetadata::get(Builder.getInt32(A->getSubGroupSize()))};
    Fn->setMetadata("intel_reqd_sub_group_size",
                    llvm::MDNode::get(Context, AttrMDArgs));
  }
}

/// Determine whether the function F ends with a return stmt.
static bool endsWithReturn(const Decl* F) {
  const Stmt *Body = nullptr;
  if (auto *FD = dyn_cast_or_null<FunctionDecl>(F))
    Body = FD->getBody();
  else if (auto *OMD = dyn_cast_or_null<ObjCMethodDecl>(F))
    Body = OMD->getBody();

  if (auto *CS = dyn_cast_or_null<CompoundStmt>(Body)) {
    auto LastStmt = CS->body_rbegin();
    if (LastStmt != CS->body_rend())
      return isa<ReturnStmt>(*LastStmt);
  }
  return false;
}

void CodeGenFunction::markAsIgnoreThreadCheckingAtRuntime(llvm::Function *Fn) {
  if (SanOpts.has(SanitizerKind::Thread)) {
    Fn->addFnAttr("sanitize_thread_no_checking_at_run_time");
    Fn->removeFnAttr(llvm::Attribute::SanitizeThread);
  }
}

/// Check if the return value of this function requires sanitization.
bool CodeGenFunction::requiresReturnValueCheck() const {
  return requiresReturnValueNullabilityCheck() ||
         (SanOpts.has(SanitizerKind::ReturnsNonnullAttribute) && CurCodeDecl &&
          CurCodeDecl->getAttr<ReturnsNonNullAttr>());
}

static bool matchesStlAllocatorFn(const Decl *D, const ASTContext &Ctx) {
  auto *MD = dyn_cast_or_null<CXXMethodDecl>(D);
  if (!MD || !MD->getDeclName().getAsIdentifierInfo() ||
      !MD->getDeclName().getAsIdentifierInfo()->isStr("allocate") ||
      (MD->getNumParams() != 1 && MD->getNumParams() != 2))
    return false;

  if (MD->parameters()[0]->getType().getCanonicalType() != Ctx.getSizeType())
    return false;

  if (MD->getNumParams() == 2) {
    auto *PT = MD->parameters()[1]->getType()->getAs<PointerType>();
    if (!PT || !PT->isVoidPointerType() ||
        !PT->getPointeeType().isConstQualified())
      return false;
  }

  return true;
}

/// Return the UBSan prologue signature for \p FD if one is available.
static llvm::Constant *getPrologueSignature(CodeGenModule &CGM,
                                            const FunctionDecl *FD) {
  if (const auto *MD = dyn_cast<CXXMethodDecl>(FD))
    if (!MD->isStatic())
      return nullptr;
  return CGM.getTargetCodeGenInfo().getUBSanFunctionSignature(CGM);
}

void CodeGenFunction::StartFunction(GlobalDecl GD, QualType RetTy,
                                    llvm::Function *Fn,
                                    const CGFunctionInfo &FnInfo,
                                    const FunctionArgList &Args,
                                    SourceLocation Loc,
                                    SourceLocation StartLoc) {
  assert(!CurFn &&
         "Do not use a CodeGenFunction object for more than one function");

  const Decl *D = GD.getDecl();

  DidCallStackSave = false;
  CurCodeDecl = D;
  const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(D);
  if (FD && FD->usesSEHTry())
    CurSEHParent = FD;
  CurFuncDecl = (D ? D->getNonClosureContext() : nullptr);
  FnRetTy = RetTy;
  CurFn = Fn;
  CurFnInfo = &FnInfo;
  assert(CurFn->isDeclaration() && "Function already has body?");

  // If this function is ignored for any of the enabled sanitizers,
  // disable the sanitizer for the function.
  do {
#define SANITIZER(NAME, ID)                                                    \
  if (SanOpts.empty())                                                         \
    break;                                                                     \
  if (SanOpts.has(SanitizerKind::ID))                                          \
    if (CGM.isInNoSanitizeList(SanitizerKind::ID, Fn, Loc))                    \
      SanOpts.set(SanitizerKind::ID, false);

#include "clang/Basic/Sanitizers.def"
#undef SANITIZER
  } while (0);

  if (D) {
    bool NoSanitizeCoverage = false;

    for (auto Attr : D->specific_attrs<NoSanitizeAttr>()) {
      // Apply the no_sanitize* attributes to SanOpts.
      SanitizerMask mask = Attr->getMask();
      SanOpts.Mask &= ~mask;
      if (mask & SanitizerKind::Address)
        SanOpts.set(SanitizerKind::KernelAddress, false);
      if (mask & SanitizerKind::KernelAddress)
        SanOpts.set(SanitizerKind::Address, false);
      if (mask & SanitizerKind::HWAddress)
        SanOpts.set(SanitizerKind::KernelHWAddress, false);
      if (mask & SanitizerKind::KernelHWAddress)
        SanOpts.set(SanitizerKind::HWAddress, false);

      // SanitizeCoverage is not handled by SanOpts.
      if (Attr->hasCoverage())
        NoSanitizeCoverage = true;
    }

    if (NoSanitizeCoverage && CGM.getCodeGenOpts().hasSanitizeCoverage())
      Fn->addFnAttr(llvm::Attribute::NoSanitizeCoverage);
  }

  // Apply sanitizer attributes to the function.
  if (SanOpts.hasOneOf(SanitizerKind::Address | SanitizerKind::KernelAddress))
    Fn->addFnAttr(llvm::Attribute::SanitizeAddress);
  if (SanOpts.hasOneOf(SanitizerKind::HWAddress | SanitizerKind::KernelHWAddress))
    Fn->addFnAttr(llvm::Attribute::SanitizeHWAddress);
  if (SanOpts.has(SanitizerKind::MemTag))
    Fn->addFnAttr(llvm::Attribute::SanitizeMemTag);
  if (SanOpts.has(SanitizerKind::Thread))
    Fn->addFnAttr(llvm::Attribute::SanitizeThread);
  if (SanOpts.hasOneOf(SanitizerKind::Memory | SanitizerKind::KernelMemory))
    Fn->addFnAttr(llvm::Attribute::SanitizeMemory);
  if (SanOpts.has(SanitizerKind::SafeStack))
    Fn->addFnAttr(llvm::Attribute::SafeStack);
  if (SanOpts.has(SanitizerKind::ShadowCallStack))
    Fn->addFnAttr(llvm::Attribute::ShadowCallStack);

  // Apply fuzzing attribute to the function.
  if (SanOpts.hasOneOf(SanitizerKind::Fuzzer | SanitizerKind::FuzzerNoLink))
    Fn->addFnAttr(llvm::Attribute::OptForFuzzing);

  // Ignore TSan memory acesses from within ObjC/ObjC++ dealloc, initialize,
  // .cxx_destruct, __destroy_helper_block_ and all of their calees at run time.
  if (SanOpts.has(SanitizerKind::Thread)) {
    if (const auto *OMD = dyn_cast_or_null<ObjCMethodDecl>(D)) {
      IdentifierInfo *II = OMD->getSelector().getIdentifierInfoForSlot(0);
      if (OMD->getMethodFamily() == OMF_dealloc ||
          OMD->getMethodFamily() == OMF_initialize ||
          (OMD->getSelector().isUnarySelector() && II->isStr(".cxx_destruct"))) {
        markAsIgnoreThreadCheckingAtRuntime(Fn);
      }
    }
  }

  // Ignore unrelated casts in STL allocate() since the allocator must cast
  // from void* to T* before object initialization completes. Don't match on the
  // namespace because not all allocators are in std::
  if (D && SanOpts.has(SanitizerKind::CFIUnrelatedCast)) {
    if (matchesStlAllocatorFn(D, getContext()))
      SanOpts.Mask &= ~SanitizerKind::CFIUnrelatedCast;
  }

  // Ignore null checks in coroutine functions since the coroutines passes
  // are not aware of how to move the extra UBSan instructions across the split
  // coroutine boundaries.
  if (D && SanOpts.has(SanitizerKind::Null))
    if (FD && FD->getBody() &&
        FD->getBody()->getStmtClass() == Stmt::CoroutineBodyStmtClass)
      SanOpts.Mask &= ~SanitizerKind::Null;

  // Apply xray attributes to the function (as a string, for now)
  bool AlwaysXRayAttr = false;
  if (const auto *XRayAttr = D ? D->getAttr<XRayInstrumentAttr>() : nullptr) {
    if (CGM.getCodeGenOpts().XRayInstrumentationBundle.has(
            XRayInstrKind::FunctionEntry) ||
        CGM.getCodeGenOpts().XRayInstrumentationBundle.has(
            XRayInstrKind::FunctionExit)) {
      if (XRayAttr->alwaysXRayInstrument() && ShouldXRayInstrumentFunction()) {
        Fn->addFnAttr("function-instrument", "xray-always");
        AlwaysXRayAttr = true;
      }
      if (XRayAttr->neverXRayInstrument())
        Fn->addFnAttr("function-instrument", "xray-never");
      if (const auto *LogArgs = D->getAttr<XRayLogArgsAttr>())
        if (ShouldXRayInstrumentFunction())
          Fn->addFnAttr("xray-log-args",
                        llvm::utostr(LogArgs->getArgumentCount()));
    }
  } else {
    if (ShouldXRayInstrumentFunction() && !CGM.imbueXRayAttrs(Fn, Loc))
      Fn->addFnAttr(
          "xray-instruction-threshold",
          llvm::itostr(CGM.getCodeGenOpts().XRayInstructionThreshold));
  }

  if (ShouldXRayInstrumentFunction()) {
    if (CGM.getCodeGenOpts().XRayIgnoreLoops)
      Fn->addFnAttr("xray-ignore-loops");

    if (!CGM.getCodeGenOpts().XRayInstrumentationBundle.has(
            XRayInstrKind::FunctionExit))
      Fn->addFnAttr("xray-skip-exit");

    if (!CGM.getCodeGenOpts().XRayInstrumentationBundle.has(
            XRayInstrKind::FunctionEntry))
      Fn->addFnAttr("xray-skip-entry");

    auto FuncGroups = CGM.getCodeGenOpts().XRayTotalFunctionGroups;
    if (FuncGroups > 1) {
      auto FuncName = llvm::makeArrayRef<uint8_t>(
          CurFn->getName().bytes_begin(), CurFn->getName().bytes_end());
      auto Group = crc32(FuncName) % FuncGroups;
      if (Group != CGM.getCodeGenOpts().XRaySelectedFunctionGroup &&
          !AlwaysXRayAttr)
        Fn->addFnAttr("function-instrument", "xray-never");
    }
  }

  if (CGM.getCodeGenOpts().getProfileInstr() != CodeGenOptions::ProfileNone)
    if (CGM.isProfileInstrExcluded(Fn, Loc))
      Fn->addFnAttr(llvm::Attribute::NoProfile);

  unsigned Count, Offset;
  if (const auto *Attr =
          D ? D->getAttr<PatchableFunctionEntryAttr>() : nullptr) {
    Count = Attr->getCount();
    Offset = Attr->getOffset();
  } else {
    Count = CGM.getCodeGenOpts().PatchableFunctionEntryCount;
    Offset = CGM.getCodeGenOpts().PatchableFunctionEntryOffset;
  }
  if (Count && Offset <= Count) {
    Fn->addFnAttr("patchable-function-entry", std::to_string(Count - Offset));
    if (Offset)
      Fn->addFnAttr("patchable-function-prefix", std::to_string(Offset));
  }

  // Add no-jump-tables value.
  if (CGM.getCodeGenOpts().NoUseJumpTables)
    Fn->addFnAttr("no-jump-tables", "true");

  // Add no-inline-line-tables value.
  if (CGM.getCodeGenOpts().NoInlineLineTables)
    Fn->addFnAttr("no-inline-line-tables");

  // Add profile-sample-accurate value.
  if (CGM.getCodeGenOpts().ProfileSampleAccurate)
    Fn->addFnAttr("profile-sample-accurate");

  if (!CGM.getCodeGenOpts().SampleProfileFile.empty())
    Fn->addFnAttr("use-sample-profile");

  if (D && D->hasAttr<CFICanonicalJumpTableAttr>())
    Fn->addFnAttr("cfi-canonical-jump-table");

  if (D && D->hasAttr<NoProfileFunctionAttr>())
    Fn->addFnAttr(llvm::Attribute::NoProfile);

  if (FD && getLangOpts().OpenCL) {
    // Add metadata for a kernel function.
    EmitOpenCLKernelMetadata(FD, Fn);
  }

  // If we are checking function types, emit a function type signature as
  // prologue data.
  if (FD && getLangOpts().CPlusPlus && SanOpts.has(SanitizerKind::Function)) {
    if (llvm::Constant *PrologueSig = getPrologueSignature(CGM, FD)) {
      // Remove any (C++17) exception specifications, to allow calling e.g. a
      // noexcept function through a non-noexcept pointer.
      auto ProtoTy = getContext().getFunctionTypeWithExceptionSpec(
          FD->getType(), EST_None);
      llvm::Constant *FTRTTIConst =
          CGM.GetAddrOfRTTIDescriptor(ProtoTy, /*ForEH=*/true);
      llvm::Constant *FTRTTIConstEncoded =
          EncodeAddrForUseInPrologue(Fn, FTRTTIConst);
      llvm::Constant *PrologueStructElems[] = {PrologueSig, FTRTTIConstEncoded};
      llvm::Constant *PrologueStructConst =
          llvm::ConstantStruct::getAnon(PrologueStructElems, /*Packed=*/true);
      Fn->setPrologueData(PrologueStructConst);
    }
  }

  // If we're checking nullability, we need to know whether we can check the
  // return value. Initialize the flag to 'true' and refine it in EmitParmDecl.
  if (SanOpts.has(SanitizerKind::NullabilityReturn)) {
    auto Nullability = FnRetTy->getNullability(getContext());
    if (Nullability && *Nullability == NullabilityKind::NonNull) {
      if (!(SanOpts.has(SanitizerKind::ReturnsNonnullAttribute) &&
            CurCodeDecl && CurCodeDecl->getAttr<ReturnsNonNullAttr>()))
        RetValNullabilityPrecondition =
            llvm::ConstantInt::getTrue(getLLVMContext());
    }
  }

  // If we're in C++ mode and the function name is "main", it is guaranteed
  // to be norecurse by the standard (3.6.1.3 "The function main shall not be
  // used within a program").
  //
  // OpenCL C 2.0 v2.2-11 s6.9.i:
  //     Recursion is not supported.
  //
  // SYCL v1.2.1 s3.10:
  //     kernels cannot include RTTI information, exception classes,
  //     recursive code, virtual functions or make use of C++ libraries that
  //     are not compiled for the device.
  if (FD && ((getLangOpts().CPlusPlus && FD->isMain()) ||
             getLangOpts().OpenCL || getLangOpts().SYCLIsDevice ||
             (getLangOpts().CUDA && FD->hasAttr<CUDAGlobalAttr>())))
    Fn->addFnAttr(llvm::Attribute::NoRecurse);

  llvm::RoundingMode RM = getLangOpts().getFPRoundingMode();
  llvm::fp::ExceptionBehavior FPExceptionBehavior =
      ToConstrainedExceptMD(getLangOpts().getFPExceptionMode());
  Builder.setDefaultConstrainedRounding(RM);
  Builder.setDefaultConstrainedExcept(FPExceptionBehavior);
  if ((FD && (FD->UsesFPIntrin() || FD->hasAttr<StrictFPAttr>())) ||
      (!FD && (FPExceptionBehavior != llvm::fp::ebIgnore ||
               RM != llvm::RoundingMode::NearestTiesToEven))) {
    Builder.setIsFPConstrained(true);
    Fn->addFnAttr(llvm::Attribute::StrictFP);
  }

  // If a custom alignment is used, force realigning to this alignment on
  // any main function which certainly will need it.
  if (FD && ((FD->isMain() || FD->isMSVCRTEntryPoint()) &&
             CGM.getCodeGenOpts().StackAlignment))
    Fn->addFnAttr("stackrealign");

  llvm::BasicBlock *EntryBB = createBasicBlock("entry", CurFn);

  // Create a marker to make it easy to insert allocas into the entryblock
  // later.  Don't create this with the builder, because we don't want it
  // folded.
  llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);
  AllocaInsertPt = new llvm::BitCastInst(Undef, Int32Ty, "allocapt", EntryBB);

  ReturnBlock = getJumpDestInCurrentScope("return");

  Builder.SetInsertPoint(EntryBB);

  // If we're checking the return value, allocate space for a pointer to a
  // precise source location of the checked return statement.
  if (requiresReturnValueCheck()) {
    ReturnLocation = CreateDefaultAlignTempAlloca(Int8PtrTy, "return.sloc.ptr");
    Builder.CreateStore(llvm::ConstantPointerNull::get(Int8PtrTy),
                        ReturnLocation);
  }

  // Emit subprogram debug descriptor.
  if (CGDebugInfo *DI = getDebugInfo()) {
    // Reconstruct the type from the argument list so that implicit parameters,
    // such as 'this' and 'vtt', show up in the debug info. Preserve the calling
    // convention.
    DI->emitFunctionStart(GD, Loc, StartLoc,
                          DI->getFunctionType(FD, RetTy, Args), CurFn,
                          CurFuncIsThunk);
  }

  if (ShouldInstrumentFunction()) {
    if (CGM.getCodeGenOpts().InstrumentFunctions)
      CurFn->addFnAttr("instrument-function-entry", "__cyg_profile_func_enter");
    if (CGM.getCodeGenOpts().InstrumentFunctionsAfterInlining)
      CurFn->addFnAttr("instrument-function-entry-inlined",
                       "__cyg_profile_func_enter");
    if (CGM.getCodeGenOpts().InstrumentFunctionEntryBare)
      CurFn->addFnAttr("instrument-function-entry-inlined",
                       "__cyg_profile_func_enter_bare");
  }

  // Since emitting the mcount call here impacts optimizations such as function
  // inlining, we just add an attribute to insert a mcount call in backend.
  // The attribute "counting-function" is set to mcount function name which is
  // architecture dependent.
  if (CGM.getCodeGenOpts().InstrumentForProfiling) {
    // Calls to fentry/mcount should not be generated if function has
    // the no_instrument_function attribute.
    if (!CurFuncDecl || !CurFuncDecl->hasAttr<NoInstrumentFunctionAttr>()) {
      if (CGM.getCodeGenOpts().CallFEntry)
        Fn->addFnAttr("fentry-call", "true");
      else {
        Fn->addFnAttr("instrument-function-entry-inlined",
                      getTarget().getMCountName());
      }
      if (CGM.getCodeGenOpts().MNopMCount) {
        if (!CGM.getCodeGenOpts().CallFEntry)
          CGM.getDiags().Report(diag::err_opt_not_valid_without_opt)
            << "-mnop-mcount" << "-mfentry";
        Fn->addFnAttr("mnop-mcount");
      }

      if (CGM.getCodeGenOpts().RecordMCount) {
        if (!CGM.getCodeGenOpts().CallFEntry)
          CGM.getDiags().Report(diag::err_opt_not_valid_without_opt)
            << "-mrecord-mcount" << "-mfentry";
        Fn->addFnAttr("mrecord-mcount");
      }
    }
  }

  if (CGM.getCodeGenOpts().PackedStack) {
    if (getContext().getTargetInfo().getTriple().getArch() !=
        llvm::Triple::systemz)
      CGM.getDiags().Report(diag::err_opt_not_valid_on_target)
        << "-mpacked-stack";
    Fn->addFnAttr("packed-stack");
  }

  if (CGM.getCodeGenOpts().WarnStackSize != UINT_MAX &&
      !CGM.getDiags().isIgnored(diag::warn_fe_backend_frame_larger_than, Loc))
    Fn->addFnAttr("warn-stack-size",
                  std::to_string(CGM.getCodeGenOpts().WarnStackSize));

  if (RetTy->isVoidType()) {
    // Void type; nothing to return.
    ReturnValue = Address::invalid();

    // Count the implicit return.
    if (!endsWithReturn(D))
      ++NumReturnExprs;
  } else if (CurFnInfo->getReturnInfo().getKind() == ABIArgInfo::Indirect) {
    // Indirect return; emit returned value directly into sret slot.
    // This reduces code size, and affects correctness in C++.
    auto AI = CurFn->arg_begin();
    if (CurFnInfo->getReturnInfo().isSRetAfterThis())
      ++AI;
    ReturnValue = Address(&*AI, CurFnInfo->getReturnInfo().getIndirectAlign());
    if (!CurFnInfo->getReturnInfo().getIndirectByVal()) {
      ReturnValuePointer =
          CreateDefaultAlignTempAlloca(Int8PtrTy, "result.ptr");
      Builder.CreateStore(Builder.CreatePointerBitCastOrAddrSpaceCast(
                              ReturnValue.getPointer(), Int8PtrTy),
                          ReturnValuePointer);
    }
  } else if (CurFnInfo->getReturnInfo().getKind() == ABIArgInfo::InAlloca &&
             !hasScalarEvaluationKind(CurFnInfo->getReturnType())) {
    // Load the sret pointer from the argument struct and return into that.
    unsigned Idx = CurFnInfo->getReturnInfo().getInAllocaFieldIndex();
    llvm::Function::arg_iterator EI = CurFn->arg_end();
    --EI;
    llvm::Value *Addr = Builder.CreateStructGEP(
        EI->getType()->getPointerElementType(), &*EI, Idx);
    llvm::Type *Ty =
        cast<llvm::GetElementPtrInst>(Addr)->getResultElementType();
    ReturnValuePointer = Address(Addr, getPointerAlign());
    Addr = Builder.CreateAlignedLoad(Ty, Addr, getPointerAlign(), "agg.result");
    ReturnValue = Address(Addr, CGM.getNaturalTypeAlignment(RetTy));
  } else {
    ReturnValue = CreateIRTemp(RetTy, "retval");

    // Tell the epilog emitter to autorelease the result.  We do this
    // now so that various specialized functions can suppress it
    // during their IR-generation.
    if (getLangOpts().ObjCAutoRefCount &&
        !CurFnInfo->isReturnsRetained() &&
        RetTy->isObjCRetainableType())
      AutoreleaseResult = true;
  }

  EmitStartEHSpec(CurCodeDecl);

  PrologueCleanupDepth = EHStack.stable_begin();

  // Emit OpenMP specific initialization of the device functions.
  if (getLangOpts().OpenMP && CurCodeDecl)
    CGM.getOpenMPRuntime().emitFunctionProlog(*this, CurCodeDecl);

  EmitFunctionProlog(*CurFnInfo, CurFn, Args);

  if (D && isa<CXXMethodDecl>(D) && cast<CXXMethodDecl>(D)->isInstance()) {
    CGM.getCXXABI().EmitInstanceFunctionProlog(*this);
    const CXXMethodDecl *MD = cast<CXXMethodDecl>(D);
    if (MD->getParent()->isLambda() &&
        MD->getOverloadedOperator() == OO_Call) {
      // We're in a lambda; figure out the captures.
      MD->getParent()->getCaptureFields(LambdaCaptureFields,
                                        LambdaThisCaptureField);
      if (LambdaThisCaptureField) {
        // If the lambda captures the object referred to by '*this' - either by
        // value or by reference, make sure CXXThisValue points to the correct
        // object.

        // Get the lvalue for the field (which is a copy of the enclosing object
        // or contains the address of the enclosing object).
        LValue ThisFieldLValue = EmitLValueForLambdaField(LambdaThisCaptureField);
        if (!LambdaThisCaptureField->getType()->isPointerType()) {
          // If the enclosing object was captured by value, just use its address.
          CXXThisValue = ThisFieldLValue.getAddress(*this).getPointer();
        } else {
          // Load the lvalue pointed to by the field, since '*this' was captured
          // by reference.
          CXXThisValue =
              EmitLoadOfLValue(ThisFieldLValue, SourceLocation()).getScalarVal();
        }
      }
      for (auto *FD : MD->getParent()->fields()) {
        if (FD->hasCapturedVLAType()) {
          auto *ExprArg = EmitLoadOfLValue(EmitLValueForLambdaField(FD),
                                           SourceLocation()).getScalarVal();
          auto VAT = FD->getCapturedVLAType();
          VLASizeMap[VAT->getSizeExpr()] = ExprArg;
        }
      }
    } else {
      // Not in a lambda; just use 'this' from the method.
      // FIXME: Should we generate a new load for each use of 'this'?  The
      // fast register allocator would be happier...
      CXXThisValue = CXXABIThisValue;
    }

    // Check the 'this' pointer once per function, if it's available.
    if (CXXABIThisValue) {
      SanitizerSet SkippedChecks;
      SkippedChecks.set(SanitizerKind::ObjectSize, true);
      QualType ThisTy = MD->getThisType();

      // If this is the call operator of a lambda with no capture-default, it
      // may have a static invoker function, which may call this operator with
      // a null 'this' pointer.
      if (isLambdaCallOperator(MD) &&
          MD->getParent()->getLambdaCaptureDefault() == LCD_None)
        SkippedChecks.set(SanitizerKind::Null, true);

      EmitTypeCheck(
          isa<CXXConstructorDecl>(MD) ? TCK_ConstructorCall : TCK_MemberCall,
          Loc, CXXABIThisValue, ThisTy, CXXABIThisAlignment, SkippedChecks);
    }
  }

  // If any of the arguments have a variably modified type, make sure to
  // emit the type size.
  for (FunctionArgList::const_iterator i = Args.begin(), e = Args.end();
       i != e; ++i) {
    const VarDecl *VD = *i;

    // Dig out the type as written from ParmVarDecls; it's unclear whether
    // the standard (C99 6.9.1p10) requires this, but we're following the
    // precedent set by gcc.
    QualType Ty;
    if (const ParmVarDecl *PVD = dyn_cast<ParmVarDecl>(VD))
      Ty = PVD->getOriginalType();
    else
      Ty = VD->getType();

    if (Ty->isVariablyModifiedType())
      EmitVariablyModifiedType(Ty);
  }
  // Emit a location at the end of the prologue.
  if (CGDebugInfo *DI = getDebugInfo())
    DI->EmitLocation(Builder, StartLoc);

  // TODO: Do we need to handle this in two places like we do with
  // target-features/target-cpu?
  if (CurFuncDecl)
    if (const auto *VecWidth = CurFuncDecl->getAttr<MinVectorWidthAttr>())
      LargestVectorWidth = VecWidth->getVectorWidth();
}

void CodeGenFunction::EmitFunctionBody(const Stmt *Body) {
  incrementProfileCounter(Body);
  if (const CompoundStmt *S = dyn_cast<CompoundStmt>(Body))
    EmitCompoundStmtWithoutScope(*S);
  else
    EmitStmt(Body);

  // This is checked after emitting the function body so we know if there
  // are any permitted infinite loops.
  if (checkIfFunctionMustProgress())
    CurFn->addFnAttr(llvm::Attribute::MustProgress);
}

/// When instrumenting to collect profile data, the counts for some blocks
/// such as switch cases need to not include the fall-through counts, so
/// emit a branch around the instrumentation code. When not instrumenting,
/// this just calls EmitBlock().
void CodeGenFunction::EmitBlockWithFallThrough(llvm::BasicBlock *BB,
                                               const Stmt *S) {
  llvm::BasicBlock *SkipCountBB = nullptr;
  if (HaveInsertPoint() && CGM.getCodeGenOpts().hasProfileClangInstr()) {
    // When instrumenting for profiling, the fallthrough to certain
    // statements needs to skip over the instrumentation code so that we
    // get an accurate count.
    SkipCountBB = createBasicBlock("skipcount");
    EmitBranch(SkipCountBB);
  }
  EmitBlock(BB);
  uint64_t CurrentCount = getCurrentProfileCount();
  incrementProfileCounter(S);
  setCurrentProfileCount(getCurrentProfileCount() + CurrentCount);
  if (SkipCountBB)
    EmitBlock(SkipCountBB);
}

/// Tries to mark the given function nounwind based on the
/// non-existence of any throwing calls within it.  We believe this is
/// lightweight enough to do at -O0.
static void TryMarkNoThrow(llvm::Function *F) {
  // LLVM treats 'nounwind' on a function as part of the type, so we
  // can't do this on functions that can be overwritten.
  if (F->isInterposable()) return;

  for (llvm::BasicBlock &BB : *F)
    for (llvm::Instruction &I : BB)
      if (I.mayThrow())
        return;

  F->setDoesNotThrow();
}

QualType CodeGenFunction::BuildFunctionArgList(GlobalDecl GD,
                                               FunctionArgList &Args) {
  const FunctionDecl *FD = cast<FunctionDecl>(GD.getDecl());
  QualType ResTy = FD->getReturnType();

  const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD);
  if (MD && MD->isInstance()) {
    if (CGM.getCXXABI().HasThisReturn(GD))
      ResTy = MD->getThisType();
    else if (CGM.getCXXABI().hasMostDerivedReturn(GD))
      ResTy = CGM.getContext().VoidPtrTy;
    CGM.getCXXABI().buildThisParam(*this, Args);
  }

  // The base version of an inheriting constructor whose constructed base is a
  // virtual base is not passed any arguments (because it doesn't actually call
  // the inherited constructor).
  bool PassedParams = true;
  if (const CXXConstructorDecl *CD = dyn_cast<CXXConstructorDecl>(FD))
    if (auto Inherited = CD->getInheritedConstructor())
      PassedParams =
          getTypes().inheritingCtorHasParams(Inherited, GD.getCtorType());

  if (PassedParams) {
    for (auto *Param : FD->parameters()) {
      Args.push_back(Param);
      if (!Param->hasAttr<PassObjectSizeAttr>())
        continue;

      auto *Implicit = ImplicitParamDecl::Create(
          getContext(), Param->getDeclContext(), Param->getLocation(),
          /*Id=*/nullptr, getContext().getSizeType(), ImplicitParamDecl::Other);
      SizeArguments[Param] = Implicit;
      Args.push_back(Implicit);
    }
  }

  if (MD && (isa<CXXConstructorDecl>(MD) || isa<CXXDestructorDecl>(MD)))
    CGM.getCXXABI().addImplicitStructorParams(*this, ResTy, Args);

  return ResTy;
}

void CodeGenFunction::GenerateCode(GlobalDecl GD, llvm::Function *Fn,
                                   const CGFunctionInfo &FnInfo) {
  const FunctionDecl *FD = cast<FunctionDecl>(GD.getDecl());
  CurGD = GD;

  FunctionArgList Args;
  QualType ResTy = BuildFunctionArgList(GD, Args);

  // When generating code for a builtin with an inline declaration, use a
  // mangled name to hold the actual body, while keeping an external definition
  // in case the function pointer is referenced somewhere.
  if (Fn) {
    if (FD->isInlineBuiltinDeclaration()) {
      std::string FDInlineName = (Fn->getName() + ".inline").str();
      llvm::Module *M = Fn->getParent();
      llvm::Function *Clone = M->getFunction(FDInlineName);
      if (!Clone) {
        Clone = llvm::Function::Create(Fn->getFunctionType(),
                                       llvm::GlobalValue::InternalLinkage,
                                       Fn->getAddressSpace(), FDInlineName, M);
        Clone->addFnAttr(llvm::Attribute::AlwaysInline);
      }
      Fn->setLinkage(llvm::GlobalValue::ExternalLinkage);
      Fn = Clone;
    }

    // Detect the unusual situation where an inline version is shadowed by a
    // non-inline version. In that case we should pick the external one
    // everywhere. That's GCC behavior too. Unfortunately, I cannot find a way
    // to detect that situation before we reach codegen, so do some late
    // replacement.
    else {
      for (const FunctionDecl *PD = FD->getPreviousDecl(); PD;
           PD = PD->getPreviousDecl()) {
        if (LLVM_UNLIKELY(PD->isInlineBuiltinDeclaration())) {
          std::string FDInlineName = (Fn->getName() + ".inline").str();
          llvm::Module *M = Fn->getParent();
          if (llvm::Function *Clone = M->getFunction(FDInlineName)) {
            Clone->replaceAllUsesWith(Fn);
            Clone->eraseFromParent();
          }
          break;
        }
      }
    }
  }

  // Check if we should generate debug info for this function.
  if (FD->hasAttr<NoDebugAttr>()) {
    // Clear non-distinct debug info that was possibly attached to the function
    // due to an earlier declaration without the nodebug attribute
    if (Fn)
      Fn->setSubprogram(nullptr);
    // Disable debug info indefinitely for this function
    DebugInfo = nullptr;
  }

  // The function might not have a body if we're generating thunks for a
  // function declaration.
  SourceRange BodyRange;
  if (Stmt *Body = FD->getBody())
    BodyRange = Body->getSourceRange();
  else
    BodyRange = FD->getLocation();
  CurEHLocation = BodyRange.getEnd();

  // Use the location of the start of the function to determine where
  // the function definition is located. By default use the location
  // of the declaration as the location for the subprogram. A function
  // may lack a declaration in the source code if it is created by code
  // gen. (examples: _GLOBAL__I_a, __cxx_global_array_dtor, thunk).
  SourceLocation Loc = FD->getLocation();

  // If this is a function specialization then use the pattern body
  // as the location for the function.
  if (const FunctionDecl *SpecDecl = FD->getTemplateInstantiationPattern())
    if (SpecDecl->hasBody(SpecDecl))
      Loc = SpecDecl->getLocation();

  Stmt *Body = FD->getBody();

  if (Body) {
    // Coroutines always emit lifetime markers.
    if (isa<CoroutineBodyStmt>(Body))
      ShouldEmitLifetimeMarkers = true;

    // Initialize helper which will detect jumps which can cause invalid
    // lifetime markers.
    if (ShouldEmitLifetimeMarkers)
      Bypasses.Init(Body);
  }

  // Emit the standard function prologue.
  StartFunction(GD, ResTy, Fn, FnInfo, Args, Loc, BodyRange.getBegin());

  // Save parameters for coroutine function.
  if (Body && isa_and_nonnull<CoroutineBodyStmt>(Body))
    for (const auto *ParamDecl : FD->parameters())
      FnArgs.push_back(ParamDecl);

  // Generate the body of the function.
  PGO.assignRegionCounters(GD, CurFn);
  if (isa<CXXDestructorDecl>(FD))
    EmitDestructorBody(Args);
  else if (isa<CXXConstructorDecl>(FD))
    EmitConstructorBody(Args);
  else if (getLangOpts().CUDA &&
           !getLangOpts().CUDAIsDevice &&
           FD->hasAttr<CUDAGlobalAttr>())
    CGM.getCUDARuntime().emitDeviceStub(*this, Args);
  else if (isa<CXXMethodDecl>(FD) &&
           cast<CXXMethodDecl>(FD)->isLambdaStaticInvoker()) {
    // The lambda static invoker function is special, because it forwards or
    // clones the body of the function call operator (but is actually static).
    EmitLambdaStaticInvokeBody(cast<CXXMethodDecl>(FD));
  } else if (FD->isDefaulted() && isa<CXXMethodDecl>(FD) &&
             (cast<CXXMethodDecl>(FD)->isCopyAssignmentOperator() ||
              cast<CXXMethodDecl>(FD)->isMoveAssignmentOperator())) {
    // Implicit copy-assignment gets the same special treatment as implicit
    // copy-constructors.
    emitImplicitAssignmentOperatorBody(Args);
  } else if (Body) {
    EmitFunctionBody(Body);
  } else
    llvm_unreachable("no definition for emitted function");

  // C++11 [stmt.return]p2:
  //   Flowing off the end of a function [...] results in undefined behavior in
  //   a value-returning function.
  // C11 6.9.1p12:
  //   If the '}' that terminates a function is reached, and the value of the
  //   function call is used by the caller, the behavior is undefined.
  if (getLangOpts().CPlusPlus && !FD->hasImplicitReturnZero() && !SawAsmBlock &&
      !FD->getReturnType()->isVoidType() && Builder.GetInsertBlock()) {
    bool ShouldEmitUnreachable =
        CGM.getCodeGenOpts().StrictReturn ||
        !CGM.MayDropFunctionReturn(FD->getASTContext(), FD->getReturnType());
    if (SanOpts.has(SanitizerKind::Return)) {
      SanitizerScope SanScope(this);
      llvm::Value *IsFalse = Builder.getFalse();
      EmitCheck(std::make_pair(IsFalse, SanitizerKind::Return),
                SanitizerHandler::MissingReturn,
                EmitCheckSourceLocation(FD->getLocation()), None);
    } else if (ShouldEmitUnreachable) {
      if (CGM.getCodeGenOpts().OptimizationLevel == 0)
        EmitTrapCall(llvm::Intrinsic::trap);
    }
    if (SanOpts.has(SanitizerKind::Return) || ShouldEmitUnreachable) {
      Builder.CreateUnreachable();
      Builder.ClearInsertionPoint();
    }
  }

  // Emit the standard function epilogue.
  FinishFunction(BodyRange.getEnd());

  // If we haven't marked the function nothrow through other means, do
  // a quick pass now to see if we can.
  if (!CurFn->doesNotThrow())
    TryMarkNoThrow(CurFn);
}

/// ContainsLabel - Return true if the statement contains a label in it.  If
/// this statement is not executed normally, it not containing a label means
/// that we can just remove the code.
bool CodeGenFunction::ContainsLabel(const Stmt *S, bool IgnoreCaseStmts) {
  // Null statement, not a label!
  if (!S) return false;

  // If this is a label, we have to emit the code, consider something like:
  // if (0) {  ...  foo:  bar(); }  goto foo;
  //
  // TODO: If anyone cared, we could track __label__'s, since we know that you
  // can't jump to one from outside their declared region.
  if (isa<LabelStmt>(S))
    return true;

  // If this is a case/default statement, and we haven't seen a switch, we have
  // to emit the code.
  if (isa<SwitchCase>(S) && !IgnoreCaseStmts)
    return true;

  // If this is a switch statement, we want to ignore cases below it.
  if (isa<SwitchStmt>(S))
    IgnoreCaseStmts = true;

  // Scan subexpressions for verboten labels.
  for (const Stmt *SubStmt : S->children())
    if (ContainsLabel(SubStmt, IgnoreCaseStmts))
      return true;

  return false;
}

/// containsBreak - Return true if the statement contains a break out of it.
/// If the statement (recursively) contains a switch or loop with a break
/// inside of it, this is fine.
bool CodeGenFunction::containsBreak(const Stmt *S) {
  // Null statement, not a label!
  if (!S) return false;

  // If this is a switch or loop that defines its own break scope, then we can
  // include it and anything inside of it.
  if (isa<SwitchStmt>(S) || isa<WhileStmt>(S) || isa<DoStmt>(S) ||
      isa<ForStmt>(S))
    return false;

  if (isa<BreakStmt>(S))
    return true;

  // Scan subexpressions for verboten breaks.
  for (const Stmt *SubStmt : S->children())
    if (containsBreak(SubStmt))
      return true;

  return false;
}

bool CodeGenFunction::mightAddDeclToScope(const Stmt *S) {
  if (!S) return false;

  // Some statement kinds add a scope and thus never add a decl to the current
  // scope. Note, this list is longer than the list of statements that might
  // have an unscoped decl nested within them, but this way is conservatively
  // correct even if more statement kinds are added.
  if (isa<IfStmt>(S) || isa<SwitchStmt>(S) || isa<WhileStmt>(S) ||
      isa<DoStmt>(S) || isa<ForStmt>(S) || isa<CompoundStmt>(S) ||
      isa<CXXForRangeStmt>(S) || isa<CXXTryStmt>(S) ||
      isa<ObjCForCollectionStmt>(S) || isa<ObjCAtTryStmt>(S))
    return false;

  if (isa<DeclStmt>(S))
    return true;

  for (const Stmt *SubStmt : S->children())
    if (mightAddDeclToScope(SubStmt))
      return true;

  return false;
}

/// ConstantFoldsToSimpleInteger - If the specified expression does not fold
/// to a constant, or if it does but contains a label, return false.  If it
/// constant folds return true and set the boolean result in Result.
bool CodeGenFunction::ConstantFoldsToSimpleInteger(const Expr *Cond,
                                                   bool &ResultBool,
                                                   bool AllowLabels) {
  llvm::APSInt ResultInt;
  if (!ConstantFoldsToSimpleInteger(Cond, ResultInt, AllowLabels))
    return false;

  ResultBool = ResultInt.getBoolValue();
  return true;
}

/// ConstantFoldsToSimpleInteger - If the specified expression does not fold
/// to a constant, or if it does but contains a label, return false.  If it
/// constant folds return true and set the folded value.
bool CodeGenFunction::ConstantFoldsToSimpleInteger(const Expr *Cond,
                                                   llvm::APSInt &ResultInt,
                                                   bool AllowLabels) {
  // FIXME: Rename and handle conversion of other evaluatable things
  // to bool.
  Expr::EvalResult Result;
  if (!Cond->EvaluateAsInt(Result, getContext()))
    return false;  // Not foldable, not integer or not fully evaluatable.

  llvm::APSInt Int = Result.Val.getInt();
  if (!AllowLabels && CodeGenFunction::ContainsLabel(Cond))
    return false;  // Contains a label.

  ResultInt = Int;
  return true;
}

/// Determine whether the given condition is an instrumentable condition
/// (i.e. no "&&" or "||").
bool CodeGenFunction::isInstrumentedCondition(const Expr *C) {
  // Bypass simplistic logical-NOT operator before determining whether the
  // condition contains any other logical operator.
  if (const UnaryOperator *UnOp = dyn_cast<UnaryOperator>(C->IgnoreParens()))
    if (UnOp->getOpcode() == UO_LNot)
      C = UnOp->getSubExpr();

  const BinaryOperator *BOp = dyn_cast<BinaryOperator>(C->IgnoreParens());
  return (!BOp || !BOp->isLogicalOp());
}

/// EmitBranchToCounterBlock - Emit a conditional branch to a new block that
/// increments a profile counter based on the semantics of the given logical
/// operator opcode.  This is used to instrument branch condition coverage for
/// logical operators.
void CodeGenFunction::EmitBranchToCounterBlock(
    const Expr *Cond, BinaryOperator::Opcode LOp, llvm::BasicBlock *TrueBlock,
    llvm::BasicBlock *FalseBlock, uint64_t TrueCount /* = 0 */,
    Stmt::Likelihood LH /* =None */, const Expr *CntrIdx /* = nullptr */) {
  // If not instrumenting, just emit a branch.
  bool InstrumentRegions = CGM.getCodeGenOpts().hasProfileClangInstr();
  if (!InstrumentRegions || !isInstrumentedCondition(Cond))
    return EmitBranchOnBoolExpr(Cond, TrueBlock, FalseBlock, TrueCount, LH);

  llvm::BasicBlock *ThenBlock = NULL;
  llvm::BasicBlock *ElseBlock = NULL;
  llvm::BasicBlock *NextBlock = NULL;

  // Create the block we'll use to increment the appropriate counter.
  llvm::BasicBlock *CounterIncrBlock = createBasicBlock("lop.rhscnt");

  // Set block pointers according to Logical-AND (BO_LAnd) semantics. This
  // means we need to evaluate the condition and increment the counter on TRUE:
  //
  // if (Cond)
  //   goto CounterIncrBlock;
  // else
  //   goto FalseBlock;
  //
  // CounterIncrBlock:
  //   Counter++;
  //   goto TrueBlock;

  if (LOp == BO_LAnd) {
    ThenBlock = CounterIncrBlock;
    ElseBlock = FalseBlock;
    NextBlock = TrueBlock;
  }

  // Set block pointers according to Logical-OR (BO_LOr) semantics. This means
  // we need to evaluate the condition and increment the counter on FALSE:
  //
  // if (Cond)
  //   goto TrueBlock;
  // else
  //   goto CounterIncrBlock;
  //
  // CounterIncrBlock:
  //   Counter++;
  //   goto FalseBlock;

  else if (LOp == BO_LOr) {
    ThenBlock = TrueBlock;
    ElseBlock = CounterIncrBlock;
    NextBlock = FalseBlock;
  } else {
    llvm_unreachable("Expected Opcode must be that of a Logical Operator");
  }

  // Emit Branch based on condition.
  EmitBranchOnBoolExpr(Cond, ThenBlock, ElseBlock, TrueCount, LH);

  // Emit the block containing the counter increment(s).
  EmitBlock(CounterIncrBlock);

  // Increment corresponding counter; if index not provided, use Cond as index.
  incrementProfileCounter(CntrIdx ? CntrIdx : Cond);

  // Go to the next block.
  EmitBranch(NextBlock);
}

/// EmitBranchOnBoolExpr - Emit a branch on a boolean condition (e.g. for an if
/// statement) to the specified blocks.  Based on the condition, this might try
/// to simplify the codegen of the conditional based on the branch.
/// \param LH The value of the likelihood attribute on the True branch.
void CodeGenFunction::EmitBranchOnBoolExpr(const Expr *Cond,
                                           llvm::BasicBlock *TrueBlock,
                                           llvm::BasicBlock *FalseBlock,
                                           uint64_t TrueCount,
                                           Stmt::Likelihood LH) {
  Cond = Cond->IgnoreParens();

  if (const BinaryOperator *CondBOp = dyn_cast<BinaryOperator>(Cond)) {

    // Handle X && Y in a condition.
    if (CondBOp->getOpcode() == BO_LAnd) {
      // If we have "1 && X", simplify the code.  "0 && X" would have constant
      // folded if the case was simple enough.
      bool ConstantBool = false;
      if (ConstantFoldsToSimpleInteger(CondBOp->getLHS(), ConstantBool) &&
          ConstantBool) {
        // br(1 && X) -> br(X).
        incrementProfileCounter(CondBOp);
        return EmitBranchToCounterBlock(CondBOp->getRHS(), BO_LAnd, TrueBlock,
                                        FalseBlock, TrueCount, LH);
      }

      // If we have "X && 1", simplify the code to use an uncond branch.
      // "X && 0" would have been constant folded to 0.
      if (ConstantFoldsToSimpleInteger(CondBOp->getRHS(), ConstantBool) &&
          ConstantBool) {
        // br(X && 1) -> br(X).
        return EmitBranchToCounterBlock(CondBOp->getLHS(), BO_LAnd, TrueBlock,
                                        FalseBlock, TrueCount, LH, CondBOp);
      }

      // Emit the LHS as a conditional.  If the LHS conditional is false, we
      // want to jump to the FalseBlock.
      llvm::BasicBlock *LHSTrue = createBasicBlock("land.lhs.true");
      // The counter tells us how often we evaluate RHS, and all of TrueCount
      // can be propagated to that branch.
      uint64_t RHSCount = getProfileCount(CondBOp->getRHS());

      ConditionalEvaluation eval(*this);
      {
        ApplyDebugLocation DL(*this, Cond);
        // Propagate the likelihood attribute like __builtin_expect
        // __builtin_expect(X && Y, 1) -> X and Y are likely
        // __builtin_expect(X && Y, 0) -> only Y is unlikely
        EmitBranchOnBoolExpr(CondBOp->getLHS(), LHSTrue, FalseBlock, RHSCount,
                             LH == Stmt::LH_Unlikely ? Stmt::LH_None : LH);
        EmitBlock(LHSTrue);
      }

      incrementProfileCounter(CondBOp);
      setCurrentProfileCount(getProfileCount(CondBOp->getRHS()));

      // Any temporaries created here are conditional.
      eval.begin(*this);
      EmitBranchToCounterBlock(CondBOp->getRHS(), BO_LAnd, TrueBlock,
                               FalseBlock, TrueCount, LH);
      eval.end(*this);

      return;
    }

    if (CondBOp->getOpcode() == BO_LOr) {
      // If we have "0 || X", simplify the code.  "1 || X" would have constant
      // folded if the case was simple enough.
      bool ConstantBool = false;
      if (ConstantFoldsToSimpleInteger(CondBOp->getLHS(), ConstantBool) &&
          !ConstantBool) {
        // br(0 || X) -> br(X).
        incrementProfileCounter(CondBOp);
        return EmitBranchToCounterBlock(CondBOp->getRHS(), BO_LOr, TrueBlock,
                                        FalseBlock, TrueCount, LH);
      }

      // If we have "X || 0", simplify the code to use an uncond branch.
      // "X || 1" would have been constant folded to 1.
      if (ConstantFoldsToSimpleInteger(CondBOp->getRHS(), ConstantBool) &&
          !ConstantBool) {
        // br(X || 0) -> br(X).
        return EmitBranchToCounterBlock(CondBOp->getLHS(), BO_LOr, TrueBlock,
                                        FalseBlock, TrueCount, LH, CondBOp);
      }

      // Emit the LHS as a conditional.  If the LHS conditional is true, we
      // want to jump to the TrueBlock.
      llvm::BasicBlock *LHSFalse = createBasicBlock("lor.lhs.false");
      // We have the count for entry to the RHS and for the whole expression
      // being true, so we can divy up True count between the short circuit and
      // the RHS.
      uint64_t LHSCount =
          getCurrentProfileCount() - getProfileCount(CondBOp->getRHS());
      uint64_t RHSCount = TrueCount - LHSCount;

      ConditionalEvaluation eval(*this);
      {
        // Propagate the likelihood attribute like __builtin_expect
        // __builtin_expect(X || Y, 1) -> only Y is likely
        // __builtin_expect(X || Y, 0) -> both X and Y are unlikely
        ApplyDebugLocation DL(*this, Cond);
        EmitBranchOnBoolExpr(CondBOp->getLHS(), TrueBlock, LHSFalse, LHSCount,
                             LH == Stmt::LH_Likely ? Stmt::LH_None : LH);
        EmitBlock(LHSFalse);
      }

      incrementProfileCounter(CondBOp);
      setCurrentProfileCount(getProfileCount(CondBOp->getRHS()));

      // Any temporaries created here are conditional.
      eval.begin(*this);
      EmitBranchToCounterBlock(CondBOp->getRHS(), BO_LOr, TrueBlock, FalseBlock,
                               RHSCount, LH);

      eval.end(*this);

      return;
    }
  }

  if (const UnaryOperator *CondUOp = dyn_cast<UnaryOperator>(Cond)) {
    // br(!x, t, f) -> br(x, f, t)
    if (CondUOp->getOpcode() == UO_LNot) {
      // Negate the count.
      uint64_t FalseCount = getCurrentProfileCount() - TrueCount;
      // The values of the enum are chosen to make this negation possible.
      LH = static_cast<Stmt::Likelihood>(-LH);
      // Negate the condition and swap the destination blocks.
      return EmitBranchOnBoolExpr(CondUOp->getSubExpr(), FalseBlock, TrueBlock,
                                  FalseCount, LH);
    }
  }

  if (const ConditionalOperator *CondOp = dyn_cast<ConditionalOperator>(Cond)) {
    // br(c ? x : y, t, f) -> br(c, br(x, t, f), br(y, t, f))
    llvm::BasicBlock *LHSBlock = createBasicBlock("cond.true");
    llvm::BasicBlock *RHSBlock = createBasicBlock("cond.false");

    // The ConditionalOperator itself has no likelihood information for its
    // true and false branches. This matches the behavior of __builtin_expect.
    ConditionalEvaluation cond(*this);
    EmitBranchOnBoolExpr(CondOp->getCond(), LHSBlock, RHSBlock,
                         getProfileCount(CondOp), Stmt::LH_None);

    // When computing PGO branch weights, we only know the overall count for
    // the true block. This code is essentially doing tail duplication of the
    // naive code-gen, introducing new edges for which counts are not
    // available. Divide the counts proportionally between the LHS and RHS of
    // the conditional operator.
    uint64_t LHSScaledTrueCount = 0;
    if (TrueCount) {
      double LHSRatio =
          getProfileCount(CondOp) / (double)getCurrentProfileCount();
      LHSScaledTrueCount = TrueCount * LHSRatio;
    }

    cond.begin(*this);
    EmitBlock(LHSBlock);
    incrementProfileCounter(CondOp);
    {
      ApplyDebugLocation DL(*this, Cond);
      EmitBranchOnBoolExpr(CondOp->getLHS(), TrueBlock, FalseBlock,
                           LHSScaledTrueCount, LH);
    }
    cond.end(*this);

    cond.begin(*this);
    EmitBlock(RHSBlock);
    EmitBranchOnBoolExpr(CondOp->getRHS(), TrueBlock, FalseBlock,
                         TrueCount - LHSScaledTrueCount, LH);
    cond.end(*this);

    return;
  }

  if (const CXXThrowExpr *Throw = dyn_cast<CXXThrowExpr>(Cond)) {
    // Conditional operator handling can give us a throw expression as a
    // condition for a case like:
    //   br(c ? throw x : y, t, f) -> br(c, br(throw x, t, f), br(y, t, f)
    // Fold this to:
    //   br(c, throw x, br(y, t, f))
    EmitCXXThrowExpr(Throw, /*KeepInsertionPoint*/false);
    return;
  }

  // Emit the code with the fully general case.
  llvm::Value *CondV;
  {
    ApplyDebugLocation DL(*this, Cond);
    CondV = EvaluateExprAsBool(Cond);
  }

  llvm::MDNode *Weights = nullptr;
  llvm::MDNode *Unpredictable = nullptr;

  // If the branch has a condition wrapped by __builtin_unpredictable,
  // create metadata that specifies that the branch is unpredictable.
  // Don't bother if not optimizing because that metadata would not be used.
  auto *Call = dyn_cast<CallExpr>(Cond->IgnoreImpCasts());
  if (Call && CGM.getCodeGenOpts().OptimizationLevel != 0) {
    auto *FD = dyn_cast_or_null<FunctionDecl>(Call->getCalleeDecl());
    if (FD && FD->getBuiltinID() == Builtin::BI__builtin_unpredictable) {
      llvm::MDBuilder MDHelper(getLLVMContext());
      Unpredictable = MDHelper.createUnpredictable();
    }
  }

  // If there is a Likelihood knowledge for the cond, lower it.
  // Note that if not optimizing this won't emit anything.
  llvm::Value *NewCondV = emitCondLikelihoodViaExpectIntrinsic(CondV, LH);
  if (CondV != NewCondV)
    CondV = NewCondV;
  else {
    // Otherwise, lower profile counts. Note that we do this even at -O0.
    uint64_t CurrentCount = std::max(getCurrentProfileCount(), TrueCount);
    Weights = createProfileWeights(TrueCount, CurrentCount - TrueCount);
  }

  Builder.CreateCondBr(CondV, TrueBlock, FalseBlock, Weights, Unpredictable);
}

/// ErrorUnsupported - Print out an error that codegen doesn't support the
/// specified stmt yet.
void CodeGenFunction::ErrorUnsupported(const Stmt *S, const char *Type) {
  CGM.ErrorUnsupported(S, Type);
}

/// emitNonZeroVLAInit - Emit the "zero" initialization of a
/// variable-length array whose elements have a non-zero bit-pattern.
///
/// \param baseType the inner-most element type of the array
/// \param src - a char* pointing to the bit-pattern for a single
/// base element of the array
/// \param sizeInChars - the total size of the VLA, in chars
static void emitNonZeroVLAInit(CodeGenFunction &CGF, QualType baseType,
                               Address dest, Address src,
                               llvm::Value *sizeInChars) {
  CGBuilderTy &Builder = CGF.Builder;

  CharUnits baseSize = CGF.getContext().getTypeSizeInChars(baseType);
  llvm::Value *baseSizeInChars
    = llvm::ConstantInt::get(CGF.IntPtrTy, baseSize.getQuantity());

  Address begin =
    Builder.CreateElementBitCast(dest, CGF.Int8Ty, "vla.begin");
  llvm::Value *end = Builder.CreateInBoundsGEP(
      begin.getElementType(), begin.getPointer(), sizeInChars, "vla.end");

  llvm::BasicBlock *originBB = CGF.Builder.GetInsertBlock();
  llvm::BasicBlock *loopBB = CGF.createBasicBlock("vla-init.loop");
  llvm::BasicBlock *contBB = CGF.createBasicBlock("vla-init.cont");

  // Make a loop over the VLA.  C99 guarantees that the VLA element
  // count must be nonzero.
  CGF.EmitBlock(loopBB);

  llvm::PHINode *cur = Builder.CreatePHI(begin.getType(), 2, "vla.cur");
  cur->addIncoming(begin.getPointer(), originBB);

  CharUnits curAlign =
    dest.getAlignment().alignmentOfArrayElement(baseSize);

  // memcpy the individual element bit-pattern.
  Builder.CreateMemCpy(Address(cur, curAlign), src, baseSizeInChars,
                       /*volatile*/ false);

  // Go to the next element.
  llvm::Value *next =
    Builder.CreateInBoundsGEP(CGF.Int8Ty, cur, baseSizeInChars, "vla.next");

  // Leave if that's the end of the VLA.
  llvm::Value *done = Builder.CreateICmpEQ(next, end, "vla-init.isdone");
  Builder.CreateCondBr(done, contBB, loopBB);
  cur->addIncoming(next, loopBB);

  CGF.EmitBlock(contBB);
}

void
CodeGenFunction::EmitNullInitialization(Address DestPtr, QualType Ty) {
  // Ignore empty classes in C++.
  if (getLangOpts().CPlusPlus) {
    if (const RecordType *RT = Ty->getAs<RecordType>()) {
      if (cast<CXXRecordDecl>(RT->getDecl())->isEmpty())
        return;
    }
  }

  // Cast the dest ptr to the appropriate i8 pointer type.
  if (DestPtr.getElementType() != Int8Ty)
    DestPtr = Builder.CreateElementBitCast(DestPtr, Int8Ty);

  // Get size and alignment info for this aggregate.
  CharUnits size = getContext().getTypeSizeInChars(Ty);

  llvm::Value *SizeVal;
  const VariableArrayType *vla;

  // Don't bother emitting a zero-byte memset.
  if (size.isZero()) {
    // But note that getTypeInfo returns 0 for a VLA.
    if (const VariableArrayType *vlaType =
          dyn_cast_or_null<VariableArrayType>(
                                          getContext().getAsArrayType(Ty))) {
      auto VlaSize = getVLASize(vlaType);
      SizeVal = VlaSize.NumElts;
      CharUnits eltSize = getContext().getTypeSizeInChars(VlaSize.Type);
      if (!eltSize.isOne())
        SizeVal = Builder.CreateNUWMul(SizeVal, CGM.getSize(eltSize));
      vla = vlaType;
    } else {
      return;
    }
  } else {
    SizeVal = CGM.getSize(size);
    vla = nullptr;
  }

  // If the type contains a pointer to data member we can't memset it to zero.
  // Instead, create a null constant and copy it to the destination.
  // TODO: there are other patterns besides zero that we can usefully memset,
  // like -1, which happens to be the pattern used by member-pointers.
  if (!CGM.getTypes().isZeroInitializable(Ty)) {
    // For a VLA, emit a single element, then splat that over the VLA.
    if (vla) Ty = getContext().getBaseElementType(vla);

    llvm::Constant *NullConstant = CGM.EmitNullConstant(Ty);

    llvm::GlobalVariable *NullVariable =
      new llvm::GlobalVariable(CGM.getModule(), NullConstant->getType(),
                               /*isConstant=*/true,
                               llvm::GlobalVariable::PrivateLinkage,
                               NullConstant, Twine());
    CharUnits NullAlign = DestPtr.getAlignment();
    NullVariable->setAlignment(NullAlign.getAsAlign());
    Address SrcPtr(Builder.CreateBitCast(NullVariable, Builder.getInt8PtrTy()),
                   NullAlign);

    if (vla) return emitNonZeroVLAInit(*this, Ty, DestPtr, SrcPtr, SizeVal);

    // Get and call the appropriate llvm.memcpy overload.
    Builder.CreateMemCpy(DestPtr, SrcPtr, SizeVal, false);
    return;
  }

  // Otherwise, just memset the whole thing to zero.  This is legal
  // because in LLVM, all default initializers (other than the ones we just
  // handled above) are guaranteed to have a bit pattern of all zeros.
  Builder.CreateMemSet(DestPtr, Builder.getInt8(0), SizeVal, false);
}

llvm::BlockAddress *CodeGenFunction::GetAddrOfLabel(const LabelDecl *L) {
  // Make sure that there is a block for the indirect goto.
  if (!IndirectBranch)
    GetIndirectGotoBlock();

  llvm::BasicBlock *BB = getJumpDestForLabel(L).getBlock();

  // Make sure the indirect branch includes all of the address-taken blocks.
  IndirectBranch->addDestination(BB);
  return llvm::BlockAddress::get(CurFn, BB);
}

llvm::BasicBlock *CodeGenFunction::GetIndirectGotoBlock() {
  // If we already made the indirect branch for indirect goto, return its block.
  if (IndirectBranch) return IndirectBranch->getParent();

  CGBuilderTy TmpBuilder(*this, createBasicBlock("indirectgoto"));

  // Create the PHI node that indirect gotos will add entries to.
  llvm::Value *DestVal = TmpBuilder.CreatePHI(Int8PtrTy, 0,
                                              "indirect.goto.dest");

  // Create the indirect branch instruction.
  IndirectBranch = TmpBuilder.CreateIndirectBr(DestVal);
  return IndirectBranch->getParent();
}

/// Computes the length of an array in elements, as well as the base
/// element type and a properly-typed first element pointer.
llvm::Value *CodeGenFunction::emitArrayLength(const ArrayType *origArrayType,
                                              QualType &baseType,
                                              Address &addr) {
  const ArrayType *arrayType = origArrayType;

  // If it's a VLA, we have to load the stored size.  Note that
  // this is the size of the VLA in bytes, not its size in elements.
  llvm::Value *numVLAElements = nullptr;
  if (isa<VariableArrayType>(arrayType)) {
    numVLAElements = getVLASize(cast<VariableArrayType>(arrayType)).NumElts;

    // Walk into all VLAs.  This doesn't require changes to addr,
    // which has type T* where T is the first non-VLA element type.
    do {
      QualType elementType = arrayType->getElementType();
      arrayType = getContext().getAsArrayType(elementType);

      // If we only have VLA components, 'addr' requires no adjustment.
      if (!arrayType) {
        baseType = elementType;
        return numVLAElements;
      }
    } while (isa<VariableArrayType>(arrayType));

    // We get out here only if we find a constant array type
    // inside the VLA.
  }

  // We have some number of constant-length arrays, so addr should
  // have LLVM type [M x [N x [...]]]*.  Build a GEP that walks
  // down to the first element of addr.
  SmallVector<llvm::Value*, 8> gepIndices;

  // GEP down to the array type.
  llvm::ConstantInt *zero = Builder.getInt32(0);
  gepIndices.push_back(zero);

  uint64_t countFromCLAs = 1;
  QualType eltType;

  llvm::ArrayType *llvmArrayType =
    dyn_cast<llvm::ArrayType>(addr.getElementType());
  while (llvmArrayType) {
    assert(isa<ConstantArrayType>(arrayType));
    assert(cast<ConstantArrayType>(arrayType)->getSize().getZExtValue()
             == llvmArrayType->getNumElements());

    gepIndices.push_back(zero);
    countFromCLAs *= llvmArrayType->getNumElements();
    eltType = arrayType->getElementType();

    llvmArrayType =
      dyn_cast<llvm::ArrayType>(llvmArrayType->getElementType());
    arrayType = getContext().getAsArrayType(arrayType->getElementType());
    assert((!llvmArrayType || arrayType) &&
           "LLVM and Clang types are out-of-synch");
  }

  if (arrayType) {
    // From this point onwards, the Clang array type has been emitted
    // as some other type (probably a packed struct). Compute the array
    // size, and just emit the 'begin' expression as a bitcast.
    while (arrayType) {
      countFromCLAs *=
          cast<ConstantArrayType>(arrayType)->getSize().getZExtValue();
      eltType = arrayType->getElementType();
      arrayType = getContext().getAsArrayType(eltType);
    }

    llvm::Type *baseType = ConvertType(eltType);
    addr = Builder.CreateElementBitCast(addr, baseType, "array.begin");
  } else {
    // Create the actual GEP.
    addr = Address(Builder.CreateInBoundsGEP(
        addr.getElementType(), addr.getPointer(), gepIndices, "array.begin"),
        addr.getAlignment());
  }

  baseType = eltType;

  llvm::Value *numElements
    = llvm::ConstantInt::get(SizeTy, countFromCLAs);

  // If we had any VLA dimensions, factor them in.
  if (numVLAElements)
    numElements = Builder.CreateNUWMul(numVLAElements, numElements);

  return numElements;
}

CodeGenFunction::VlaSizePair CodeGenFunction::getVLASize(QualType type) {
  const VariableArrayType *vla = getContext().getAsVariableArrayType(type);
  assert(vla && "type was not a variable array type!");
  return getVLASize(vla);
}

CodeGenFunction::VlaSizePair
CodeGenFunction::getVLASize(const VariableArrayType *type) {
  // The number of elements so far; always size_t.
  llvm::Value *numElements = nullptr;

  QualType elementType;
  do {
    elementType = type->getElementType();
    llvm::Value *vlaSize = VLASizeMap[type->getSizeExpr()];
    assert(vlaSize && "no size for VLA!");
    assert(vlaSize->getType() == SizeTy);

    if (!numElements) {
      numElements = vlaSize;
    } else {
      // It's undefined behavior if this wraps around, so mark it that way.
      // FIXME: Teach -fsanitize=undefined to trap this.
      numElements = Builder.CreateNUWMul(numElements, vlaSize);
    }
  } while ((type = getContext().getAsVariableArrayType(elementType)));

  return { numElements, elementType };
}

CodeGenFunction::VlaSizePair
CodeGenFunction::getVLAElements1D(QualType type) {
  const VariableArrayType *vla = getContext().getAsVariableArrayType(type);
  assert(vla && "type was not a variable array type!");
  return getVLAElements1D(vla);
}

CodeGenFunction::VlaSizePair
CodeGenFunction::getVLAElements1D(const VariableArrayType *Vla) {
  llvm::Value *VlaSize = VLASizeMap[Vla->getSizeExpr()];
  assert(VlaSize && "no size for VLA!");
  assert(VlaSize->getType() == SizeTy);
  return { VlaSize, Vla->getElementType() };
}

void CodeGenFunction::EmitVariablyModifiedType(QualType type) {
  assert(type->isVariablyModifiedType() &&
         "Must pass variably modified type to EmitVLASizes!");

  EnsureInsertPoint();

  // We're going to walk down into the type and look for VLA
  // expressions.
  do {
    assert(type->isVariablyModifiedType());

    const Type *ty = type.getTypePtr();
    switch (ty->getTypeClass()) {

#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_TYPE(Class, Base)
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class, Base)
#include "clang/AST/TypeNodes.inc"
      llvm_unreachable("unexpected dependent type!");

    // These types are never variably-modified.
    case Type::Builtin:
    case Type::Complex:
    case Type::Vector:
    case Type::ExtVector:
    case Type::ConstantMatrix:
    case Type::Record:
    case Type::Enum:
    case Type::Elaborated:
    case Type::TemplateSpecialization:
    case Type::ObjCTypeParam:
    case Type::ObjCObject:
    case Type::ObjCInterface:
    case Type::ObjCObjectPointer:
    case Type::ExtInt:
      llvm_unreachable("type class is never variably-modified!");

    case Type::Adjusted:
      type = cast<AdjustedType>(ty)->getAdjustedType();
      break;

    case Type::Decayed:
      type = cast<DecayedType>(ty)->getPointeeType();
      break;

    case Type::Pointer:
      type = cast<PointerType>(ty)->getPointeeType();
      break;

    case Type::BlockPointer:
      type = cast<BlockPointerType>(ty)->getPointeeType();
      break;

    case Type::LValueReference:
    case Type::RValueReference:
      type = cast<ReferenceType>(ty)->getPointeeType();
      break;

    case Type::MemberPointer:
      type = cast<MemberPointerType>(ty)->getPointeeType();
      break;

    case Type::ConstantArray:
    case Type::IncompleteArray:
      // Losing element qualification here is fine.
      type = cast<ArrayType>(ty)->getElementType();
      break;

    case Type::VariableArray: {
      // Losing element qualification here is fine.
      const VariableArrayType *vat = cast<VariableArrayType>(ty);

      // Unknown size indication requires no size computation.
      // Otherwise, evaluate and record it.
      if (const Expr *size = vat->getSizeExpr()) {
        // It's possible that we might have emitted this already,
        // e.g. with a typedef and a pointer to it.
        llvm::Value *&entry = VLASizeMap[size];
        if (!entry) {
          llvm::Value *Size = EmitScalarExpr(size);

          // C11 6.7.6.2p5:
          //   If the size is an expression that is not an integer constant
          //   expression [...] each time it is evaluated it shall have a value
          //   greater than zero.
          if (SanOpts.has(SanitizerKind::VLABound) &&
              size->getType()->isSignedIntegerType()) {
            SanitizerScope SanScope(this);
            llvm::Value *Zero = llvm::Constant::getNullValue(Size->getType());
            llvm::Constant *StaticArgs[] = {
                EmitCheckSourceLocation(size->getBeginLoc()),
                EmitCheckTypeDescriptor(size->getType())};
            EmitCheck(std::make_pair(Builder.CreateICmpSGT(Size, Zero),
                                     SanitizerKind::VLABound),
                      SanitizerHandler::VLABoundNotPositive, StaticArgs, Size);
          }

          // Always zexting here would be wrong if it weren't
          // undefined behavior to have a negative bound.
          entry = Builder.CreateIntCast(Size, SizeTy, /*signed*/ false);
        }
      }
      type = vat->getElementType();
      break;
    }

    case Type::FunctionProto:
    case Type::FunctionNoProto:
      type = cast<FunctionType>(ty)->getReturnType();
      break;

    case Type::Paren:
    case Type::TypeOf:
    case Type::UnaryTransform:
    case Type::Attributed:
    case Type::SubstTemplateTypeParm:
    case Type::MacroQualified:
      // Keep walking after single level desugaring.
      type = type.getSingleStepDesugaredType(getContext());
      break;

    case Type::Typedef:
    case Type::Decltype:
    case Type::Auto:
    case Type::DeducedTemplateSpecialization:
      // Stop walking: nothing to do.
      return;

    case Type::TypeOfExpr:
      // Stop walking: emit typeof expression.
      EmitIgnoredExpr(cast<TypeOfExprType>(ty)->getUnderlyingExpr());
      return;

    case Type::Atomic:
      type = cast<AtomicType>(ty)->getValueType();
      break;

    case Type::Pipe:
      type = cast<PipeType>(ty)->getElementType();
      break;
    }
  } while (type->isVariablyModifiedType());
}

Address CodeGenFunction::EmitVAListRef(const Expr* E) {
  if (getContext().getBuiltinVaListType()->isArrayType())
    return EmitPointerWithAlignment(E);
  return EmitLValue(E).getAddress(*this);
}

Address CodeGenFunction::EmitMSVAListRef(const Expr *E) {
  return EmitLValue(E).getAddress(*this);
}

void CodeGenFunction::EmitDeclRefExprDbgValue(const DeclRefExpr *E,
                                              const APValue &Init) {
  assert(Init.hasValue() && "Invalid DeclRefExpr initializer!");
  if (CGDebugInfo *Dbg = getDebugInfo())
    if (CGM.getCodeGenOpts().hasReducedDebugInfo())
      Dbg->EmitGlobalVariable(E->getDecl(), Init);
}

CodeGenFunction::PeepholeProtection
CodeGenFunction::protectFromPeepholes(RValue rvalue) {
  // At the moment, the only aggressive peephole we do in IR gen
  // is trunc(zext) folding, but if we add more, we can easily
  // extend this protection.

  if (!rvalue.isScalar()) return PeepholeProtection();
  llvm::Value *value = rvalue.getScalarVal();
  if (!isa<llvm::ZExtInst>(value)) return PeepholeProtection();

  // Just make an extra bitcast.
  assert(HaveInsertPoint());
  llvm::Instruction *inst = new llvm::BitCastInst(value, value->getType(), "",
                                                  Builder.GetInsertBlock());

  PeepholeProtection protection;
  protection.Inst = inst;
  return protection;
}

void CodeGenFunction::unprotectFromPeepholes(PeepholeProtection protection) {
  if (!protection.Inst) return;

  // In theory, we could try to duplicate the peepholes now, but whatever.
  protection.Inst->eraseFromParent();
}

void CodeGenFunction::emitAlignmentAssumption(llvm::Value *PtrValue,
                                              QualType Ty, SourceLocation Loc,
                                              SourceLocation AssumptionLoc,
                                              llvm::Value *Alignment,
                                              llvm::Value *OffsetValue) {
  if (Alignment->getType() != IntPtrTy)
    Alignment =
        Builder.CreateIntCast(Alignment, IntPtrTy, false, "casted.align");
  if (OffsetValue && OffsetValue->getType() != IntPtrTy)
    OffsetValue =
        Builder.CreateIntCast(OffsetValue, IntPtrTy, true, "casted.offset");
  llvm::Value *TheCheck = nullptr;
  if (SanOpts.has(SanitizerKind::Alignment)) {
    llvm::Value *PtrIntValue =
        Builder.CreatePtrToInt(PtrValue, IntPtrTy, "ptrint");

    if (OffsetValue) {
      bool IsOffsetZero = false;
      if (const auto *CI = dyn_cast<llvm::ConstantInt>(OffsetValue))
        IsOffsetZero = CI->isZero();

      if (!IsOffsetZero)
        PtrIntValue = Builder.CreateSub(PtrIntValue, OffsetValue, "offsetptr");
    }

    llvm::Value *Zero = llvm::ConstantInt::get(IntPtrTy, 0);
    llvm::Value *Mask =
        Builder.CreateSub(Alignment, llvm::ConstantInt::get(IntPtrTy, 1));
    llvm::Value *MaskedPtr = Builder.CreateAnd(PtrIntValue, Mask, "maskedptr");
    TheCheck = Builder.CreateICmpEQ(MaskedPtr, Zero, "maskcond");
  }
  llvm::Instruction *Assumption = Builder.CreateAlignmentAssumption(
      CGM.getDataLayout(), PtrValue, Alignment, OffsetValue);

  if (!SanOpts.has(SanitizerKind::Alignment))
    return;
  emitAlignmentAssumptionCheck(PtrValue, Ty, Loc, AssumptionLoc, Alignment,
                               OffsetValue, TheCheck, Assumption);
}

void CodeGenFunction::emitAlignmentAssumption(llvm::Value *PtrValue,
                                              const Expr *E,
                                              SourceLocation AssumptionLoc,
                                              llvm::Value *Alignment,
                                              llvm::Value *OffsetValue) {
  if (auto *CE = dyn_cast<CastExpr>(E))
    E = CE->getSubExprAsWritten();
  QualType Ty = E->getType();
  SourceLocation Loc = E->getExprLoc();

  emitAlignmentAssumption(PtrValue, Ty, Loc, AssumptionLoc, Alignment,
                          OffsetValue);
}

llvm::Value *CodeGenFunction::EmitAnnotationCall(llvm::Function *AnnotationFn,
                                                 llvm::Value *AnnotatedVal,
                                                 StringRef AnnotationStr,
                                                 SourceLocation Location,
                                                 const AnnotateAttr *Attr) {
  SmallVector<llvm::Value *, 5> Args = {
      AnnotatedVal,
      Builder.CreateBitCast(CGM.EmitAnnotationString(AnnotationStr), Int8PtrTy),
      Builder.CreateBitCast(CGM.EmitAnnotationUnit(Location), Int8PtrTy),
      CGM.EmitAnnotationLineNo(Location),
  };
  if (Attr)
    Args.push_back(CGM.EmitAnnotationArgs(Attr));
  return Builder.CreateCall(AnnotationFn, Args);
}

void CodeGenFunction::EmitVarAnnotations(const VarDecl *D, llvm::Value *V) {
  assert(D->hasAttr<AnnotateAttr>() && "no annotate attribute");
  // FIXME We create a new bitcast for every annotation because that's what
  // llvm-gcc was doing.
  for (const auto *I : D->specific_attrs<AnnotateAttr>())
    EmitAnnotationCall(CGM.getIntrinsic(llvm::Intrinsic::var_annotation),
                       Builder.CreateBitCast(V, CGM.Int8PtrTy, V->getName()),
                       I->getAnnotation(), D->getLocation(), I);
}

Address CodeGenFunction::EmitFieldAnnotations(const FieldDecl *D,
                                              Address Addr) {
  assert(D->hasAttr<AnnotateAttr>() && "no annotate attribute");
  llvm::Value *V = Addr.getPointer();
  llvm::Type *VTy = V->getType();
  auto *PTy = dyn_cast<llvm::PointerType>(VTy);
  unsigned AS = PTy ? PTy->getAddressSpace() : 0;
  llvm::PointerType *IntrinTy =
      llvm::PointerType::getWithSamePointeeType(CGM.Int8PtrTy, AS);
  llvm::Function *F =
      CGM.getIntrinsic(llvm::Intrinsic::ptr_annotation, IntrinTy);

  for (const auto *I : D->specific_attrs<AnnotateAttr>()) {
    // FIXME Always emit the cast inst so we can differentiate between
    // annotation on the first field of a struct and annotation on the struct
    // itself.
    if (VTy != IntrinTy)
      V = Builder.CreateBitCast(V, IntrinTy);
    V = EmitAnnotationCall(F, V, I->getAnnotation(), D->getLocation(), I);
    V = Builder.CreateBitCast(V, VTy);
  }

  return Address(V, Addr.getAlignment());
}

CodeGenFunction::CGCapturedStmtInfo::~CGCapturedStmtInfo() { }

CodeGenFunction::SanitizerScope::SanitizerScope(CodeGenFunction *CGF)
    : CGF(CGF) {
  assert(!CGF->IsSanitizerScope);
  CGF->IsSanitizerScope = true;
}

CodeGenFunction::SanitizerScope::~SanitizerScope() {
  CGF->IsSanitizerScope = false;
}

void CodeGenFunction::InsertHelper(llvm::Instruction *I,
                                   const llvm::Twine &Name,
                                   llvm::BasicBlock *BB,
                                   llvm::BasicBlock::iterator InsertPt) const {
  LoopStack.InsertHelper(I);
  if (IsSanitizerScope)
    CGM.getSanitizerMetadata()->disableSanitizerForInstruction(I);
}

void CGBuilderInserter::InsertHelper(
    llvm::Instruction *I, const llvm::Twine &Name, llvm::BasicBlock *BB,
    llvm::BasicBlock::iterator InsertPt) const {
  llvm::IRBuilderDefaultInserter::InsertHelper(I, Name, BB, InsertPt);
  if (CGF)
    CGF->InsertHelper(I, Name, BB, InsertPt);
}

// Emits an error if we don't have a valid set of target features for the
// called function.
void CodeGenFunction::checkTargetFeatures(const CallExpr *E,
                                          const FunctionDecl *TargetDecl) {
  return checkTargetFeatures(E->getBeginLoc(), TargetDecl);
}

// Emits an error if we don't have a valid set of target features for the
// called function.
void CodeGenFunction::checkTargetFeatures(SourceLocation Loc,
                                          const FunctionDecl *TargetDecl) {
  // Early exit if this is an indirect call.
  if (!TargetDecl)
    return;

  // Get the current enclosing function if it exists. If it doesn't
  // we can't check the target features anyhow.
  const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(CurCodeDecl);
  if (!FD)
    return;

  // Grab the required features for the call. For a builtin this is listed in
  // the td file with the default cpu, for an always_inline function this is any
  // listed cpu and any listed features.
  unsigned BuiltinID = TargetDecl->getBuiltinID();
  std::string MissingFeature;
  llvm::StringMap<bool> CallerFeatureMap;
  CGM.getContext().getFunctionFeatureMap(CallerFeatureMap, FD);
  if (BuiltinID) {
    StringRef FeatureList(
        CGM.getContext().BuiltinInfo.getRequiredFeatures(BuiltinID));
    // Return if the builtin doesn't have any required features.
    if (FeatureList.empty())
      return;
    assert(!FeatureList.contains(' ') && "Space in feature list");
    TargetFeatures TF(CallerFeatureMap);
    if (!TF.hasRequiredFeatures(FeatureList))
      CGM.getDiags().Report(Loc, diag::err_builtin_needs_feature)
          << TargetDecl->getDeclName() << FeatureList;
  } else if (!TargetDecl->isMultiVersion() &&
             TargetDecl->hasAttr<TargetAttr>()) {
    // Get the required features for the callee.

    const TargetAttr *TD = TargetDecl->getAttr<TargetAttr>();
    ParsedTargetAttr ParsedAttr =
        CGM.getContext().filterFunctionTargetAttrs(TD);

    SmallVector<StringRef, 1> ReqFeatures;
    llvm::StringMap<bool> CalleeFeatureMap;
    CGM.getContext().getFunctionFeatureMap(CalleeFeatureMap, TargetDecl);

    for (const auto &F : ParsedAttr.Features) {
      if (F[0] == '+' && CalleeFeatureMap.lookup(F.substr(1)))
        ReqFeatures.push_back(StringRef(F).substr(1));
    }

    for (const auto &F : CalleeFeatureMap) {
      // Only positive features are "required".
      if (F.getValue())
        ReqFeatures.push_back(F.getKey());
    }
    if (!llvm::all_of(ReqFeatures, [&](StringRef Feature) {
      if (!CallerFeatureMap.lookup(Feature)) {
        MissingFeature = Feature.str();
        return false;
      }
      return true;
    }))
      CGM.getDiags().Report(Loc, diag::err_function_needs_feature)
          << FD->getDeclName() << TargetDecl->getDeclName() << MissingFeature;
  }
}

void CodeGenFunction::EmitSanitizerStatReport(llvm::SanitizerStatKind SSK) {
  if (!CGM.getCodeGenOpts().SanitizeStats)
    return;

  llvm::IRBuilder<> IRB(Builder.GetInsertBlock(), Builder.GetInsertPoint());
  IRB.SetCurrentDebugLocation(Builder.getCurrentDebugLocation());
  CGM.getSanStats().create(IRB, SSK);
}

llvm::Value *
CodeGenFunction::FormResolverCondition(const MultiVersionResolverOption &RO) {
  llvm::Value *Condition = nullptr;

  if (!RO.Conditions.Architecture.empty())
    Condition = EmitX86CpuIs(RO.Conditions.Architecture);

  if (!RO.Conditions.Features.empty()) {
    llvm::Value *FeatureCond = EmitX86CpuSupports(RO.Conditions.Features);
    Condition =
        Condition ? Builder.CreateAnd(Condition, FeatureCond) : FeatureCond;
  }
  return Condition;
}

static void CreateMultiVersionResolverReturn(CodeGenModule &CGM,
                                             llvm::Function *Resolver,
                                             CGBuilderTy &Builder,
                                             llvm::Function *FuncToReturn,
                                             bool SupportsIFunc) {
  if (SupportsIFunc) {
    Builder.CreateRet(FuncToReturn);
    return;
  }

  llvm::SmallVector<llvm::Value *, 10> Args;
  llvm::for_each(Resolver->args(),
                 [&](llvm::Argument &Arg) { Args.push_back(&Arg); });

  llvm::CallInst *Result = Builder.CreateCall(FuncToReturn, Args);
  Result->setTailCallKind(llvm::CallInst::TCK_MustTail);

  if (Resolver->getReturnType()->isVoidTy())
    Builder.CreateRetVoid();
  else
    Builder.CreateRet(Result);
}

void CodeGenFunction::EmitMultiVersionResolver(
    llvm::Function *Resolver, ArrayRef<MultiVersionResolverOption> Options) {
  assert(getContext().getTargetInfo().getTriple().isX86() &&
         "Only implemented for x86 targets");

  bool SupportsIFunc = getContext().getTargetInfo().supportsIFunc();

  // Main function's basic block.
  llvm::BasicBlock *CurBlock = createBasicBlock("resolver_entry", Resolver);
  Builder.SetInsertPoint(CurBlock);
  EmitX86CpuInit();

  for (const MultiVersionResolverOption &RO : Options) {
    Builder.SetInsertPoint(CurBlock);
    llvm::Value *Condition = FormResolverCondition(RO);

    // The 'default' or 'generic' case.
    if (!Condition) {
      assert(&RO == Options.end() - 1 &&
             "Default or Generic case must be last");
      CreateMultiVersionResolverReturn(CGM, Resolver, Builder, RO.Function,
                                       SupportsIFunc);
      return;
    }

    llvm::BasicBlock *RetBlock = createBasicBlock("resolver_return", Resolver);
    CGBuilderTy RetBuilder(*this, RetBlock);
    CreateMultiVersionResolverReturn(CGM, Resolver, RetBuilder, RO.Function,
                                     SupportsIFunc);
    CurBlock = createBasicBlock("resolver_else", Resolver);
    Builder.CreateCondBr(Condition, RetBlock, CurBlock);
  }

  // If no generic/default, emit an unreachable.
  Builder.SetInsertPoint(CurBlock);
  llvm::CallInst *TrapCall = EmitTrapCall(llvm::Intrinsic::trap);
  TrapCall->setDoesNotReturn();
  TrapCall->setDoesNotThrow();
  Builder.CreateUnreachable();
  Builder.ClearInsertionPoint();
}

// Loc - where the diagnostic will point, where in the source code this
//  alignment has failed.
// SecondaryLoc - if present (will be present if sufficiently different from
//  Loc), the diagnostic will additionally point a "Note:" to this location.
//  It should be the location where the __attribute__((assume_aligned))
//  was written e.g.
void CodeGenFunction::emitAlignmentAssumptionCheck(
    llvm::Value *Ptr, QualType Ty, SourceLocation Loc,
    SourceLocation SecondaryLoc, llvm::Value *Alignment,
    llvm::Value *OffsetValue, llvm::Value *TheCheck,
    llvm::Instruction *Assumption) {
  assert(Assumption && isa<llvm::CallInst>(Assumption) &&
         cast<llvm::CallInst>(Assumption)->getCalledOperand() ==
             llvm::Intrinsic::getDeclaration(
                 Builder.GetInsertBlock()->getParent()->getParent(),
                 llvm::Intrinsic::assume) &&
         "Assumption should be a call to llvm.assume().");
  assert(&(Builder.GetInsertBlock()->back()) == Assumption &&
         "Assumption should be the last instruction of the basic block, "
         "since the basic block is still being generated.");

  if (!SanOpts.has(SanitizerKind::Alignment))
    return;

  // Don't check pointers to volatile data. The behavior here is implementation-
  // defined.
  if (Ty->getPointeeType().isVolatileQualified())
    return;

  // We need to temorairly remove the assumption so we can insert the
  // sanitizer check before it, else the check will be dropped by optimizations.
  Assumption->removeFromParent();

  {
    SanitizerScope SanScope(this);

    if (!OffsetValue)
      OffsetValue = Builder.getInt1(0); // no offset.

    llvm::Constant *StaticData[] = {EmitCheckSourceLocation(Loc),
                                    EmitCheckSourceLocation(SecondaryLoc),
                                    EmitCheckTypeDescriptor(Ty)};
    llvm::Value *DynamicData[] = {EmitCheckValue(Ptr),
                                  EmitCheckValue(Alignment),
                                  EmitCheckValue(OffsetValue)};
    EmitCheck({std::make_pair(TheCheck, SanitizerKind::Alignment)},
              SanitizerHandler::AlignmentAssumption, StaticData, DynamicData);
  }

  // We are now in the (new, empty) "cont" basic block.
  // Reintroduce the assumption.
  Builder.Insert(Assumption);
  // FIXME: Assumption still has it's original basic block as it's Parent.
}

llvm::DebugLoc CodeGenFunction::SourceLocToDebugLoc(SourceLocation Location) {
  if (CGDebugInfo *DI = getDebugInfo())
    return DI->SourceLocToDebugLoc(Location);

  return llvm::DebugLoc();
}

llvm::Value *
CodeGenFunction::emitCondLikelihoodViaExpectIntrinsic(llvm::Value *Cond,
                                                      Stmt::Likelihood LH) {
  switch (LH) {
  case Stmt::LH_None:
    return Cond;
  case Stmt::LH_Likely:
  case Stmt::LH_Unlikely:
    // Don't generate llvm.expect on -O0 as the backend won't use it for
    // anything.
    if (CGM.getCodeGenOpts().OptimizationLevel == 0)
      return Cond;
    llvm::Type *CondTy = Cond->getType();
    assert(CondTy->isIntegerTy(1) && "expecting condition to be a boolean");
    llvm::Function *FnExpect =
        CGM.getIntrinsic(llvm::Intrinsic::expect, CondTy);
    llvm::Value *ExpectedValueOfCond =
        llvm::ConstantInt::getBool(CondTy, LH == Stmt::LH_Likely);
    return Builder.CreateCall(FnExpect, {Cond, ExpectedValueOfCond},
                              Cond->getName() + ".expval");
  }
  llvm_unreachable("Unknown Likelihood");
}
