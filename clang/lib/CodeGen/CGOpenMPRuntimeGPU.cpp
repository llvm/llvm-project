//===---- CGOpenMPRuntimeGPU.cpp - Interface to OpenMP GPU Runtimes ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides a generalized class for OpenMP runtime code generation
// specialized by GPU targets NVPTX and AMDGCN.
//
//===----------------------------------------------------------------------===//

#include "CGOpenMPRuntimeGPU.h"
#include "CodeGenFunction.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclOpenMP.h"
#include "clang/AST/OpenMPClause.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Cuda.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"
#include "llvm/Support/MathExtras.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm::omp;

namespace {
/// Pre(post)-action for different OpenMP constructs specialized for NVPTX.
class NVPTXActionTy final : public PrePostActionTy {
  llvm::FunctionCallee EnterCallee = nullptr;
  ArrayRef<llvm::Value *> EnterArgs;
  llvm::FunctionCallee ExitCallee = nullptr;
  ArrayRef<llvm::Value *> ExitArgs;
  bool Conditional = false;
  llvm::BasicBlock *ContBlock = nullptr;

public:
  NVPTXActionTy(llvm::FunctionCallee EnterCallee,
                ArrayRef<llvm::Value *> EnterArgs,
                llvm::FunctionCallee ExitCallee,
                ArrayRef<llvm::Value *> ExitArgs, bool Conditional = false)
      : EnterCallee(EnterCallee), EnterArgs(EnterArgs), ExitCallee(ExitCallee),
        ExitArgs(ExitArgs), Conditional(Conditional) {}
  void Enter(CodeGenFunction &CGF) override {
    llvm::Value *EnterRes = CGF.EmitRuntimeCall(EnterCallee, EnterArgs);
    if (Conditional) {
      llvm::Value *CallBool = CGF.Builder.CreateIsNotNull(EnterRes);
      auto *ThenBlock = CGF.createBasicBlock("omp_if.then");
      ContBlock = CGF.createBasicBlock("omp_if.end");
      // Generate the branch (If-stmt)
      CGF.Builder.CreateCondBr(CallBool, ThenBlock, ContBlock);
      CGF.EmitBlock(ThenBlock);
    }
  }
  void Done(CodeGenFunction &CGF) {
    // Emit the rest of blocks/branches
    CGF.EmitBranch(ContBlock);
    CGF.EmitBlock(ContBlock, true);
  }
  void Exit(CodeGenFunction &CGF) override {
    CGF.EmitRuntimeCall(ExitCallee, ExitArgs);
  }
};

/// A class to track the execution mode when codegening directives within
/// a target region. The appropriate mode (SPMD|NON-SPMD) is set on entry
/// to the target region and used by containing directives such as 'parallel'
/// to emit optimized code.
class ExecutionRuntimeModesRAII {
private:
  CGOpenMPRuntimeGPU::ExecutionMode SavedExecMode =
      CGOpenMPRuntimeGPU::EM_Unknown;
  CGOpenMPRuntimeGPU::ExecutionMode &ExecMode;

public:
  ExecutionRuntimeModesRAII(CGOpenMPRuntimeGPU::ExecutionMode &ExecMode,
                            CGOpenMPRuntimeGPU::ExecutionMode EntryMode)
      : ExecMode(ExecMode) {
    SavedExecMode = ExecMode;
    ExecMode = EntryMode;
  }
  ~ExecutionRuntimeModesRAII() { ExecMode = SavedExecMode; }
};

/// GPU Configuration:  This information can be derived from cuda registers,
/// however, providing compile time constants helps generate more efficient
/// code.  For all practical purposes this is fine because the configuration
/// is the same for all known NVPTX architectures.
enum MachineConfiguration : unsigned {
  /// See "llvm/Frontend/OpenMP/OMPGridValues.h" for various related target
  /// specific Grid Values like GV_Warp_Size, GV_Slot_Size

  /// Global memory alignment for performance.
  GlobalMemoryAlignment = 128,
};

static const ValueDecl *getPrivateItem(const Expr *RefExpr) {
  RefExpr = RefExpr->IgnoreParens();
  if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(RefExpr)) {
    const Expr *Base = ASE->getBase()->IgnoreParenImpCasts();
    while (const auto *TempASE = dyn_cast<ArraySubscriptExpr>(Base))
      Base = TempASE->getBase()->IgnoreParenImpCasts();
    RefExpr = Base;
  } else if (auto *OASE = dyn_cast<OMPArraySectionExpr>(RefExpr)) {
    const Expr *Base = OASE->getBase()->IgnoreParenImpCasts();
    while (const auto *TempOASE = dyn_cast<OMPArraySectionExpr>(Base))
      Base = TempOASE->getBase()->IgnoreParenImpCasts();
    while (const auto *TempASE = dyn_cast<ArraySubscriptExpr>(Base))
      Base = TempASE->getBase()->IgnoreParenImpCasts();
    RefExpr = Base;
  }
  RefExpr = RefExpr->IgnoreParenImpCasts();
  if (const auto *DE = dyn_cast<DeclRefExpr>(RefExpr))
    return cast<ValueDecl>(DE->getDecl()->getCanonicalDecl());
  const auto *ME = cast<MemberExpr>(RefExpr);
  return cast<ValueDecl>(ME->getMemberDecl()->getCanonicalDecl());
}


static RecordDecl *buildRecordForGlobalizedVars(
    ASTContext &C, ArrayRef<const ValueDecl *> EscapedDecls,
    ArrayRef<const ValueDecl *> EscapedDeclsForTeams,
    llvm::SmallDenseMap<const ValueDecl *, const FieldDecl *>
        &MappedDeclsFields, int BufSize) {
  using VarsDataTy = std::pair<CharUnits /*Align*/, const ValueDecl *>;
  if (EscapedDecls.empty() && EscapedDeclsForTeams.empty())
    return nullptr;
  SmallVector<VarsDataTy, 4> GlobalizedVars;
  for (const ValueDecl *D : EscapedDecls)
    GlobalizedVars.emplace_back(
        CharUnits::fromQuantity(std::max(
            C.getDeclAlign(D).getQuantity(),
            static_cast<CharUnits::QuantityType>(GlobalMemoryAlignment))),
        D);
  for (const ValueDecl *D : EscapedDeclsForTeams)
    GlobalizedVars.emplace_back(C.getDeclAlign(D), D);
  llvm::stable_sort(GlobalizedVars, [](VarsDataTy L, VarsDataTy R) {
    return L.first > R.first;
  });

  // Build struct _globalized_locals_ty {
  //         /*  globalized vars  */[WarSize] align (max(decl_align,
  //         GlobalMemoryAlignment))
  //         /*  globalized vars  */ for EscapedDeclsForTeams
  //       };
  RecordDecl *GlobalizedRD = C.buildImplicitRecord("_globalized_locals_ty");
  GlobalizedRD->startDefinition();
  llvm::SmallPtrSet<const ValueDecl *, 16> SingleEscaped(
      EscapedDeclsForTeams.begin(), EscapedDeclsForTeams.end());
  for (const auto &Pair : GlobalizedVars) {
    const ValueDecl *VD = Pair.second;
    QualType Type = VD->getType();
    if (Type->isLValueReferenceType())
      Type = C.getPointerType(Type.getNonReferenceType());
    else
      Type = Type.getNonReferenceType();
    SourceLocation Loc = VD->getLocation();
    FieldDecl *Field;
    if (SingleEscaped.count(VD)) {
      Field = FieldDecl::Create(
          C, GlobalizedRD, Loc, Loc, VD->getIdentifier(), Type,
          C.getTrivialTypeSourceInfo(Type, SourceLocation()),
          /*BW=*/nullptr, /*Mutable=*/false,
          /*InitStyle=*/ICIS_NoInit);
      Field->setAccess(AS_public);
      if (VD->hasAttrs()) {
        for (specific_attr_iterator<AlignedAttr> I(VD->getAttrs().begin()),
             E(VD->getAttrs().end());
             I != E; ++I)
          Field->addAttr(*I);
      }
    } else {
      llvm::APInt ArraySize(32, BufSize);
      Type = C.getConstantArrayType(Type, ArraySize, nullptr, ArrayType::Normal,
                                    0);
      Field = FieldDecl::Create(
          C, GlobalizedRD, Loc, Loc, VD->getIdentifier(), Type,
          C.getTrivialTypeSourceInfo(Type, SourceLocation()),
          /*BW=*/nullptr, /*Mutable=*/false,
          /*InitStyle=*/ICIS_NoInit);
      Field->setAccess(AS_public);
      llvm::APInt Align(32, std::max(C.getDeclAlign(VD).getQuantity(),
                                     static_cast<CharUnits::QuantityType>(
                                         GlobalMemoryAlignment)));
      Field->addAttr(AlignedAttr::CreateImplicit(
          C, /*IsAlignmentExpr=*/true,
          IntegerLiteral::Create(C, Align,
                                 C.getIntTypeForBitwidth(32, /*Signed=*/0),
                                 SourceLocation()),
          {}, AlignedAttr::GNU_aligned));
    }
    GlobalizedRD->addDecl(Field);
    MappedDeclsFields.try_emplace(VD, Field);
  }
  GlobalizedRD->completeDefinition();
  return GlobalizedRD;
}

/// Get the list of variables that can escape their declaration context.
class CheckVarsEscapingDeclContext final
    : public ConstStmtVisitor<CheckVarsEscapingDeclContext> {
  CodeGenFunction &CGF;
  llvm::SetVector<const ValueDecl *> EscapedDecls;
  llvm::SetVector<const ValueDecl *> EscapedVariableLengthDecls;
  llvm::SetVector<const ValueDecl *> DelayedVariableLengthDecls;
  llvm::SmallPtrSet<const Decl *, 4> EscapedParameters;
  RecordDecl *GlobalizedRD = nullptr;
  llvm::SmallDenseMap<const ValueDecl *, const FieldDecl *> MappedDeclsFields;
  bool AllEscaped = false;
  bool IsForCombinedParallelRegion = false;

  void markAsEscaped(const ValueDecl *VD) {
    // Do not globalize declare target variables.
    if (!isa<VarDecl>(VD) ||
        OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(VD))
      return;
    VD = cast<ValueDecl>(VD->getCanonicalDecl());
    // Use user-specified allocation.
    if (VD->hasAttrs() && VD->hasAttr<OMPAllocateDeclAttr>())
      return;
    // Variables captured by value must be globalized.
    bool IsCaptured = false;
    if (auto *CSI = CGF.CapturedStmtInfo) {
      if (const FieldDecl *FD = CSI->lookup(cast<VarDecl>(VD))) {
        // Check if need to capture the variable that was already captured by
        // value in the outer region.
        IsCaptured = true;
        if (!IsForCombinedParallelRegion) {
          if (!FD->hasAttrs())
            return;
          const auto *Attr = FD->getAttr<OMPCaptureKindAttr>();
          if (!Attr)
            return;
          if (((Attr->getCaptureKind() != OMPC_map) &&
               !isOpenMPPrivate(Attr->getCaptureKind())) ||
              ((Attr->getCaptureKind() == OMPC_map) &&
               !FD->getType()->isAnyPointerType()))
            return;
        }
        if (!FD->getType()->isReferenceType()) {
          assert(!VD->getType()->isVariablyModifiedType() &&
                 "Parameter captured by value with variably modified type");
          EscapedParameters.insert(VD);
        } else if (!IsForCombinedParallelRegion) {
          return;
        }
      }
    }
    if ((!CGF.CapturedStmtInfo ||
         (IsForCombinedParallelRegion && CGF.CapturedStmtInfo)) &&
        VD->getType()->isReferenceType())
      // Do not globalize variables with reference type.
      return;
    if (VD->getType()->isVariablyModifiedType()) {
      // If not captured at the target region level then mark the escaped
      // variable as delayed.
      if (IsCaptured)
        EscapedVariableLengthDecls.insert(VD);
      else
        DelayedVariableLengthDecls.insert(VD);
    } else
      EscapedDecls.insert(VD);
  }

  void VisitValueDecl(const ValueDecl *VD) {
    if (VD->getType()->isLValueReferenceType())
      markAsEscaped(VD);
    if (const auto *VarD = dyn_cast<VarDecl>(VD)) {
      if (!isa<ParmVarDecl>(VarD) && VarD->hasInit()) {
        const bool SavedAllEscaped = AllEscaped;
        AllEscaped = VD->getType()->isLValueReferenceType();
        Visit(VarD->getInit());
        AllEscaped = SavedAllEscaped;
      }
    }
  }
  void VisitOpenMPCapturedStmt(const CapturedStmt *S,
                               ArrayRef<OMPClause *> Clauses,
                               bool IsCombinedParallelRegion) {
    if (!S)
      return;
    for (const CapturedStmt::Capture &C : S->captures()) {
      if (C.capturesVariable() && !C.capturesVariableByCopy()) {
        const ValueDecl *VD = C.getCapturedVar();
        bool SavedIsForCombinedParallelRegion = IsForCombinedParallelRegion;
        if (IsCombinedParallelRegion) {
          // Check if the variable is privatized in the combined construct and
          // those private copies must be shared in the inner parallel
          // directive.
          IsForCombinedParallelRegion = false;
          for (const OMPClause *C : Clauses) {
            if (!isOpenMPPrivate(C->getClauseKind()) ||
                C->getClauseKind() == OMPC_reduction ||
                C->getClauseKind() == OMPC_linear ||
                C->getClauseKind() == OMPC_private)
              continue;
            ArrayRef<const Expr *> Vars;
            if (const auto *PC = dyn_cast<OMPFirstprivateClause>(C))
              Vars = PC->getVarRefs();
            else if (const auto *PC = dyn_cast<OMPLastprivateClause>(C))
              Vars = PC->getVarRefs();
            else
              llvm_unreachable("Unexpected clause.");
            for (const auto *E : Vars) {
              const Decl *D =
                  cast<DeclRefExpr>(E)->getDecl()->getCanonicalDecl();
              if (D == VD->getCanonicalDecl()) {
                IsForCombinedParallelRegion = true;
                break;
              }
            }
            if (IsForCombinedParallelRegion)
              break;
          }
        }
        markAsEscaped(VD);
        if (isa<OMPCapturedExprDecl>(VD))
          VisitValueDecl(VD);
        IsForCombinedParallelRegion = SavedIsForCombinedParallelRegion;
      }
    }
  }

  void buildRecordForGlobalizedVars(bool IsInTTDRegion) {
    assert(!GlobalizedRD &&
           "Record for globalized variables is built already.");
    ArrayRef<const ValueDecl *> EscapedDeclsForParallel, EscapedDeclsForTeams;
    unsigned WarpSize = CGF.getTarget().getGridValue().GV_Warp_Size;
    if (IsInTTDRegion)
      EscapedDeclsForTeams = EscapedDecls.getArrayRef();
    else
      EscapedDeclsForParallel = EscapedDecls.getArrayRef();
    GlobalizedRD = ::buildRecordForGlobalizedVars(
        CGF.getContext(), EscapedDeclsForParallel, EscapedDeclsForTeams,
        MappedDeclsFields, WarpSize);
  }

public:
  CheckVarsEscapingDeclContext(CodeGenFunction &CGF,
                               ArrayRef<const ValueDecl *> TeamsReductions)
      : CGF(CGF), EscapedDecls(TeamsReductions.begin(), TeamsReductions.end()) {
  }
  virtual ~CheckVarsEscapingDeclContext() = default;
  void VisitDeclStmt(const DeclStmt *S) {
    if (!S)
      return;
    for (const Decl *D : S->decls())
      if (const auto *VD = dyn_cast_or_null<ValueDecl>(D))
        VisitValueDecl(VD);
  }
  void VisitOMPExecutableDirective(const OMPExecutableDirective *D) {
    if (!D)
      return;
    if (!D->hasAssociatedStmt())
      return;
    if (const auto *S =
            dyn_cast_or_null<CapturedStmt>(D->getAssociatedStmt())) {
      // Do not analyze directives that do not actually require capturing,
      // like `omp for` or `omp simd` directives.
      llvm::SmallVector<OpenMPDirectiveKind, 4> CaptureRegions;
      getOpenMPCaptureRegions(CaptureRegions, D->getDirectiveKind());
      if (CaptureRegions.size() == 1 && CaptureRegions.back() == OMPD_unknown) {
        VisitStmt(S->getCapturedStmt());
        return;
      }
      VisitOpenMPCapturedStmt(
          S, D->clauses(),
          CaptureRegions.back() == OMPD_parallel &&
              isOpenMPDistributeDirective(D->getDirectiveKind()));
    }
  }
  void VisitCapturedStmt(const CapturedStmt *S) {
    if (!S)
      return;
    for (const CapturedStmt::Capture &C : S->captures()) {
      if (C.capturesVariable() && !C.capturesVariableByCopy()) {
        const ValueDecl *VD = C.getCapturedVar();
        markAsEscaped(VD);
        if (isa<OMPCapturedExprDecl>(VD))
          VisitValueDecl(VD);
      }
    }
  }
  void VisitLambdaExpr(const LambdaExpr *E) {
    if (!E)
      return;
    for (const LambdaCapture &C : E->captures()) {
      if (C.capturesVariable()) {
        if (C.getCaptureKind() == LCK_ByRef) {
          const ValueDecl *VD = C.getCapturedVar();
          markAsEscaped(VD);
          if (E->isInitCapture(&C) || isa<OMPCapturedExprDecl>(VD))
            VisitValueDecl(VD);
        }
      }
    }
  }
  void VisitBlockExpr(const BlockExpr *E) {
    if (!E)
      return;
    for (const BlockDecl::Capture &C : E->getBlockDecl()->captures()) {
      if (C.isByRef()) {
        const VarDecl *VD = C.getVariable();
        markAsEscaped(VD);
        if (isa<OMPCapturedExprDecl>(VD) || VD->isInitCapture())
          VisitValueDecl(VD);
      }
    }
  }
  void VisitCallExpr(const CallExpr *E) {
    if (!E)
      return;
    for (const Expr *Arg : E->arguments()) {
      if (!Arg)
        continue;
      if (Arg->isLValue()) {
        const bool SavedAllEscaped = AllEscaped;
        AllEscaped = true;
        Visit(Arg);
        AllEscaped = SavedAllEscaped;
      } else {
        Visit(Arg);
      }
    }
    Visit(E->getCallee());
  }
  void VisitDeclRefExpr(const DeclRefExpr *E) {
    if (!E)
      return;
    const ValueDecl *VD = E->getDecl();
    if (AllEscaped)
      markAsEscaped(VD);
    if (isa<OMPCapturedExprDecl>(VD))
      VisitValueDecl(VD);
    else if (VD->isInitCapture())
      VisitValueDecl(VD);
  }
  void VisitUnaryOperator(const UnaryOperator *E) {
    if (!E)
      return;
    if (E->getOpcode() == UO_AddrOf) {
      const bool SavedAllEscaped = AllEscaped;
      AllEscaped = true;
      Visit(E->getSubExpr());
      AllEscaped = SavedAllEscaped;
    } else {
      Visit(E->getSubExpr());
    }
  }
  void VisitImplicitCastExpr(const ImplicitCastExpr *E) {
    if (!E)
      return;
    if (E->getCastKind() == CK_ArrayToPointerDecay) {
      const bool SavedAllEscaped = AllEscaped;
      AllEscaped = true;
      Visit(E->getSubExpr());
      AllEscaped = SavedAllEscaped;
    } else {
      Visit(E->getSubExpr());
    }
  }
  void VisitExpr(const Expr *E) {
    if (!E)
      return;
    bool SavedAllEscaped = AllEscaped;
    if (!E->isLValue())
      AllEscaped = false;
    for (const Stmt *Child : E->children())
      if (Child)
        Visit(Child);
    AllEscaped = SavedAllEscaped;
  }
  void VisitStmt(const Stmt *S) {
    if (!S)
      return;
    for (const Stmt *Child : S->children())
      if (Child)
        Visit(Child);
  }

  /// Returns the record that handles all the escaped local variables and used
  /// instead of their original storage.
  const RecordDecl *getGlobalizedRecord(bool IsInTTDRegion) {
    if (!GlobalizedRD)
      buildRecordForGlobalizedVars(IsInTTDRegion);
    return GlobalizedRD;
  }

  /// Returns the field in the globalized record for the escaped variable.
  const FieldDecl *getFieldForGlobalizedVar(const ValueDecl *VD) const {
    assert(GlobalizedRD &&
           "Record for globalized variables must be generated already.");
    return MappedDeclsFields.lookup(VD);
  }

  /// Returns the list of the escaped local variables/parameters.
  ArrayRef<const ValueDecl *> getEscapedDecls() const {
    return EscapedDecls.getArrayRef();
  }

  /// Checks if the escaped local variable is actually a parameter passed by
  /// value.
  const llvm::SmallPtrSetImpl<const Decl *> &getEscapedParameters() const {
    return EscapedParameters;
  }

  /// Returns the list of the escaped variables with the variably modified
  /// types.
  ArrayRef<const ValueDecl *> getEscapedVariableLengthDecls() const {
    return EscapedVariableLengthDecls.getArrayRef();
  }

  /// Returns the list of the delayed variables with the variably modified
  /// types.
  ArrayRef<const ValueDecl *> getDelayedVariableLengthDecls() const {
    return DelayedVariableLengthDecls.getArrayRef();
  }
};
} // anonymous namespace

/// Get the id of the warp in the block.
/// We assume that the warp size is 32, which is always the case
/// on the NVPTX device, to generate more efficient code.
static llvm::Value *getNVPTXWarpID(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  unsigned LaneIDBits =
      llvm::Log2_32(CGF.getTarget().getGridValue().GV_Warp_Size);
  auto &RT = static_cast<CGOpenMPRuntimeGPU &>(CGF.CGM.getOpenMPRuntime());
  return Bld.CreateAShr(RT.getGPUThreadID(CGF), LaneIDBits, "nvptx_warp_id");
}

/// Get the id of the current lane in the Warp.
/// We assume that the warp size is 32, which is always the case
/// on the NVPTX device, to generate more efficient code.
static llvm::Value *getNVPTXLaneID(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  unsigned LaneIDBits =
      llvm::Log2_32(CGF.getTarget().getGridValue().GV_Warp_Size);
  assert(LaneIDBits < 32 && "Invalid LaneIDBits size in NVPTX device.");
  unsigned LaneIDMask = ~0u >> (32u - LaneIDBits);
  auto &RT = static_cast<CGOpenMPRuntimeGPU &>(CGF.CGM.getOpenMPRuntime());
  return Bld.CreateAnd(RT.getGPUThreadID(CGF), Bld.getInt32(LaneIDMask),
                       "nvptx_lane_id");
}

CGOpenMPRuntimeGPU::ExecutionMode
CGOpenMPRuntimeGPU::getExecutionMode() const {
  return CurrentExecutionMode;
}

CGOpenMPRuntimeGPU::DataSharingMode
CGOpenMPRuntimeGPU::getDataSharingMode() const {
  return CurrentDataSharingMode;
}

/// Check for inner (nested) SPMD construct, if any
static bool hasNestedSPMDDirective(ASTContext &Ctx,
                                   const OMPExecutableDirective &D) {
  const auto *CS = D.getInnermostCapturedStmt();
  const auto *Body =
      CS->getCapturedStmt()->IgnoreContainers(/*IgnoreCaptured=*/true);
  const Stmt *ChildStmt = CGOpenMPRuntime::getSingleCompoundChild(Ctx, Body);

  if (const auto *NestedDir =
          dyn_cast_or_null<OMPExecutableDirective>(ChildStmt)) {
    OpenMPDirectiveKind DKind = NestedDir->getDirectiveKind();
    switch (D.getDirectiveKind()) {
    case OMPD_target:
      if (isOpenMPParallelDirective(DKind))
        return true;
      if (DKind == OMPD_teams) {
        Body = NestedDir->getInnermostCapturedStmt()->IgnoreContainers(
            /*IgnoreCaptured=*/true);
        if (!Body)
          return false;
        ChildStmt = CGOpenMPRuntime::getSingleCompoundChild(Ctx, Body);
        if (const auto *NND =
                dyn_cast_or_null<OMPExecutableDirective>(ChildStmt)) {
          DKind = NND->getDirectiveKind();
          if (isOpenMPParallelDirective(DKind))
            return true;
        }
      }
      return false;
    case OMPD_target_teams:
      return isOpenMPParallelDirective(DKind);
    case OMPD_target_simd:
    case OMPD_target_parallel:
    case OMPD_target_parallel_for:
    case OMPD_target_parallel_for_simd:
    case OMPD_target_teams_distribute:
    case OMPD_target_teams_distribute_simd:
    case OMPD_target_teams_distribute_parallel_for:
    case OMPD_target_teams_distribute_parallel_for_simd:
    case OMPD_parallel:
    case OMPD_for:
    case OMPD_parallel_for:
    case OMPD_parallel_master:
    case OMPD_parallel_sections:
    case OMPD_for_simd:
    case OMPD_parallel_for_simd:
    case OMPD_cancel:
    case OMPD_cancellation_point:
    case OMPD_ordered:
    case OMPD_threadprivate:
    case OMPD_allocate:
    case OMPD_task:
    case OMPD_simd:
    case OMPD_sections:
    case OMPD_section:
    case OMPD_single:
    case OMPD_master:
    case OMPD_critical:
    case OMPD_taskyield:
    case OMPD_barrier:
    case OMPD_taskwait:
    case OMPD_taskgroup:
    case OMPD_atomic:
    case OMPD_flush:
    case OMPD_depobj:
    case OMPD_scan:
    case OMPD_teams:
    case OMPD_target_data:
    case OMPD_target_exit_data:
    case OMPD_target_enter_data:
    case OMPD_distribute:
    case OMPD_distribute_simd:
    case OMPD_distribute_parallel_for:
    case OMPD_distribute_parallel_for_simd:
    case OMPD_teams_distribute:
    case OMPD_teams_distribute_simd:
    case OMPD_teams_distribute_parallel_for:
    case OMPD_teams_distribute_parallel_for_simd:
    case OMPD_target_update:
    case OMPD_declare_simd:
    case OMPD_declare_variant:
    case OMPD_begin_declare_variant:
    case OMPD_end_declare_variant:
    case OMPD_declare_target:
    case OMPD_end_declare_target:
    case OMPD_declare_reduction:
    case OMPD_declare_mapper:
    case OMPD_taskloop:
    case OMPD_taskloop_simd:
    case OMPD_master_taskloop:
    case OMPD_master_taskloop_simd:
    case OMPD_parallel_master_taskloop:
    case OMPD_parallel_master_taskloop_simd:
    case OMPD_requires:
    case OMPD_unknown:
    default:
      llvm_unreachable("Unexpected directive.");
    }
  }

  return false;
}

static bool supportsSPMDExecutionMode(ASTContext &Ctx,
                                      const OMPExecutableDirective &D) {
  OpenMPDirectiveKind DirectiveKind = D.getDirectiveKind();
  switch (DirectiveKind) {
  case OMPD_target:
  case OMPD_target_teams:
    return hasNestedSPMDDirective(Ctx, D);
  case OMPD_target_teams_loop:
  case OMPD_target_parallel_loop:
  case OMPD_target_parallel:
  case OMPD_target_parallel_for:
  case OMPD_target_parallel_for_simd:
  case OMPD_target_teams_distribute_parallel_for:
  case OMPD_target_teams_distribute_parallel_for_simd:
  case OMPD_target_simd:
  case OMPD_target_teams_distribute_simd:
    return true;
  case OMPD_target_teams_distribute:
    return false;
  case OMPD_parallel:
  case OMPD_for:
  case OMPD_parallel_for:
  case OMPD_parallel_master:
  case OMPD_parallel_sections:
  case OMPD_for_simd:
  case OMPD_parallel_for_simd:
  case OMPD_cancel:
  case OMPD_cancellation_point:
  case OMPD_ordered:
  case OMPD_threadprivate:
  case OMPD_allocate:
  case OMPD_task:
  case OMPD_simd:
  case OMPD_sections:
  case OMPD_section:
  case OMPD_single:
  case OMPD_master:
  case OMPD_critical:
  case OMPD_taskyield:
  case OMPD_barrier:
  case OMPD_taskwait:
  case OMPD_taskgroup:
  case OMPD_atomic:
  case OMPD_flush:
  case OMPD_depobj:
  case OMPD_scan:
  case OMPD_teams:
  case OMPD_target_data:
  case OMPD_target_exit_data:
  case OMPD_target_enter_data:
  case OMPD_distribute:
  case OMPD_distribute_simd:
  case OMPD_distribute_parallel_for:
  case OMPD_distribute_parallel_for_simd:
  case OMPD_teams_distribute:
  case OMPD_teams_distribute_simd:
  case OMPD_teams_distribute_parallel_for:
  case OMPD_teams_distribute_parallel_for_simd:
  case OMPD_target_update:
  case OMPD_declare_simd:
  case OMPD_declare_variant:
  case OMPD_begin_declare_variant:
  case OMPD_end_declare_variant:
  case OMPD_declare_target:
  case OMPD_end_declare_target:
  case OMPD_declare_reduction:
  case OMPD_declare_mapper:
  case OMPD_taskloop:
  case OMPD_taskloop_simd:
  case OMPD_master_taskloop:
  case OMPD_master_taskloop_simd:
  case OMPD_parallel_master_taskloop:
  case OMPD_parallel_master_taskloop_simd:
  case OMPD_requires:
  case OMPD_unknown:
  default:
    break;
  }
  llvm_unreachable(
      "Unknown programming model for OpenMP directive on NVPTX target.");
}

void CGOpenMPRuntimeGPU::emitNonSPMDKernel(const OMPExecutableDirective &D,
                                             StringRef ParentName,
                                             llvm::Function *&OutlinedFn,
                                             llvm::Constant *&OutlinedFnID,
                                             bool IsOffloadEntry,
                                             const RegionCodeGenTy &CodeGen) {
  ExecutionRuntimeModesRAII ModeRAII(CurrentExecutionMode, EM_NonSPMD);
  EntryFunctionState EST;
  WrapperFunctionsMap.clear();

  [[maybe_unused]] bool IsBareKernel = D.getSingleClause<OMPXBareClause>();
  assert(!IsBareKernel && "bare kernel should not be at generic mode");

  // Emit target region as a standalone region.
  class NVPTXPrePostActionTy : public PrePostActionTy {
    CGOpenMPRuntimeGPU::EntryFunctionState &EST;

  public:
    NVPTXPrePostActionTy(CGOpenMPRuntimeGPU::EntryFunctionState &EST)
        : EST(EST) {}
    void Enter(CodeGenFunction &CGF) override {
      auto &RT = static_cast<CGOpenMPRuntimeGPU &>(CGF.CGM.getOpenMPRuntime());
      RT.emitKernelInit(CGF, EST, /* IsSPMD */ false);
      // Skip target region initialization.
      RT.setLocThreadIdInsertPt(CGF, /*AtCurrentPoint=*/true);
    }
    void Exit(CodeGenFunction &CGF) override {
      auto &RT = static_cast<CGOpenMPRuntimeGPU &>(CGF.CGM.getOpenMPRuntime());
      RT.clearLocThreadIdInsertPt(CGF);
      RT.emitKernelDeinit(CGF, EST, /* IsSPMD */ false);
    }
  } Action(EST);
  CodeGen.setAction(Action);
  IsInTTDRegion = true;
  emitTargetOutlinedFunctionHelper(D, ParentName, OutlinedFn, OutlinedFnID,
                                   IsOffloadEntry, CodeGen);
  IsInTTDRegion = false;
}

void CGOpenMPRuntimeGPU::emitKernelInit(CodeGenFunction &CGF,
                                        EntryFunctionState &EST, bool IsSPMD) {
  CGBuilderTy &Bld = CGF.Builder;
  Bld.restoreIP(OMPBuilder.createTargetInit(Bld, IsSPMD));
  if (!IsSPMD)
    emitGenericVarsProlog(CGF, EST.Loc);
}

void CGOpenMPRuntimeGPU::emitKernelDeinit(CodeGenFunction &CGF,
                                          EntryFunctionState &EST,
                                          bool IsSPMD) {
  if (!IsSPMD)
    emitGenericVarsEpilog(CGF);

  CGBuilderTy &Bld = CGF.Builder;
  OMPBuilder.createTargetDeinit(Bld);
}

void CGOpenMPRuntimeGPU::emitSPMDKernel(const OMPExecutableDirective &D,
                                          StringRef ParentName,
                                          llvm::Function *&OutlinedFn,
                                          llvm::Constant *&OutlinedFnID,
                                          bool IsOffloadEntry,
                                          const RegionCodeGenTy &CodeGen) {
  ExecutionRuntimeModesRAII ModeRAII(CurrentExecutionMode, EM_SPMD);
  EntryFunctionState EST;

  bool IsBareKernel = D.getSingleClause<OMPXBareClause>();

  // Emit target region as a standalone region.
  class NVPTXPrePostActionTy : public PrePostActionTy {
    CGOpenMPRuntimeGPU &RT;
    CGOpenMPRuntimeGPU::EntryFunctionState &EST;
    bool IsBareKernel;
    DataSharingMode Mode;

  public:
    NVPTXPrePostActionTy(CGOpenMPRuntimeGPU &RT,
                         CGOpenMPRuntimeGPU::EntryFunctionState &EST,
                         bool IsBareKernel)
        : RT(RT), EST(EST), IsBareKernel(IsBareKernel),
          Mode(RT.CurrentDataSharingMode) {}
    void Enter(CodeGenFunction &CGF) override {
      if (IsBareKernel) {
        RT.CurrentDataSharingMode = DataSharingMode::DS_CUDA;
        return;
      }
      RT.emitKernelInit(CGF, EST, /* IsSPMD */ true);
      // Skip target region initialization.
      RT.setLocThreadIdInsertPt(CGF, /*AtCurrentPoint=*/true);
    }
    void Exit(CodeGenFunction &CGF) override {
      if (IsBareKernel) {
        RT.CurrentDataSharingMode = Mode;
        return;
      }
      RT.clearLocThreadIdInsertPt(CGF);
      RT.emitKernelDeinit(CGF, EST, /* IsSPMD */ true);
    }
  } Action(*this, EST, IsBareKernel);
  CodeGen.setAction(Action);
  IsInTTDRegion = true;
  emitTargetOutlinedFunctionHelper(D, ParentName, OutlinedFn, OutlinedFnID,
                                   IsOffloadEntry, CodeGen);
  IsInTTDRegion = false;
}

void CGOpenMPRuntimeGPU::emitTargetOutlinedFunction(
    const OMPExecutableDirective &D, StringRef ParentName,
    llvm::Function *&OutlinedFn, llvm::Constant *&OutlinedFnID,
    bool IsOffloadEntry, const RegionCodeGenTy &CodeGen) {
  if (!IsOffloadEntry) // Nothing to do.
    return;

  assert(!ParentName.empty() && "Invalid target region parent name!");

  bool Mode = supportsSPMDExecutionMode(CGM.getContext(), D);
  bool IsBareKernel = D.getSingleClause<OMPXBareClause>();
  if (Mode || IsBareKernel)
    emitSPMDKernel(D, ParentName, OutlinedFn, OutlinedFnID, IsOffloadEntry,
                   CodeGen);
  else
    emitNonSPMDKernel(D, ParentName, OutlinedFn, OutlinedFnID, IsOffloadEntry,
                      CodeGen);
}

CGOpenMPRuntimeGPU::CGOpenMPRuntimeGPU(CodeGenModule &CGM)
    : CGOpenMPRuntime(CGM) {
  llvm::OpenMPIRBuilderConfig Config(
      CGM.getLangOpts().OpenMPIsTargetDevice, isGPU(),
      CGM.getLangOpts().OpenMPOffloadMandatory,
      /*HasRequiresReverseOffload*/ false, /*HasRequiresUnifiedAddress*/ false,
      hasRequiresUnifiedSharedMemory(), /*HasRequiresDynamicAllocators*/ false);
  OMPBuilder.setConfig(Config);

  if (!CGM.getLangOpts().OpenMPIsTargetDevice)
    llvm_unreachable("OpenMP can only handle device code.");

  if (CGM.getLangOpts().OpenMPCUDAMode)
    CurrentDataSharingMode = CGOpenMPRuntimeGPU::DS_CUDA;

  llvm::OpenMPIRBuilder &OMPBuilder = getOMPBuilder();
  if (CGM.getLangOpts().NoGPULib || CGM.getLangOpts().OMPHostIRFile.empty())
    return;

  OMPBuilder.createGlobalFlag(CGM.getLangOpts().OpenMPTargetDebug,
                              "__omp_rtl_debug_kind");
  OMPBuilder.createGlobalFlag(CGM.getLangOpts().OpenMPTeamSubscription,
                              "__omp_rtl_assume_teams_oversubscription");
  OMPBuilder.createGlobalFlag(CGM.getLangOpts().OpenMPThreadSubscription,
                              "__omp_rtl_assume_threads_oversubscription");
  OMPBuilder.createGlobalFlag(CGM.getLangOpts().OpenMPNoThreadState,
                              "__omp_rtl_assume_no_thread_state");
  OMPBuilder.createGlobalFlag(CGM.getLangOpts().OpenMPNoNestedParallelism,
                              "__omp_rtl_assume_no_nested_parallelism");
}

void CGOpenMPRuntimeGPU::emitProcBindClause(CodeGenFunction &CGF,
                                              ProcBindKind ProcBind,
                                              SourceLocation Loc) {
  // Do nothing in case of SPMD mode and L0 parallel.
  if (getExecutionMode() == CGOpenMPRuntimeGPU::EM_SPMD)
    return;

  CGOpenMPRuntime::emitProcBindClause(CGF, ProcBind, Loc);
}

void CGOpenMPRuntimeGPU::emitNumThreadsClause(CodeGenFunction &CGF,
                                                llvm::Value *NumThreads,
                                                SourceLocation Loc) {
  // Nothing to do.
}

void CGOpenMPRuntimeGPU::emitNumTeamsClause(CodeGenFunction &CGF,
                                              const Expr *NumTeams,
                                              const Expr *ThreadLimit,
                                              SourceLocation Loc) {}

llvm::Function *CGOpenMPRuntimeGPU::emitParallelOutlinedFunction(
    CodeGenFunction &CGF, const OMPExecutableDirective &D,
    const VarDecl *ThreadIDVar, OpenMPDirectiveKind InnermostKind,
    const RegionCodeGenTy &CodeGen) {
  // Emit target region as a standalone region.
  bool PrevIsInTTDRegion = IsInTTDRegion;
  IsInTTDRegion = false;
  auto *OutlinedFun =
      cast<llvm::Function>(CGOpenMPRuntime::emitParallelOutlinedFunction(
          CGF, D, ThreadIDVar, InnermostKind, CodeGen));
  IsInTTDRegion = PrevIsInTTDRegion;
  if (getExecutionMode() != CGOpenMPRuntimeGPU::EM_SPMD) {
    llvm::Function *WrapperFun =
        createParallelDataSharingWrapper(OutlinedFun, D);
    WrapperFunctionsMap[OutlinedFun] = WrapperFun;
  }

  return OutlinedFun;
}

/// Get list of lastprivate variables from the teams distribute ... or
/// teams {distribute ...} directives.
static void
getDistributeLastprivateVars(ASTContext &Ctx, const OMPExecutableDirective &D,
                             llvm::SmallVectorImpl<const ValueDecl *> &Vars) {
  assert(isOpenMPTeamsDirective(D.getDirectiveKind()) &&
         "expected teams directive.");
  const OMPExecutableDirective *Dir = &D;
  if (!isOpenMPDistributeDirective(D.getDirectiveKind())) {
    if (const Stmt *S = CGOpenMPRuntime::getSingleCompoundChild(
            Ctx,
            D.getInnermostCapturedStmt()->getCapturedStmt()->IgnoreContainers(
                /*IgnoreCaptured=*/true))) {
      Dir = dyn_cast_or_null<OMPExecutableDirective>(S);
      if (Dir && !isOpenMPDistributeDirective(Dir->getDirectiveKind()))
        Dir = nullptr;
    }
  }
  if (!Dir)
    return;
  for (const auto *C : Dir->getClausesOfKind<OMPLastprivateClause>()) {
    for (const Expr *E : C->getVarRefs())
      Vars.push_back(getPrivateItem(E));
  }
}

/// Get list of reduction variables from the teams ... directives.
static void
getTeamsReductionVars(ASTContext &Ctx, const OMPExecutableDirective &D,
                      llvm::SmallVectorImpl<const ValueDecl *> &Vars) {
  assert(isOpenMPTeamsDirective(D.getDirectiveKind()) &&
         "expected teams directive.");
  for (const auto *C : D.getClausesOfKind<OMPReductionClause>()) {
    for (const Expr *E : C->privates())
      Vars.push_back(getPrivateItem(E));
  }
}

llvm::Function *CGOpenMPRuntimeGPU::emitTeamsOutlinedFunction(
    CodeGenFunction &CGF, const OMPExecutableDirective &D,
    const VarDecl *ThreadIDVar, OpenMPDirectiveKind InnermostKind,
    const RegionCodeGenTy &CodeGen) {
  SourceLocation Loc = D.getBeginLoc();

  const RecordDecl *GlobalizedRD = nullptr;
  llvm::SmallVector<const ValueDecl *, 4> LastPrivatesReductions;
  llvm::SmallDenseMap<const ValueDecl *, const FieldDecl *> MappedDeclsFields;
  unsigned WarpSize = CGM.getTarget().getGridValue().GV_Warp_Size;
  // Globalize team reductions variable unconditionally in all modes.
  if (getExecutionMode() != CGOpenMPRuntimeGPU::EM_SPMD)
    getTeamsReductionVars(CGM.getContext(), D, LastPrivatesReductions);
  if (getExecutionMode() == CGOpenMPRuntimeGPU::EM_SPMD) {
    getDistributeLastprivateVars(CGM.getContext(), D, LastPrivatesReductions);
    if (!LastPrivatesReductions.empty()) {
      GlobalizedRD = ::buildRecordForGlobalizedVars(
          CGM.getContext(), std::nullopt, LastPrivatesReductions,
          MappedDeclsFields, WarpSize);
    }
  } else if (!LastPrivatesReductions.empty()) {
    assert(!TeamAndReductions.first &&
           "Previous team declaration is not expected.");
    TeamAndReductions.first = D.getCapturedStmt(OMPD_teams)->getCapturedDecl();
    std::swap(TeamAndReductions.second, LastPrivatesReductions);
  }

  // Emit target region as a standalone region.
  class NVPTXPrePostActionTy : public PrePostActionTy {
    SourceLocation &Loc;
    const RecordDecl *GlobalizedRD;
    llvm::SmallDenseMap<const ValueDecl *, const FieldDecl *>
        &MappedDeclsFields;

  public:
    NVPTXPrePostActionTy(
        SourceLocation &Loc, const RecordDecl *GlobalizedRD,
        llvm::SmallDenseMap<const ValueDecl *, const FieldDecl *>
            &MappedDeclsFields)
        : Loc(Loc), GlobalizedRD(GlobalizedRD),
          MappedDeclsFields(MappedDeclsFields) {}
    void Enter(CodeGenFunction &CGF) override {
      auto &Rt =
          static_cast<CGOpenMPRuntimeGPU &>(CGF.CGM.getOpenMPRuntime());
      if (GlobalizedRD) {
        auto I = Rt.FunctionGlobalizedDecls.try_emplace(CGF.CurFn).first;
        I->getSecond().MappedParams =
            std::make_unique<CodeGenFunction::OMPMapVars>();
        DeclToAddrMapTy &Data = I->getSecond().LocalVarData;
        for (const auto &Pair : MappedDeclsFields) {
          assert(Pair.getFirst()->isCanonicalDecl() &&
                 "Expected canonical declaration");
          Data.insert(std::make_pair(Pair.getFirst(), MappedVarData()));
        }
      }
      Rt.emitGenericVarsProlog(CGF, Loc);
    }
    void Exit(CodeGenFunction &CGF) override {
      static_cast<CGOpenMPRuntimeGPU &>(CGF.CGM.getOpenMPRuntime())
          .emitGenericVarsEpilog(CGF);
    }
  } Action(Loc, GlobalizedRD, MappedDeclsFields);
  CodeGen.setAction(Action);
  llvm::Function *OutlinedFun = CGOpenMPRuntime::emitTeamsOutlinedFunction(
      CGF, D, ThreadIDVar, InnermostKind, CodeGen);

  return OutlinedFun;
}

void CGOpenMPRuntimeGPU::emitGenericVarsProlog(CodeGenFunction &CGF,
                                                 SourceLocation Loc,
                                                 bool WithSPMDCheck) {
  if (getDataSharingMode() != CGOpenMPRuntimeGPU::DS_Generic &&
      getExecutionMode() != CGOpenMPRuntimeGPU::EM_SPMD)
    return;

  CGBuilderTy &Bld = CGF.Builder;

  const auto I = FunctionGlobalizedDecls.find(CGF.CurFn);
  if (I == FunctionGlobalizedDecls.end())
    return;

  for (auto &Rec : I->getSecond().LocalVarData) {
    const auto *VD = cast<VarDecl>(Rec.first);
    bool EscapedParam = I->getSecond().EscapedParameters.count(Rec.first);
    QualType VarTy = VD->getType();

    // Get the local allocation of a firstprivate variable before sharing
    llvm::Value *ParValue;
    if (EscapedParam) {
      LValue ParLVal =
          CGF.MakeAddrLValue(CGF.GetAddrOfLocalVar(VD), VD->getType());
      ParValue = CGF.EmitLoadOfScalar(ParLVal, Loc);
    }

    // Allocate space for the variable to be globalized
    llvm::Value *AllocArgs[] = {CGF.getTypeSize(VD->getType())};
    llvm::CallBase *VoidPtr =
        CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                                CGM.getModule(), OMPRTL___kmpc_alloc_shared),
                            AllocArgs, VD->getName());
    // FIXME: We should use the variables actual alignment as an argument.
    VoidPtr->addRetAttr(llvm::Attribute::get(
        CGM.getLLVMContext(), llvm::Attribute::Alignment,
        CGM.getContext().getTargetInfo().getNewAlign() / 8));

    // Cast the void pointer and get the address of the globalized variable.
    llvm::PointerType *VarPtrTy = CGF.ConvertTypeForMem(VarTy)->getPointerTo();
    llvm::Value *CastedVoidPtr = Bld.CreatePointerBitCastOrAddrSpaceCast(
        VoidPtr, VarPtrTy, VD->getName() + "_on_stack");
    LValue VarAddr = CGF.MakeNaturalAlignAddrLValue(CastedVoidPtr, VarTy);
    Rec.second.PrivateAddr = VarAddr.getAddress(CGF);
    Rec.second.GlobalizedVal = VoidPtr;

    // Assign the local allocation to the newly globalized location.
    if (EscapedParam) {
      CGF.EmitStoreOfScalar(ParValue, VarAddr);
      I->getSecond().MappedParams->setVarAddr(CGF, VD, VarAddr.getAddress(CGF));
    }
    if (auto *DI = CGF.getDebugInfo())
      VoidPtr->setDebugLoc(DI->SourceLocToDebugLoc(VD->getLocation()));
  }

  for (const auto *ValueD : I->getSecond().EscapedVariableLengthDecls) {
    const auto *VD = cast<VarDecl>(ValueD);
    std::pair<llvm::Value *, llvm::Value *> AddrSizePair =
        getKmpcAllocShared(CGF, VD);
    I->getSecond().EscapedVariableLengthDeclsAddrs.emplace_back(AddrSizePair);
    LValue Base = CGF.MakeAddrLValue(AddrSizePair.first, VD->getType(),
                                     CGM.getContext().getDeclAlign(VD),
                                     AlignmentSource::Decl);
    I->getSecond().MappedParams->setVarAddr(CGF, VD, Base.getAddress(CGF));
  }
  I->getSecond().MappedParams->apply(CGF);
}

bool CGOpenMPRuntimeGPU::isDelayedVariableLengthDecl(CodeGenFunction &CGF,
                                                     const VarDecl *VD) const {
  const auto I = FunctionGlobalizedDecls.find(CGF.CurFn);
  if (I == FunctionGlobalizedDecls.end())
    return false;

  // Check variable declaration is delayed:
  return llvm::is_contained(I->getSecond().DelayedVariableLengthDecls, VD);
}

std::pair<llvm::Value *, llvm::Value *>
CGOpenMPRuntimeGPU::getKmpcAllocShared(CodeGenFunction &CGF,
                                       const VarDecl *VD) {
  CGBuilderTy &Bld = CGF.Builder;

  // Compute size and alignment.
  llvm::Value *Size = CGF.getTypeSize(VD->getType());
  CharUnits Align = CGM.getContext().getDeclAlign(VD);
  Size = Bld.CreateNUWAdd(
      Size, llvm::ConstantInt::get(CGF.SizeTy, Align.getQuantity() - 1));
  llvm::Value *AlignVal =
      llvm::ConstantInt::get(CGF.SizeTy, Align.getQuantity());
  Size = Bld.CreateUDiv(Size, AlignVal);
  Size = Bld.CreateNUWMul(Size, AlignVal);

  // Allocate space for this VLA object to be globalized.
  llvm::Value *AllocArgs[] = {Size};
  llvm::CallBase *VoidPtr =
      CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                              CGM.getModule(), OMPRTL___kmpc_alloc_shared),
                          AllocArgs, VD->getName());
  VoidPtr->addRetAttr(llvm::Attribute::get(
      CGM.getLLVMContext(), llvm::Attribute::Alignment, Align.getQuantity()));

  return std::make_pair(VoidPtr, Size);
}

void CGOpenMPRuntimeGPU::getKmpcFreeShared(
    CodeGenFunction &CGF,
    const std::pair<llvm::Value *, llvm::Value *> &AddrSizePair) {
  // Deallocate the memory for each globalized VLA object
  CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                          CGM.getModule(), OMPRTL___kmpc_free_shared),
                      {AddrSizePair.first, AddrSizePair.second});
}

void CGOpenMPRuntimeGPU::emitGenericVarsEpilog(CodeGenFunction &CGF,
                                                 bool WithSPMDCheck) {
  if (getDataSharingMode() != CGOpenMPRuntimeGPU::DS_Generic &&
      getExecutionMode() != CGOpenMPRuntimeGPU::EM_SPMD)
    return;

  const auto I = FunctionGlobalizedDecls.find(CGF.CurFn);
  if (I != FunctionGlobalizedDecls.end()) {
    // Deallocate the memory for each globalized VLA object that was
    // globalized in the prolog (i.e. emitGenericVarsProlog).
    for (const auto &AddrSizePair :
         llvm::reverse(I->getSecond().EscapedVariableLengthDeclsAddrs)) {
      CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                              CGM.getModule(), OMPRTL___kmpc_free_shared),
                          {AddrSizePair.first, AddrSizePair.second});
    }
    // Deallocate the memory for each globalized value
    for (auto &Rec : llvm::reverse(I->getSecond().LocalVarData)) {
      const auto *VD = cast<VarDecl>(Rec.first);
      I->getSecond().MappedParams->restore(CGF);

      llvm::Value *FreeArgs[] = {Rec.second.GlobalizedVal,
                                 CGF.getTypeSize(VD->getType())};
      CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                              CGM.getModule(), OMPRTL___kmpc_free_shared),
                          FreeArgs);
    }
  }
}

void CGOpenMPRuntimeGPU::emitTeamsCall(CodeGenFunction &CGF,
                                         const OMPExecutableDirective &D,
                                         SourceLocation Loc,
                                         llvm::Function *OutlinedFn,
                                         ArrayRef<llvm::Value *> CapturedVars) {
  if (!CGF.HaveInsertPoint())
    return;

  bool IsBareKernel = D.getSingleClause<OMPXBareClause>();

  Address ZeroAddr = CGF.CreateDefaultAlignTempAlloca(CGF.Int32Ty,
                                                      /*Name=*/".zero.addr");
  CGF.Builder.CreateStore(CGF.Builder.getInt32(/*C*/ 0), ZeroAddr);
  llvm::SmallVector<llvm::Value *, 16> OutlinedFnArgs;
  // We don't emit any thread id function call in bare kernel, but because the
  // outlined function has a pointer argument, we emit a nullptr here.
  if (IsBareKernel)
    OutlinedFnArgs.push_back(llvm::ConstantPointerNull::get(CGM.VoidPtrTy));
  else
    OutlinedFnArgs.push_back(emitThreadIDAddress(CGF, Loc).getPointer());
  OutlinedFnArgs.push_back(ZeroAddr.getPointer());
  OutlinedFnArgs.append(CapturedVars.begin(), CapturedVars.end());
  emitOutlinedFunctionCall(CGF, Loc, OutlinedFn, OutlinedFnArgs);
}

void CGOpenMPRuntimeGPU::emitParallelCall(CodeGenFunction &CGF,
                                          SourceLocation Loc,
                                          llvm::Function *OutlinedFn,
                                          ArrayRef<llvm::Value *> CapturedVars,
                                          const Expr *IfCond,
                                          llvm::Value *NumThreads) {
  if (!CGF.HaveInsertPoint())
    return;

  auto &&ParallelGen = [this, Loc, OutlinedFn, CapturedVars, IfCond,
                        NumThreads](CodeGenFunction &CGF,
                                    PrePostActionTy &Action) {
    CGBuilderTy &Bld = CGF.Builder;
    llvm::Value *NumThreadsVal = NumThreads;
    llvm::Function *WFn = WrapperFunctionsMap[OutlinedFn];
    llvm::Value *ID = llvm::ConstantPointerNull::get(CGM.Int8PtrTy);
    if (WFn)
      ID = Bld.CreateBitOrPointerCast(WFn, CGM.Int8PtrTy);
    llvm::Value *FnPtr = Bld.CreateBitOrPointerCast(OutlinedFn, CGM.Int8PtrTy);

    // Create a private scope that will globalize the arguments
    // passed from the outside of the target region.
    // TODO: Is that needed?
    CodeGenFunction::OMPPrivateScope PrivateArgScope(CGF);

    Address CapturedVarsAddrs = CGF.CreateDefaultAlignTempAlloca(
        llvm::ArrayType::get(CGM.VoidPtrTy, CapturedVars.size()),
        "captured_vars_addrs");
    // There's something to share.
    if (!CapturedVars.empty()) {
      // Prepare for parallel region. Indicate the outlined function.
      ASTContext &Ctx = CGF.getContext();
      unsigned Idx = 0;
      for (llvm::Value *V : CapturedVars) {
        Address Dst = Bld.CreateConstArrayGEP(CapturedVarsAddrs, Idx);
        llvm::Value *PtrV;
        if (V->getType()->isIntegerTy())
          PtrV = Bld.CreateIntToPtr(V, CGF.VoidPtrTy);
        else
          PtrV = Bld.CreatePointerBitCastOrAddrSpaceCast(V, CGF.VoidPtrTy);
        CGF.EmitStoreOfScalar(PtrV, Dst, /*Volatile=*/false,
                              Ctx.getPointerType(Ctx.VoidPtrTy));
        ++Idx;
      }
    }

    llvm::Value *IfCondVal = nullptr;
    if (IfCond)
      IfCondVal = Bld.CreateIntCast(CGF.EvaluateExprAsBool(IfCond), CGF.Int32Ty,
                                    /* isSigned */ false);
    else
      IfCondVal = llvm::ConstantInt::get(CGF.Int32Ty, 1);

    if (!NumThreadsVal)
      NumThreadsVal = llvm::ConstantInt::get(CGF.Int32Ty, -1);
    else
      NumThreadsVal = Bld.CreateZExtOrTrunc(NumThreadsVal, CGF.Int32Ty),

      assert(IfCondVal && "Expected a value");
    llvm::Value *RTLoc = emitUpdateLocation(CGF, Loc);
    llvm::Value *Args[] = {
        RTLoc,
        getThreadID(CGF, Loc),
        IfCondVal,
        NumThreadsVal,
        llvm::ConstantInt::get(CGF.Int32Ty, -1),
        FnPtr,
        ID,
        Bld.CreateBitOrPointerCast(CapturedVarsAddrs.getPointer(),
                                   CGF.VoidPtrPtrTy),
        llvm::ConstantInt::get(CGM.SizeTy, CapturedVars.size())};
    CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                            CGM.getModule(), OMPRTL___kmpc_parallel_51),
                        Args);
  };

  RegionCodeGenTy RCG(ParallelGen);
  RCG(CGF);
}

void CGOpenMPRuntimeGPU::syncCTAThreads(CodeGenFunction &CGF) {
  // Always emit simple barriers!
  if (!CGF.HaveInsertPoint())
    return;
  // Build call __kmpc_barrier_simple_spmd(nullptr, 0);
  // This function does not use parameters, so we can emit just default values.
  llvm::Value *Args[] = {
      llvm::ConstantPointerNull::get(
          cast<llvm::PointerType>(getIdentTyPointerTy())),
      llvm::ConstantInt::get(CGF.Int32Ty, /*V=*/0, /*isSigned=*/true)};
  CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                          CGM.getModule(), OMPRTL___kmpc_barrier_simple_spmd),
                      Args);
}

void CGOpenMPRuntimeGPU::emitBarrierCall(CodeGenFunction &CGF,
                                           SourceLocation Loc,
                                           OpenMPDirectiveKind Kind, bool,
                                           bool) {
  // Always emit simple barriers!
  if (!CGF.HaveInsertPoint())
    return;
  // Build call __kmpc_cancel_barrier(loc, thread_id);
  unsigned Flags = getDefaultFlagsForBarriers(Kind);
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc, Flags),
                         getThreadID(CGF, Loc)};

  CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                          CGM.getModule(), OMPRTL___kmpc_barrier),
                      Args);
}

void CGOpenMPRuntimeGPU::emitCriticalRegion(
    CodeGenFunction &CGF, StringRef CriticalName,
    const RegionCodeGenTy &CriticalOpGen, SourceLocation Loc,
    const Expr *Hint) {
  llvm::BasicBlock *LoopBB = CGF.createBasicBlock("omp.critical.loop");
  llvm::BasicBlock *TestBB = CGF.createBasicBlock("omp.critical.test");
  llvm::BasicBlock *SyncBB = CGF.createBasicBlock("omp.critical.sync");
  llvm::BasicBlock *BodyBB = CGF.createBasicBlock("omp.critical.body");
  llvm::BasicBlock *ExitBB = CGF.createBasicBlock("omp.critical.exit");

  auto &RT = static_cast<CGOpenMPRuntimeGPU &>(CGF.CGM.getOpenMPRuntime());

  // Get the mask of active threads in the warp.
  llvm::Value *Mask = CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
      CGM.getModule(), OMPRTL___kmpc_warp_active_thread_mask));
  // Fetch team-local id of the thread.
  llvm::Value *ThreadID = RT.getGPUThreadID(CGF);

  // Get the width of the team.
  llvm::Value *TeamWidth = RT.getGPUNumThreads(CGF);

  // Initialize the counter variable for the loop.
  QualType Int32Ty =
      CGF.getContext().getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/0);
  Address Counter = CGF.CreateMemTemp(Int32Ty, "critical_counter");
  LValue CounterLVal = CGF.MakeAddrLValue(Counter, Int32Ty);
  CGF.EmitStoreOfScalar(llvm::Constant::getNullValue(CGM.Int32Ty), CounterLVal,
                        /*isInit=*/true);

  // Block checks if loop counter exceeds upper bound.
  CGF.EmitBlock(LoopBB);
  llvm::Value *CounterVal = CGF.EmitLoadOfScalar(CounterLVal, Loc);
  llvm::Value *CmpLoopBound = CGF.Builder.CreateICmpSLT(CounterVal, TeamWidth);
  CGF.Builder.CreateCondBr(CmpLoopBound, TestBB, ExitBB);

  // Block tests which single thread should execute region, and which threads
  // should go straight to synchronisation point.
  CGF.EmitBlock(TestBB);
  CounterVal = CGF.EmitLoadOfScalar(CounterLVal, Loc);
  llvm::Value *CmpThreadToCounter =
      CGF.Builder.CreateICmpEQ(ThreadID, CounterVal);
  CGF.Builder.CreateCondBr(CmpThreadToCounter, BodyBB, SyncBB);

  // Block emits the body of the critical region.
  CGF.EmitBlock(BodyBB);

  // Output the critical statement.
  CGOpenMPRuntime::emitCriticalRegion(CGF, CriticalName, CriticalOpGen, Loc,
                                      Hint);

  // After the body surrounded by the critical region, the single executing
  // thread will jump to the synchronisation point.
  // Block waits for all threads in current team to finish then increments the
  // counter variable and returns to the loop.
  CGF.EmitBlock(SyncBB);
  // Reconverge active threads in the warp.
  (void)CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                                CGM.getModule(), OMPRTL___kmpc_syncwarp),
                            Mask);

  llvm::Value *IncCounterVal =
      CGF.Builder.CreateNSWAdd(CounterVal, CGF.Builder.getInt32(1));
  CGF.EmitStoreOfScalar(IncCounterVal, CounterLVal);
  CGF.EmitBranch(LoopBB);

  // Block that is reached when  all threads in the team complete the region.
  CGF.EmitBlock(ExitBB, /*IsFinished=*/true);
}

/// Cast value to the specified type.
static llvm::Value *castValueToType(CodeGenFunction &CGF, llvm::Value *Val,
                                    QualType ValTy, QualType CastTy,
                                    SourceLocation Loc) {
  assert(!CGF.getContext().getTypeSizeInChars(CastTy).isZero() &&
         "Cast type must sized.");
  assert(!CGF.getContext().getTypeSizeInChars(ValTy).isZero() &&
         "Val type must sized.");
  llvm::Type *LLVMCastTy = CGF.ConvertTypeForMem(CastTy);
  if (ValTy == CastTy)
    return Val;
  if (CGF.getContext().getTypeSizeInChars(ValTy) ==
      CGF.getContext().getTypeSizeInChars(CastTy))
    return CGF.Builder.CreateBitCast(Val, LLVMCastTy);
  if (CastTy->isIntegerType() && ValTy->isIntegerType())
    return CGF.Builder.CreateIntCast(Val, LLVMCastTy,
                                     CastTy->hasSignedIntegerRepresentation());
  Address CastItem = CGF.CreateMemTemp(CastTy);
  Address ValCastItem = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      CastItem, Val->getType()->getPointerTo(CastItem.getAddressSpace()),
      Val->getType());
  CGF.EmitStoreOfScalar(Val, ValCastItem, /*Volatile=*/false, ValTy,
                        LValueBaseInfo(AlignmentSource::Type),
                        TBAAAccessInfo());
  return CGF.EmitLoadOfScalar(CastItem, /*Volatile=*/false, CastTy, Loc,
                              LValueBaseInfo(AlignmentSource::Type),
                              TBAAAccessInfo());
}

/// This function creates calls to one of two shuffle functions to copy
/// variables between lanes in a warp.
static llvm::Value *createRuntimeShuffleFunction(CodeGenFunction &CGF,
                                                 llvm::Value *Elem,
                                                 QualType ElemType,
                                                 llvm::Value *Offset,
                                                 SourceLocation Loc) {
  CodeGenModule &CGM = CGF.CGM;
  CGBuilderTy &Bld = CGF.Builder;
  CGOpenMPRuntimeGPU &RT =
      *(static_cast<CGOpenMPRuntimeGPU *>(&CGM.getOpenMPRuntime()));
  llvm::OpenMPIRBuilder &OMPBuilder = RT.getOMPBuilder();

  CharUnits Size = CGF.getContext().getTypeSizeInChars(ElemType);
  assert(Size.getQuantity() <= 8 &&
         "Unsupported bitwidth in shuffle instruction.");

  RuntimeFunction ShuffleFn = Size.getQuantity() <= 4
                                  ? OMPRTL___kmpc_shuffle_int32
                                  : OMPRTL___kmpc_shuffle_int64;

  // Cast all types to 32- or 64-bit values before calling shuffle routines.
  QualType CastTy = CGF.getContext().getIntTypeForBitwidth(
      Size.getQuantity() <= 4 ? 32 : 64, /*Signed=*/1);
  llvm::Value *ElemCast = castValueToType(CGF, Elem, ElemType, CastTy, Loc);
  llvm::Value *WarpSize =
      Bld.CreateIntCast(RT.getGPUWarpSize(CGF), CGM.Int16Ty, /*isSigned=*/true);

  llvm::Value *ShuffledVal = CGF.EmitRuntimeCall(
      OMPBuilder.getOrCreateRuntimeFunction(CGM.getModule(), ShuffleFn),
      {ElemCast, Offset, WarpSize});

  return castValueToType(CGF, ShuffledVal, CastTy, ElemType, Loc);
}

static void shuffleAndStore(CodeGenFunction &CGF, Address SrcAddr,
                            Address DestAddr, QualType ElemType,
                            llvm::Value *Offset, SourceLocation Loc) {
  CGBuilderTy &Bld = CGF.Builder;

  CharUnits Size = CGF.getContext().getTypeSizeInChars(ElemType);
  // Create the loop over the big sized data.
  // ptr = (void*)Elem;
  // ptrEnd = (void*) Elem + 1;
  // Step = 8;
  // while (ptr + Step < ptrEnd)
  //   shuffle((int64_t)*ptr);
  // Step = 4;
  // while (ptr + Step < ptrEnd)
  //   shuffle((int32_t)*ptr);
  // ...
  Address ElemPtr = DestAddr;
  Address Ptr = SrcAddr;
  Address PtrEnd = Bld.CreatePointerBitCastOrAddrSpaceCast(
      Bld.CreateConstGEP(SrcAddr, 1), CGF.VoidPtrTy, CGF.Int8Ty);
  for (int IntSize = 8; IntSize >= 1; IntSize /= 2) {
    if (Size < CharUnits::fromQuantity(IntSize))
      continue;
    QualType IntType = CGF.getContext().getIntTypeForBitwidth(
        CGF.getContext().toBits(CharUnits::fromQuantity(IntSize)),
        /*Signed=*/1);
    llvm::Type *IntTy = CGF.ConvertTypeForMem(IntType);
    Ptr = Bld.CreatePointerBitCastOrAddrSpaceCast(Ptr, IntTy->getPointerTo(),
                                                  IntTy);
    ElemPtr = Bld.CreatePointerBitCastOrAddrSpaceCast(
        ElemPtr, IntTy->getPointerTo(), IntTy);
    if (Size.getQuantity() / IntSize > 1) {
      llvm::BasicBlock *PreCondBB = CGF.createBasicBlock(".shuffle.pre_cond");
      llvm::BasicBlock *ThenBB = CGF.createBasicBlock(".shuffle.then");
      llvm::BasicBlock *ExitBB = CGF.createBasicBlock(".shuffle.exit");
      llvm::BasicBlock *CurrentBB = Bld.GetInsertBlock();
      CGF.EmitBlock(PreCondBB);
      llvm::PHINode *PhiSrc =
          Bld.CreatePHI(Ptr.getType(), /*NumReservedValues=*/2);
      PhiSrc->addIncoming(Ptr.getPointer(), CurrentBB);
      llvm::PHINode *PhiDest =
          Bld.CreatePHI(ElemPtr.getType(), /*NumReservedValues=*/2);
      PhiDest->addIncoming(ElemPtr.getPointer(), CurrentBB);
      Ptr = Address(PhiSrc, Ptr.getElementType(), Ptr.getAlignment());
      ElemPtr =
          Address(PhiDest, ElemPtr.getElementType(), ElemPtr.getAlignment());
      llvm::Value *PtrDiff = Bld.CreatePtrDiff(
          CGF.Int8Ty, PtrEnd.getPointer(),
          Bld.CreatePointerBitCastOrAddrSpaceCast(Ptr.getPointer(),
                                                  CGF.VoidPtrTy));
      Bld.CreateCondBr(Bld.CreateICmpSGT(PtrDiff, Bld.getInt64(IntSize - 1)),
                       ThenBB, ExitBB);
      CGF.EmitBlock(ThenBB);
      llvm::Value *Res = createRuntimeShuffleFunction(
          CGF,
          CGF.EmitLoadOfScalar(Ptr, /*Volatile=*/false, IntType, Loc,
                               LValueBaseInfo(AlignmentSource::Type),
                               TBAAAccessInfo()),
          IntType, Offset, Loc);
      CGF.EmitStoreOfScalar(Res, ElemPtr, /*Volatile=*/false, IntType,
                            LValueBaseInfo(AlignmentSource::Type),
                            TBAAAccessInfo());
      Address LocalPtr = Bld.CreateConstGEP(Ptr, 1);
      Address LocalElemPtr = Bld.CreateConstGEP(ElemPtr, 1);
      PhiSrc->addIncoming(LocalPtr.getPointer(), ThenBB);
      PhiDest->addIncoming(LocalElemPtr.getPointer(), ThenBB);
      CGF.EmitBranch(PreCondBB);
      CGF.EmitBlock(ExitBB);
    } else {
      llvm::Value *Res = createRuntimeShuffleFunction(
          CGF,
          CGF.EmitLoadOfScalar(Ptr, /*Volatile=*/false, IntType, Loc,
                               LValueBaseInfo(AlignmentSource::Type),
                               TBAAAccessInfo()),
          IntType, Offset, Loc);
      CGF.EmitStoreOfScalar(Res, ElemPtr, /*Volatile=*/false, IntType,
                            LValueBaseInfo(AlignmentSource::Type),
                            TBAAAccessInfo());
      Ptr = Bld.CreateConstGEP(Ptr, 1);
      ElemPtr = Bld.CreateConstGEP(ElemPtr, 1);
    }
    Size = Size % IntSize;
  }
}

namespace {
enum CopyAction : unsigned {
  // RemoteLaneToThread: Copy over a Reduce list from a remote lane in
  // the warp using shuffle instructions.
  RemoteLaneToThread,
  // ThreadCopy: Make a copy of a Reduce list on the thread's stack.
  ThreadCopy,
  // ThreadToScratchpad: Copy a team-reduced array to the scratchpad.
  ThreadToScratchpad,
  // ScratchpadToThread: Copy from a scratchpad array in global memory
  // containing team-reduced data to a thread's stack.
  ScratchpadToThread,
};
} // namespace

struct CopyOptionsTy {
  llvm::Value *RemoteLaneOffset;
  llvm::Value *ScratchpadIndex;
  llvm::Value *ScratchpadWidth;
};

/// Emit instructions to copy a Reduce list, which contains partially
/// aggregated values, in the specified direction.
static void emitReductionListCopy(
    CopyAction Action, CodeGenFunction &CGF, QualType ReductionArrayTy,
    ArrayRef<const Expr *> Privates, Address SrcBase, Address DestBase,
    CopyOptionsTy CopyOptions = {nullptr, nullptr, nullptr}) {

  CodeGenModule &CGM = CGF.CGM;
  ASTContext &C = CGM.getContext();
  CGBuilderTy &Bld = CGF.Builder;

  llvm::Value *RemoteLaneOffset = CopyOptions.RemoteLaneOffset;
  llvm::Value *ScratchpadIndex = CopyOptions.ScratchpadIndex;
  llvm::Value *ScratchpadWidth = CopyOptions.ScratchpadWidth;

  // Iterates, element-by-element, through the source Reduce list and
  // make a copy.
  unsigned Idx = 0;
  unsigned Size = Privates.size();
  for (const Expr *Private : Privates) {
    Address SrcElementAddr = Address::invalid();
    Address DestElementAddr = Address::invalid();
    Address DestElementPtrAddr = Address::invalid();
    // Should we shuffle in an element from a remote lane?
    bool ShuffleInElement = false;
    // Set to true to update the pointer in the dest Reduce list to a
    // newly created element.
    bool UpdateDestListPtr = false;
    // Increment the src or dest pointer to the scratchpad, for each
    // new element.
    bool IncrScratchpadSrc = false;
    bool IncrScratchpadDest = false;
    QualType PrivatePtrType = C.getPointerType(Private->getType());
    llvm::Type *PrivateLlvmPtrType = CGF.ConvertType(PrivatePtrType);

    switch (Action) {
    case RemoteLaneToThread: {
      // Step 1.1: Get the address for the src element in the Reduce list.
      Address SrcElementPtrAddr = Bld.CreateConstArrayGEP(SrcBase, Idx);
      SrcElementAddr = CGF.EmitLoadOfPointer(
          SrcElementPtrAddr.withElementType(PrivateLlvmPtrType),
          PrivatePtrType->castAs<PointerType>());

      // Step 1.2: Create a temporary to store the element in the destination
      // Reduce list.
      DestElementPtrAddr = Bld.CreateConstArrayGEP(DestBase, Idx);
      DestElementAddr =
          CGF.CreateMemTemp(Private->getType(), ".omp.reduction.element");
      ShuffleInElement = true;
      UpdateDestListPtr = true;
      break;
    }
    case ThreadCopy: {
      // Step 1.1: Get the address for the src element in the Reduce list.
      Address SrcElementPtrAddr = Bld.CreateConstArrayGEP(SrcBase, Idx);
      SrcElementAddr = CGF.EmitLoadOfPointer(
          SrcElementPtrAddr.withElementType(PrivateLlvmPtrType),
          PrivatePtrType->castAs<PointerType>());

      // Step 1.2: Get the address for dest element.  The destination
      // element has already been created on the thread's stack.
      DestElementPtrAddr = Bld.CreateConstArrayGEP(DestBase, Idx);
      DestElementAddr = CGF.EmitLoadOfPointer(
          DestElementPtrAddr.withElementType(PrivateLlvmPtrType),
          PrivatePtrType->castAs<PointerType>());
      break;
    }
    case ThreadToScratchpad: {
      // Step 1.1: Get the address for the src element in the Reduce list.
      Address SrcElementPtrAddr = Bld.CreateConstArrayGEP(SrcBase, Idx);
      SrcElementAddr = CGF.EmitLoadOfPointer(
          SrcElementPtrAddr.withElementType(PrivateLlvmPtrType),
          PrivatePtrType->castAs<PointerType>());

      // Step 1.2: Get the address for dest element:
      // address = base + index * ElementSizeInChars.
      llvm::Value *ElementSizeInChars = CGF.getTypeSize(Private->getType());
      llvm::Value *CurrentOffset =
          Bld.CreateNUWMul(ElementSizeInChars, ScratchpadIndex);
      llvm::Value *ScratchPadElemAbsolutePtrVal =
          Bld.CreateNUWAdd(DestBase.getPointer(), CurrentOffset);
      ScratchPadElemAbsolutePtrVal =
          Bld.CreateIntToPtr(ScratchPadElemAbsolutePtrVal, CGF.VoidPtrTy);
      DestElementAddr = Address(ScratchPadElemAbsolutePtrVal, CGF.Int8Ty,
                                C.getTypeAlignInChars(Private->getType()));
      IncrScratchpadDest = true;
      break;
    }
    case ScratchpadToThread: {
      // Step 1.1: Get the address for the src element in the scratchpad.
      // address = base + index * ElementSizeInChars.
      llvm::Value *ElementSizeInChars = CGF.getTypeSize(Private->getType());
      llvm::Value *CurrentOffset =
          Bld.CreateNUWMul(ElementSizeInChars, ScratchpadIndex);
      llvm::Value *ScratchPadElemAbsolutePtrVal =
          Bld.CreateNUWAdd(SrcBase.getPointer(), CurrentOffset);
      ScratchPadElemAbsolutePtrVal =
          Bld.CreateIntToPtr(ScratchPadElemAbsolutePtrVal, CGF.VoidPtrTy);
      SrcElementAddr = Address(ScratchPadElemAbsolutePtrVal, CGF.Int8Ty,
                               C.getTypeAlignInChars(Private->getType()));
      IncrScratchpadSrc = true;

      // Step 1.2: Create a temporary to store the element in the destination
      // Reduce list.
      DestElementPtrAddr = Bld.CreateConstArrayGEP(DestBase, Idx);
      DestElementAddr =
          CGF.CreateMemTemp(Private->getType(), ".omp.reduction.element");
      UpdateDestListPtr = true;
      break;
    }
    }

    // Regardless of src and dest of copy, we emit the load of src
    // element as this is required in all directions
    SrcElementAddr = SrcElementAddr.withElementType(
        CGF.ConvertTypeForMem(Private->getType()));
    DestElementAddr =
        DestElementAddr.withElementType(SrcElementAddr.getElementType());

    // Now that all active lanes have read the element in the
    // Reduce list, shuffle over the value from the remote lane.
    if (ShuffleInElement) {
      shuffleAndStore(CGF, SrcElementAddr, DestElementAddr, Private->getType(),
                      RemoteLaneOffset, Private->getExprLoc());
    } else {
      switch (CGF.getEvaluationKind(Private->getType())) {
      case TEK_Scalar: {
        llvm::Value *Elem = CGF.EmitLoadOfScalar(
            SrcElementAddr, /*Volatile=*/false, Private->getType(),
            Private->getExprLoc(), LValueBaseInfo(AlignmentSource::Type),
            TBAAAccessInfo());
        // Store the source element value to the dest element address.
        CGF.EmitStoreOfScalar(
            Elem, DestElementAddr, /*Volatile=*/false, Private->getType(),
            LValueBaseInfo(AlignmentSource::Type), TBAAAccessInfo());
        break;
      }
      case TEK_Complex: {
        CodeGenFunction::ComplexPairTy Elem = CGF.EmitLoadOfComplex(
            CGF.MakeAddrLValue(SrcElementAddr, Private->getType()),
            Private->getExprLoc());
        CGF.EmitStoreOfComplex(
            Elem, CGF.MakeAddrLValue(DestElementAddr, Private->getType()),
            /*isInit=*/false);
        break;
      }
      case TEK_Aggregate:
        CGF.EmitAggregateCopy(
            CGF.MakeAddrLValue(DestElementAddr, Private->getType()),
            CGF.MakeAddrLValue(SrcElementAddr, Private->getType()),
            Private->getType(), AggValueSlot::DoesNotOverlap);
        break;
      }
    }

    // Step 3.1: Modify reference in dest Reduce list as needed.
    // Modifying the reference in Reduce list to point to the newly
    // created element.  The element is live in the current function
    // scope and that of functions it invokes (i.e., reduce_function).
    // RemoteReduceData[i] = (void*)&RemoteElem
    if (UpdateDestListPtr) {
      CGF.EmitStoreOfScalar(Bld.CreatePointerBitCastOrAddrSpaceCast(
                                DestElementAddr.getPointer(), CGF.VoidPtrTy),
                            DestElementPtrAddr, /*Volatile=*/false,
                            C.VoidPtrTy);
    }

    // Step 4.1: Increment SrcBase/DestBase so that it points to the starting
    // address of the next element in scratchpad memory, unless we're currently
    // processing the last one.  Memory alignment is also taken care of here.
    if ((IncrScratchpadDest || IncrScratchpadSrc) && (Idx + 1 < Size)) {
      // FIXME: This code doesn't make any sense, it's trying to perform
      // integer arithmetic on pointers.
      llvm::Value *ScratchpadBasePtr =
          IncrScratchpadDest ? DestBase.getPointer() : SrcBase.getPointer();
      llvm::Value *ElementSizeInChars = CGF.getTypeSize(Private->getType());
      ScratchpadBasePtr = Bld.CreateNUWAdd(
          ScratchpadBasePtr,
          Bld.CreateNUWMul(ScratchpadWidth, ElementSizeInChars));

      // Take care of global memory alignment for performance
      ScratchpadBasePtr = Bld.CreateNUWSub(
          ScratchpadBasePtr, llvm::ConstantInt::get(CGM.SizeTy, 1));
      ScratchpadBasePtr = Bld.CreateUDiv(
          ScratchpadBasePtr,
          llvm::ConstantInt::get(CGM.SizeTy, GlobalMemoryAlignment));
      ScratchpadBasePtr = Bld.CreateNUWAdd(
          ScratchpadBasePtr, llvm::ConstantInt::get(CGM.SizeTy, 1));
      ScratchpadBasePtr = Bld.CreateNUWMul(
          ScratchpadBasePtr,
          llvm::ConstantInt::get(CGM.SizeTy, GlobalMemoryAlignment));

      if (IncrScratchpadDest)
        DestBase =
            Address(ScratchpadBasePtr, CGF.VoidPtrTy, CGF.getPointerAlign());
      else /* IncrScratchpadSrc = true */
        SrcBase =
            Address(ScratchpadBasePtr, CGF.VoidPtrTy, CGF.getPointerAlign());
    }

    ++Idx;
  }
}

/// This function emits a helper that gathers Reduce lists from the first
/// lane of every active warp to lanes in the first warp.
///
/// void inter_warp_copy_func(void* reduce_data, num_warps)
///   shared smem[warp_size];
///   For all data entries D in reduce_data:
///     sync
///     If (I am the first lane in each warp)
///       Copy my local D to smem[warp_id]
///     sync
///     if (I am the first warp)
///       Copy smem[thread_id] to my local D
static llvm::Value *emitInterWarpCopyFunction(CodeGenModule &CGM,
                                              ArrayRef<const Expr *> Privates,
                                              QualType ReductionArrayTy,
                                              SourceLocation Loc) {
  ASTContext &C = CGM.getContext();
  llvm::Module &M = CGM.getModule();

  // ReduceList: thread local Reduce list.
  // At the stage of the computation when this function is called, partially
  // aggregated values reside in the first lane of every active warp.
  ImplicitParamDecl ReduceListArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                  C.VoidPtrTy, ImplicitParamDecl::Other);
  // NumWarps: number of warps active in the parallel region.  This could
  // be smaller than 32 (max warps in a CTA) for partial block reduction.
  ImplicitParamDecl NumWarpsArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                C.getIntTypeForBitwidth(32, /* Signed */ true),
                                ImplicitParamDecl::Other);
  FunctionArgList Args;
  Args.push_back(&ReduceListArg);
  Args.push_back(&NumWarpsArg);

  const CGFunctionInfo &CGFI =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  auto *Fn = llvm::Function::Create(CGM.getTypes().GetFunctionType(CGFI),
                                    llvm::GlobalValue::InternalLinkage,
                                    "_omp_reduction_inter_warp_copy_func", &M);
  CGM.SetInternalFunctionAttributes(GlobalDecl(), Fn, CGFI);
  Fn->setDoesNotRecurse();
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, CGFI, Args, Loc, Loc);

  CGBuilderTy &Bld = CGF.Builder;

  // This array is used as a medium to transfer, one reduce element at a time,
  // the data from the first lane of every warp to lanes in the first warp
  // in order to perform the final step of a reduction in a parallel region
  // (reduction across warps).  The array is placed in NVPTX __shared__ memory
  // for reduced latency, as well as to have a distinct copy for concurrently
  // executing target regions.  The array is declared with common linkage so
  // as to be shared across compilation units.
  StringRef TransferMediumName =
      "__openmp_nvptx_data_transfer_temporary_storage";
  llvm::GlobalVariable *TransferMedium =
      M.getGlobalVariable(TransferMediumName);
  unsigned WarpSize = CGF.getTarget().getGridValue().GV_Warp_Size;
  if (!TransferMedium) {
    auto *Ty = llvm::ArrayType::get(CGM.Int32Ty, WarpSize);
    unsigned SharedAddressSpace = C.getTargetAddressSpace(LangAS::cuda_shared);
    TransferMedium = new llvm::GlobalVariable(
        M, Ty, /*isConstant=*/false, llvm::GlobalVariable::WeakAnyLinkage,
        llvm::UndefValue::get(Ty), TransferMediumName,
        /*InsertBefore=*/nullptr, llvm::GlobalVariable::NotThreadLocal,
        SharedAddressSpace);
    CGM.addCompilerUsedGlobal(TransferMedium);
  }

  auto &RT = static_cast<CGOpenMPRuntimeGPU &>(CGF.CGM.getOpenMPRuntime());
  // Get the CUDA thread id of the current OpenMP thread on the GPU.
  llvm::Value *ThreadID = RT.getGPUThreadID(CGF);
  // nvptx_lane_id = nvptx_id % warpsize
  llvm::Value *LaneID = getNVPTXLaneID(CGF);
  // nvptx_warp_id = nvptx_id / warpsize
  llvm::Value *WarpID = getNVPTXWarpID(CGF);

  Address AddrReduceListArg = CGF.GetAddrOfLocalVar(&ReduceListArg);
  llvm::Type *ElemTy = CGF.ConvertTypeForMem(ReductionArrayTy);
  Address LocalReduceList(
      Bld.CreatePointerBitCastOrAddrSpaceCast(
          CGF.EmitLoadOfScalar(
              AddrReduceListArg, /*Volatile=*/false, C.VoidPtrTy, Loc,
              LValueBaseInfo(AlignmentSource::Type), TBAAAccessInfo()),
          ElemTy->getPointerTo()),
      ElemTy, CGF.getPointerAlign());

  unsigned Idx = 0;
  for (const Expr *Private : Privates) {
    //
    // Warp master copies reduce element to transfer medium in __shared__
    // memory.
    //
    unsigned RealTySize =
        C.getTypeSizeInChars(Private->getType())
            .alignTo(C.getTypeAlignInChars(Private->getType()))
            .getQuantity();
    for (unsigned TySize = 4; TySize > 0 && RealTySize > 0; TySize /=2) {
      unsigned NumIters = RealTySize / TySize;
      if (NumIters == 0)
        continue;
      QualType CType = C.getIntTypeForBitwidth(
          C.toBits(CharUnits::fromQuantity(TySize)), /*Signed=*/1);
      llvm::Type *CopyType = CGF.ConvertTypeForMem(CType);
      CharUnits Align = CharUnits::fromQuantity(TySize);
      llvm::Value *Cnt = nullptr;
      Address CntAddr = Address::invalid();
      llvm::BasicBlock *PrecondBB = nullptr;
      llvm::BasicBlock *ExitBB = nullptr;
      if (NumIters > 1) {
        CntAddr = CGF.CreateMemTemp(C.IntTy, ".cnt.addr");
        CGF.EmitStoreOfScalar(llvm::Constant::getNullValue(CGM.IntTy), CntAddr,
                              /*Volatile=*/false, C.IntTy);
        PrecondBB = CGF.createBasicBlock("precond");
        ExitBB = CGF.createBasicBlock("exit");
        llvm::BasicBlock *BodyBB = CGF.createBasicBlock("body");
        // There is no need to emit line number for unconditional branch.
        (void)ApplyDebugLocation::CreateEmpty(CGF);
        CGF.EmitBlock(PrecondBB);
        Cnt = CGF.EmitLoadOfScalar(CntAddr, /*Volatile=*/false, C.IntTy, Loc);
        llvm::Value *Cmp =
            Bld.CreateICmpULT(Cnt, llvm::ConstantInt::get(CGM.IntTy, NumIters));
        Bld.CreateCondBr(Cmp, BodyBB, ExitBB);
        CGF.EmitBlock(BodyBB);
      }
      // kmpc_barrier.
      CGM.getOpenMPRuntime().emitBarrierCall(CGF, Loc, OMPD_unknown,
                                             /*EmitChecks=*/false,
                                             /*ForceSimpleCall=*/true);
      llvm::BasicBlock *ThenBB = CGF.createBasicBlock("then");
      llvm::BasicBlock *ElseBB = CGF.createBasicBlock("else");
      llvm::BasicBlock *MergeBB = CGF.createBasicBlock("ifcont");

      // if (lane_id == 0)
      llvm::Value *IsWarpMaster = Bld.CreateIsNull(LaneID, "warp_master");
      Bld.CreateCondBr(IsWarpMaster, ThenBB, ElseBB);
      CGF.EmitBlock(ThenBB);

      // Reduce element = LocalReduceList[i]
      Address ElemPtrPtrAddr = Bld.CreateConstArrayGEP(LocalReduceList, Idx);
      llvm::Value *ElemPtrPtr = CGF.EmitLoadOfScalar(
          ElemPtrPtrAddr, /*Volatile=*/false, C.VoidPtrTy, SourceLocation());
      // elemptr = ((CopyType*)(elemptrptr)) + I
      Address ElemPtr(ElemPtrPtr, CopyType, Align);
      if (NumIters > 1)
        ElemPtr = Bld.CreateGEP(ElemPtr, Cnt);

      // Get pointer to location in transfer medium.
      // MediumPtr = &medium[warp_id]
      llvm::Value *MediumPtrVal = Bld.CreateInBoundsGEP(
          TransferMedium->getValueType(), TransferMedium,
          {llvm::Constant::getNullValue(CGM.Int64Ty), WarpID});
      // Casting to actual data type.
      // MediumPtr = (CopyType*)MediumPtrAddr;
      Address MediumPtr(
          Bld.CreateBitCast(
              MediumPtrVal,
              CopyType->getPointerTo(
                  MediumPtrVal->getType()->getPointerAddressSpace())),
          CopyType, Align);

      // elem = *elemptr
      //*MediumPtr = elem
      llvm::Value *Elem = CGF.EmitLoadOfScalar(
          ElemPtr, /*Volatile=*/false, CType, Loc,
          LValueBaseInfo(AlignmentSource::Type), TBAAAccessInfo());
      // Store the source element value to the dest element address.
      CGF.EmitStoreOfScalar(Elem, MediumPtr, /*Volatile=*/true, CType,
                            LValueBaseInfo(AlignmentSource::Type),
                            TBAAAccessInfo());

      Bld.CreateBr(MergeBB);

      CGF.EmitBlock(ElseBB);
      Bld.CreateBr(MergeBB);

      CGF.EmitBlock(MergeBB);

      // kmpc_barrier.
      CGM.getOpenMPRuntime().emitBarrierCall(CGF, Loc, OMPD_unknown,
                                             /*EmitChecks=*/false,
                                             /*ForceSimpleCall=*/true);

      //
      // Warp 0 copies reduce element from transfer medium.
      //
      llvm::BasicBlock *W0ThenBB = CGF.createBasicBlock("then");
      llvm::BasicBlock *W0ElseBB = CGF.createBasicBlock("else");
      llvm::BasicBlock *W0MergeBB = CGF.createBasicBlock("ifcont");

      Address AddrNumWarpsArg = CGF.GetAddrOfLocalVar(&NumWarpsArg);
      llvm::Value *NumWarpsVal = CGF.EmitLoadOfScalar(
          AddrNumWarpsArg, /*Volatile=*/false, C.IntTy, Loc);

      // Up to 32 threads in warp 0 are active.
      llvm::Value *IsActiveThread =
          Bld.CreateICmpULT(ThreadID, NumWarpsVal, "is_active_thread");
      Bld.CreateCondBr(IsActiveThread, W0ThenBB, W0ElseBB);

      CGF.EmitBlock(W0ThenBB);

      // SrcMediumPtr = &medium[tid]
      llvm::Value *SrcMediumPtrVal = Bld.CreateInBoundsGEP(
          TransferMedium->getValueType(), TransferMedium,
          {llvm::Constant::getNullValue(CGM.Int64Ty), ThreadID});
      // SrcMediumVal = *SrcMediumPtr;
      Address SrcMediumPtr(
          Bld.CreateBitCast(
              SrcMediumPtrVal,
              CopyType->getPointerTo(
                  SrcMediumPtrVal->getType()->getPointerAddressSpace())),
          CopyType, Align);

      // TargetElemPtr = (CopyType*)(SrcDataAddr[i]) + I
      Address TargetElemPtrPtr = Bld.CreateConstArrayGEP(LocalReduceList, Idx);
      llvm::Value *TargetElemPtrVal = CGF.EmitLoadOfScalar(
          TargetElemPtrPtr, /*Volatile=*/false, C.VoidPtrTy, Loc);
      Address TargetElemPtr(TargetElemPtrVal, CopyType, Align);
      if (NumIters > 1)
        TargetElemPtr = Bld.CreateGEP(TargetElemPtr, Cnt);

      // *TargetElemPtr = SrcMediumVal;
      llvm::Value *SrcMediumValue =
          CGF.EmitLoadOfScalar(SrcMediumPtr, /*Volatile=*/true, CType, Loc);
      CGF.EmitStoreOfScalar(SrcMediumValue, TargetElemPtr, /*Volatile=*/false,
                            CType);
      Bld.CreateBr(W0MergeBB);

      CGF.EmitBlock(W0ElseBB);
      Bld.CreateBr(W0MergeBB);

      CGF.EmitBlock(W0MergeBB);

      if (NumIters > 1) {
        Cnt = Bld.CreateNSWAdd(Cnt, llvm::ConstantInt::get(CGM.IntTy, /*V=*/1));
        CGF.EmitStoreOfScalar(Cnt, CntAddr, /*Volatile=*/false, C.IntTy);
        CGF.EmitBranch(PrecondBB);
        (void)ApplyDebugLocation::CreateEmpty(CGF);
        CGF.EmitBlock(ExitBB);
      }
      RealTySize %= TySize;
    }
    ++Idx;
  }

  CGF.FinishFunction();
  return Fn;
}

/// Emit a helper that reduces data across two OpenMP threads (lanes)
/// in the same warp.  It uses shuffle instructions to copy over data from
/// a remote lane's stack.  The reduction algorithm performed is specified
/// by the fourth parameter.
///
/// Algorithm Versions.
/// Full Warp Reduce (argument value 0):
///   This algorithm assumes that all 32 lanes are active and gathers
///   data from these 32 lanes, producing a single resultant value.
/// Contiguous Partial Warp Reduce (argument value 1):
///   This algorithm assumes that only a *contiguous* subset of lanes
///   are active.  This happens for the last warp in a parallel region
///   when the user specified num_threads is not an integer multiple of
///   32.  This contiguous subset always starts with the zeroth lane.
/// Partial Warp Reduce (argument value 2):
///   This algorithm gathers data from any number of lanes at any position.
/// All reduced values are stored in the lowest possible lane.  The set
/// of problems every algorithm addresses is a super set of those
/// addressable by algorithms with a lower version number.  Overhead
/// increases as algorithm version increases.
///
/// Terminology
/// Reduce element:
///   Reduce element refers to the individual data field with primitive
///   data types to be combined and reduced across threads.
/// Reduce list:
///   Reduce list refers to a collection of local, thread-private
///   reduce elements.
/// Remote Reduce list:
///   Remote Reduce list refers to a collection of remote (relative to
///   the current thread) reduce elements.
///
/// We distinguish between three states of threads that are important to
/// the implementation of this function.
/// Alive threads:
///   Threads in a warp executing the SIMT instruction, as distinguished from
///   threads that are inactive due to divergent control flow.
/// Active threads:
///   The minimal set of threads that has to be alive upon entry to this
///   function.  The computation is correct iff active threads are alive.
///   Some threads are alive but they are not active because they do not
///   contribute to the computation in any useful manner.  Turning them off
///   may introduce control flow overheads without any tangible benefits.
/// Effective threads:
///   In order to comply with the argument requirements of the shuffle
///   function, we must keep all lanes holding data alive.  But at most
///   half of them perform value aggregation; we refer to this half of
///   threads as effective. The other half is simply handing off their
///   data.
///
/// Procedure
/// Value shuffle:
///   In this step active threads transfer data from higher lane positions
///   in the warp to lower lane positions, creating Remote Reduce list.
/// Value aggregation:
///   In this step, effective threads combine their thread local Reduce list
///   with Remote Reduce list and store the result in the thread local
///   Reduce list.
/// Value copy:
///   In this step, we deal with the assumption made by algorithm 2
///   (i.e. contiguity assumption).  When we have an odd number of lanes
///   active, say 2k+1, only k threads will be effective and therefore k
///   new values will be produced.  However, the Reduce list owned by the
///   (2k+1)th thread is ignored in the value aggregation.  Therefore
///   we copy the Reduce list from the (2k+1)th lane to (k+1)th lane so
///   that the contiguity assumption still holds.
static llvm::Function *emitShuffleAndReduceFunction(
    CodeGenModule &CGM, ArrayRef<const Expr *> Privates,
    QualType ReductionArrayTy, llvm::Function *ReduceFn, SourceLocation Loc) {
  ASTContext &C = CGM.getContext();

  // Thread local Reduce list used to host the values of data to be reduced.
  ImplicitParamDecl ReduceListArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                  C.VoidPtrTy, ImplicitParamDecl::Other);
  // Current lane id; could be logical.
  ImplicitParamDecl LaneIDArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, C.ShortTy,
                              ImplicitParamDecl::Other);
  // Offset of the remote source lane relative to the current lane.
  ImplicitParamDecl RemoteLaneOffsetArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                        C.ShortTy, ImplicitParamDecl::Other);
  // Algorithm version.  This is expected to be known at compile time.
  ImplicitParamDecl AlgoVerArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                               C.ShortTy, ImplicitParamDecl::Other);
  FunctionArgList Args;
  Args.push_back(&ReduceListArg);
  Args.push_back(&LaneIDArg);
  Args.push_back(&RemoteLaneOffsetArg);
  Args.push_back(&AlgoVerArg);

  const CGFunctionInfo &CGFI =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  auto *Fn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(CGFI), llvm::GlobalValue::InternalLinkage,
      "_omp_reduction_shuffle_and_reduce_func", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), Fn, CGFI);
  Fn->setDoesNotRecurse();

  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, CGFI, Args, Loc, Loc);

  CGBuilderTy &Bld = CGF.Builder;

  Address AddrReduceListArg = CGF.GetAddrOfLocalVar(&ReduceListArg);
  llvm::Type *ElemTy = CGF.ConvertTypeForMem(ReductionArrayTy);
  Address LocalReduceList(
      Bld.CreatePointerBitCastOrAddrSpaceCast(
          CGF.EmitLoadOfScalar(AddrReduceListArg, /*Volatile=*/false,
                               C.VoidPtrTy, SourceLocation()),
          ElemTy->getPointerTo()),
      ElemTy, CGF.getPointerAlign());

  Address AddrLaneIDArg = CGF.GetAddrOfLocalVar(&LaneIDArg);
  llvm::Value *LaneIDArgVal = CGF.EmitLoadOfScalar(
      AddrLaneIDArg, /*Volatile=*/false, C.ShortTy, SourceLocation());

  Address AddrRemoteLaneOffsetArg = CGF.GetAddrOfLocalVar(&RemoteLaneOffsetArg);
  llvm::Value *RemoteLaneOffsetArgVal = CGF.EmitLoadOfScalar(
      AddrRemoteLaneOffsetArg, /*Volatile=*/false, C.ShortTy, SourceLocation());

  Address AddrAlgoVerArg = CGF.GetAddrOfLocalVar(&AlgoVerArg);
  llvm::Value *AlgoVerArgVal = CGF.EmitLoadOfScalar(
      AddrAlgoVerArg, /*Volatile=*/false, C.ShortTy, SourceLocation());

  // Create a local thread-private variable to host the Reduce list
  // from a remote lane.
  Address RemoteReduceList =
      CGF.CreateMemTemp(ReductionArrayTy, ".omp.reduction.remote_reduce_list");

  // This loop iterates through the list of reduce elements and copies,
  // element by element, from a remote lane in the warp to RemoteReduceList,
  // hosted on the thread's stack.
  emitReductionListCopy(RemoteLaneToThread, CGF, ReductionArrayTy, Privates,
                        LocalReduceList, RemoteReduceList,
                        {/*RemoteLaneOffset=*/RemoteLaneOffsetArgVal,
                         /*ScratchpadIndex=*/nullptr,
                         /*ScratchpadWidth=*/nullptr});

  // The actions to be performed on the Remote Reduce list is dependent
  // on the algorithm version.
  //
  //  if (AlgoVer==0) || (AlgoVer==1 && (LaneId < Offset)) || (AlgoVer==2 &&
  //  LaneId % 2 == 0 && Offset > 0):
  //    do the reduction value aggregation
  //
  //  The thread local variable Reduce list is mutated in place to host the
  //  reduced data, which is the aggregated value produced from local and
  //  remote lanes.
  //
  //  Note that AlgoVer is expected to be a constant integer known at compile
  //  time.
  //  When AlgoVer==0, the first conjunction evaluates to true, making
  //    the entire predicate true during compile time.
  //  When AlgoVer==1, the second conjunction has only the second part to be
  //    evaluated during runtime.  Other conjunctions evaluates to false
  //    during compile time.
  //  When AlgoVer==2, the third conjunction has only the second part to be
  //    evaluated during runtime.  Other conjunctions evaluates to false
  //    during compile time.
  llvm::Value *CondAlgo0 = Bld.CreateIsNull(AlgoVerArgVal);

  llvm::Value *Algo1 = Bld.CreateICmpEQ(AlgoVerArgVal, Bld.getInt16(1));
  llvm::Value *CondAlgo1 = Bld.CreateAnd(
      Algo1, Bld.CreateICmpULT(LaneIDArgVal, RemoteLaneOffsetArgVal));

  llvm::Value *Algo2 = Bld.CreateICmpEQ(AlgoVerArgVal, Bld.getInt16(2));
  llvm::Value *CondAlgo2 = Bld.CreateAnd(
      Algo2, Bld.CreateIsNull(Bld.CreateAnd(LaneIDArgVal, Bld.getInt16(1))));
  CondAlgo2 = Bld.CreateAnd(
      CondAlgo2, Bld.CreateICmpSGT(RemoteLaneOffsetArgVal, Bld.getInt16(0)));

  llvm::Value *CondReduce = Bld.CreateOr(CondAlgo0, CondAlgo1);
  CondReduce = Bld.CreateOr(CondReduce, CondAlgo2);

  llvm::BasicBlock *ThenBB = CGF.createBasicBlock("then");
  llvm::BasicBlock *ElseBB = CGF.createBasicBlock("else");
  llvm::BasicBlock *MergeBB = CGF.createBasicBlock("ifcont");
  Bld.CreateCondBr(CondReduce, ThenBB, ElseBB);

  CGF.EmitBlock(ThenBB);
  // reduce_function(LocalReduceList, RemoteReduceList)
  llvm::Value *LocalReduceListPtr = Bld.CreatePointerBitCastOrAddrSpaceCast(
      LocalReduceList.getPointer(), CGF.VoidPtrTy);
  llvm::Value *RemoteReduceListPtr = Bld.CreatePointerBitCastOrAddrSpaceCast(
      RemoteReduceList.getPointer(), CGF.VoidPtrTy);
  CGM.getOpenMPRuntime().emitOutlinedFunctionCall(
      CGF, Loc, ReduceFn, {LocalReduceListPtr, RemoteReduceListPtr});
  Bld.CreateBr(MergeBB);

  CGF.EmitBlock(ElseBB);
  Bld.CreateBr(MergeBB);

  CGF.EmitBlock(MergeBB);

  // if (AlgoVer==1 && (LaneId >= Offset)) copy Remote Reduce list to local
  // Reduce list.
  Algo1 = Bld.CreateICmpEQ(AlgoVerArgVal, Bld.getInt16(1));
  llvm::Value *CondCopy = Bld.CreateAnd(
      Algo1, Bld.CreateICmpUGE(LaneIDArgVal, RemoteLaneOffsetArgVal));

  llvm::BasicBlock *CpyThenBB = CGF.createBasicBlock("then");
  llvm::BasicBlock *CpyElseBB = CGF.createBasicBlock("else");
  llvm::BasicBlock *CpyMergeBB = CGF.createBasicBlock("ifcont");
  Bld.CreateCondBr(CondCopy, CpyThenBB, CpyElseBB);

  CGF.EmitBlock(CpyThenBB);
  emitReductionListCopy(ThreadCopy, CGF, ReductionArrayTy, Privates,
                        RemoteReduceList, LocalReduceList);
  Bld.CreateBr(CpyMergeBB);

  CGF.EmitBlock(CpyElseBB);
  Bld.CreateBr(CpyMergeBB);

  CGF.EmitBlock(CpyMergeBB);

  CGF.FinishFunction();
  return Fn;
}

/// This function emits a helper that copies all the reduction variables from
/// the team into the provided global buffer for the reduction variables.
///
/// void list_to_global_copy_func(void *buffer, int Idx, void *reduce_data)
///   For all data entries D in reduce_data:
///     Copy local D to buffer.D[Idx]
static llvm::Value *emitListToGlobalCopyFunction(
    CodeGenModule &CGM, ArrayRef<const Expr *> Privates,
    QualType ReductionArrayTy, SourceLocation Loc,
    const RecordDecl *TeamReductionRec,
    const llvm::SmallDenseMap<const ValueDecl *, const FieldDecl *>
        &VarFieldMap) {
  ASTContext &C = CGM.getContext();

  // Buffer: global reduction buffer.
  ImplicitParamDecl BufferArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                              C.VoidPtrTy, ImplicitParamDecl::Other);
  // Idx: index of the buffer.
  ImplicitParamDecl IdxArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, C.IntTy,
                           ImplicitParamDecl::Other);
  // ReduceList: thread local Reduce list.
  ImplicitParamDecl ReduceListArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                  C.VoidPtrTy, ImplicitParamDecl::Other);
  FunctionArgList Args;
  Args.push_back(&BufferArg);
  Args.push_back(&IdxArg);
  Args.push_back(&ReduceListArg);

  const CGFunctionInfo &CGFI =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  auto *Fn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(CGFI), llvm::GlobalValue::InternalLinkage,
      "_omp_reduction_list_to_global_copy_func", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), Fn, CGFI);
  Fn->setDoesNotRecurse();
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, CGFI, Args, Loc, Loc);

  CGBuilderTy &Bld = CGF.Builder;

  Address AddrReduceListArg = CGF.GetAddrOfLocalVar(&ReduceListArg);
  Address AddrBufferArg = CGF.GetAddrOfLocalVar(&BufferArg);
  llvm::Type *ElemTy = CGF.ConvertTypeForMem(ReductionArrayTy);
  Address LocalReduceList(
      Bld.CreatePointerBitCastOrAddrSpaceCast(
          CGF.EmitLoadOfScalar(AddrReduceListArg, /*Volatile=*/false,
                               C.VoidPtrTy, Loc),
          ElemTy->getPointerTo()),
      ElemTy, CGF.getPointerAlign());
  QualType StaticTy = C.getRecordType(TeamReductionRec);
  llvm::Type *LLVMReductionsBufferTy =
      CGM.getTypes().ConvertTypeForMem(StaticTy);
  llvm::Value *BufferArrPtr = Bld.CreatePointerBitCastOrAddrSpaceCast(
      CGF.EmitLoadOfScalar(AddrBufferArg, /*Volatile=*/false, C.VoidPtrTy, Loc),
      LLVMReductionsBufferTy->getPointerTo());
  llvm::Value *Idxs[] = {llvm::ConstantInt::getNullValue(CGF.Int32Ty),
                         CGF.EmitLoadOfScalar(CGF.GetAddrOfLocalVar(&IdxArg),
                                              /*Volatile=*/false, C.IntTy,
                                              Loc)};
  unsigned Idx = 0;
  for (const Expr *Private : Privates) {
    // Reduce element = LocalReduceList[i]
    Address ElemPtrPtrAddr = Bld.CreateConstArrayGEP(LocalReduceList, Idx);
    llvm::Value *ElemPtrPtr = CGF.EmitLoadOfScalar(
        ElemPtrPtrAddr, /*Volatile=*/false, C.VoidPtrTy, SourceLocation());
    // elemptr = ((CopyType*)(elemptrptr)) + I
    ElemTy = CGF.ConvertTypeForMem(Private->getType());
    ElemPtrPtr = Bld.CreatePointerBitCastOrAddrSpaceCast(
        ElemPtrPtr, ElemTy->getPointerTo());
    Address ElemPtr =
        Address(ElemPtrPtr, ElemTy, C.getTypeAlignInChars(Private->getType()));
    const ValueDecl *VD = cast<DeclRefExpr>(Private)->getDecl();
    // Global = Buffer.VD[Idx];
    const FieldDecl *FD = VarFieldMap.lookup(VD);
    LValue GlobLVal = CGF.EmitLValueForField(
        CGF.MakeNaturalAlignAddrLValue(BufferArrPtr, StaticTy), FD);
    Address GlobAddr = GlobLVal.getAddress(CGF);
    llvm::Value *BufferPtr = Bld.CreateInBoundsGEP(GlobAddr.getElementType(),
                                                   GlobAddr.getPointer(), Idxs);
    GlobLVal.setAddress(Address(BufferPtr,
                                CGF.ConvertTypeForMem(Private->getType()),
                                GlobAddr.getAlignment()));
    switch (CGF.getEvaluationKind(Private->getType())) {
    case TEK_Scalar: {
      llvm::Value *V = CGF.EmitLoadOfScalar(
          ElemPtr, /*Volatile=*/false, Private->getType(), Loc,
          LValueBaseInfo(AlignmentSource::Type), TBAAAccessInfo());
      CGF.EmitStoreOfScalar(V, GlobLVal);
      break;
    }
    case TEK_Complex: {
      CodeGenFunction::ComplexPairTy V = CGF.EmitLoadOfComplex(
          CGF.MakeAddrLValue(ElemPtr, Private->getType()), Loc);
      CGF.EmitStoreOfComplex(V, GlobLVal, /*isInit=*/false);
      break;
    }
    case TEK_Aggregate:
      CGF.EmitAggregateCopy(GlobLVal,
                            CGF.MakeAddrLValue(ElemPtr, Private->getType()),
                            Private->getType(), AggValueSlot::DoesNotOverlap);
      break;
    }
    ++Idx;
  }

  CGF.FinishFunction();
  return Fn;
}

/// This function emits a helper that reduces all the reduction variables from
/// the team into the provided global buffer for the reduction variables.
///
/// void list_to_global_reduce_func(void *buffer, int Idx, void *reduce_data)
///  void *GlobPtrs[];
///  GlobPtrs[0] = (void*)&buffer.D0[Idx];
///  ...
///  GlobPtrs[N] = (void*)&buffer.DN[Idx];
///  reduce_function(GlobPtrs, reduce_data);
static llvm::Value *emitListToGlobalReduceFunction(
    CodeGenModule &CGM, ArrayRef<const Expr *> Privates,
    QualType ReductionArrayTy, SourceLocation Loc,
    const RecordDecl *TeamReductionRec,
    const llvm::SmallDenseMap<const ValueDecl *, const FieldDecl *>
        &VarFieldMap,
    llvm::Function *ReduceFn) {
  ASTContext &C = CGM.getContext();

  // Buffer: global reduction buffer.
  ImplicitParamDecl BufferArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                              C.VoidPtrTy, ImplicitParamDecl::Other);
  // Idx: index of the buffer.
  ImplicitParamDecl IdxArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, C.IntTy,
                           ImplicitParamDecl::Other);
  // ReduceList: thread local Reduce list.
  ImplicitParamDecl ReduceListArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                  C.VoidPtrTy, ImplicitParamDecl::Other);
  FunctionArgList Args;
  Args.push_back(&BufferArg);
  Args.push_back(&IdxArg);
  Args.push_back(&ReduceListArg);

  const CGFunctionInfo &CGFI =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  auto *Fn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(CGFI), llvm::GlobalValue::InternalLinkage,
      "_omp_reduction_list_to_global_reduce_func", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), Fn, CGFI);
  Fn->setDoesNotRecurse();
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, CGFI, Args, Loc, Loc);

  CGBuilderTy &Bld = CGF.Builder;

  Address AddrBufferArg = CGF.GetAddrOfLocalVar(&BufferArg);
  QualType StaticTy = C.getRecordType(TeamReductionRec);
  llvm::Type *LLVMReductionsBufferTy =
      CGM.getTypes().ConvertTypeForMem(StaticTy);
  llvm::Value *BufferArrPtr = Bld.CreatePointerBitCastOrAddrSpaceCast(
      CGF.EmitLoadOfScalar(AddrBufferArg, /*Volatile=*/false, C.VoidPtrTy, Loc),
      LLVMReductionsBufferTy->getPointerTo());

  // 1. Build a list of reduction variables.
  // void *RedList[<n>] = {<ReductionVars>[0], ..., <ReductionVars>[<n>-1]};
  Address ReductionList =
      CGF.CreateMemTemp(ReductionArrayTy, ".omp.reduction.red_list");
  auto IPriv = Privates.begin();
  llvm::Value *Idxs[] = {llvm::ConstantInt::getNullValue(CGF.Int32Ty),
                         CGF.EmitLoadOfScalar(CGF.GetAddrOfLocalVar(&IdxArg),
                                              /*Volatile=*/false, C.IntTy,
                                              Loc)};
  unsigned Idx = 0;
  for (unsigned I = 0, E = Privates.size(); I < E; ++I, ++IPriv, ++Idx) {
    Address Elem = CGF.Builder.CreateConstArrayGEP(ReductionList, Idx);
    // Global = Buffer.VD[Idx];
    const ValueDecl *VD = cast<DeclRefExpr>(*IPriv)->getDecl();
    const FieldDecl *FD = VarFieldMap.lookup(VD);
    LValue GlobLVal = CGF.EmitLValueForField(
        CGF.MakeNaturalAlignAddrLValue(BufferArrPtr, StaticTy), FD);
    Address GlobAddr = GlobLVal.getAddress(CGF);
    llvm::Value *BufferPtr = Bld.CreateInBoundsGEP(
        GlobAddr.getElementType(), GlobAddr.getPointer(), Idxs);
    CGF.EmitStoreOfScalar(BufferPtr, Elem, /*Volatile=*/false, C.VoidPtrTy);
    if ((*IPriv)->getType()->isVariablyModifiedType()) {
      // Store array size.
      ++Idx;
      Elem = CGF.Builder.CreateConstArrayGEP(ReductionList, Idx);
      llvm::Value *Size = CGF.Builder.CreateIntCast(
          CGF.getVLASize(
                 CGF.getContext().getAsVariableArrayType((*IPriv)->getType()))
              .NumElts,
          CGF.SizeTy, /*isSigned=*/false);
      CGF.Builder.CreateStore(CGF.Builder.CreateIntToPtr(Size, CGF.VoidPtrTy),
                              Elem);
    }
  }

  // Call reduce_function(GlobalReduceList, ReduceList)
  llvm::Value *GlobalReduceList = ReductionList.getPointer();
  Address AddrReduceListArg = CGF.GetAddrOfLocalVar(&ReduceListArg);
  llvm::Value *ReducedPtr = CGF.EmitLoadOfScalar(
      AddrReduceListArg, /*Volatile=*/false, C.VoidPtrTy, Loc);
  CGM.getOpenMPRuntime().emitOutlinedFunctionCall(
      CGF, Loc, ReduceFn, {GlobalReduceList, ReducedPtr});
  CGF.FinishFunction();
  return Fn;
}

/// This function emits a helper that copies all the reduction variables from
/// the team into the provided global buffer for the reduction variables.
///
/// void list_to_global_copy_func(void *buffer, int Idx, void *reduce_data)
///   For all data entries D in reduce_data:
///     Copy buffer.D[Idx] to local D;
static llvm::Value *emitGlobalToListCopyFunction(
    CodeGenModule &CGM, ArrayRef<const Expr *> Privates,
    QualType ReductionArrayTy, SourceLocation Loc,
    const RecordDecl *TeamReductionRec,
    const llvm::SmallDenseMap<const ValueDecl *, const FieldDecl *>
        &VarFieldMap) {
  ASTContext &C = CGM.getContext();

  // Buffer: global reduction buffer.
  ImplicitParamDecl BufferArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                              C.VoidPtrTy, ImplicitParamDecl::Other);
  // Idx: index of the buffer.
  ImplicitParamDecl IdxArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, C.IntTy,
                           ImplicitParamDecl::Other);
  // ReduceList: thread local Reduce list.
  ImplicitParamDecl ReduceListArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                  C.VoidPtrTy, ImplicitParamDecl::Other);
  FunctionArgList Args;
  Args.push_back(&BufferArg);
  Args.push_back(&IdxArg);
  Args.push_back(&ReduceListArg);

  const CGFunctionInfo &CGFI =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  auto *Fn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(CGFI), llvm::GlobalValue::InternalLinkage,
      "_omp_reduction_global_to_list_copy_func", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), Fn, CGFI);
  Fn->setDoesNotRecurse();
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, CGFI, Args, Loc, Loc);

  CGBuilderTy &Bld = CGF.Builder;

  Address AddrReduceListArg = CGF.GetAddrOfLocalVar(&ReduceListArg);
  Address AddrBufferArg = CGF.GetAddrOfLocalVar(&BufferArg);
  llvm::Type *ElemTy = CGF.ConvertTypeForMem(ReductionArrayTy);
  Address LocalReduceList(
      Bld.CreatePointerBitCastOrAddrSpaceCast(
          CGF.EmitLoadOfScalar(AddrReduceListArg, /*Volatile=*/false,
                               C.VoidPtrTy, Loc),
          ElemTy->getPointerTo()),
      ElemTy, CGF.getPointerAlign());
  QualType StaticTy = C.getRecordType(TeamReductionRec);
  llvm::Type *LLVMReductionsBufferTy =
      CGM.getTypes().ConvertTypeForMem(StaticTy);
  llvm::Value *BufferArrPtr = Bld.CreatePointerBitCastOrAddrSpaceCast(
      CGF.EmitLoadOfScalar(AddrBufferArg, /*Volatile=*/false, C.VoidPtrTy, Loc),
      LLVMReductionsBufferTy->getPointerTo());

  llvm::Value *Idxs[] = {llvm::ConstantInt::getNullValue(CGF.Int32Ty),
                         CGF.EmitLoadOfScalar(CGF.GetAddrOfLocalVar(&IdxArg),
                                              /*Volatile=*/false, C.IntTy,
                                              Loc)};
  unsigned Idx = 0;
  for (const Expr *Private : Privates) {
    // Reduce element = LocalReduceList[i]
    Address ElemPtrPtrAddr = Bld.CreateConstArrayGEP(LocalReduceList, Idx);
    llvm::Value *ElemPtrPtr = CGF.EmitLoadOfScalar(
        ElemPtrPtrAddr, /*Volatile=*/false, C.VoidPtrTy, SourceLocation());
    // elemptr = ((CopyType*)(elemptrptr)) + I
    ElemTy = CGF.ConvertTypeForMem(Private->getType());
    ElemPtrPtr = Bld.CreatePointerBitCastOrAddrSpaceCast(
        ElemPtrPtr, ElemTy->getPointerTo());
    Address ElemPtr =
        Address(ElemPtrPtr, ElemTy, C.getTypeAlignInChars(Private->getType()));
    const ValueDecl *VD = cast<DeclRefExpr>(Private)->getDecl();
    // Global = Buffer.VD[Idx];
    const FieldDecl *FD = VarFieldMap.lookup(VD);
    LValue GlobLVal = CGF.EmitLValueForField(
        CGF.MakeNaturalAlignAddrLValue(BufferArrPtr, StaticTy), FD);
    Address GlobAddr = GlobLVal.getAddress(CGF);
    llvm::Value *BufferPtr = Bld.CreateInBoundsGEP(GlobAddr.getElementType(),
                                                   GlobAddr.getPointer(), Idxs);
    GlobLVal.setAddress(Address(BufferPtr,
                                CGF.ConvertTypeForMem(Private->getType()),
                                GlobAddr.getAlignment()));
    switch (CGF.getEvaluationKind(Private->getType())) {
    case TEK_Scalar: {
      llvm::Value *V = CGF.EmitLoadOfScalar(GlobLVal, Loc);
      CGF.EmitStoreOfScalar(V, ElemPtr, /*Volatile=*/false, Private->getType(),
                            LValueBaseInfo(AlignmentSource::Type),
                            TBAAAccessInfo());
      break;
    }
    case TEK_Complex: {
      CodeGenFunction::ComplexPairTy V = CGF.EmitLoadOfComplex(GlobLVal, Loc);
      CGF.EmitStoreOfComplex(V, CGF.MakeAddrLValue(ElemPtr, Private->getType()),
                             /*isInit=*/false);
      break;
    }
    case TEK_Aggregate:
      CGF.EmitAggregateCopy(CGF.MakeAddrLValue(ElemPtr, Private->getType()),
                            GlobLVal, Private->getType(),
                            AggValueSlot::DoesNotOverlap);
      break;
    }
    ++Idx;
  }

  CGF.FinishFunction();
  return Fn;
}

/// This function emits a helper that reduces all the reduction variables from
/// the team into the provided global buffer for the reduction variables.
///
/// void global_to_list_reduce_func(void *buffer, int Idx, void *reduce_data)
///  void *GlobPtrs[];
///  GlobPtrs[0] = (void*)&buffer.D0[Idx];
///  ...
///  GlobPtrs[N] = (void*)&buffer.DN[Idx];
///  reduce_function(reduce_data, GlobPtrs);
static llvm::Value *emitGlobalToListReduceFunction(
    CodeGenModule &CGM, ArrayRef<const Expr *> Privates,
    QualType ReductionArrayTy, SourceLocation Loc,
    const RecordDecl *TeamReductionRec,
    const llvm::SmallDenseMap<const ValueDecl *, const FieldDecl *>
        &VarFieldMap,
    llvm::Function *ReduceFn) {
  ASTContext &C = CGM.getContext();

  // Buffer: global reduction buffer.
  ImplicitParamDecl BufferArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                              C.VoidPtrTy, ImplicitParamDecl::Other);
  // Idx: index of the buffer.
  ImplicitParamDecl IdxArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, C.IntTy,
                           ImplicitParamDecl::Other);
  // ReduceList: thread local Reduce list.
  ImplicitParamDecl ReduceListArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                  C.VoidPtrTy, ImplicitParamDecl::Other);
  FunctionArgList Args;
  Args.push_back(&BufferArg);
  Args.push_back(&IdxArg);
  Args.push_back(&ReduceListArg);

  const CGFunctionInfo &CGFI =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  auto *Fn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(CGFI), llvm::GlobalValue::InternalLinkage,
      "_omp_reduction_global_to_list_reduce_func", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), Fn, CGFI);
  Fn->setDoesNotRecurse();
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, CGFI, Args, Loc, Loc);

  CGBuilderTy &Bld = CGF.Builder;

  Address AddrBufferArg = CGF.GetAddrOfLocalVar(&BufferArg);
  QualType StaticTy = C.getRecordType(TeamReductionRec);
  llvm::Type *LLVMReductionsBufferTy =
      CGM.getTypes().ConvertTypeForMem(StaticTy);
  llvm::Value *BufferArrPtr = Bld.CreatePointerBitCastOrAddrSpaceCast(
      CGF.EmitLoadOfScalar(AddrBufferArg, /*Volatile=*/false, C.VoidPtrTy, Loc),
      LLVMReductionsBufferTy->getPointerTo());

  // 1. Build a list of reduction variables.
  // void *RedList[<n>] = {<ReductionVars>[0], ..., <ReductionVars>[<n>-1]};
  Address ReductionList =
      CGF.CreateMemTemp(ReductionArrayTy, ".omp.reduction.red_list");
  auto IPriv = Privates.begin();
  llvm::Value *Idxs[] = {llvm::ConstantInt::getNullValue(CGF.Int32Ty),
                         CGF.EmitLoadOfScalar(CGF.GetAddrOfLocalVar(&IdxArg),
                                              /*Volatile=*/false, C.IntTy,
                                              Loc)};
  unsigned Idx = 0;
  for (unsigned I = 0, E = Privates.size(); I < E; ++I, ++IPriv, ++Idx) {
    Address Elem = CGF.Builder.CreateConstArrayGEP(ReductionList, Idx);
    // Global = Buffer.VD[Idx];
    const ValueDecl *VD = cast<DeclRefExpr>(*IPriv)->getDecl();
    const FieldDecl *FD = VarFieldMap.lookup(VD);
    LValue GlobLVal = CGF.EmitLValueForField(
        CGF.MakeNaturalAlignAddrLValue(BufferArrPtr, StaticTy), FD);
    Address GlobAddr = GlobLVal.getAddress(CGF);
    llvm::Value *BufferPtr = Bld.CreateInBoundsGEP(
        GlobAddr.getElementType(), GlobAddr.getPointer(), Idxs);
    CGF.EmitStoreOfScalar(BufferPtr, Elem, /*Volatile=*/false, C.VoidPtrTy);
    if ((*IPriv)->getType()->isVariablyModifiedType()) {
      // Store array size.
      ++Idx;
      Elem = CGF.Builder.CreateConstArrayGEP(ReductionList, Idx);
      llvm::Value *Size = CGF.Builder.CreateIntCast(
          CGF.getVLASize(
                 CGF.getContext().getAsVariableArrayType((*IPriv)->getType()))
              .NumElts,
          CGF.SizeTy, /*isSigned=*/false);
      CGF.Builder.CreateStore(CGF.Builder.CreateIntToPtr(Size, CGF.VoidPtrTy),
                              Elem);
    }
  }

  // Call reduce_function(ReduceList, GlobalReduceList)
  llvm::Value *GlobalReduceList = ReductionList.getPointer();
  Address AddrReduceListArg = CGF.GetAddrOfLocalVar(&ReduceListArg);
  llvm::Value *ReducedPtr = CGF.EmitLoadOfScalar(
      AddrReduceListArg, /*Volatile=*/false, C.VoidPtrTy, Loc);
  CGM.getOpenMPRuntime().emitOutlinedFunctionCall(
      CGF, Loc, ReduceFn, {ReducedPtr, GlobalReduceList});
  CGF.FinishFunction();
  return Fn;
}

///
/// Design of OpenMP reductions on the GPU
///
/// Consider a typical OpenMP program with one or more reduction
/// clauses:
///
/// float foo;
/// double bar;
/// #pragma omp target teams distribute parallel for \
///             reduction(+:foo) reduction(*:bar)
/// for (int i = 0; i < N; i++) {
///   foo += A[i]; bar *= B[i];
/// }
///
/// where 'foo' and 'bar' are reduced across all OpenMP threads in
/// all teams.  In our OpenMP implementation on the NVPTX device an
/// OpenMP team is mapped to a CUDA threadblock and OpenMP threads
/// within a team are mapped to CUDA threads within a threadblock.
/// Our goal is to efficiently aggregate values across all OpenMP
/// threads such that:
///
///   - the compiler and runtime are logically concise, and
///   - the reduction is performed efficiently in a hierarchical
///     manner as follows: within OpenMP threads in the same warp,
///     across warps in a threadblock, and finally across teams on
///     the NVPTX device.
///
/// Introduction to Decoupling
///
/// We would like to decouple the compiler and the runtime so that the
/// latter is ignorant of the reduction variables (number, data types)
/// and the reduction operators.  This allows a simpler interface
/// and implementation while still attaining good performance.
///
/// Pseudocode for the aforementioned OpenMP program generated by the
/// compiler is as follows:
///
/// 1. Create private copies of reduction variables on each OpenMP
///    thread: 'foo_private', 'bar_private'
/// 2. Each OpenMP thread reduces the chunk of 'A' and 'B' assigned
///    to it and writes the result in 'foo_private' and 'bar_private'
///    respectively.
/// 3. Call the OpenMP runtime on the GPU to reduce within a team
///    and store the result on the team master:
///
///     __kmpc_nvptx_parallel_reduce_nowait_v2(...,
///        reduceData, shuffleReduceFn, interWarpCpyFn)
///
///     where:
///       struct ReduceData {
///         double *foo;
///         double *bar;
///       } reduceData
///       reduceData.foo = &foo_private
///       reduceData.bar = &bar_private
///
///     'shuffleReduceFn' and 'interWarpCpyFn' are pointers to two
///     auxiliary functions generated by the compiler that operate on
///     variables of type 'ReduceData'.  They aid the runtime perform
///     algorithmic steps in a data agnostic manner.
///
///     'shuffleReduceFn' is a pointer to a function that reduces data
///     of type 'ReduceData' across two OpenMP threads (lanes) in the
///     same warp.  It takes the following arguments as input:
///
///     a. variable of type 'ReduceData' on the calling lane,
///     b. its lane_id,
///     c. an offset relative to the current lane_id to generate a
///        remote_lane_id.  The remote lane contains the second
///        variable of type 'ReduceData' that is to be reduced.
///     d. an algorithm version parameter determining which reduction
///        algorithm to use.
///
///     'shuffleReduceFn' retrieves data from the remote lane using
///     efficient GPU shuffle intrinsics and reduces, using the
///     algorithm specified by the 4th parameter, the two operands
///     element-wise.  The result is written to the first operand.
///
///     Different reduction algorithms are implemented in different
///     runtime functions, all calling 'shuffleReduceFn' to perform
///     the essential reduction step.  Therefore, based on the 4th
///     parameter, this function behaves slightly differently to
///     cooperate with the runtime to ensure correctness under
///     different circumstances.
///
///     'InterWarpCpyFn' is a pointer to a function that transfers
///     reduced variables across warps.  It tunnels, through CUDA
///     shared memory, the thread-private data of type 'ReduceData'
///     from lane 0 of each warp to a lane in the first warp.
/// 4. Call the OpenMP runtime on the GPU to reduce across teams.
///    The last team writes the global reduced value to memory.
///
///     ret = __kmpc_nvptx_teams_reduce_nowait(...,
///             reduceData, shuffleReduceFn, interWarpCpyFn,
///             scratchpadCopyFn, loadAndReduceFn)
///
///     'scratchpadCopyFn' is a helper that stores reduced
///     data from the team master to a scratchpad array in
///     global memory.
///
///     'loadAndReduceFn' is a helper that loads data from
///     the scratchpad array and reduces it with the input
///     operand.
///
///     These compiler generated functions hide address
///     calculation and alignment information from the runtime.
/// 5. if ret == 1:
///     The team master of the last team stores the reduced
///     result to the globals in memory.
///     foo += reduceData.foo; bar *= reduceData.bar
///
///
/// Warp Reduction Algorithms
///
/// On the warp level, we have three algorithms implemented in the
/// OpenMP runtime depending on the number of active lanes:
///
/// Full Warp Reduction
///
/// The reduce algorithm within a warp where all lanes are active
/// is implemented in the runtime as follows:
///
/// full_warp_reduce(void *reduce_data,
///                  kmp_ShuffleReductFctPtr ShuffleReduceFn) {
///   for (int offset = WARPSIZE/2; offset > 0; offset /= 2)
///     ShuffleReduceFn(reduce_data, 0, offset, 0);
/// }
///
/// The algorithm completes in log(2, WARPSIZE) steps.
///
/// 'ShuffleReduceFn' is used here with lane_id set to 0 because it is
/// not used therefore we save instructions by not retrieving lane_id
/// from the corresponding special registers.  The 4th parameter, which
/// represents the version of the algorithm being used, is set to 0 to
/// signify full warp reduction.
///
/// In this version, 'ShuffleReduceFn' behaves, per element, as follows:
///
/// #reduce_elem refers to an element in the local lane's data structure
/// #remote_elem is retrieved from a remote lane
/// remote_elem = shuffle_down(reduce_elem, offset, WARPSIZE);
/// reduce_elem = reduce_elem REDUCE_OP remote_elem;
///
/// Contiguous Partial Warp Reduction
///
/// This reduce algorithm is used within a warp where only the first
/// 'n' (n <= WARPSIZE) lanes are active.  It is typically used when the
/// number of OpenMP threads in a parallel region is not a multiple of
/// WARPSIZE.  The algorithm is implemented in the runtime as follows:
///
/// void
/// contiguous_partial_reduce(void *reduce_data,
///                           kmp_ShuffleReductFctPtr ShuffleReduceFn,
///                           int size, int lane_id) {
///   int curr_size;
///   int offset;
///   curr_size = size;
///   mask = curr_size/2;
///   while (offset>0) {
///     ShuffleReduceFn(reduce_data, lane_id, offset, 1);
///     curr_size = (curr_size+1)/2;
///     offset = curr_size/2;
///   }
/// }
///
/// In this version, 'ShuffleReduceFn' behaves, per element, as follows:
///
/// remote_elem = shuffle_down(reduce_elem, offset, WARPSIZE);
/// if (lane_id < offset)
///     reduce_elem = reduce_elem REDUCE_OP remote_elem
/// else
///     reduce_elem = remote_elem
///
/// This algorithm assumes that the data to be reduced are located in a
/// contiguous subset of lanes starting from the first.  When there is
/// an odd number of active lanes, the data in the last lane is not
/// aggregated with any other lane's dat but is instead copied over.
///
/// Dispersed Partial Warp Reduction
///
/// This algorithm is used within a warp when any discontiguous subset of
/// lanes are active.  It is used to implement the reduction operation
/// across lanes in an OpenMP simd region or in a nested parallel region.
///
/// void
/// dispersed_partial_reduce(void *reduce_data,
///                          kmp_ShuffleReductFctPtr ShuffleReduceFn) {
///   int size, remote_id;
///   int logical_lane_id = number_of_active_lanes_before_me() * 2;
///   do {
///       remote_id = next_active_lane_id_right_after_me();
///       # the above function returns 0 of no active lane
///       # is present right after the current lane.
///       size = number_of_active_lanes_in_this_warp();
///       logical_lane_id /= 2;
///       ShuffleReduceFn(reduce_data, logical_lane_id,
///                       remote_id-1-threadIdx.x, 2);
///   } while (logical_lane_id % 2 == 0 && size > 1);
/// }
///
/// There is no assumption made about the initial state of the reduction.
/// Any number of lanes (>=1) could be active at any position.  The reduction
/// result is returned in the first active lane.
///
/// In this version, 'ShuffleReduceFn' behaves, per element, as follows:
///
/// remote_elem = shuffle_down(reduce_elem, offset, WARPSIZE);
/// if (lane_id % 2 == 0 && offset > 0)
///     reduce_elem = reduce_elem REDUCE_OP remote_elem
/// else
///     reduce_elem = remote_elem
///
///
/// Intra-Team Reduction
///
/// This function, as implemented in the runtime call
/// '__kmpc_nvptx_parallel_reduce_nowait_v2', aggregates data across OpenMP
/// threads in a team.  It first reduces within a warp using the
/// aforementioned algorithms.  We then proceed to gather all such
/// reduced values at the first warp.
///
/// The runtime makes use of the function 'InterWarpCpyFn', which copies
/// data from each of the "warp master" (zeroth lane of each warp, where
/// warp-reduced data is held) to the zeroth warp.  This step reduces (in
/// a mathematical sense) the problem of reduction across warp masters in
/// a block to the problem of warp reduction.
///
///
/// Inter-Team Reduction
///
/// Once a team has reduced its data to a single value, it is stored in
/// a global scratchpad array.  Since each team has a distinct slot, this
/// can be done without locking.
///
/// The last team to write to the scratchpad array proceeds to reduce the
/// scratchpad array.  One or more workers in the last team use the helper
/// 'loadAndReduceDataFn' to load and reduce values from the array, i.e.,
/// the k'th worker reduces every k'th element.
///
/// Finally, a call is made to '__kmpc_nvptx_parallel_reduce_nowait_v2' to
/// reduce across workers and compute a globally reduced value.
///
void CGOpenMPRuntimeGPU::emitReduction(
    CodeGenFunction &CGF, SourceLocation Loc, ArrayRef<const Expr *> Privates,
    ArrayRef<const Expr *> LHSExprs, ArrayRef<const Expr *> RHSExprs,
    ArrayRef<const Expr *> ReductionOps, ReductionOptionsTy Options) {
  if (!CGF.HaveInsertPoint())
    return;

  bool ParallelReduction = isOpenMPParallelDirective(Options.ReductionKind);
#ifndef NDEBUG
  bool TeamsReduction = isOpenMPTeamsDirective(Options.ReductionKind);
#endif

  if (Options.SimpleReduction) {
    assert(!TeamsReduction && !ParallelReduction &&
           "Invalid reduction selection in emitReduction.");
    CGOpenMPRuntime::emitReduction(CGF, Loc, Privates, LHSExprs, RHSExprs,
                                   ReductionOps, Options);
    return;
  }

  assert((TeamsReduction || ParallelReduction) &&
         "Invalid reduction selection in emitReduction.");

  // Build res = __kmpc_reduce{_nowait}(<gtid>, <n>, sizeof(RedList),
  // RedList, shuffle_reduce_func, interwarp_copy_func);
  // or
  // Build res = __kmpc_reduce_teams_nowait_simple(<loc>, <gtid>, <lck>);
  llvm::Value *RTLoc = emitUpdateLocation(CGF, Loc);
  llvm::Value *ThreadId = getThreadID(CGF, Loc);

  llvm::Value *Res;
  ASTContext &C = CGM.getContext();
  // 1. Build a list of reduction variables.
  // void *RedList[<n>] = {<ReductionVars>[0], ..., <ReductionVars>[<n>-1]};
  auto Size = RHSExprs.size();
  for (const Expr *E : Privates) {
    if (E->getType()->isVariablyModifiedType())
      // Reserve place for array size.
      ++Size;
  }
  llvm::APInt ArraySize(/*unsigned int numBits=*/32, Size);
  QualType ReductionArrayTy =
      C.getConstantArrayType(C.VoidPtrTy, ArraySize, nullptr, ArrayType::Normal,
                             /*IndexTypeQuals=*/0);
  Address ReductionList =
      CGF.CreateMemTemp(ReductionArrayTy, ".omp.reduction.red_list");
  auto IPriv = Privates.begin();
  unsigned Idx = 0;
  for (unsigned I = 0, E = RHSExprs.size(); I < E; ++I, ++IPriv, ++Idx) {
    Address Elem = CGF.Builder.CreateConstArrayGEP(ReductionList, Idx);
    CGF.Builder.CreateStore(
        CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
            CGF.EmitLValue(RHSExprs[I]).getPointer(CGF), CGF.VoidPtrTy),
        Elem);
    if ((*IPriv)->getType()->isVariablyModifiedType()) {
      // Store array size.
      ++Idx;
      Elem = CGF.Builder.CreateConstArrayGEP(ReductionList, Idx);
      llvm::Value *Size = CGF.Builder.CreateIntCast(
          CGF.getVLASize(
                 CGF.getContext().getAsVariableArrayType((*IPriv)->getType()))
              .NumElts,
          CGF.SizeTy, /*isSigned=*/false);
      CGF.Builder.CreateStore(CGF.Builder.CreateIntToPtr(Size, CGF.VoidPtrTy),
                              Elem);
    }
  }

  llvm::Value *RL = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      ReductionList.getPointer(), CGF.VoidPtrTy);
  llvm::Function *ReductionFn = emitReductionFunction(
      CGF.CurFn->getName(), Loc, CGF.ConvertTypeForMem(ReductionArrayTy),
      Privates, LHSExprs, RHSExprs, ReductionOps);
  llvm::Value *ReductionArrayTySize = CGF.getTypeSize(ReductionArrayTy);
  llvm::Function *ShuffleAndReduceFn = emitShuffleAndReduceFunction(
      CGM, Privates, ReductionArrayTy, ReductionFn, Loc);
  llvm::Value *InterWarpCopyFn =
      emitInterWarpCopyFunction(CGM, Privates, ReductionArrayTy, Loc);

  if (ParallelReduction) {
    llvm::Value *Args[] = {RTLoc,
                           ThreadId,
                           CGF.Builder.getInt32(RHSExprs.size()),
                           ReductionArrayTySize,
                           RL,
                           ShuffleAndReduceFn,
                           InterWarpCopyFn};

    Res = CGF.EmitRuntimeCall(
        OMPBuilder.getOrCreateRuntimeFunction(
            CGM.getModule(), OMPRTL___kmpc_nvptx_parallel_reduce_nowait_v2),
        Args);
  } else {
    assert(TeamsReduction && "expected teams reduction.");
    llvm::SmallDenseMap<const ValueDecl *, const FieldDecl *> VarFieldMap;
    llvm::SmallVector<const ValueDecl *, 4> PrivatesReductions(Privates.size());
    int Cnt = 0;
    for (const Expr *DRE : Privates) {
      PrivatesReductions[Cnt] = cast<DeclRefExpr>(DRE)->getDecl();
      ++Cnt;
    }
    const RecordDecl *TeamReductionRec = ::buildRecordForGlobalizedVars(
        CGM.getContext(), PrivatesReductions, std::nullopt, VarFieldMap,
        C.getLangOpts().OpenMPCUDAReductionBufNum);
    TeamsReductions.push_back(TeamReductionRec);
    if (!KernelTeamsReductionPtr) {
      KernelTeamsReductionPtr = new llvm::GlobalVariable(
          CGM.getModule(), CGM.VoidPtrTy, /*isConstant=*/true,
          llvm::GlobalValue::InternalLinkage, nullptr,
          "_openmp_teams_reductions_buffer_$_$ptr");
    }
    llvm::Value *GlobalBufferPtr = CGF.EmitLoadOfScalar(
        Address(KernelTeamsReductionPtr, CGF.VoidPtrTy, CGM.getPointerAlign()),
        /*Volatile=*/false, C.getPointerType(C.VoidPtrTy), Loc);
    llvm::Value *GlobalToBufferCpyFn = ::emitListToGlobalCopyFunction(
        CGM, Privates, ReductionArrayTy, Loc, TeamReductionRec, VarFieldMap);
    llvm::Value *GlobalToBufferRedFn = ::emitListToGlobalReduceFunction(
        CGM, Privates, ReductionArrayTy, Loc, TeamReductionRec, VarFieldMap,
        ReductionFn);
    llvm::Value *BufferToGlobalCpyFn = ::emitGlobalToListCopyFunction(
        CGM, Privates, ReductionArrayTy, Loc, TeamReductionRec, VarFieldMap);
    llvm::Value *BufferToGlobalRedFn = ::emitGlobalToListReduceFunction(
        CGM, Privates, ReductionArrayTy, Loc, TeamReductionRec, VarFieldMap,
        ReductionFn);

    llvm::Value *Args[] = {
        RTLoc,
        ThreadId,
        GlobalBufferPtr,
        CGF.Builder.getInt32(C.getLangOpts().OpenMPCUDAReductionBufNum),
        RL,
        ShuffleAndReduceFn,
        InterWarpCopyFn,
        GlobalToBufferCpyFn,
        GlobalToBufferRedFn,
        BufferToGlobalCpyFn,
        BufferToGlobalRedFn};

    Res = CGF.EmitRuntimeCall(
        OMPBuilder.getOrCreateRuntimeFunction(
            CGM.getModule(), OMPRTL___kmpc_nvptx_teams_reduce_nowait_v2),
        Args);
  }

  // 5. Build if (res == 1)
  llvm::BasicBlock *ExitBB = CGF.createBasicBlock(".omp.reduction.done");
  llvm::BasicBlock *ThenBB = CGF.createBasicBlock(".omp.reduction.then");
  llvm::Value *Cond = CGF.Builder.CreateICmpEQ(
      Res, llvm::ConstantInt::get(CGM.Int32Ty, /*V=*/1));
  CGF.Builder.CreateCondBr(Cond, ThenBB, ExitBB);

  // 6. Build then branch: where we have reduced values in the master
  //    thread in each team.
  //    __kmpc_end_reduce{_nowait}(<gtid>);
  //    break;
  CGF.EmitBlock(ThenBB);

  // Add emission of __kmpc_end_reduce{_nowait}(<gtid>);
  auto &&CodeGen = [Privates, LHSExprs, RHSExprs, ReductionOps,
                    this](CodeGenFunction &CGF, PrePostActionTy &Action) {
    auto IPriv = Privates.begin();
    auto ILHS = LHSExprs.begin();
    auto IRHS = RHSExprs.begin();
    for (const Expr *E : ReductionOps) {
      emitSingleReductionCombiner(CGF, E, *IPriv, cast<DeclRefExpr>(*ILHS),
                                  cast<DeclRefExpr>(*IRHS));
      ++IPriv;
      ++ILHS;
      ++IRHS;
    }
  };
  llvm::Value *EndArgs[] = {ThreadId};
  RegionCodeGenTy RCG(CodeGen);
  NVPTXActionTy Action(
      nullptr, std::nullopt,
      OMPBuilder.getOrCreateRuntimeFunction(
          CGM.getModule(), OMPRTL___kmpc_nvptx_end_reduce_nowait),
      EndArgs);
  RCG.setAction(Action);
  RCG(CGF);
  // There is no need to emit line number for unconditional branch.
  (void)ApplyDebugLocation::CreateEmpty(CGF);
  CGF.EmitBlock(ExitBB, /*IsFinished=*/true);
}

const VarDecl *
CGOpenMPRuntimeGPU::translateParameter(const FieldDecl *FD,
                                       const VarDecl *NativeParam) const {
  if (!NativeParam->getType()->isReferenceType())
    return NativeParam;
  QualType ArgType = NativeParam->getType();
  QualifierCollector QC;
  const Type *NonQualTy = QC.strip(ArgType);
  QualType PointeeTy = cast<ReferenceType>(NonQualTy)->getPointeeType();
  if (const auto *Attr = FD->getAttr<OMPCaptureKindAttr>()) {
    if (Attr->getCaptureKind() == OMPC_map) {
      PointeeTy = CGM.getContext().getAddrSpaceQualType(PointeeTy,
                                                        LangAS::opencl_global);
    }
  }
  ArgType = CGM.getContext().getPointerType(PointeeTy);
  QC.addRestrict();
  enum { NVPTX_local_addr = 5 };
  QC.addAddressSpace(getLangASFromTargetAS(NVPTX_local_addr));
  ArgType = QC.apply(CGM.getContext(), ArgType);
  if (isa<ImplicitParamDecl>(NativeParam))
    return ImplicitParamDecl::Create(
        CGM.getContext(), /*DC=*/nullptr, NativeParam->getLocation(),
        NativeParam->getIdentifier(), ArgType, ImplicitParamDecl::Other);
  return ParmVarDecl::Create(
      CGM.getContext(),
      const_cast<DeclContext *>(NativeParam->getDeclContext()),
      NativeParam->getBeginLoc(), NativeParam->getLocation(),
      NativeParam->getIdentifier(), ArgType,
      /*TInfo=*/nullptr, SC_None, /*DefArg=*/nullptr);
}

Address
CGOpenMPRuntimeGPU::getParameterAddress(CodeGenFunction &CGF,
                                          const VarDecl *NativeParam,
                                          const VarDecl *TargetParam) const {
  assert(NativeParam != TargetParam &&
         NativeParam->getType()->isReferenceType() &&
         "Native arg must not be the same as target arg.");
  Address LocalAddr = CGF.GetAddrOfLocalVar(TargetParam);
  QualType NativeParamType = NativeParam->getType();
  QualifierCollector QC;
  const Type *NonQualTy = QC.strip(NativeParamType);
  QualType NativePointeeTy = cast<ReferenceType>(NonQualTy)->getPointeeType();
  unsigned NativePointeeAddrSpace =
      CGF.getTypes().getTargetAddressSpace(NativePointeeTy);
  QualType TargetTy = TargetParam->getType();
  llvm::Value *TargetAddr = CGF.EmitLoadOfScalar(LocalAddr, /*Volatile=*/false,
                                                 TargetTy, SourceLocation());
  // First cast to generic.
  TargetAddr = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      TargetAddr,
      llvm::PointerType::get(CGF.getLLVMContext(), /*AddrSpace=*/0));
  // Cast from generic to native address space.
  TargetAddr = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      TargetAddr,
      llvm::PointerType::get(CGF.getLLVMContext(), NativePointeeAddrSpace));
  Address NativeParamAddr = CGF.CreateMemTemp(NativeParamType);
  CGF.EmitStoreOfScalar(TargetAddr, NativeParamAddr, /*Volatile=*/false,
                        NativeParamType);
  return NativeParamAddr;
}

void CGOpenMPRuntimeGPU::emitOutlinedFunctionCall(
    CodeGenFunction &CGF, SourceLocation Loc, llvm::FunctionCallee OutlinedFn,
    ArrayRef<llvm::Value *> Args) const {
  SmallVector<llvm::Value *, 4> TargetArgs;
  TargetArgs.reserve(Args.size());
  auto *FnType = OutlinedFn.getFunctionType();
  for (unsigned I = 0, E = Args.size(); I < E; ++I) {
    if (FnType->isVarArg() && FnType->getNumParams() <= I) {
      TargetArgs.append(std::next(Args.begin(), I), Args.end());
      break;
    }
    llvm::Type *TargetType = FnType->getParamType(I);
    llvm::Value *NativeArg = Args[I];
    if (!TargetType->isPointerTy()) {
      TargetArgs.emplace_back(NativeArg);
      continue;
    }
    llvm::Value *TargetArg = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
        NativeArg,
        llvm::PointerType::get(CGF.getLLVMContext(), /*AddrSpace*/ 0));
    TargetArgs.emplace_back(
        CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(TargetArg, TargetType));
  }
  CGOpenMPRuntime::emitOutlinedFunctionCall(CGF, Loc, OutlinedFn, TargetArgs);
}

/// Emit function which wraps the outline parallel region
/// and controls the arguments which are passed to this function.
/// The wrapper ensures that the outlined function is called
/// with the correct arguments when data is shared.
llvm::Function *CGOpenMPRuntimeGPU::createParallelDataSharingWrapper(
    llvm::Function *OutlinedParallelFn, const OMPExecutableDirective &D) {
  ASTContext &Ctx = CGM.getContext();
  const auto &CS = *D.getCapturedStmt(OMPD_parallel);

  // Create a function that takes as argument the source thread.
  FunctionArgList WrapperArgs;
  QualType Int16QTy =
      Ctx.getIntTypeForBitwidth(/*DestWidth=*/16, /*Signed=*/false);
  QualType Int32QTy =
      Ctx.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/false);
  ImplicitParamDecl ParallelLevelArg(Ctx, /*DC=*/nullptr, D.getBeginLoc(),
                                     /*Id=*/nullptr, Int16QTy,
                                     ImplicitParamDecl::Other);
  ImplicitParamDecl WrapperArg(Ctx, /*DC=*/nullptr, D.getBeginLoc(),
                               /*Id=*/nullptr, Int32QTy,
                               ImplicitParamDecl::Other);
  WrapperArgs.emplace_back(&ParallelLevelArg);
  WrapperArgs.emplace_back(&WrapperArg);

  const CGFunctionInfo &CGFI =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(Ctx.VoidTy, WrapperArgs);

  auto *Fn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(CGFI), llvm::GlobalValue::InternalLinkage,
      Twine(OutlinedParallelFn->getName(), "_wrapper"), &CGM.getModule());

  // Ensure we do not inline the function. This is trivially true for the ones
  // passed to __kmpc_fork_call but the ones calles in serialized regions
  // could be inlined. This is not a perfect but it is closer to the invariant
  // we want, namely, every data environment starts with a new function.
  // TODO: We should pass the if condition to the runtime function and do the
  //       handling there. Much cleaner code.
  Fn->addFnAttr(llvm::Attribute::NoInline);

  CGM.SetInternalFunctionAttributes(GlobalDecl(), Fn, CGFI);
  Fn->setLinkage(llvm::GlobalValue::InternalLinkage);
  Fn->setDoesNotRecurse();

  CodeGenFunction CGF(CGM, /*suppressNewContext=*/true);
  CGF.StartFunction(GlobalDecl(), Ctx.VoidTy, Fn, CGFI, WrapperArgs,
                    D.getBeginLoc(), D.getBeginLoc());

  const auto *RD = CS.getCapturedRecordDecl();
  auto CurField = RD->field_begin();

  Address ZeroAddr = CGF.CreateDefaultAlignTempAlloca(CGF.Int32Ty,
                                                      /*Name=*/".zero.addr");
  CGF.Builder.CreateStore(CGF.Builder.getInt32(/*C*/ 0), ZeroAddr);
  // Get the array of arguments.
  SmallVector<llvm::Value *, 8> Args;

  Args.emplace_back(CGF.GetAddrOfLocalVar(&WrapperArg).getPointer());
  Args.emplace_back(ZeroAddr.getPointer());

  CGBuilderTy &Bld = CGF.Builder;
  auto CI = CS.capture_begin();

  // Use global memory for data sharing.
  // Handle passing of global args to workers.
  Address GlobalArgs =
      CGF.CreateDefaultAlignTempAlloca(CGF.VoidPtrPtrTy, "global_args");
  llvm::Value *GlobalArgsPtr = GlobalArgs.getPointer();
  llvm::Value *DataSharingArgs[] = {GlobalArgsPtr};
  CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                          CGM.getModule(), OMPRTL___kmpc_get_shared_variables),
                      DataSharingArgs);

  // Retrieve the shared variables from the list of references returned
  // by the runtime. Pass the variables to the outlined function.
  Address SharedArgListAddress = Address::invalid();
  if (CS.capture_size() > 0 ||
      isOpenMPLoopBoundSharingDirective(D.getDirectiveKind())) {
    SharedArgListAddress = CGF.EmitLoadOfPointer(
        GlobalArgs, CGF.getContext()
                        .getPointerType(CGF.getContext().VoidPtrTy)
                        .castAs<PointerType>());
  }
  unsigned Idx = 0;
  if (isOpenMPLoopBoundSharingDirective(D.getDirectiveKind())) {
    Address Src = Bld.CreateConstInBoundsGEP(SharedArgListAddress, Idx);
    Address TypedAddress = Bld.CreatePointerBitCastOrAddrSpaceCast(
        Src, CGF.SizeTy->getPointerTo(), CGF.SizeTy);
    llvm::Value *LB = CGF.EmitLoadOfScalar(
        TypedAddress,
        /*Volatile=*/false,
        CGF.getContext().getPointerType(CGF.getContext().getSizeType()),
        cast<OMPLoopDirective>(D).getLowerBoundVariable()->getExprLoc());
    Args.emplace_back(LB);
    ++Idx;
    Src = Bld.CreateConstInBoundsGEP(SharedArgListAddress, Idx);
    TypedAddress = Bld.CreatePointerBitCastOrAddrSpaceCast(
        Src, CGF.SizeTy->getPointerTo(), CGF.SizeTy);
    llvm::Value *UB = CGF.EmitLoadOfScalar(
        TypedAddress,
        /*Volatile=*/false,
        CGF.getContext().getPointerType(CGF.getContext().getSizeType()),
        cast<OMPLoopDirective>(D).getUpperBoundVariable()->getExprLoc());
    Args.emplace_back(UB);
    ++Idx;
  }
  if (CS.capture_size() > 0) {
    ASTContext &CGFContext = CGF.getContext();
    for (unsigned I = 0, E = CS.capture_size(); I < E; ++I, ++CI, ++CurField) {
      QualType ElemTy = CurField->getType();
      Address Src = Bld.CreateConstInBoundsGEP(SharedArgListAddress, I + Idx);
      Address TypedAddress = Bld.CreatePointerBitCastOrAddrSpaceCast(
          Src, CGF.ConvertTypeForMem(CGFContext.getPointerType(ElemTy)),
          CGF.ConvertTypeForMem(ElemTy));
      llvm::Value *Arg = CGF.EmitLoadOfScalar(TypedAddress,
                                              /*Volatile=*/false,
                                              CGFContext.getPointerType(ElemTy),
                                              CI->getLocation());
      if (CI->capturesVariableByCopy() &&
          !CI->getCapturedVar()->getType()->isAnyPointerType()) {
        Arg = castValueToType(CGF, Arg, ElemTy, CGFContext.getUIntPtrType(),
                              CI->getLocation());
      }
      Args.emplace_back(Arg);
    }
  }

  emitOutlinedFunctionCall(CGF, D.getBeginLoc(), OutlinedParallelFn, Args);
  CGF.FinishFunction();
  return Fn;
}

void CGOpenMPRuntimeGPU::emitFunctionProlog(CodeGenFunction &CGF,
                                              const Decl *D) {
  if (getDataSharingMode() != CGOpenMPRuntimeGPU::DS_Generic)
    return;

  assert(D && "Expected function or captured|block decl.");
  assert(FunctionGlobalizedDecls.count(CGF.CurFn) == 0 &&
         "Function is registered already.");
  assert((!TeamAndReductions.first || TeamAndReductions.first == D) &&
         "Team is set but not processed.");
  const Stmt *Body = nullptr;
  bool NeedToDelayGlobalization = false;
  if (const auto *FD = dyn_cast<FunctionDecl>(D)) {
    Body = FD->getBody();
  } else if (const auto *BD = dyn_cast<BlockDecl>(D)) {
    Body = BD->getBody();
  } else if (const auto *CD = dyn_cast<CapturedDecl>(D)) {
    Body = CD->getBody();
    NeedToDelayGlobalization = CGF.CapturedStmtInfo->getKind() == CR_OpenMP;
    if (NeedToDelayGlobalization &&
        getExecutionMode() == CGOpenMPRuntimeGPU::EM_SPMD)
      return;
  }
  if (!Body)
    return;
  CheckVarsEscapingDeclContext VarChecker(CGF, TeamAndReductions.second);
  VarChecker.Visit(Body);
  const RecordDecl *GlobalizedVarsRecord =
      VarChecker.getGlobalizedRecord(IsInTTDRegion);
  TeamAndReductions.first = nullptr;
  TeamAndReductions.second.clear();
  ArrayRef<const ValueDecl *> EscapedVariableLengthDecls =
      VarChecker.getEscapedVariableLengthDecls();
  ArrayRef<const ValueDecl *> DelayedVariableLengthDecls =
      VarChecker.getDelayedVariableLengthDecls();
  if (!GlobalizedVarsRecord && EscapedVariableLengthDecls.empty() &&
      DelayedVariableLengthDecls.empty())
    return;
  auto I = FunctionGlobalizedDecls.try_emplace(CGF.CurFn).first;
  I->getSecond().MappedParams =
      std::make_unique<CodeGenFunction::OMPMapVars>();
  I->getSecond().EscapedParameters.insert(
      VarChecker.getEscapedParameters().begin(),
      VarChecker.getEscapedParameters().end());
  I->getSecond().EscapedVariableLengthDecls.append(
      EscapedVariableLengthDecls.begin(), EscapedVariableLengthDecls.end());
  I->getSecond().DelayedVariableLengthDecls.append(
      DelayedVariableLengthDecls.begin(), DelayedVariableLengthDecls.end());
  DeclToAddrMapTy &Data = I->getSecond().LocalVarData;
  for (const ValueDecl *VD : VarChecker.getEscapedDecls()) {
    assert(VD->isCanonicalDecl() && "Expected canonical declaration");
    Data.insert(std::make_pair(VD, MappedVarData()));
  }
  if (!NeedToDelayGlobalization) {
    emitGenericVarsProlog(CGF, D->getBeginLoc(), /*WithSPMDCheck=*/true);
    struct GlobalizationScope final : EHScopeStack::Cleanup {
      GlobalizationScope() = default;

      void Emit(CodeGenFunction &CGF, Flags flags) override {
        static_cast<CGOpenMPRuntimeGPU &>(CGF.CGM.getOpenMPRuntime())
            .emitGenericVarsEpilog(CGF, /*WithSPMDCheck=*/true);
      }
    };
    CGF.EHStack.pushCleanup<GlobalizationScope>(NormalAndEHCleanup);
  }
}

Address CGOpenMPRuntimeGPU::getAddressOfLocalVariable(CodeGenFunction &CGF,
                                                        const VarDecl *VD) {
  if (VD && VD->hasAttr<OMPAllocateDeclAttr>()) {
    const auto *A = VD->getAttr<OMPAllocateDeclAttr>();
    auto AS = LangAS::Default;
    switch (A->getAllocatorType()) {
      // Use the default allocator here as by default local vars are
      // threadlocal.
    case OMPAllocateDeclAttr::OMPNullMemAlloc:
    case OMPAllocateDeclAttr::OMPDefaultMemAlloc:
    case OMPAllocateDeclAttr::OMPThreadMemAlloc:
    case OMPAllocateDeclAttr::OMPHighBWMemAlloc:
    case OMPAllocateDeclAttr::OMPLowLatMemAlloc:
      // Follow the user decision - use default allocation.
      return Address::invalid();
    case OMPAllocateDeclAttr::OMPUserDefinedMemAlloc:
      // TODO: implement aupport for user-defined allocators.
      return Address::invalid();
    case OMPAllocateDeclAttr::OMPConstMemAlloc:
      AS = LangAS::cuda_constant;
      break;
    case OMPAllocateDeclAttr::OMPPTeamMemAlloc:
      AS = LangAS::cuda_shared;
      break;
    case OMPAllocateDeclAttr::OMPLargeCapMemAlloc:
    case OMPAllocateDeclAttr::OMPCGroupMemAlloc:
      break;
    }
    llvm::Type *VarTy = CGF.ConvertTypeForMem(VD->getType());
    auto *GV = new llvm::GlobalVariable(
        CGM.getModule(), VarTy, /*isConstant=*/false,
        llvm::GlobalValue::InternalLinkage, llvm::PoisonValue::get(VarTy),
        VD->getName(),
        /*InsertBefore=*/nullptr, llvm::GlobalValue::NotThreadLocal,
        CGM.getContext().getTargetAddressSpace(AS));
    CharUnits Align = CGM.getContext().getDeclAlign(VD);
    GV->setAlignment(Align.getAsAlign());
    return Address(
        CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
            GV, VarTy->getPointerTo(CGM.getContext().getTargetAddressSpace(
                    VD->getType().getAddressSpace()))),
        VarTy, Align);
  }

  if (getDataSharingMode() != CGOpenMPRuntimeGPU::DS_Generic)
    return Address::invalid();

  VD = VD->getCanonicalDecl();
  auto I = FunctionGlobalizedDecls.find(CGF.CurFn);
  if (I == FunctionGlobalizedDecls.end())
    return Address::invalid();
  auto VDI = I->getSecond().LocalVarData.find(VD);
  if (VDI != I->getSecond().LocalVarData.end())
    return VDI->second.PrivateAddr;
  if (VD->hasAttrs()) {
    for (specific_attr_iterator<OMPReferencedVarAttr> IT(VD->attr_begin()),
         E(VD->attr_end());
         IT != E; ++IT) {
      auto VDI = I->getSecond().LocalVarData.find(
          cast<VarDecl>(cast<DeclRefExpr>(IT->getRef())->getDecl())
              ->getCanonicalDecl());
      if (VDI != I->getSecond().LocalVarData.end())
        return VDI->second.PrivateAddr;
    }
  }

  return Address::invalid();
}

void CGOpenMPRuntimeGPU::functionFinished(CodeGenFunction &CGF) {
  FunctionGlobalizedDecls.erase(CGF.CurFn);
  CGOpenMPRuntime::functionFinished(CGF);
}

void CGOpenMPRuntimeGPU::getDefaultDistScheduleAndChunk(
    CodeGenFunction &CGF, const OMPLoopDirective &S,
    OpenMPDistScheduleClauseKind &ScheduleKind,
    llvm::Value *&Chunk) const {
  auto &RT = static_cast<CGOpenMPRuntimeGPU &>(CGF.CGM.getOpenMPRuntime());
  if (getExecutionMode() == CGOpenMPRuntimeGPU::EM_SPMD) {
    ScheduleKind = OMPC_DIST_SCHEDULE_static;
    Chunk = CGF.EmitScalarConversion(
        RT.getGPUNumThreads(CGF),
        CGF.getContext().getIntTypeForBitwidth(32, /*Signed=*/0),
        S.getIterationVariable()->getType(), S.getBeginLoc());
    return;
  }
  CGOpenMPRuntime::getDefaultDistScheduleAndChunk(
      CGF, S, ScheduleKind, Chunk);
}

void CGOpenMPRuntimeGPU::getDefaultScheduleAndChunk(
    CodeGenFunction &CGF, const OMPLoopDirective &S,
    OpenMPScheduleClauseKind &ScheduleKind,
    const Expr *&ChunkExpr) const {
  ScheduleKind = OMPC_SCHEDULE_static;
  // Chunk size is 1 in this case.
  llvm::APInt ChunkSize(32, 1);
  ChunkExpr = IntegerLiteral::Create(CGF.getContext(), ChunkSize,
      CGF.getContext().getIntTypeForBitwidth(32, /*Signed=*/0),
      SourceLocation());
}

void CGOpenMPRuntimeGPU::adjustTargetSpecificDataForLambdas(
    CodeGenFunction &CGF, const OMPExecutableDirective &D) const {
  assert(isOpenMPTargetExecutionDirective(D.getDirectiveKind()) &&
         " Expected target-based directive.");
  const CapturedStmt *CS = D.getCapturedStmt(OMPD_target);
  for (const CapturedStmt::Capture &C : CS->captures()) {
    // Capture variables captured by reference in lambdas for target-based
    // directives.
    if (!C.capturesVariable())
      continue;
    const VarDecl *VD = C.getCapturedVar();
    const auto *RD = VD->getType()
                         .getCanonicalType()
                         .getNonReferenceType()
                         ->getAsCXXRecordDecl();
    if (!RD || !RD->isLambda())
      continue;
    Address VDAddr = CGF.GetAddrOfLocalVar(VD);
    LValue VDLVal;
    if (VD->getType().getCanonicalType()->isReferenceType())
      VDLVal = CGF.EmitLoadOfReferenceLValue(VDAddr, VD->getType());
    else
      VDLVal = CGF.MakeAddrLValue(
          VDAddr, VD->getType().getCanonicalType().getNonReferenceType());
    llvm::DenseMap<const ValueDecl *, FieldDecl *> Captures;
    FieldDecl *ThisCapture = nullptr;
    RD->getCaptureFields(Captures, ThisCapture);
    if (ThisCapture && CGF.CapturedStmtInfo->isCXXThisExprCaptured()) {
      LValue ThisLVal =
          CGF.EmitLValueForFieldInitialization(VDLVal, ThisCapture);
      llvm::Value *CXXThis = CGF.LoadCXXThis();
      CGF.EmitStoreOfScalar(CXXThis, ThisLVal);
    }
    for (const LambdaCapture &LC : RD->captures()) {
      if (LC.getCaptureKind() != LCK_ByRef)
        continue;
      const ValueDecl *VD = LC.getCapturedVar();
      // FIXME: For now VD is always a VarDecl because OpenMP does not support
      //  capturing structured bindings in lambdas yet.
      if (!CS->capturesVariable(cast<VarDecl>(VD)))
        continue;
      auto It = Captures.find(VD);
      assert(It != Captures.end() && "Found lambda capture without field.");
      LValue VarLVal = CGF.EmitLValueForFieldInitialization(VDLVal, It->second);
      Address VDAddr = CGF.GetAddrOfLocalVar(cast<VarDecl>(VD));
      if (VD->getType().getCanonicalType()->isReferenceType())
        VDAddr = CGF.EmitLoadOfReferenceLValue(VDAddr,
                                               VD->getType().getCanonicalType())
                     .getAddress(CGF);
      CGF.EmitStoreOfScalar(VDAddr.getPointer(), VarLVal);
    }
  }
}

bool CGOpenMPRuntimeGPU::hasAllocateAttributeForGlobalVar(const VarDecl *VD,
                                                            LangAS &AS) {
  if (!VD || !VD->hasAttr<OMPAllocateDeclAttr>())
    return false;
  const auto *A = VD->getAttr<OMPAllocateDeclAttr>();
  switch(A->getAllocatorType()) {
  case OMPAllocateDeclAttr::OMPNullMemAlloc:
  case OMPAllocateDeclAttr::OMPDefaultMemAlloc:
  // Not supported, fallback to the default mem space.
  case OMPAllocateDeclAttr::OMPThreadMemAlloc:
  case OMPAllocateDeclAttr::OMPLargeCapMemAlloc:
  case OMPAllocateDeclAttr::OMPCGroupMemAlloc:
  case OMPAllocateDeclAttr::OMPHighBWMemAlloc:
  case OMPAllocateDeclAttr::OMPLowLatMemAlloc:
    AS = LangAS::Default;
    return true;
  case OMPAllocateDeclAttr::OMPConstMemAlloc:
    AS = LangAS::cuda_constant;
    return true;
  case OMPAllocateDeclAttr::OMPPTeamMemAlloc:
    AS = LangAS::cuda_shared;
    return true;
  case OMPAllocateDeclAttr::OMPUserDefinedMemAlloc:
    llvm_unreachable("Expected predefined allocator for the variables with the "
                     "static storage.");
  }
  return false;
}

// Get current CudaArch and ignore any unknown values
static CudaArch getCudaArch(CodeGenModule &CGM) {
  if (!CGM.getTarget().hasFeature("ptx"))
    return CudaArch::UNKNOWN;
  for (const auto &Feature : CGM.getTarget().getTargetOpts().FeatureMap) {
    if (Feature.getValue()) {
      CudaArch Arch = StringToCudaArch(Feature.getKey());
      if (Arch != CudaArch::UNKNOWN)
        return Arch;
    }
  }
  return CudaArch::UNKNOWN;
}

/// Check to see if target architecture supports unified addressing which is
/// a restriction for OpenMP requires clause "unified_shared_memory".
void CGOpenMPRuntimeGPU::processRequiresDirective(
    const OMPRequiresDecl *D) {
  for (const OMPClause *Clause : D->clauselists()) {
    if (Clause->getClauseKind() == OMPC_unified_shared_memory) {
      CudaArch Arch = getCudaArch(CGM);
      switch (Arch) {
      case CudaArch::SM_20:
      case CudaArch::SM_21:
      case CudaArch::SM_30:
      case CudaArch::SM_32:
      case CudaArch::SM_35:
      case CudaArch::SM_37:
      case CudaArch::SM_50:
      case CudaArch::SM_52:
      case CudaArch::SM_53: {
        SmallString<256> Buffer;
        llvm::raw_svector_ostream Out(Buffer);
        Out << "Target architecture " << CudaArchToString(Arch)
            << " does not support unified addressing";
        CGM.Error(Clause->getBeginLoc(), Out.str());
        return;
      }
      case CudaArch::SM_60:
      case CudaArch::SM_61:
      case CudaArch::SM_62:
      case CudaArch::SM_70:
      case CudaArch::SM_72:
      case CudaArch::SM_75:
      case CudaArch::SM_80:
      case CudaArch::SM_86:
      case CudaArch::SM_87:
      case CudaArch::SM_89:
      case CudaArch::SM_90:
      case CudaArch::GFX600:
      case CudaArch::GFX601:
      case CudaArch::GFX602:
      case CudaArch::GFX700:
      case CudaArch::GFX701:
      case CudaArch::GFX702:
      case CudaArch::GFX703:
      case CudaArch::GFX704:
      case CudaArch::GFX705:
      case CudaArch::GFX801:
      case CudaArch::GFX802:
      case CudaArch::GFX803:
      case CudaArch::GFX805:
      case CudaArch::GFX810:
      case CudaArch::GFX900:
      case CudaArch::GFX902:
      case CudaArch::GFX904:
      case CudaArch::GFX906:
      case CudaArch::GFX908:
      case CudaArch::GFX909:
      case CudaArch::GFX90a:
      case CudaArch::GFX90c:
      case CudaArch::GFX940:
      case CudaArch::GFX941:
      case CudaArch::GFX942:
      case CudaArch::GFX1010:
      case CudaArch::GFX1011:
      case CudaArch::GFX1012:
      case CudaArch::GFX1013:
      case CudaArch::GFX1030:
      case CudaArch::GFX1031:
      case CudaArch::GFX1032:
      case CudaArch::GFX1033:
      case CudaArch::GFX1034:
      case CudaArch::GFX1035:
      case CudaArch::GFX1036:
      case CudaArch::GFX1100:
      case CudaArch::GFX1101:
      case CudaArch::GFX1102:
      case CudaArch::GFX1103:
      case CudaArch::GFX1105:
      case CudaArch::GFX1150:
      case CudaArch::GFX1151:
      case CudaArch::GFX1200:
      case CudaArch::GFX1201:
      case CudaArch::GFX1210:
      case CudaArch::Generic:
      case CudaArch::UNUSED:
      case CudaArch::UNKNOWN:
        break;
      case CudaArch::LAST:
        llvm_unreachable("Unexpected Cuda arch.");
      }
    }
  }
  CGOpenMPRuntime::processRequiresDirective(D);
}

void CGOpenMPRuntimeGPU::clear() {

  if (!TeamsReductions.empty()) {
    ASTContext &C = CGM.getContext();
    RecordDecl *StaticRD = C.buildImplicitRecord(
        "_openmp_teams_reduction_type_$_", RecordDecl::TagKind::TTK_Union);
    StaticRD->startDefinition();
    for (const RecordDecl *TeamReductionRec : TeamsReductions) {
      QualType RecTy = C.getRecordType(TeamReductionRec);
      auto *Field = FieldDecl::Create(
          C, StaticRD, SourceLocation(), SourceLocation(), nullptr, RecTy,
          C.getTrivialTypeSourceInfo(RecTy, SourceLocation()),
          /*BW=*/nullptr, /*Mutable=*/false,
          /*InitStyle=*/ICIS_NoInit);
      Field->setAccess(AS_public);
      StaticRD->addDecl(Field);
    }
    StaticRD->completeDefinition();
    QualType StaticTy = C.getRecordType(StaticRD);
    llvm::Type *LLVMReductionsBufferTy =
        CGM.getTypes().ConvertTypeForMem(StaticTy);
    // FIXME: nvlink does not handle weak linkage correctly (object with the
    // different size are reported as erroneous).
    // Restore CommonLinkage as soon as nvlink is fixed.
    auto *GV = new llvm::GlobalVariable(
        CGM.getModule(), LLVMReductionsBufferTy,
        /*isConstant=*/false, llvm::GlobalValue::InternalLinkage,
        llvm::Constant::getNullValue(LLVMReductionsBufferTy),
        "_openmp_teams_reductions_buffer_$_");
    KernelTeamsReductionPtr->setInitializer(
        llvm::ConstantExpr::getPointerBitCastOrAddrSpaceCast(GV,
                                                             CGM.VoidPtrTy));
  }
  CGOpenMPRuntime::clear();
}

llvm::Value *CGOpenMPRuntimeGPU::getGPUNumThreads(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  llvm::Module *M = &CGF.CGM.getModule();
  const char *LocSize = "__kmpc_get_hardware_num_threads_in_block";
  llvm::Function *F = M->getFunction(LocSize);
  if (!F) {
    F = llvm::Function::Create(
        llvm::FunctionType::get(CGF.Int32Ty, std::nullopt, false),
        llvm::GlobalVariable::ExternalLinkage, LocSize, &CGF.CGM.getModule());
  }
  return Bld.CreateCall(F, std::nullopt, "nvptx_num_threads");
}

llvm::Value *CGOpenMPRuntimeGPU::getGPUThreadID(CodeGenFunction &CGF) {
  ArrayRef<llvm::Value *> Args{};
  return CGF.EmitRuntimeCall(
      OMPBuilder.getOrCreateRuntimeFunction(
          CGM.getModule(), OMPRTL___kmpc_get_hardware_thread_id_in_block),
      Args);
}

llvm::Value *CGOpenMPRuntimeGPU::getGPUWarpSize(CodeGenFunction &CGF) {
  ArrayRef<llvm::Value *> Args{};
  return CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                                 CGM.getModule(), OMPRTL___kmpc_get_warp_size),
                             Args);
}
