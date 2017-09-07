//===----- CGOpenMPRuntime.cpp - Interface to OpenMP Runtimes -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides a class for OpenMP runtime code generation.
//
//===----------------------------------------------------------------------===//

#include "CGCXXABI.h"
#include "CGCleanup.h"
#include "CGOpenMPRuntime.h"
#include "CodeGenFunction.h"
#include "clang/CodeGen/ConstantInitBuilder.h"
#include "clang/AST/Decl.h"
#include "clang/AST/StmtOpenMP.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace clang;
using namespace CodeGen;

namespace {
/// \brief Base class for handling code generation inside OpenMP regions.
class CGOpenMPRegionInfo : public CodeGenFunction::CGCapturedStmtInfo {
public:
  /// \brief Kinds of OpenMP regions used in codegen.
  enum CGOpenMPRegionKind {
    /// \brief Region with outlined function for standalone 'parallel'
    /// directive.
    ParallelOutlinedRegion,
    /// \brief Region with outlined function for standalone 'task' directive.
    TaskOutlinedRegion,
    /// \brief Region for constructs that do not require function outlining,
    /// like 'for', 'sections', 'atomic' etc. directives.
    InlinedRegion,
    /// \brief Region with outlined function for standalone 'target' directive.
    TargetRegion,
  };

  CGOpenMPRegionInfo(const CapturedStmt &CS,
                     const CGOpenMPRegionKind RegionKind,
                     const RegionCodeGenTy &CodeGen, OpenMPDirectiveKind Kind,
                     bool HasCancel)
      : CGCapturedStmtInfo(CS, CR_OpenMP), RegionKind(RegionKind),
        CodeGen(CodeGen), Kind(Kind), HasCancel(HasCancel) {}

  CGOpenMPRegionInfo(const CGOpenMPRegionKind RegionKind,
                     const RegionCodeGenTy &CodeGen, OpenMPDirectiveKind Kind,
                     bool HasCancel)
      : CGCapturedStmtInfo(CR_OpenMP), RegionKind(RegionKind), CodeGen(CodeGen),
        Kind(Kind), HasCancel(HasCancel) {}

  /// \brief Get a variable or parameter for storing global thread id
  /// inside OpenMP construct.
  virtual const VarDecl *getThreadIDVariable() const = 0;

  /// \brief Emit the captured statement body.
  void EmitBody(CodeGenFunction &CGF, const Stmt *S) override;

  /// \brief Get an LValue for the current ThreadID variable.
  /// \return LValue for thread id variable. This LValue always has type int32*.
  virtual LValue getThreadIDVariableLValue(CodeGenFunction &CGF);

  virtual void emitUntiedSwitch(CodeGenFunction & /*CGF*/) {}

  CGOpenMPRegionKind getRegionKind() const { return RegionKind; }

  OpenMPDirectiveKind getDirectiveKind() const { return Kind; }

  bool hasCancel() const { return HasCancel; }

  static bool classof(const CGCapturedStmtInfo *Info) {
    return Info->getKind() == CR_OpenMP;
  }

  ~CGOpenMPRegionInfo() override = default;

protected:
  CGOpenMPRegionKind RegionKind;
  RegionCodeGenTy CodeGen;
  OpenMPDirectiveKind Kind;
  bool HasCancel;
};

/// \brief API for captured statement code generation in OpenMP constructs.
class CGOpenMPOutlinedRegionInfo final : public CGOpenMPRegionInfo {
public:
  CGOpenMPOutlinedRegionInfo(const CapturedStmt &CS, const VarDecl *ThreadIDVar,
                             const RegionCodeGenTy &CodeGen,
                             OpenMPDirectiveKind Kind, bool HasCancel,
                             StringRef HelperName)
      : CGOpenMPRegionInfo(CS, ParallelOutlinedRegion, CodeGen, Kind,
                           HasCancel),
        ThreadIDVar(ThreadIDVar), HelperName(HelperName) {
    assert(ThreadIDVar != nullptr && "No ThreadID in OpenMP region.");
  }

  /// \brief Get a variable or parameter for storing global thread id
  /// inside OpenMP construct.
  const VarDecl *getThreadIDVariable() const override { return ThreadIDVar; }

  /// \brief Get the name of the capture helper.
  StringRef getHelperName() const override { return HelperName; }

  static bool classof(const CGCapturedStmtInfo *Info) {
    return CGOpenMPRegionInfo::classof(Info) &&
           cast<CGOpenMPRegionInfo>(Info)->getRegionKind() ==
               ParallelOutlinedRegion;
  }

private:
  /// \brief A variable or parameter storing global thread id for OpenMP
  /// constructs.
  const VarDecl *ThreadIDVar;
  StringRef HelperName;
};

/// \brief API for captured statement code generation in OpenMP constructs.
class CGOpenMPTaskOutlinedRegionInfo final : public CGOpenMPRegionInfo {
public:
  class UntiedTaskActionTy final : public PrePostActionTy {
    bool Untied;
    const VarDecl *PartIDVar;
    const RegionCodeGenTy UntiedCodeGen;
    llvm::SwitchInst *UntiedSwitch = nullptr;

  public:
    UntiedTaskActionTy(bool Tied, const VarDecl *PartIDVar,
                       const RegionCodeGenTy &UntiedCodeGen)
        : Untied(!Tied), PartIDVar(PartIDVar), UntiedCodeGen(UntiedCodeGen) {}
    void Enter(CodeGenFunction &CGF) override {
      if (Untied) {
        // Emit task switching point.
        auto PartIdLVal = CGF.EmitLoadOfPointerLValue(
            CGF.GetAddrOfLocalVar(PartIDVar),
            PartIDVar->getType()->castAs<PointerType>());
        auto *Res = CGF.EmitLoadOfScalar(PartIdLVal, SourceLocation());
        auto *DoneBB = CGF.createBasicBlock(".untied.done.");
        UntiedSwitch = CGF.Builder.CreateSwitch(Res, DoneBB);
        CGF.EmitBlock(DoneBB);
        CGF.EmitBranchThroughCleanup(CGF.ReturnBlock);
        CGF.EmitBlock(CGF.createBasicBlock(".untied.jmp."));
        UntiedSwitch->addCase(CGF.Builder.getInt32(0),
                              CGF.Builder.GetInsertBlock());
        emitUntiedSwitch(CGF);
      }
    }
    void emitUntiedSwitch(CodeGenFunction &CGF) const {
      if (Untied) {
        auto PartIdLVal = CGF.EmitLoadOfPointerLValue(
            CGF.GetAddrOfLocalVar(PartIDVar),
            PartIDVar->getType()->castAs<PointerType>());
        CGF.EmitStoreOfScalar(CGF.Builder.getInt32(UntiedSwitch->getNumCases()),
                              PartIdLVal);
        UntiedCodeGen(CGF);
        CodeGenFunction::JumpDest CurPoint =
            CGF.getJumpDestInCurrentScope(".untied.next.");
        CGF.EmitBranchThroughCleanup(CGF.ReturnBlock);
        CGF.EmitBlock(CGF.createBasicBlock(".untied.jmp."));
        UntiedSwitch->addCase(CGF.Builder.getInt32(UntiedSwitch->getNumCases()),
                              CGF.Builder.GetInsertBlock());
        CGF.EmitBranchThroughCleanup(CurPoint);
        CGF.EmitBlock(CurPoint.getBlock());
      }
    }
    unsigned getNumberOfParts() const { return UntiedSwitch->getNumCases(); }
  };
  CGOpenMPTaskOutlinedRegionInfo(const CapturedStmt &CS,
                                 const VarDecl *ThreadIDVar,
                                 const RegionCodeGenTy &CodeGen,
                                 OpenMPDirectiveKind Kind, bool HasCancel,
                                 const UntiedTaskActionTy &Action)
      : CGOpenMPRegionInfo(CS, TaskOutlinedRegion, CodeGen, Kind, HasCancel),
        ThreadIDVar(ThreadIDVar), Action(Action) {
    assert(ThreadIDVar != nullptr && "No ThreadID in OpenMP region.");
  }

  /// \brief Get a variable or parameter for storing global thread id
  /// inside OpenMP construct.
  const VarDecl *getThreadIDVariable() const override { return ThreadIDVar; }

  /// \brief Get an LValue for the current ThreadID variable.
  LValue getThreadIDVariableLValue(CodeGenFunction &CGF) override;

  /// \brief Get the name of the capture helper.
  StringRef getHelperName() const override { return ".omp_outlined."; }

  void emitUntiedSwitch(CodeGenFunction &CGF) override {
    Action.emitUntiedSwitch(CGF);
  }

  static bool classof(const CGCapturedStmtInfo *Info) {
    return CGOpenMPRegionInfo::classof(Info) &&
           cast<CGOpenMPRegionInfo>(Info)->getRegionKind() ==
               TaskOutlinedRegion;
  }

private:
  /// \brief A variable or parameter storing global thread id for OpenMP
  /// constructs.
  const VarDecl *ThreadIDVar;
  /// Action for emitting code for untied tasks.
  const UntiedTaskActionTy &Action;
};

/// \brief API for inlined captured statement code generation in OpenMP
/// constructs.
class CGOpenMPInlinedRegionInfo : public CGOpenMPRegionInfo {
public:
  CGOpenMPInlinedRegionInfo(CodeGenFunction::CGCapturedStmtInfo *OldCSI,
                            const RegionCodeGenTy &CodeGen,
                            OpenMPDirectiveKind Kind, bool HasCancel)
      : CGOpenMPRegionInfo(InlinedRegion, CodeGen, Kind, HasCancel),
        OldCSI(OldCSI),
        OuterRegionInfo(dyn_cast_or_null<CGOpenMPRegionInfo>(OldCSI)) {}

  // \brief Retrieve the value of the context parameter.
  llvm::Value *getContextValue() const override {
    if (OuterRegionInfo)
      return OuterRegionInfo->getContextValue();
    llvm_unreachable("No context value for inlined OpenMP region");
  }

  void setContextValue(llvm::Value *V) override {
    if (OuterRegionInfo) {
      OuterRegionInfo->setContextValue(V);
      return;
    }
    llvm_unreachable("No context value for inlined OpenMP region");
  }

  /// \brief Lookup the captured field decl for a variable.
  const FieldDecl *lookup(const VarDecl *VD) const override {
    if (OuterRegionInfo)
      return OuterRegionInfo->lookup(VD);
    // If there is no outer outlined region,no need to lookup in a list of
    // captured variables, we can use the original one.
    return nullptr;
  }

  FieldDecl *getThisFieldDecl() const override {
    if (OuterRegionInfo)
      return OuterRegionInfo->getThisFieldDecl();
    return nullptr;
  }

  /// \brief Get a variable or parameter for storing global thread id
  /// inside OpenMP construct.
  const VarDecl *getThreadIDVariable() const override {
    if (OuterRegionInfo)
      return OuterRegionInfo->getThreadIDVariable();
    return nullptr;
  }

  /// \brief Get the name of the capture helper.
  StringRef getHelperName() const override {
    if (auto *OuterRegionInfo = getOldCSI())
      return OuterRegionInfo->getHelperName();
    llvm_unreachable("No helper name for inlined OpenMP construct");
  }

  void emitUntiedSwitch(CodeGenFunction &CGF) override {
    if (OuterRegionInfo)
      OuterRegionInfo->emitUntiedSwitch(CGF);
  }

  CodeGenFunction::CGCapturedStmtInfo *getOldCSI() const { return OldCSI; }

  static bool classof(const CGCapturedStmtInfo *Info) {
    return CGOpenMPRegionInfo::classof(Info) &&
           cast<CGOpenMPRegionInfo>(Info)->getRegionKind() == InlinedRegion;
  }

  ~CGOpenMPInlinedRegionInfo() override = default;

private:
  /// \brief CodeGen info about outer OpenMP region.
  CodeGenFunction::CGCapturedStmtInfo *OldCSI;
  CGOpenMPRegionInfo *OuterRegionInfo;
};

/// \brief API for captured statement code generation in OpenMP target
/// constructs. For this captures, implicit parameters are used instead of the
/// captured fields. The name of the target region has to be unique in a given
/// application so it is provided by the client, because only the client has
/// the information to generate that.
class CGOpenMPTargetRegionInfo final : public CGOpenMPRegionInfo {
public:
  CGOpenMPTargetRegionInfo(const CapturedStmt &CS,
                           const RegionCodeGenTy &CodeGen, StringRef HelperName)
      : CGOpenMPRegionInfo(CS, TargetRegion, CodeGen, OMPD_target,
                           /*HasCancel=*/false),
        HelperName(HelperName) {}

  /// \brief This is unused for target regions because each starts executing
  /// with a single thread.
  const VarDecl *getThreadIDVariable() const override { return nullptr; }

  /// \brief Get the name of the capture helper.
  StringRef getHelperName() const override { return HelperName; }

  static bool classof(const CGCapturedStmtInfo *Info) {
    return CGOpenMPRegionInfo::classof(Info) &&
           cast<CGOpenMPRegionInfo>(Info)->getRegionKind() == TargetRegion;
  }

private:
  StringRef HelperName;
};

static void EmptyCodeGen(CodeGenFunction &, PrePostActionTy &) {
  llvm_unreachable("No codegen for expressions");
}
/// \brief API for generation of expressions captured in a innermost OpenMP
/// region.
class CGOpenMPInnerExprInfo final : public CGOpenMPInlinedRegionInfo {
public:
  CGOpenMPInnerExprInfo(CodeGenFunction &CGF, const CapturedStmt &CS)
      : CGOpenMPInlinedRegionInfo(CGF.CapturedStmtInfo, EmptyCodeGen,
                                  OMPD_unknown,
                                  /*HasCancel=*/false),
        PrivScope(CGF) {
    // Make sure the globals captured in the provided statement are local by
    // using the privatization logic. We assume the same variable is not
    // captured more than once.
    for (auto &C : CS.captures()) {
      if (!C.capturesVariable() && !C.capturesVariableByCopy())
        continue;

      const VarDecl *VD = C.getCapturedVar();
      if (VD->isLocalVarDeclOrParm())
        continue;

      DeclRefExpr DRE(const_cast<VarDecl *>(VD),
                      /*RefersToEnclosingVariableOrCapture=*/false,
                      VD->getType().getNonReferenceType(), VK_LValue,
                      SourceLocation());
      PrivScope.addPrivate(VD, [&CGF, &DRE]() -> Address {
        return CGF.EmitLValue(&DRE).getAddress();
      });
    }
    (void)PrivScope.Privatize();
  }

  /// \brief Lookup the captured field decl for a variable.
  const FieldDecl *lookup(const VarDecl *VD) const override {
    if (auto *FD = CGOpenMPInlinedRegionInfo::lookup(VD))
      return FD;
    return nullptr;
  }

  /// \brief Emit the captured statement body.
  void EmitBody(CodeGenFunction &CGF, const Stmt *S) override {
    llvm_unreachable("No body for expressions");
  }

  /// \brief Get a variable or parameter for storing global thread id
  /// inside OpenMP construct.
  const VarDecl *getThreadIDVariable() const override {
    llvm_unreachable("No thread id for expressions");
  }

  /// \brief Get the name of the capture helper.
  StringRef getHelperName() const override {
    llvm_unreachable("No helper name for expressions");
  }

  static bool classof(const CGCapturedStmtInfo *Info) { return false; }

private:
  /// Private scope to capture global variables.
  CodeGenFunction::OMPPrivateScope PrivScope;
};

/// \brief RAII for emitting code of OpenMP constructs.
class InlinedOpenMPRegionRAII {
  CodeGenFunction &CGF;
  llvm::DenseMap<const VarDecl *, FieldDecl *> LambdaCaptureFields;
  FieldDecl *LambdaThisCaptureField = nullptr;

public:
  /// \brief Constructs region for combined constructs.
  /// \param CodeGen Code generation sequence for combined directives. Includes
  /// a list of functions used for code generation of implicitly inlined
  /// regions.
  InlinedOpenMPRegionRAII(CodeGenFunction &CGF, const RegionCodeGenTy &CodeGen,
                          OpenMPDirectiveKind Kind, bool HasCancel)
      : CGF(CGF) {
    // Start emission for the construct.
    CGF.CapturedStmtInfo = new CGOpenMPInlinedRegionInfo(
        CGF.CapturedStmtInfo, CodeGen, Kind, HasCancel);
    std::swap(CGF.LambdaCaptureFields, LambdaCaptureFields);
    LambdaThisCaptureField = CGF.LambdaThisCaptureField;
    CGF.LambdaThisCaptureField = nullptr;
  }

  ~InlinedOpenMPRegionRAII() {
    // Restore original CapturedStmtInfo only if we're done with code emission.
    auto *OldCSI =
        cast<CGOpenMPInlinedRegionInfo>(CGF.CapturedStmtInfo)->getOldCSI();
    delete CGF.CapturedStmtInfo;
    CGF.CapturedStmtInfo = OldCSI;
    std::swap(CGF.LambdaCaptureFields, LambdaCaptureFields);
    CGF.LambdaThisCaptureField = LambdaThisCaptureField;
  }
};

/// \brief Values for bit flags used in the ident_t to describe the fields.
/// All enumeric elements are named and described in accordance with the code
/// from http://llvm.org/svn/llvm-project/openmp/trunk/runtime/src/kmp.h
enum OpenMPLocationFlags : unsigned {
  /// \brief Use trampoline for internal microtask.
  OMP_IDENT_IMD = 0x01,
  /// \brief Use c-style ident structure.
  OMP_IDENT_KMPC = 0x02,
  /// \brief Atomic reduction option for kmpc_reduce.
  OMP_ATOMIC_REDUCE = 0x10,
  /// \brief Explicit 'barrier' directive.
  OMP_IDENT_BARRIER_EXPL = 0x20,
  /// \brief Implicit barrier in code.
  OMP_IDENT_BARRIER_IMPL = 0x40,
  /// \brief Implicit barrier in 'for' directive.
  OMP_IDENT_BARRIER_IMPL_FOR = 0x40,
  /// \brief Implicit barrier in 'sections' directive.
  OMP_IDENT_BARRIER_IMPL_SECTIONS = 0xC0,
  /// \brief Implicit barrier in 'single' directive.
  OMP_IDENT_BARRIER_IMPL_SINGLE = 0x140,
  /// Call of __kmp_for_static_init for static loop.
  OMP_IDENT_WORK_LOOP = 0x200,
  /// Call of __kmp_for_static_init for sections.
  OMP_IDENT_WORK_SECTIONS = 0x400,
  /// Call of __kmp_for_static_init for distribute.
  OMP_IDENT_WORK_DISTRIBUTE = 0x800,
  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/OMP_IDENT_WORK_DISTRIBUTE)
};

/// \brief Describes ident structure that describes a source location.
/// All descriptions are taken from
/// http://llvm.org/svn/llvm-project/openmp/trunk/runtime/src/kmp.h
/// Original structure:
/// typedef struct ident {
///    kmp_int32 reserved_1;   /**<  might be used in Fortran;
///                                  see above  */
///    kmp_int32 flags;        /**<  also f.flags; KMP_IDENT_xxx flags;
///                                  KMP_IDENT_KMPC identifies this union
///                                  member  */
///    kmp_int32 reserved_2;   /**<  not really used in Fortran any more;
///                                  see above */
///#if USE_ITT_BUILD
///                            /*  but currently used for storing
///                                region-specific ITT */
///                            /*  contextual information. */
///#endif /* USE_ITT_BUILD */
///    kmp_int32 reserved_3;   /**< source[4] in Fortran, do not use for
///                                 C++  */
///    char const *psource;    /**< String describing the source location.
///                            The string is composed of semi-colon separated
//                             fields which describe the source file,
///                            the function and a pair of line numbers that
///                            delimit the construct.
///                             */
/// } ident_t;
enum IdentFieldIndex {
  /// \brief might be used in Fortran
  IdentField_Reserved_1,
  /// \brief OMP_IDENT_xxx flags; OMP_IDENT_KMPC identifies this union member.
  IdentField_Flags,
  /// \brief Not really used in Fortran any more
  IdentField_Reserved_2,
  /// \brief Source[4] in Fortran, do not use for C++
  IdentField_Reserved_3,
  /// \brief String describing the source location. The string is composed of
  /// semi-colon separated fields which describe the source file, the function
  /// and a pair of line numbers that delimit the construct.
  IdentField_PSource
};

/// \brief Schedule types for 'omp for' loops (these enumerators are taken from
/// the enum sched_type in kmp.h).
enum OpenMPSchedType {
  /// \brief Lower bound for default (unordered) versions.
  OMP_sch_lower = 32,
  OMP_sch_static_chunked = 33,
  OMP_sch_static = 34,
  OMP_sch_dynamic_chunked = 35,
  OMP_sch_guided_chunked = 36,
  OMP_sch_runtime = 37,
  OMP_sch_auto = 38,
  /// static with chunk adjustment (e.g., simd)
  OMP_sch_static_balanced_chunked = 45,
  /// \brief Lower bound for 'ordered' versions.
  OMP_ord_lower = 64,
  OMP_ord_static_chunked = 65,
  OMP_ord_static = 66,
  OMP_ord_dynamic_chunked = 67,
  OMP_ord_guided_chunked = 68,
  OMP_ord_runtime = 69,
  OMP_ord_auto = 70,
  OMP_sch_default = OMP_sch_static,
  /// \brief dist_schedule types
  OMP_dist_sch_static_chunked = 91,
  OMP_dist_sch_static = 92,
  /// Support for OpenMP 4.5 monotonic and nonmonotonic schedule modifiers.
  /// Set if the monotonic schedule modifier was present.
  OMP_sch_modifier_monotonic = (1 << 29),
  /// Set if the nonmonotonic schedule modifier was present.
  OMP_sch_modifier_nonmonotonic = (1 << 30),
};

enum OpenMPRTLFunction {
  /// \brief Call to void __kmpc_fork_call(ident_t *loc, kmp_int32 argc,
  /// kmpc_micro microtask, ...);
  OMPRTL__kmpc_fork_call,
  /// \brief Call to void *__kmpc_threadprivate_cached(ident_t *loc,
  /// kmp_int32 global_tid, void *data, size_t size, void ***cache);
  OMPRTL__kmpc_threadprivate_cached,
  /// \brief Call to void __kmpc_threadprivate_register( ident_t *,
  /// void *data, kmpc_ctor ctor, kmpc_cctor cctor, kmpc_dtor dtor);
  OMPRTL__kmpc_threadprivate_register,
  // Call to __kmpc_int32 kmpc_global_thread_num(ident_t *loc);
  OMPRTL__kmpc_global_thread_num,
  // Call to void __kmpc_critical(ident_t *loc, kmp_int32 global_tid,
  // kmp_critical_name *crit);
  OMPRTL__kmpc_critical,
  // Call to void __kmpc_critical_with_hint(ident_t *loc, kmp_int32
  // global_tid, kmp_critical_name *crit, uintptr_t hint);
  OMPRTL__kmpc_critical_with_hint,
  // Call to void __kmpc_end_critical(ident_t *loc, kmp_int32 global_tid,
  // kmp_critical_name *crit);
  OMPRTL__kmpc_end_critical,
  // Call to kmp_int32 __kmpc_cancel_barrier(ident_t *loc, kmp_int32
  // global_tid);
  OMPRTL__kmpc_cancel_barrier,
  // Call to void __kmpc_barrier(ident_t *loc, kmp_int32 global_tid);
  OMPRTL__kmpc_barrier,
  // Call to void __kmpc_for_static_fini(ident_t *loc, kmp_int32 global_tid);
  OMPRTL__kmpc_for_static_fini,
  // Call to void __kmpc_serialized_parallel(ident_t *loc, kmp_int32
  // global_tid);
  OMPRTL__kmpc_serialized_parallel,
  // Call to void __kmpc_end_serialized_parallel(ident_t *loc, kmp_int32
  // global_tid);
  OMPRTL__kmpc_end_serialized_parallel,
  // Call to void __kmpc_push_num_threads(ident_t *loc, kmp_int32 global_tid,
  // kmp_int32 num_threads);
  OMPRTL__kmpc_push_num_threads,
  // Call to void __kmpc_flush(ident_t *loc);
  OMPRTL__kmpc_flush,
  // Call to kmp_int32 __kmpc_master(ident_t *, kmp_int32 global_tid);
  OMPRTL__kmpc_master,
  // Call to void __kmpc_end_master(ident_t *, kmp_int32 global_tid);
  OMPRTL__kmpc_end_master,
  // Call to kmp_int32 __kmpc_omp_taskyield(ident_t *, kmp_int32 global_tid,
  // int end_part);
  OMPRTL__kmpc_omp_taskyield,
  // Call to kmp_int32 __kmpc_single(ident_t *, kmp_int32 global_tid);
  OMPRTL__kmpc_single,
  // Call to void __kmpc_end_single(ident_t *, kmp_int32 global_tid);
  OMPRTL__kmpc_end_single,
  // Call to kmp_task_t * __kmpc_omp_task_alloc(ident_t *, kmp_int32 gtid,
  // kmp_int32 flags, size_t sizeof_kmp_task_t, size_t sizeof_shareds,
  // kmp_routine_entry_t *task_entry);
  OMPRTL__kmpc_omp_task_alloc,
  // Call to kmp_int32 __kmpc_omp_task(ident_t *, kmp_int32 gtid, kmp_task_t *
  // new_task);
  OMPRTL__kmpc_omp_task,
  // Call to void __kmpc_copyprivate(ident_t *loc, kmp_int32 global_tid,
  // size_t cpy_size, void *cpy_data, void(*cpy_func)(void *, void *),
  // kmp_int32 didit);
  OMPRTL__kmpc_copyprivate,
  // Call to kmp_int32 __kmpc_reduce(ident_t *loc, kmp_int32 global_tid,
  // kmp_int32 num_vars, size_t reduce_size, void *reduce_data, void
  // (*reduce_func)(void *lhs_data, void *rhs_data), kmp_critical_name *lck);
  OMPRTL__kmpc_reduce,
  // Call to kmp_int32 __kmpc_reduce_nowait(ident_t *loc, kmp_int32
  // global_tid, kmp_int32 num_vars, size_t reduce_size, void *reduce_data,
  // void (*reduce_func)(void *lhs_data, void *rhs_data), kmp_critical_name
  // *lck);
  OMPRTL__kmpc_reduce_nowait,
  // Call to void __kmpc_end_reduce(ident_t *loc, kmp_int32 global_tid,
  // kmp_critical_name *lck);
  OMPRTL__kmpc_end_reduce,
  // Call to void __kmpc_end_reduce_nowait(ident_t *loc, kmp_int32 global_tid,
  // kmp_critical_name *lck);
  OMPRTL__kmpc_end_reduce_nowait,
  // Call to void __kmpc_omp_task_begin_if0(ident_t *, kmp_int32 gtid,
  // kmp_task_t * new_task);
  OMPRTL__kmpc_omp_task_begin_if0,
  // Call to void __kmpc_omp_task_complete_if0(ident_t *, kmp_int32 gtid,
  // kmp_task_t * new_task);
  OMPRTL__kmpc_omp_task_complete_if0,
  // Call to void __kmpc_ordered(ident_t *loc, kmp_int32 global_tid);
  OMPRTL__kmpc_ordered,
  // Call to void __kmpc_end_ordered(ident_t *loc, kmp_int32 global_tid);
  OMPRTL__kmpc_end_ordered,
  // Call to kmp_int32 __kmpc_omp_taskwait(ident_t *loc, kmp_int32
  // global_tid);
  OMPRTL__kmpc_omp_taskwait,
  // Call to void __kmpc_taskgroup(ident_t *loc, kmp_int32 global_tid);
  OMPRTL__kmpc_taskgroup,
  // Call to void __kmpc_end_taskgroup(ident_t *loc, kmp_int32 global_tid);
  OMPRTL__kmpc_end_taskgroup,
  // Call to void __kmpc_push_proc_bind(ident_t *loc, kmp_int32 global_tid,
  // int proc_bind);
  OMPRTL__kmpc_push_proc_bind,
  // Call to kmp_int32 __kmpc_omp_task_with_deps(ident_t *loc_ref, kmp_int32
  // gtid, kmp_task_t * new_task, kmp_int32 ndeps, kmp_depend_info_t
  // *dep_list, kmp_int32 ndeps_noalias, kmp_depend_info_t *noalias_dep_list);
  OMPRTL__kmpc_omp_task_with_deps,
  // Call to void __kmpc_omp_wait_deps(ident_t *loc_ref, kmp_int32
  // gtid, kmp_int32 ndeps, kmp_depend_info_t *dep_list, kmp_int32
  // ndeps_noalias, kmp_depend_info_t *noalias_dep_list);
  OMPRTL__kmpc_omp_wait_deps,
  // Call to kmp_int32 __kmpc_cancellationpoint(ident_t *loc, kmp_int32
  // global_tid, kmp_int32 cncl_kind);
  OMPRTL__kmpc_cancellationpoint,
  // Call to kmp_int32 __kmpc_cancel(ident_t *loc, kmp_int32 global_tid,
  // kmp_int32 cncl_kind);
  OMPRTL__kmpc_cancel,
  // Call to void __kmpc_push_num_teams(ident_t *loc, kmp_int32 global_tid,
  // kmp_int32 num_teams, kmp_int32 thread_limit);
  OMPRTL__kmpc_push_num_teams,
  // Call to void __kmpc_fork_teams(ident_t *loc, kmp_int32 argc, kmpc_micro
  // microtask, ...);
  OMPRTL__kmpc_fork_teams,
  // Call to void __kmpc_taskloop(ident_t *loc, int gtid, kmp_task_t *task, int
  // if_val, kmp_uint64 *lb, kmp_uint64 *ub, kmp_int64 st, int nogroup, int
  // sched, kmp_uint64 grainsize, void *task_dup);
  OMPRTL__kmpc_taskloop,
  // Call to void __kmpc_doacross_init(ident_t *loc, kmp_int32 gtid, kmp_int32
  // num_dims, struct kmp_dim *dims);
  OMPRTL__kmpc_doacross_init,
  // Call to void __kmpc_doacross_fini(ident_t *loc, kmp_int32 gtid);
  OMPRTL__kmpc_doacross_fini,
  // Call to void __kmpc_doacross_post(ident_t *loc, kmp_int32 gtid, kmp_int64
  // *vec);
  OMPRTL__kmpc_doacross_post,
  // Call to void __kmpc_doacross_wait(ident_t *loc, kmp_int32 gtid, kmp_int64
  // *vec);
  OMPRTL__kmpc_doacross_wait,
  // Call to void *__kmpc_task_reduction_init(int gtid, int num_data, void
  // *data);
  OMPRTL__kmpc_task_reduction_init,
  // Call to void *__kmpc_task_reduction_get_th_data(int gtid, void *tg, void
  // *d);
  OMPRTL__kmpc_task_reduction_get_th_data,

  //
  // Offloading related calls
  //
  // Call to int32_t __tgt_target(int32_t device_id, void *host_ptr, int32_t
  // arg_num, void** args_base, void **args, size_t *arg_sizes, int32_t
  // *arg_types);
  OMPRTL__tgt_target,
  // Call to int32_t __tgt_target_teams(int32_t device_id, void *host_ptr,
  // int32_t arg_num, void** args_base, void **args, size_t *arg_sizes,
  // int32_t *arg_types, int32_t num_teams, int32_t thread_limit);
  OMPRTL__tgt_target_teams,
  // Call to void __tgt_register_lib(__tgt_bin_desc *desc);
  OMPRTL__tgt_register_lib,
  // Call to void __tgt_unregister_lib(__tgt_bin_desc *desc);
  OMPRTL__tgt_unregister_lib,
  // Call to void __tgt_target_data_begin(int32_t device_id, int32_t arg_num,
  // void** args_base, void **args, size_t *arg_sizes, int32_t *arg_types);
  OMPRTL__tgt_target_data_begin,
  // Call to void __tgt_target_data_end(int32_t device_id, int32_t arg_num,
  // void** args_base, void **args, size_t *arg_sizes, int32_t *arg_types);
  OMPRTL__tgt_target_data_end,
  // Call to void __tgt_target_data_update(int32_t device_id, int32_t arg_num,
  // void** args_base, void **args, size_t *arg_sizes, int32_t *arg_types);
  OMPRTL__tgt_target_data_update,
};

/// A basic class for pre|post-action for advanced codegen sequence for OpenMP
/// region.
class CleanupTy final : public EHScopeStack::Cleanup {
  PrePostActionTy *Action;

public:
  explicit CleanupTy(PrePostActionTy *Action) : Action(Action) {}
  void Emit(CodeGenFunction &CGF, Flags /*flags*/) override {
    if (!CGF.HaveInsertPoint())
      return;
    Action->Exit(CGF);
  }
};

} // anonymous namespace

void RegionCodeGenTy::operator()(CodeGenFunction &CGF) const {
  CodeGenFunction::RunCleanupsScope Scope(CGF);
  if (PrePostAction) {
    CGF.EHStack.pushCleanup<CleanupTy>(NormalAndEHCleanup, PrePostAction);
    Callback(CodeGen, CGF, *PrePostAction);
  } else {
    PrePostActionTy Action;
    Callback(CodeGen, CGF, Action);
  }
}

/// Check if the combiner is a call to UDR combiner and if it is so return the
/// UDR decl used for reduction.
static const OMPDeclareReductionDecl *
getReductionInit(const Expr *ReductionOp) {
  if (auto *CE = dyn_cast<CallExpr>(ReductionOp))
    if (auto *OVE = dyn_cast<OpaqueValueExpr>(CE->getCallee()))
      if (auto *DRE =
              dyn_cast<DeclRefExpr>(OVE->getSourceExpr()->IgnoreImpCasts()))
        if (auto *DRD = dyn_cast<OMPDeclareReductionDecl>(DRE->getDecl()))
          return DRD;
  return nullptr;
}

static void emitInitWithReductionInitializer(CodeGenFunction &CGF,
                                             const OMPDeclareReductionDecl *DRD,
                                             const Expr *InitOp,
                                             Address Private, Address Original,
                                             QualType Ty) {
  if (DRD->getInitializer()) {
    std::pair<llvm::Function *, llvm::Function *> Reduction =
        CGF.CGM.getOpenMPRuntime().getUserDefinedReduction(DRD);
    auto *CE = cast<CallExpr>(InitOp);
    auto *OVE = cast<OpaqueValueExpr>(CE->getCallee());
    const Expr *LHS = CE->getArg(/*Arg=*/0)->IgnoreParenImpCasts();
    const Expr *RHS = CE->getArg(/*Arg=*/1)->IgnoreParenImpCasts();
    auto *LHSDRE = cast<DeclRefExpr>(cast<UnaryOperator>(LHS)->getSubExpr());
    auto *RHSDRE = cast<DeclRefExpr>(cast<UnaryOperator>(RHS)->getSubExpr());
    CodeGenFunction::OMPPrivateScope PrivateScope(CGF);
    PrivateScope.addPrivate(cast<VarDecl>(LHSDRE->getDecl()),
                            [=]() -> Address { return Private; });
    PrivateScope.addPrivate(cast<VarDecl>(RHSDRE->getDecl()),
                            [=]() -> Address { return Original; });
    (void)PrivateScope.Privatize();
    RValue Func = RValue::get(Reduction.second);
    CodeGenFunction::OpaqueValueMapping Map(CGF, OVE, Func);
    CGF.EmitIgnoredExpr(InitOp);
  } else {
    llvm::Constant *Init = CGF.CGM.EmitNullConstant(Ty);
    auto *GV = new llvm::GlobalVariable(
        CGF.CGM.getModule(), Init->getType(), /*isConstant=*/true,
        llvm::GlobalValue::PrivateLinkage, Init, ".init");
    LValue LV = CGF.MakeNaturalAlignAddrLValue(GV, Ty);
    RValue InitRVal;
    switch (CGF.getEvaluationKind(Ty)) {
    case TEK_Scalar:
      InitRVal = CGF.EmitLoadOfLValue(LV, SourceLocation());
      break;
    case TEK_Complex:
      InitRVal =
          RValue::getComplex(CGF.EmitLoadOfComplex(LV, SourceLocation()));
      break;
    case TEK_Aggregate:
      InitRVal = RValue::getAggregate(LV.getAddress());
      break;
    }
    OpaqueValueExpr OVE(SourceLocation(), Ty, VK_RValue);
    CodeGenFunction::OpaqueValueMapping OpaqueMap(CGF, &OVE, InitRVal);
    CGF.EmitAnyExprToMem(&OVE, Private, Ty.getQualifiers(),
                         /*IsInitializer=*/false);
  }
}

/// \brief Emit initialization of arrays of complex types.
/// \param DestAddr Address of the array.
/// \param Type Type of array.
/// \param Init Initial expression of array.
/// \param SrcAddr Address of the original array.
static void EmitOMPAggregateInit(CodeGenFunction &CGF, Address DestAddr,
                                 QualType Type, const Expr *Init,
                                 const OMPDeclareReductionDecl *DRD,
                                 Address SrcAddr = Address::invalid()) {
  // Perform element-by-element initialization.
  QualType ElementTy;

  // Drill down to the base element type on both arrays.
  auto ArrayTy = Type->getAsArrayTypeUnsafe();
  auto NumElements = CGF.emitArrayLength(ArrayTy, ElementTy, DestAddr);
  DestAddr =
      CGF.Builder.CreateElementBitCast(DestAddr, DestAddr.getElementType());
  if (DRD)
    SrcAddr =
        CGF.Builder.CreateElementBitCast(SrcAddr, DestAddr.getElementType());

  llvm::Value *SrcBegin = nullptr;
  if (DRD)
    SrcBegin = SrcAddr.getPointer();
  auto DestBegin = DestAddr.getPointer();
  // Cast from pointer to array type to pointer to single element.
  auto DestEnd = CGF.Builder.CreateGEP(DestBegin, NumElements);
  // The basic structure here is a while-do loop.
  auto BodyBB = CGF.createBasicBlock("omp.arrayinit.body");
  auto DoneBB = CGF.createBasicBlock("omp.arrayinit.done");
  auto IsEmpty =
      CGF.Builder.CreateICmpEQ(DestBegin, DestEnd, "omp.arrayinit.isempty");
  CGF.Builder.CreateCondBr(IsEmpty, DoneBB, BodyBB);

  // Enter the loop body, making that address the current address.
  auto EntryBB = CGF.Builder.GetInsertBlock();
  CGF.EmitBlock(BodyBB);

  CharUnits ElementSize = CGF.getContext().getTypeSizeInChars(ElementTy);

  llvm::PHINode *SrcElementPHI = nullptr;
  Address SrcElementCurrent = Address::invalid();
  if (DRD) {
    SrcElementPHI = CGF.Builder.CreatePHI(SrcBegin->getType(), 2,
                                          "omp.arraycpy.srcElementPast");
    SrcElementPHI->addIncoming(SrcBegin, EntryBB);
    SrcElementCurrent =
        Address(SrcElementPHI,
                SrcAddr.getAlignment().alignmentOfArrayElement(ElementSize));
  }
  llvm::PHINode *DestElementPHI = CGF.Builder.CreatePHI(
      DestBegin->getType(), 2, "omp.arraycpy.destElementPast");
  DestElementPHI->addIncoming(DestBegin, EntryBB);
  Address DestElementCurrent =
      Address(DestElementPHI,
              DestAddr.getAlignment().alignmentOfArrayElement(ElementSize));

  // Emit copy.
  {
    CodeGenFunction::RunCleanupsScope InitScope(CGF);
    if (DRD && (DRD->getInitializer() || !Init)) {
      emitInitWithReductionInitializer(CGF, DRD, Init, DestElementCurrent,
                                       SrcElementCurrent, ElementTy);
    } else
      CGF.EmitAnyExprToMem(Init, DestElementCurrent, ElementTy.getQualifiers(),
                           /*IsInitializer=*/false);
  }

  if (DRD) {
    // Shift the address forward by one element.
    auto SrcElementNext = CGF.Builder.CreateConstGEP1_32(
        SrcElementPHI, /*Idx0=*/1, "omp.arraycpy.dest.element");
    SrcElementPHI->addIncoming(SrcElementNext, CGF.Builder.GetInsertBlock());
  }

  // Shift the address forward by one element.
  auto DestElementNext = CGF.Builder.CreateConstGEP1_32(
      DestElementPHI, /*Idx0=*/1, "omp.arraycpy.dest.element");
  // Check whether we've reached the end.
  auto Done =
      CGF.Builder.CreateICmpEQ(DestElementNext, DestEnd, "omp.arraycpy.done");
  CGF.Builder.CreateCondBr(Done, DoneBB, BodyBB);
  DestElementPHI->addIncoming(DestElementNext, CGF.Builder.GetInsertBlock());

  // Done.
  CGF.EmitBlock(DoneBB, /*IsFinished=*/true);
}

LValue ReductionCodeGen::emitSharedLValue(CodeGenFunction &CGF, const Expr *E) {
  if (const auto *OASE = dyn_cast<OMPArraySectionExpr>(E))
    return CGF.EmitOMPArraySectionExpr(OASE);
  if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(E))
    return CGF.EmitLValue(ASE);
  auto *OrigVD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
  DeclRefExpr DRE(const_cast<VarDecl *>(OrigVD),
                  CGF.CapturedStmtInfo &&
                      CGF.CapturedStmtInfo->lookup(OrigVD) != nullptr,
                  E->getType(), VK_LValue, E->getExprLoc());
  // Store the address of the original variable associated with the LHS
  // implicit variable.
  return CGF.EmitLValue(&DRE);
}

LValue ReductionCodeGen::emitSharedLValueUB(CodeGenFunction &CGF,
                                            const Expr *E) {
  if (const auto *OASE = dyn_cast<OMPArraySectionExpr>(E))
    return CGF.EmitOMPArraySectionExpr(OASE, /*IsLowerBound=*/false);
  return LValue();
}

void ReductionCodeGen::emitAggregateInitialization(
    CodeGenFunction &CGF, unsigned N, Address PrivateAddr, LValue SharedLVal,
    const OMPDeclareReductionDecl *DRD) {
  // Emit VarDecl with copy init for arrays.
  // Get the address of the original variable captured in current
  // captured region.
  auto *PrivateVD =
      cast<VarDecl>(cast<DeclRefExpr>(ClausesData[N].Private)->getDecl());
  EmitOMPAggregateInit(CGF, PrivateAddr, PrivateVD->getType(),
                       DRD ? ClausesData[N].ReductionOp : PrivateVD->getInit(),
                       DRD, SharedLVal.getAddress());
}

ReductionCodeGen::ReductionCodeGen(ArrayRef<const Expr *> Shareds,
                                   ArrayRef<const Expr *> Privates,
                                   ArrayRef<const Expr *> ReductionOps) {
  ClausesData.reserve(Shareds.size());
  SharedAddresses.reserve(Shareds.size());
  Sizes.reserve(Shareds.size());
  BaseDecls.reserve(Shareds.size());
  auto IPriv = Privates.begin();
  auto IRed = ReductionOps.begin();
  for (const auto *Ref : Shareds) {
    ClausesData.emplace_back(Ref, *IPriv, *IRed);
    std::advance(IPriv, 1);
    std::advance(IRed, 1);
  }
}

void ReductionCodeGen::emitSharedLValue(CodeGenFunction &CGF, unsigned N) {
  assert(SharedAddresses.size() == N &&
         "Number of generated lvalues must be exactly N.");
  SharedAddresses.emplace_back(emitSharedLValue(CGF, ClausesData[N].Ref),
                               emitSharedLValueUB(CGF, ClausesData[N].Ref));
}

void ReductionCodeGen::emitAggregateType(CodeGenFunction &CGF, unsigned N) {
  auto *PrivateVD =
      cast<VarDecl>(cast<DeclRefExpr>(ClausesData[N].Private)->getDecl());
  QualType PrivateType = PrivateVD->getType();
  bool AsArraySection = isa<OMPArraySectionExpr>(ClausesData[N].Ref);
  if (!AsArraySection && !PrivateType->isVariablyModifiedType()) {
    Sizes.emplace_back(
        CGF.getTypeSize(
            SharedAddresses[N].first.getType().getNonReferenceType()),
        nullptr);
    return;
  }
  llvm::Value *Size;
  llvm::Value *SizeInChars;
  llvm::Type *ElemType =
      cast<llvm::PointerType>(SharedAddresses[N].first.getPointer()->getType())
          ->getElementType();
  auto *ElemSizeOf = llvm::ConstantExpr::getSizeOf(ElemType);
  if (AsArraySection) {
    Size = CGF.Builder.CreatePtrDiff(SharedAddresses[N].second.getPointer(),
                                     SharedAddresses[N].first.getPointer());
    Size = CGF.Builder.CreateNUWAdd(
        Size, llvm::ConstantInt::get(Size->getType(), /*V=*/1));
    SizeInChars = CGF.Builder.CreateNUWMul(Size, ElemSizeOf);
  } else {
    SizeInChars = CGF.getTypeSize(
        SharedAddresses[N].first.getType().getNonReferenceType());
    Size = CGF.Builder.CreateExactUDiv(SizeInChars, ElemSizeOf);
  }
  Sizes.emplace_back(SizeInChars, Size);
  CodeGenFunction::OpaqueValueMapping OpaqueMap(
      CGF,
      cast<OpaqueValueExpr>(
          CGF.getContext().getAsVariableArrayType(PrivateType)->getSizeExpr()),
      RValue::get(Size));
  CGF.EmitVariablyModifiedType(PrivateType);
}

void ReductionCodeGen::emitAggregateType(CodeGenFunction &CGF, unsigned N,
                                         llvm::Value *Size) {
  auto *PrivateVD =
      cast<VarDecl>(cast<DeclRefExpr>(ClausesData[N].Private)->getDecl());
  QualType PrivateType = PrivateVD->getType();
  bool AsArraySection = isa<OMPArraySectionExpr>(ClausesData[N].Ref);
  if (!AsArraySection && !PrivateType->isVariablyModifiedType()) {
    assert(!Size && !Sizes[N].second &&
           "Size should be nullptr for non-variably modified redution "
           "items.");
    return;
  }
  CodeGenFunction::OpaqueValueMapping OpaqueMap(
      CGF,
      cast<OpaqueValueExpr>(
          CGF.getContext().getAsVariableArrayType(PrivateType)->getSizeExpr()),
      RValue::get(Size));
  CGF.EmitVariablyModifiedType(PrivateType);
}

void ReductionCodeGen::emitInitialization(
    CodeGenFunction &CGF, unsigned N, Address PrivateAddr, LValue SharedLVal,
    llvm::function_ref<bool(CodeGenFunction &)> DefaultInit) {
  assert(SharedAddresses.size() > N && "No variable was generated");
  auto *PrivateVD =
      cast<VarDecl>(cast<DeclRefExpr>(ClausesData[N].Private)->getDecl());
  auto *DRD = getReductionInit(ClausesData[N].ReductionOp);
  QualType PrivateType = PrivateVD->getType();
  PrivateAddr = CGF.Builder.CreateElementBitCast(
      PrivateAddr, CGF.ConvertTypeForMem(PrivateType));
  QualType SharedType = SharedAddresses[N].first.getType();
  SharedLVal = CGF.MakeAddrLValue(
      CGF.Builder.CreateElementBitCast(SharedLVal.getAddress(),
                                       CGF.ConvertTypeForMem(SharedType)),
      SharedType, SharedAddresses[N].first.getBaseInfo());
  if (isa<OMPArraySectionExpr>(ClausesData[N].Ref) ||
      CGF.getContext().getAsArrayType(PrivateVD->getType())) {
    emitAggregateInitialization(CGF, N, PrivateAddr, SharedLVal, DRD);
  } else if (DRD && (DRD->getInitializer() || !PrivateVD->hasInit())) {
    emitInitWithReductionInitializer(CGF, DRD, ClausesData[N].ReductionOp,
                                     PrivateAddr, SharedLVal.getAddress(),
                                     SharedLVal.getType());
  } else if (!DefaultInit(CGF) && PrivateVD->hasInit() &&
             !CGF.isTrivialInitializer(PrivateVD->getInit())) {
    CGF.EmitAnyExprToMem(PrivateVD->getInit(), PrivateAddr,
                         PrivateVD->getType().getQualifiers(),
                         /*IsInitializer=*/false);
  }
}

bool ReductionCodeGen::needCleanups(unsigned N) {
  auto *PrivateVD =
      cast<VarDecl>(cast<DeclRefExpr>(ClausesData[N].Private)->getDecl());
  QualType PrivateType = PrivateVD->getType();
  QualType::DestructionKind DTorKind = PrivateType.isDestructedType();
  return DTorKind != QualType::DK_none;
}

void ReductionCodeGen::emitCleanups(CodeGenFunction &CGF, unsigned N,
                                    Address PrivateAddr) {
  auto *PrivateVD =
      cast<VarDecl>(cast<DeclRefExpr>(ClausesData[N].Private)->getDecl());
  QualType PrivateType = PrivateVD->getType();
  QualType::DestructionKind DTorKind = PrivateType.isDestructedType();
  if (needCleanups(N)) {
    PrivateAddr = CGF.Builder.CreateElementBitCast(
        PrivateAddr, CGF.ConvertTypeForMem(PrivateType));
    CGF.pushDestroy(DTorKind, PrivateAddr, PrivateType);
  }
}

static LValue loadToBegin(CodeGenFunction &CGF, QualType BaseTy, QualType ElTy,
                          LValue BaseLV) {
  BaseTy = BaseTy.getNonReferenceType();
  while ((BaseTy->isPointerType() || BaseTy->isReferenceType()) &&
         !CGF.getContext().hasSameType(BaseTy, ElTy)) {
    if (auto *PtrTy = BaseTy->getAs<PointerType>())
      BaseLV = CGF.EmitLoadOfPointerLValue(BaseLV.getAddress(), PtrTy);
    else {
      BaseLV = CGF.EmitLoadOfReferenceLValue(BaseLV.getAddress(),
                                             BaseTy->castAs<ReferenceType>());
    }
    BaseTy = BaseTy->getPointeeType();
  }
  return CGF.MakeAddrLValue(
      CGF.Builder.CreateElementBitCast(BaseLV.getAddress(),
                                       CGF.ConvertTypeForMem(ElTy)),
      BaseLV.getType(), BaseLV.getBaseInfo());
}

static Address castToBase(CodeGenFunction &CGF, QualType BaseTy, QualType ElTy,
                          llvm::Type *BaseLVType, CharUnits BaseLVAlignment,
                          llvm::Value *Addr) {
  Address Tmp = Address::invalid();
  Address TopTmp = Address::invalid();
  Address MostTopTmp = Address::invalid();
  BaseTy = BaseTy.getNonReferenceType();
  while ((BaseTy->isPointerType() || BaseTy->isReferenceType()) &&
         !CGF.getContext().hasSameType(BaseTy, ElTy)) {
    Tmp = CGF.CreateMemTemp(BaseTy);
    if (TopTmp.isValid())
      CGF.Builder.CreateStore(Tmp.getPointer(), TopTmp);
    else
      MostTopTmp = Tmp;
    TopTmp = Tmp;
    BaseTy = BaseTy->getPointeeType();
  }
  llvm::Type *Ty = BaseLVType;
  if (Tmp.isValid())
    Ty = Tmp.getElementType();
  Addr = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(Addr, Ty);
  if (Tmp.isValid()) {
    CGF.Builder.CreateStore(Addr, Tmp);
    return MostTopTmp;
  }
  return Address(Addr, BaseLVAlignment);
}

Address ReductionCodeGen::adjustPrivateAddress(CodeGenFunction &CGF, unsigned N,
                                               Address PrivateAddr) {
  const DeclRefExpr *DE;
  const VarDecl *OrigVD = nullptr;
  if (auto *OASE = dyn_cast<OMPArraySectionExpr>(ClausesData[N].Ref)) {
    auto *Base = OASE->getBase()->IgnoreParenImpCasts();
    while (auto *TempOASE = dyn_cast<OMPArraySectionExpr>(Base))
      Base = TempOASE->getBase()->IgnoreParenImpCasts();
    while (auto *TempASE = dyn_cast<ArraySubscriptExpr>(Base))
      Base = TempASE->getBase()->IgnoreParenImpCasts();
    DE = cast<DeclRefExpr>(Base);
    OrigVD = cast<VarDecl>(DE->getDecl());
  } else if (auto *ASE = dyn_cast<ArraySubscriptExpr>(ClausesData[N].Ref)) {
    auto *Base = ASE->getBase()->IgnoreParenImpCasts();
    while (auto *TempASE = dyn_cast<ArraySubscriptExpr>(Base))
      Base = TempASE->getBase()->IgnoreParenImpCasts();
    DE = cast<DeclRefExpr>(Base);
    OrigVD = cast<VarDecl>(DE->getDecl());
  }
  if (OrigVD) {
    BaseDecls.emplace_back(OrigVD);
    auto OriginalBaseLValue = CGF.EmitLValue(DE);
    LValue BaseLValue =
        loadToBegin(CGF, OrigVD->getType(), SharedAddresses[N].first.getType(),
                    OriginalBaseLValue);
    llvm::Value *Adjustment = CGF.Builder.CreatePtrDiff(
        BaseLValue.getPointer(), SharedAddresses[N].first.getPointer());
    llvm::Value *Ptr =
        CGF.Builder.CreateGEP(PrivateAddr.getPointer(), Adjustment);
    return castToBase(CGF, OrigVD->getType(),
                      SharedAddresses[N].first.getType(),
                      OriginalBaseLValue.getPointer()->getType(),
                      OriginalBaseLValue.getAlignment(), Ptr);
  }
  BaseDecls.emplace_back(
      cast<VarDecl>(cast<DeclRefExpr>(ClausesData[N].Ref)->getDecl()));
  return PrivateAddr;
}

bool ReductionCodeGen::usesReductionInitializer(unsigned N) const {
  auto *DRD = getReductionInit(ClausesData[N].ReductionOp);
  return DRD && DRD->getInitializer();
}

LValue CGOpenMPRegionInfo::getThreadIDVariableLValue(CodeGenFunction &CGF) {
  return CGF.EmitLoadOfPointerLValue(
      CGF.GetAddrOfLocalVar(getThreadIDVariable()),
      getThreadIDVariable()->getType()->castAs<PointerType>());
}

void CGOpenMPRegionInfo::EmitBody(CodeGenFunction &CGF, const Stmt * /*S*/) {
  if (!CGF.HaveInsertPoint())
    return;
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CGF.EHStack.pushTerminate();
  CodeGen(CGF);
  CGF.EHStack.popTerminate();
}

LValue CGOpenMPTaskOutlinedRegionInfo::getThreadIDVariableLValue(
    CodeGenFunction &CGF) {
  return CGF.MakeAddrLValue(CGF.GetAddrOfLocalVar(getThreadIDVariable()),
                            getThreadIDVariable()->getType(),
                            LValueBaseInfo(AlignmentSource::Decl, false));
}

CGOpenMPRuntime::CGOpenMPRuntime(CodeGenModule &CGM)
    : CGM(CGM), OffloadEntriesInfoManager(CGM) {
  IdentTy = llvm::StructType::create(
      "ident_t", CGM.Int32Ty /* reserved_1 */, CGM.Int32Ty /* flags */,
      CGM.Int32Ty /* reserved_2 */, CGM.Int32Ty /* reserved_3 */,
      CGM.Int8PtrTy /* psource */);
  KmpCriticalNameTy = llvm::ArrayType::get(CGM.Int32Ty, /*NumElements*/ 8);

  loadOffloadInfoMetadata();
}

void CGOpenMPRuntime::clear() {
  InternalVars.clear();
}

static llvm::Function *
emitCombinerOrInitializer(CodeGenModule &CGM, QualType Ty,
                          const Expr *CombinerInitializer, const VarDecl *In,
                          const VarDecl *Out, bool IsCombiner) {
  // void .omp_combiner.(Ty *in, Ty *out);
  auto &C = CGM.getContext();
  QualType PtrTy = C.getPointerType(Ty).withRestrict();
  FunctionArgList Args;
  ImplicitParamDecl OmpOutParm(C, /*DC=*/nullptr, Out->getLocation(),
                               /*Id=*/nullptr, PtrTy, ImplicitParamDecl::Other);
  ImplicitParamDecl OmpInParm(C, /*DC=*/nullptr, In->getLocation(),
                              /*Id=*/nullptr, PtrTy, ImplicitParamDecl::Other);
  Args.push_back(&OmpOutParm);
  Args.push_back(&OmpInParm);
  auto &FnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  auto *FnTy = CGM.getTypes().GetFunctionType(FnInfo);
  auto *Fn = llvm::Function::Create(
      FnTy, llvm::GlobalValue::InternalLinkage,
      IsCombiner ? ".omp_combiner." : ".omp_initializer.", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*D=*/nullptr, Fn, FnInfo);
  Fn->removeFnAttr(llvm::Attribute::NoInline);
  Fn->removeFnAttr(llvm::Attribute::OptimizeNone);
  Fn->addFnAttr(llvm::Attribute::AlwaysInline);
  CodeGenFunction CGF(CGM);
  // Map "T omp_in;" variable to "*omp_in_parm" value in all expressions.
  // Map "T omp_out;" variable to "*omp_out_parm" value in all expressions.
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, FnInfo, Args);
  CodeGenFunction::OMPPrivateScope Scope(CGF);
  Address AddrIn = CGF.GetAddrOfLocalVar(&OmpInParm);
  Scope.addPrivate(In, [&CGF, AddrIn, PtrTy]() -> Address {
    return CGF.EmitLoadOfPointerLValue(AddrIn, PtrTy->castAs<PointerType>())
        .getAddress();
  });
  Address AddrOut = CGF.GetAddrOfLocalVar(&OmpOutParm);
  Scope.addPrivate(Out, [&CGF, AddrOut, PtrTy]() -> Address {
    return CGF.EmitLoadOfPointerLValue(AddrOut, PtrTy->castAs<PointerType>())
        .getAddress();
  });
  (void)Scope.Privatize();
  if (!IsCombiner && Out->hasInit() &&
      !CGF.isTrivialInitializer(Out->getInit())) {
    CGF.EmitAnyExprToMem(Out->getInit(), CGF.GetAddrOfLocalVar(Out),
                         Out->getType().getQualifiers(),
                         /*IsInitializer=*/true);
  }
  if (CombinerInitializer)
    CGF.EmitIgnoredExpr(CombinerInitializer);
  Scope.ForceCleanup();
  CGF.FinishFunction();
  return Fn;
}

void CGOpenMPRuntime::emitUserDefinedReduction(
    CodeGenFunction *CGF, const OMPDeclareReductionDecl *D) {
  if (UDRMap.count(D) > 0)
    return;
  auto &C = CGM.getContext();
  if (!In || !Out) {
    In = &C.Idents.get("omp_in");
    Out = &C.Idents.get("omp_out");
  }
  llvm::Function *Combiner = emitCombinerOrInitializer(
      CGM, D->getType(), D->getCombiner(), cast<VarDecl>(D->lookup(In).front()),
      cast<VarDecl>(D->lookup(Out).front()),
      /*IsCombiner=*/true);
  llvm::Function *Initializer = nullptr;
  if (auto *Init = D->getInitializer()) {
    if (!Priv || !Orig) {
      Priv = &C.Idents.get("omp_priv");
      Orig = &C.Idents.get("omp_orig");
    }
    Initializer = emitCombinerOrInitializer(
        CGM, D->getType(),
        D->getInitializerKind() == OMPDeclareReductionDecl::CallInit ? Init
                                                                     : nullptr,
        cast<VarDecl>(D->lookup(Orig).front()),
        cast<VarDecl>(D->lookup(Priv).front()),
        /*IsCombiner=*/false);
  }
  UDRMap.insert(std::make_pair(D, std::make_pair(Combiner, Initializer)));
  if (CGF) {
    auto &Decls = FunctionUDRMap.FindAndConstruct(CGF->CurFn);
    Decls.second.push_back(D);
  }
}

std::pair<llvm::Function *, llvm::Function *>
CGOpenMPRuntime::getUserDefinedReduction(const OMPDeclareReductionDecl *D) {
  auto I = UDRMap.find(D);
  if (I != UDRMap.end())
    return I->second;
  emitUserDefinedReduction(/*CGF=*/nullptr, D);
  return UDRMap.lookup(D);
}

// Layout information for ident_t.
static CharUnits getIdentAlign(CodeGenModule &CGM) {
  return CGM.getPointerAlign();
}
static CharUnits getIdentSize(CodeGenModule &CGM) {
  assert((4 * CGM.getPointerSize()).isMultipleOf(CGM.getPointerAlign()));
  return CharUnits::fromQuantity(16) + CGM.getPointerSize();
}
static CharUnits getOffsetOfIdentField(IdentFieldIndex Field) {
  // All the fields except the last are i32, so this works beautifully.
  return unsigned(Field) * CharUnits::fromQuantity(4);
}
static Address createIdentFieldGEP(CodeGenFunction &CGF, Address Addr,
                                   IdentFieldIndex Field,
                                   const llvm::Twine &Name = "") {
  auto Offset = getOffsetOfIdentField(Field);
  return CGF.Builder.CreateStructGEP(Addr, Field, Offset, Name);
}

static llvm::Value *emitParallelOrTeamsOutlinedFunction(
    CodeGenModule &CGM, const OMPExecutableDirective &D, const CapturedStmt *CS,
    const VarDecl *ThreadIDVar, OpenMPDirectiveKind InnermostKind,
    const StringRef OutlinedHelperName, const RegionCodeGenTy &CodeGen) {
  assert(ThreadIDVar->getType()->isPointerType() &&
         "thread id variable must be of type kmp_int32 *");
  CodeGenFunction CGF(CGM, true);
  bool HasCancel = false;
  if (auto *OPD = dyn_cast<OMPParallelDirective>(&D))
    HasCancel = OPD->hasCancel();
  else if (auto *OPSD = dyn_cast<OMPParallelSectionsDirective>(&D))
    HasCancel = OPSD->hasCancel();
  else if (auto *OPFD = dyn_cast<OMPParallelForDirective>(&D))
    HasCancel = OPFD->hasCancel();
  CGOpenMPOutlinedRegionInfo CGInfo(*CS, ThreadIDVar, CodeGen, InnermostKind,
                                    HasCancel, OutlinedHelperName);
  CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CGInfo);
  return CGF.GenerateOpenMPCapturedStmtFunction(*CS);
}

llvm::Value *CGOpenMPRuntime::emitParallelOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
    OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen) {
  const CapturedStmt *CS = D.getCapturedStmt(OMPD_parallel);
  return emitParallelOrTeamsOutlinedFunction(
      CGM, D, CS, ThreadIDVar, InnermostKind, getOutlinedHelperName(), CodeGen);
}

llvm::Value *CGOpenMPRuntime::emitTeamsOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
    OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen) {
  const CapturedStmt *CS = D.getCapturedStmt(OMPD_teams);
  return emitParallelOrTeamsOutlinedFunction(
      CGM, D, CS, ThreadIDVar, InnermostKind, getOutlinedHelperName(), CodeGen);
}

llvm::Value *CGOpenMPRuntime::emitTaskOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
    const VarDecl *PartIDVar, const VarDecl *TaskTVar,
    OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen,
    bool Tied, unsigned &NumberOfParts) {
  auto &&UntiedCodeGen = [this, &D, TaskTVar](CodeGenFunction &CGF,
                                              PrePostActionTy &) {
    auto *ThreadID = getThreadID(CGF, D.getLocStart());
    auto *UpLoc = emitUpdateLocation(CGF, D.getLocStart());
    llvm::Value *TaskArgs[] = {
        UpLoc, ThreadID,
        CGF.EmitLoadOfPointerLValue(CGF.GetAddrOfLocalVar(TaskTVar),
                                    TaskTVar->getType()->castAs<PointerType>())
            .getPointer()};
    CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_omp_task), TaskArgs);
  };
  CGOpenMPTaskOutlinedRegionInfo::UntiedTaskActionTy Action(Tied, PartIDVar,
                                                            UntiedCodeGen);
  CodeGen.setAction(Action);
  assert(!ThreadIDVar->getType()->isPointerType() &&
         "thread id variable must be of type kmp_int32 for tasks");
  auto *CS = cast<CapturedStmt>(D.getAssociatedStmt());
  auto *TD = dyn_cast<OMPTaskDirective>(&D);
  CodeGenFunction CGF(CGM, true);
  CGOpenMPTaskOutlinedRegionInfo CGInfo(*CS, ThreadIDVar, CodeGen,
                                        InnermostKind,
                                        TD ? TD->hasCancel() : false, Action);
  CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CGInfo);
  auto *Res = CGF.GenerateCapturedStmtFunction(*CS);
  if (!Tied)
    NumberOfParts = Action.getNumberOfParts();
  return Res;
}

Address CGOpenMPRuntime::getOrCreateDefaultLocation(unsigned Flags) {
  CharUnits Align = getIdentAlign(CGM);
  llvm::Value *Entry = OpenMPDefaultLocMap.lookup(Flags);
  if (!Entry) {
    if (!DefaultOpenMPPSource) {
      // Initialize default location for psource field of ident_t structure of
      // all ident_t objects. Format is ";file;function;line;column;;".
      // Taken from
      // http://llvm.org/svn/llvm-project/openmp/trunk/runtime/src/kmp_str.c
      DefaultOpenMPPSource =
          CGM.GetAddrOfConstantCString(";unknown;unknown;0;0;;").getPointer();
      DefaultOpenMPPSource =
          llvm::ConstantExpr::getBitCast(DefaultOpenMPPSource, CGM.Int8PtrTy);
    }

    ConstantInitBuilder builder(CGM);
    auto fields = builder.beginStruct(IdentTy);
    fields.addInt(CGM.Int32Ty, 0);
    fields.addInt(CGM.Int32Ty, Flags);
    fields.addInt(CGM.Int32Ty, 0);
    fields.addInt(CGM.Int32Ty, 0);
    fields.add(DefaultOpenMPPSource);
    auto DefaultOpenMPLocation =
      fields.finishAndCreateGlobal("", Align, /*isConstant*/ true,
                                   llvm::GlobalValue::PrivateLinkage);
    DefaultOpenMPLocation->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);

    OpenMPDefaultLocMap[Flags] = Entry = DefaultOpenMPLocation;
  }
  return Address(Entry, Align);
}

llvm::Value *CGOpenMPRuntime::emitUpdateLocation(CodeGenFunction &CGF,
                                                 SourceLocation Loc,
                                                 unsigned Flags) {
  Flags |= OMP_IDENT_KMPC;
  // If no debug info is generated - return global default location.
  if (CGM.getCodeGenOpts().getDebugInfo() == codegenoptions::NoDebugInfo ||
      Loc.isInvalid())
    return getOrCreateDefaultLocation(Flags).getPointer();

  assert(CGF.CurFn && "No function in current CodeGenFunction.");

  Address LocValue = Address::invalid();
  auto I = OpenMPLocThreadIDMap.find(CGF.CurFn);
  if (I != OpenMPLocThreadIDMap.end())
    LocValue = Address(I->second.DebugLoc, getIdentAlign(CGF.CGM));

  // OpenMPLocThreadIDMap may have null DebugLoc and non-null ThreadID, if
  // GetOpenMPThreadID was called before this routine.
  if (!LocValue.isValid()) {
    // Generate "ident_t .kmpc_loc.addr;"
    Address AI = CGF.CreateTempAlloca(IdentTy, getIdentAlign(CGF.CGM),
                                      ".kmpc_loc.addr");
    auto &Elem = OpenMPLocThreadIDMap.FindAndConstruct(CGF.CurFn);
    Elem.second.DebugLoc = AI.getPointer();
    LocValue = AI;

    CGBuilderTy::InsertPointGuard IPG(CGF.Builder);
    CGF.Builder.SetInsertPoint(CGF.AllocaInsertPt);
    CGF.Builder.CreateMemCpy(LocValue, getOrCreateDefaultLocation(Flags),
                             CGM.getSize(getIdentSize(CGF.CGM)));
  }

  // char **psource = &.kmpc_loc_<flags>.addr.psource;
  Address PSource = createIdentFieldGEP(CGF, LocValue, IdentField_PSource);

  auto OMPDebugLoc = OpenMPDebugLocMap.lookup(Loc.getRawEncoding());
  if (OMPDebugLoc == nullptr) {
    SmallString<128> Buffer2;
    llvm::raw_svector_ostream OS2(Buffer2);
    // Build debug location
    PresumedLoc PLoc = CGF.getContext().getSourceManager().getPresumedLoc(Loc);
    OS2 << ";" << PLoc.getFilename() << ";";
    if (const FunctionDecl *FD =
            dyn_cast_or_null<FunctionDecl>(CGF.CurFuncDecl)) {
      OS2 << FD->getQualifiedNameAsString();
    }
    OS2 << ";" << PLoc.getLine() << ";" << PLoc.getColumn() << ";;";
    OMPDebugLoc = CGF.Builder.CreateGlobalStringPtr(OS2.str());
    OpenMPDebugLocMap[Loc.getRawEncoding()] = OMPDebugLoc;
  }
  // *psource = ";<File>;<Function>;<Line>;<Column>;;";
  CGF.Builder.CreateStore(OMPDebugLoc, PSource);

  // Our callers always pass this to a runtime function, so for
  // convenience, go ahead and return a naked pointer.
  return LocValue.getPointer();
}

llvm::Value *CGOpenMPRuntime::getThreadID(CodeGenFunction &CGF,
                                          SourceLocation Loc) {
  assert(CGF.CurFn && "No function in current CodeGenFunction.");

  llvm::Value *ThreadID = nullptr;
  // Check whether we've already cached a load of the thread id in this
  // function.
  auto I = OpenMPLocThreadIDMap.find(CGF.CurFn);
  if (I != OpenMPLocThreadIDMap.end()) {
    ThreadID = I->second.ThreadID;
    if (ThreadID != nullptr)
      return ThreadID;
  }
  // If exceptions are enabled, do not use parameter to avoid possible crash.
  if (!CGF.getInvokeDest()) {
    if (auto *OMPRegionInfo =
            dyn_cast_or_null<CGOpenMPRegionInfo>(CGF.CapturedStmtInfo)) {
      if (OMPRegionInfo->getThreadIDVariable()) {
        // Check if this an outlined function with thread id passed as argument.
        auto LVal = OMPRegionInfo->getThreadIDVariableLValue(CGF);
        ThreadID = CGF.EmitLoadOfLValue(LVal, Loc).getScalarVal();
        // If value loaded in entry block, cache it and use it everywhere in
        // function.
        if (CGF.Builder.GetInsertBlock() == CGF.AllocaInsertPt->getParent()) {
          auto &Elem = OpenMPLocThreadIDMap.FindAndConstruct(CGF.CurFn);
          Elem.second.ThreadID = ThreadID;
        }
        return ThreadID;
      }
    }
  }

  // This is not an outlined function region - need to call __kmpc_int32
  // kmpc_global_thread_num(ident_t *loc).
  // Generate thread id value and cache this value for use across the
  // function.
  CGBuilderTy::InsertPointGuard IPG(CGF.Builder);
  CGF.Builder.SetInsertPoint(CGF.AllocaInsertPt);
  ThreadID =
      CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_global_thread_num),
                          emitUpdateLocation(CGF, Loc));
  auto &Elem = OpenMPLocThreadIDMap.FindAndConstruct(CGF.CurFn);
  Elem.second.ThreadID = ThreadID;
  return ThreadID;
}

void CGOpenMPRuntime::functionFinished(CodeGenFunction &CGF) {
  assert(CGF.CurFn && "No function in current CodeGenFunction.");
  if (OpenMPLocThreadIDMap.count(CGF.CurFn))
    OpenMPLocThreadIDMap.erase(CGF.CurFn);
  if (FunctionUDRMap.count(CGF.CurFn) > 0) {
    for(auto *D : FunctionUDRMap[CGF.CurFn]) {
      UDRMap.erase(D);
    }
    FunctionUDRMap.erase(CGF.CurFn);
  }
}

llvm::Type *CGOpenMPRuntime::getIdentTyPointerTy() {
  if (!IdentTy) {
  }
  return llvm::PointerType::getUnqual(IdentTy);
}

llvm::Type *CGOpenMPRuntime::getKmpc_MicroPointerTy() {
  if (!Kmpc_MicroTy) {
    // Build void (*kmpc_micro)(kmp_int32 *global_tid, kmp_int32 *bound_tid,...)
    llvm::Type *MicroParams[] = {llvm::PointerType::getUnqual(CGM.Int32Ty),
                                 llvm::PointerType::getUnqual(CGM.Int32Ty)};
    Kmpc_MicroTy = llvm::FunctionType::get(CGM.VoidTy, MicroParams, true);
  }
  return llvm::PointerType::getUnqual(Kmpc_MicroTy);
}

llvm::Constant *
CGOpenMPRuntime::createRuntimeFunction(unsigned Function) {
  llvm::Constant *RTLFn = nullptr;
  switch (static_cast<OpenMPRTLFunction>(Function)) {
  case OMPRTL__kmpc_fork_call: {
    // Build void __kmpc_fork_call(ident_t *loc, kmp_int32 argc, kmpc_micro
    // microtask, ...);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty,
                                getKmpc_MicroPointerTy()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ true);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_fork_call");
    break;
  }
  case OMPRTL__kmpc_global_thread_num: {
    // Build kmp_int32 __kmpc_global_thread_num(ident_t *loc);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_global_thread_num");
    break;
  }
  case OMPRTL__kmpc_threadprivate_cached: {
    // Build void *__kmpc_threadprivate_cached(ident_t *loc,
    // kmp_int32 global_tid, void *data, size_t size, void ***cache);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty,
                                CGM.VoidPtrTy, CGM.SizeTy,
                                CGM.VoidPtrTy->getPointerTo()->getPointerTo()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidPtrTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_threadprivate_cached");
    break;
  }
  case OMPRTL__kmpc_critical: {
    // Build void __kmpc_critical(ident_t *loc, kmp_int32 global_tid,
    // kmp_critical_name *crit);
    llvm::Type *TypeParams[] = {
        getIdentTyPointerTy(), CGM.Int32Ty,
        llvm::PointerType::getUnqual(KmpCriticalNameTy)};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_critical");
    break;
  }
  case OMPRTL__kmpc_critical_with_hint: {
    // Build void __kmpc_critical_with_hint(ident_t *loc, kmp_int32 global_tid,
    // kmp_critical_name *crit, uintptr_t hint);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty,
                                llvm::PointerType::getUnqual(KmpCriticalNameTy),
                                CGM.IntPtrTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_critical_with_hint");
    break;
  }
  case OMPRTL__kmpc_threadprivate_register: {
    // Build void __kmpc_threadprivate_register(ident_t *, void *data,
    // kmpc_ctor ctor, kmpc_cctor cctor, kmpc_dtor dtor);
    // typedef void *(*kmpc_ctor)(void *);
    auto KmpcCtorTy =
        llvm::FunctionType::get(CGM.VoidPtrTy, CGM.VoidPtrTy,
                                /*isVarArg*/ false)->getPointerTo();
    // typedef void *(*kmpc_cctor)(void *, void *);
    llvm::Type *KmpcCopyCtorTyArgs[] = {CGM.VoidPtrTy, CGM.VoidPtrTy};
    auto KmpcCopyCtorTy =
        llvm::FunctionType::get(CGM.VoidPtrTy, KmpcCopyCtorTyArgs,
                                /*isVarArg*/ false)->getPointerTo();
    // typedef void (*kmpc_dtor)(void *);
    auto KmpcDtorTy =
        llvm::FunctionType::get(CGM.VoidTy, CGM.VoidPtrTy, /*isVarArg*/ false)
            ->getPointerTo();
    llvm::Type *FnTyArgs[] = {getIdentTyPointerTy(), CGM.VoidPtrTy, KmpcCtorTy,
                              KmpcCopyCtorTy, KmpcDtorTy};
    auto FnTy = llvm::FunctionType::get(CGM.VoidTy, FnTyArgs,
                                        /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_threadprivate_register");
    break;
  }
  case OMPRTL__kmpc_end_critical: {
    // Build void __kmpc_end_critical(ident_t *loc, kmp_int32 global_tid,
    // kmp_critical_name *crit);
    llvm::Type *TypeParams[] = {
        getIdentTyPointerTy(), CGM.Int32Ty,
        llvm::PointerType::getUnqual(KmpCriticalNameTy)};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_end_critical");
    break;
  }
  case OMPRTL__kmpc_cancel_barrier: {
    // Build kmp_int32 __kmpc_cancel_barrier(ident_t *loc, kmp_int32
    // global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name*/ "__kmpc_cancel_barrier");
    break;
  }
  case OMPRTL__kmpc_barrier: {
    // Build void __kmpc_barrier(ident_t *loc, kmp_int32 global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name*/ "__kmpc_barrier");
    break;
  }
  case OMPRTL__kmpc_for_static_fini: {
    // Build void __kmpc_for_static_fini(ident_t *loc, kmp_int32 global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_for_static_fini");
    break;
  }
  case OMPRTL__kmpc_push_num_threads: {
    // Build void __kmpc_push_num_threads(ident_t *loc, kmp_int32 global_tid,
    // kmp_int32 num_threads)
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty,
                                CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_push_num_threads");
    break;
  }
  case OMPRTL__kmpc_serialized_parallel: {
    // Build void __kmpc_serialized_parallel(ident_t *loc, kmp_int32
    // global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_serialized_parallel");
    break;
  }
  case OMPRTL__kmpc_end_serialized_parallel: {
    // Build void __kmpc_end_serialized_parallel(ident_t *loc, kmp_int32
    // global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_end_serialized_parallel");
    break;
  }
  case OMPRTL__kmpc_flush: {
    // Build void __kmpc_flush(ident_t *loc);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_flush");
    break;
  }
  case OMPRTL__kmpc_master: {
    // Build kmp_int32 __kmpc_master(ident_t *loc, kmp_int32 global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_master");
    break;
  }
  case OMPRTL__kmpc_end_master: {
    // Build void __kmpc_end_master(ident_t *loc, kmp_int32 global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_end_master");
    break;
  }
  case OMPRTL__kmpc_omp_taskyield: {
    // Build kmp_int32 __kmpc_omp_taskyield(ident_t *, kmp_int32 global_tid,
    // int end_part);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty, CGM.IntTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_omp_taskyield");
    break;
  }
  case OMPRTL__kmpc_single: {
    // Build kmp_int32 __kmpc_single(ident_t *loc, kmp_int32 global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_single");
    break;
  }
  case OMPRTL__kmpc_end_single: {
    // Build void __kmpc_end_single(ident_t *loc, kmp_int32 global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_end_single");
    break;
  }
  case OMPRTL__kmpc_omp_task_alloc: {
    // Build kmp_task_t *__kmpc_omp_task_alloc(ident_t *, kmp_int32 gtid,
    // kmp_int32 flags, size_t sizeof_kmp_task_t, size_t sizeof_shareds,
    // kmp_routine_entry_t *task_entry);
    assert(KmpRoutineEntryPtrTy != nullptr &&
           "Type kmp_routine_entry_t must be created.");
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty, CGM.Int32Ty,
                                CGM.SizeTy, CGM.SizeTy, KmpRoutineEntryPtrTy};
    // Return void * and then cast to particular kmp_task_t type.
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidPtrTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_omp_task_alloc");
    break;
  }
  case OMPRTL__kmpc_omp_task: {
    // Build kmp_int32 __kmpc_omp_task(ident_t *, kmp_int32 gtid, kmp_task_t
    // *new_task);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty,
                                CGM.VoidPtrTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_omp_task");
    break;
  }
  case OMPRTL__kmpc_copyprivate: {
    // Build void __kmpc_copyprivate(ident_t *loc, kmp_int32 global_tid,
    // size_t cpy_size, void *cpy_data, void(*cpy_func)(void *, void *),
    // kmp_int32 didit);
    llvm::Type *CpyTypeParams[] = {CGM.VoidPtrTy, CGM.VoidPtrTy};
    auto *CpyFnTy =
        llvm::FunctionType::get(CGM.VoidTy, CpyTypeParams, /*isVarArg=*/false);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty, CGM.SizeTy,
                                CGM.VoidPtrTy, CpyFnTy->getPointerTo(),
                                CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_copyprivate");
    break;
  }
  case OMPRTL__kmpc_reduce: {
    // Build kmp_int32 __kmpc_reduce(ident_t *loc, kmp_int32 global_tid,
    // kmp_int32 num_vars, size_t reduce_size, void *reduce_data, void
    // (*reduce_func)(void *lhs_data, void *rhs_data), kmp_critical_name *lck);
    llvm::Type *ReduceTypeParams[] = {CGM.VoidPtrTy, CGM.VoidPtrTy};
    auto *ReduceFnTy = llvm::FunctionType::get(CGM.VoidTy, ReduceTypeParams,
                                               /*isVarArg=*/false);
    llvm::Type *TypeParams[] = {
        getIdentTyPointerTy(), CGM.Int32Ty, CGM.Int32Ty, CGM.SizeTy,
        CGM.VoidPtrTy, ReduceFnTy->getPointerTo(),
        llvm::PointerType::getUnqual(KmpCriticalNameTy)};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_reduce");
    break;
  }
  case OMPRTL__kmpc_reduce_nowait: {
    // Build kmp_int32 __kmpc_reduce_nowait(ident_t *loc, kmp_int32
    // global_tid, kmp_int32 num_vars, size_t reduce_size, void *reduce_data,
    // void (*reduce_func)(void *lhs_data, void *rhs_data), kmp_critical_name
    // *lck);
    llvm::Type *ReduceTypeParams[] = {CGM.VoidPtrTy, CGM.VoidPtrTy};
    auto *ReduceFnTy = llvm::FunctionType::get(CGM.VoidTy, ReduceTypeParams,
                                               /*isVarArg=*/false);
    llvm::Type *TypeParams[] = {
        getIdentTyPointerTy(), CGM.Int32Ty, CGM.Int32Ty, CGM.SizeTy,
        CGM.VoidPtrTy, ReduceFnTy->getPointerTo(),
        llvm::PointerType::getUnqual(KmpCriticalNameTy)};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_reduce_nowait");
    break;
  }
  case OMPRTL__kmpc_end_reduce: {
    // Build void __kmpc_end_reduce(ident_t *loc, kmp_int32 global_tid,
    // kmp_critical_name *lck);
    llvm::Type *TypeParams[] = {
        getIdentTyPointerTy(), CGM.Int32Ty,
        llvm::PointerType::getUnqual(KmpCriticalNameTy)};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_end_reduce");
    break;
  }
  case OMPRTL__kmpc_end_reduce_nowait: {
    // Build __kmpc_end_reduce_nowait(ident_t *loc, kmp_int32 global_tid,
    // kmp_critical_name *lck);
    llvm::Type *TypeParams[] = {
        getIdentTyPointerTy(), CGM.Int32Ty,
        llvm::PointerType::getUnqual(KmpCriticalNameTy)};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn =
        CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_end_reduce_nowait");
    break;
  }
  case OMPRTL__kmpc_omp_task_begin_if0: {
    // Build void __kmpc_omp_task(ident_t *, kmp_int32 gtid, kmp_task_t
    // *new_task);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty,
                                CGM.VoidPtrTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn =
        CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_omp_task_begin_if0");
    break;
  }
  case OMPRTL__kmpc_omp_task_complete_if0: {
    // Build void __kmpc_omp_task(ident_t *, kmp_int32 gtid, kmp_task_t
    // *new_task);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty,
                                CGM.VoidPtrTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy,
                                      /*Name=*/"__kmpc_omp_task_complete_if0");
    break;
  }
  case OMPRTL__kmpc_ordered: {
    // Build void __kmpc_ordered(ident_t *loc, kmp_int32 global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_ordered");
    break;
  }
  case OMPRTL__kmpc_end_ordered: {
    // Build void __kmpc_end_ordered(ident_t *loc, kmp_int32 global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_end_ordered");
    break;
  }
  case OMPRTL__kmpc_omp_taskwait: {
    // Build kmp_int32 __kmpc_omp_taskwait(ident_t *loc, kmp_int32 global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_omp_taskwait");
    break;
  }
  case OMPRTL__kmpc_taskgroup: {
    // Build void __kmpc_taskgroup(ident_t *loc, kmp_int32 global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_taskgroup");
    break;
  }
  case OMPRTL__kmpc_end_taskgroup: {
    // Build void __kmpc_end_taskgroup(ident_t *loc, kmp_int32 global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_end_taskgroup");
    break;
  }
  case OMPRTL__kmpc_push_proc_bind: {
    // Build void __kmpc_push_proc_bind(ident_t *loc, kmp_int32 global_tid,
    // int proc_bind)
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty, CGM.IntTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_push_proc_bind");
    break;
  }
  case OMPRTL__kmpc_omp_task_with_deps: {
    // Build kmp_int32 __kmpc_omp_task_with_deps(ident_t *, kmp_int32 gtid,
    // kmp_task_t *new_task, kmp_int32 ndeps, kmp_depend_info_t *dep_list,
    // kmp_int32 ndeps_noalias, kmp_depend_info_t *noalias_dep_list);
    llvm::Type *TypeParams[] = {
        getIdentTyPointerTy(), CGM.Int32Ty, CGM.VoidPtrTy, CGM.Int32Ty,
        CGM.VoidPtrTy,         CGM.Int32Ty, CGM.VoidPtrTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn =
        CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_omp_task_with_deps");
    break;
  }
  case OMPRTL__kmpc_omp_wait_deps: {
    // Build void __kmpc_omp_wait_deps(ident_t *, kmp_int32 gtid,
    // kmp_int32 ndeps, kmp_depend_info_t *dep_list, kmp_int32 ndeps_noalias,
    // kmp_depend_info_t *noalias_dep_list);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty,
                                CGM.Int32Ty,           CGM.VoidPtrTy,
                                CGM.Int32Ty,           CGM.VoidPtrTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_omp_wait_deps");
    break;
  }
  case OMPRTL__kmpc_cancellationpoint: {
    // Build kmp_int32 __kmpc_cancellationpoint(ident_t *loc, kmp_int32
    // global_tid, kmp_int32 cncl_kind)
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty, CGM.IntTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_cancellationpoint");
    break;
  }
  case OMPRTL__kmpc_cancel: {
    // Build kmp_int32 __kmpc_cancel(ident_t *loc, kmp_int32 global_tid,
    // kmp_int32 cncl_kind)
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty, CGM.IntTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_cancel");
    break;
  }
  case OMPRTL__kmpc_push_num_teams: {
    // Build void kmpc_push_num_teams (ident_t loc, kmp_int32 global_tid,
    // kmp_int32 num_teams, kmp_int32 num_threads)
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty, CGM.Int32Ty,
        CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_push_num_teams");
    break;
  }
  case OMPRTL__kmpc_fork_teams: {
    // Build void __kmpc_fork_teams(ident_t *loc, kmp_int32 argc, kmpc_micro
    // microtask, ...);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty,
                                getKmpc_MicroPointerTy()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ true);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_fork_teams");
    break;
  }
  case OMPRTL__kmpc_taskloop: {
    // Build void __kmpc_taskloop(ident_t *loc, int gtid, kmp_task_t *task, int
    // if_val, kmp_uint64 *lb, kmp_uint64 *ub, kmp_int64 st, int nogroup, int
    // sched, kmp_uint64 grainsize, void *task_dup);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(),
                                CGM.IntTy,
                                CGM.VoidPtrTy,
                                CGM.IntTy,
                                CGM.Int64Ty->getPointerTo(),
                                CGM.Int64Ty->getPointerTo(),
                                CGM.Int64Ty,
                                CGM.IntTy,
                                CGM.IntTy,
                                CGM.Int64Ty,
                                CGM.VoidPtrTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_taskloop");
    break;
  }
  case OMPRTL__kmpc_doacross_init: {
    // Build void __kmpc_doacross_init(ident_t *loc, kmp_int32 gtid, kmp_int32
    // num_dims, struct kmp_dim *dims);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(),
                                CGM.Int32Ty,
                                CGM.Int32Ty,
                                CGM.VoidPtrTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_doacross_init");
    break;
  }
  case OMPRTL__kmpc_doacross_fini: {
    // Build void __kmpc_doacross_fini(ident_t *loc, kmp_int32 gtid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_doacross_fini");
    break;
  }
  case OMPRTL__kmpc_doacross_post: {
    // Build void __kmpc_doacross_post(ident_t *loc, kmp_int32 gtid, kmp_int64
    // *vec);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty,
                                CGM.Int64Ty->getPointerTo()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_doacross_post");
    break;
  }
  case OMPRTL__kmpc_doacross_wait: {
    // Build void __kmpc_doacross_wait(ident_t *loc, kmp_int32 gtid, kmp_int64
    // *vec);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty,
                                CGM.Int64Ty->getPointerTo()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_doacross_wait");
    break;
  }
  case OMPRTL__kmpc_task_reduction_init: {
    // Build void *__kmpc_task_reduction_init(int gtid, int num_data, void
    // *data);
    llvm::Type *TypeParams[] = {CGM.IntTy, CGM.IntTy, CGM.VoidPtrTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidPtrTy, TypeParams, /*isVarArg=*/false);
    RTLFn =
        CGM.CreateRuntimeFunction(FnTy, /*Name=*/"__kmpc_task_reduction_init");
    break;
  }
  case OMPRTL__kmpc_task_reduction_get_th_data: {
    // Build void *__kmpc_task_reduction_get_th_data(int gtid, void *tg, void
    // *d);
    llvm::Type *TypeParams[] = {CGM.IntTy, CGM.VoidPtrTy, CGM.VoidPtrTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidPtrTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(
        FnTy, /*Name=*/"__kmpc_task_reduction_get_th_data");
    break;
  }
  case OMPRTL__tgt_target: {
    // Build int32_t __tgt_target(int32_t device_id, void *host_ptr, int32_t
    // arg_num, void** args_base, void **args, size_t *arg_sizes, int32_t
    // *arg_types);
    llvm::Type *TypeParams[] = {CGM.Int32Ty,
                                CGM.VoidPtrTy,
                                CGM.Int32Ty,
                                CGM.VoidPtrPtrTy,
                                CGM.VoidPtrPtrTy,
                                CGM.SizeTy->getPointerTo(),
                                CGM.Int32Ty->getPointerTo()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__tgt_target");
    break;
  }
  case OMPRTL__tgt_target_teams: {
    // Build int32_t __tgt_target_teams(int32_t device_id, void *host_ptr,
    // int32_t arg_num, void** args_base, void **args, size_t *arg_sizes,
    // int32_t *arg_types, int32_t num_teams, int32_t thread_limit);
    llvm::Type *TypeParams[] = {CGM.Int32Ty,
                                CGM.VoidPtrTy,
                                CGM.Int32Ty,
                                CGM.VoidPtrPtrTy,
                                CGM.VoidPtrPtrTy,
                                CGM.SizeTy->getPointerTo(),
                                CGM.Int32Ty->getPointerTo(),
                                CGM.Int32Ty,
                                CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__tgt_target_teams");
    break;
  }
  case OMPRTL__tgt_register_lib: {
    // Build void __tgt_register_lib(__tgt_bin_desc *desc);
    QualType ParamTy =
        CGM.getContext().getPointerType(getTgtBinaryDescriptorQTy());
    llvm::Type *TypeParams[] = {CGM.getTypes().ConvertTypeForMem(ParamTy)};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__tgt_register_lib");
    break;
  }
  case OMPRTL__tgt_unregister_lib: {
    // Build void __tgt_unregister_lib(__tgt_bin_desc *desc);
    QualType ParamTy =
        CGM.getContext().getPointerType(getTgtBinaryDescriptorQTy());
    llvm::Type *TypeParams[] = {CGM.getTypes().ConvertTypeForMem(ParamTy)};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__tgt_unregister_lib");
    break;
  }
  case OMPRTL__tgt_target_data_begin: {
    // Build void __tgt_target_data_begin(int32_t device_id, int32_t arg_num,
    // void** args_base, void **args, size_t *arg_sizes, int32_t *arg_types);
    llvm::Type *TypeParams[] = {CGM.Int32Ty,
                                CGM.Int32Ty,
                                CGM.VoidPtrPtrTy,
                                CGM.VoidPtrPtrTy,
                                CGM.SizeTy->getPointerTo(),
                                CGM.Int32Ty->getPointerTo()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__tgt_target_data_begin");
    break;
  }
  case OMPRTL__tgt_target_data_end: {
    // Build void __tgt_target_data_end(int32_t device_id, int32_t arg_num,
    // void** args_base, void **args, size_t *arg_sizes, int32_t *arg_types);
    llvm::Type *TypeParams[] = {CGM.Int32Ty,
                                CGM.Int32Ty,
                                CGM.VoidPtrPtrTy,
                                CGM.VoidPtrPtrTy,
                                CGM.SizeTy->getPointerTo(),
                                CGM.Int32Ty->getPointerTo()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__tgt_target_data_end");
    break;
  }
  case OMPRTL__tgt_target_data_update: {
    // Build void __tgt_target_data_update(int32_t device_id, int32_t arg_num,
    // void** args_base, void **args, size_t *arg_sizes, int32_t *arg_types);
    llvm::Type *TypeParams[] = {CGM.Int32Ty,
                                CGM.Int32Ty,
                                CGM.VoidPtrPtrTy,
                                CGM.VoidPtrPtrTy,
                                CGM.SizeTy->getPointerTo(),
                                CGM.Int32Ty->getPointerTo()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__tgt_target_data_update");
    break;
  }
  }
  assert(RTLFn && "Unable to find OpenMP runtime function");
  return RTLFn;
}

llvm::Constant *CGOpenMPRuntime::createForStaticInitFunction(unsigned IVSize,
                                                             bool IVSigned) {
  assert((IVSize == 32 || IVSize == 64) &&
         "IV size is not compatible with the omp runtime");
  auto Name = IVSize == 32 ? (IVSigned ? "__kmpc_for_static_init_4"
                                       : "__kmpc_for_static_init_4u")
                           : (IVSigned ? "__kmpc_for_static_init_8"
                                       : "__kmpc_for_static_init_8u");
  auto ITy = IVSize == 32 ? CGM.Int32Ty : CGM.Int64Ty;
  auto PtrTy = llvm::PointerType::getUnqual(ITy);
  llvm::Type *TypeParams[] = {
    getIdentTyPointerTy(),                     // loc
    CGM.Int32Ty,                               // tid
    CGM.Int32Ty,                               // schedtype
    llvm::PointerType::getUnqual(CGM.Int32Ty), // p_lastiter
    PtrTy,                                     // p_lower
    PtrTy,                                     // p_upper
    PtrTy,                                     // p_stride
    ITy,                                       // incr
    ITy                                        // chunk
  };
  llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
  return CGM.CreateRuntimeFunction(FnTy, Name);
}

llvm::Constant *CGOpenMPRuntime::createDispatchInitFunction(unsigned IVSize,
                                                            bool IVSigned) {
  assert((IVSize == 32 || IVSize == 64) &&
         "IV size is not compatible with the omp runtime");
  auto Name =
      IVSize == 32
          ? (IVSigned ? "__kmpc_dispatch_init_4" : "__kmpc_dispatch_init_4u")
          : (IVSigned ? "__kmpc_dispatch_init_8" : "__kmpc_dispatch_init_8u");
  auto ITy = IVSize == 32 ? CGM.Int32Ty : CGM.Int64Ty;
  llvm::Type *TypeParams[] = { getIdentTyPointerTy(), // loc
                               CGM.Int32Ty,           // tid
                               CGM.Int32Ty,           // schedtype
                               ITy,                   // lower
                               ITy,                   // upper
                               ITy,                   // stride
                               ITy                    // chunk
  };
  llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
  return CGM.CreateRuntimeFunction(FnTy, Name);
}

llvm::Constant *CGOpenMPRuntime::createDispatchFiniFunction(unsigned IVSize,
                                                            bool IVSigned) {
  assert((IVSize == 32 || IVSize == 64) &&
         "IV size is not compatible with the omp runtime");
  auto Name =
      IVSize == 32
          ? (IVSigned ? "__kmpc_dispatch_fini_4" : "__kmpc_dispatch_fini_4u")
          : (IVSigned ? "__kmpc_dispatch_fini_8" : "__kmpc_dispatch_fini_8u");
  llvm::Type *TypeParams[] = {
      getIdentTyPointerTy(), // loc
      CGM.Int32Ty,           // tid
  };
  llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
  return CGM.CreateRuntimeFunction(FnTy, Name);
}

llvm::Constant *CGOpenMPRuntime::createDispatchNextFunction(unsigned IVSize,
                                                            bool IVSigned) {
  assert((IVSize == 32 || IVSize == 64) &&
         "IV size is not compatible with the omp runtime");
  auto Name =
      IVSize == 32
          ? (IVSigned ? "__kmpc_dispatch_next_4" : "__kmpc_dispatch_next_4u")
          : (IVSigned ? "__kmpc_dispatch_next_8" : "__kmpc_dispatch_next_8u");
  auto ITy = IVSize == 32 ? CGM.Int32Ty : CGM.Int64Ty;
  auto PtrTy = llvm::PointerType::getUnqual(ITy);
  llvm::Type *TypeParams[] = {
    getIdentTyPointerTy(),                     // loc
    CGM.Int32Ty,                               // tid
    llvm::PointerType::getUnqual(CGM.Int32Ty), // p_lastiter
    PtrTy,                                     // p_lower
    PtrTy,                                     // p_upper
    PtrTy                                      // p_stride
  };
  llvm::FunctionType *FnTy =
      llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg*/ false);
  return CGM.CreateRuntimeFunction(FnTy, Name);
}

llvm::Constant *
CGOpenMPRuntime::getOrCreateThreadPrivateCache(const VarDecl *VD) {
  assert(!CGM.getLangOpts().OpenMPUseTLS ||
         !CGM.getContext().getTargetInfo().isTLSSupported());
  // Lookup the entry, lazily creating it if necessary.
  return getOrCreateInternalVariable(CGM.Int8PtrPtrTy,
                                     Twine(CGM.getMangledName(VD)) + ".cache.");
}

Address CGOpenMPRuntime::getAddrOfThreadPrivate(CodeGenFunction &CGF,
                                                const VarDecl *VD,
                                                Address VDAddr,
                                                SourceLocation Loc) {
  if (CGM.getLangOpts().OpenMPUseTLS &&
      CGM.getContext().getTargetInfo().isTLSSupported())
    return VDAddr;

  auto VarTy = VDAddr.getElementType();
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc),
                         CGF.Builder.CreatePointerCast(VDAddr.getPointer(),
                                                       CGM.Int8PtrTy),
                         CGM.getSize(CGM.GetTargetTypeStoreSize(VarTy)),
                         getOrCreateThreadPrivateCache(VD)};
  return Address(CGF.EmitRuntimeCall(
      createRuntimeFunction(OMPRTL__kmpc_threadprivate_cached), Args),
                 VDAddr.getAlignment());
}

void CGOpenMPRuntime::emitThreadPrivateVarInit(
    CodeGenFunction &CGF, Address VDAddr, llvm::Value *Ctor,
    llvm::Value *CopyCtor, llvm::Value *Dtor, SourceLocation Loc) {
  // Call kmp_int32 __kmpc_global_thread_num(&loc) to init OpenMP runtime
  // library.
  auto OMPLoc = emitUpdateLocation(CGF, Loc);
  CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_global_thread_num),
                      OMPLoc);
  // Call __kmpc_threadprivate_register(&loc, &var, ctor, cctor/*NULL*/, dtor)
  // to register constructor/destructor for variable.
  llvm::Value *Args[] = {OMPLoc,
                         CGF.Builder.CreatePointerCast(VDAddr.getPointer(),
                                                       CGM.VoidPtrTy),
                         Ctor, CopyCtor, Dtor};
  CGF.EmitRuntimeCall(
      createRuntimeFunction(OMPRTL__kmpc_threadprivate_register), Args);
}

llvm::Function *CGOpenMPRuntime::emitThreadPrivateVarDefinition(
    const VarDecl *VD, Address VDAddr, SourceLocation Loc,
    bool PerformInit, CodeGenFunction *CGF) {
  if (CGM.getLangOpts().OpenMPUseTLS &&
      CGM.getContext().getTargetInfo().isTLSSupported())
    return nullptr;

  VD = VD->getDefinition(CGM.getContext());
  if (VD && ThreadPrivateWithDefinition.count(VD) == 0) {
    ThreadPrivateWithDefinition.insert(VD);
    QualType ASTTy = VD->getType();

    llvm::Value *Ctor = nullptr, *CopyCtor = nullptr, *Dtor = nullptr;
    auto Init = VD->getAnyInitializer();
    if (CGM.getLangOpts().CPlusPlus && PerformInit) {
      // Generate function that re-emits the declaration's initializer into the
      // threadprivate copy of the variable VD
      CodeGenFunction CtorCGF(CGM);
      FunctionArgList Args;
      ImplicitParamDecl Dst(CGM.getContext(), CGM.getContext().VoidPtrTy,
                            ImplicitParamDecl::Other);
      Args.push_back(&Dst);

      auto &FI = CGM.getTypes().arrangeBuiltinFunctionDeclaration(
          CGM.getContext().VoidPtrTy, Args);
      auto FTy = CGM.getTypes().GetFunctionType(FI);
      auto Fn = CGM.CreateGlobalInitOrDestructFunction(
          FTy, ".__kmpc_global_ctor_.", FI, Loc);
      CtorCGF.StartFunction(GlobalDecl(), CGM.getContext().VoidPtrTy, Fn, FI,
                            Args, SourceLocation());
      auto ArgVal = CtorCGF.EmitLoadOfScalar(
          CtorCGF.GetAddrOfLocalVar(&Dst), /*Volatile=*/false,
          CGM.getContext().VoidPtrTy, Dst.getLocation());
      Address Arg = Address(ArgVal, VDAddr.getAlignment());
      Arg = CtorCGF.Builder.CreateElementBitCast(Arg,
                                             CtorCGF.ConvertTypeForMem(ASTTy));
      CtorCGF.EmitAnyExprToMem(Init, Arg, Init->getType().getQualifiers(),
                               /*IsInitializer=*/true);
      ArgVal = CtorCGF.EmitLoadOfScalar(
          CtorCGF.GetAddrOfLocalVar(&Dst), /*Volatile=*/false,
          CGM.getContext().VoidPtrTy, Dst.getLocation());
      CtorCGF.Builder.CreateStore(ArgVal, CtorCGF.ReturnValue);
      CtorCGF.FinishFunction();
      Ctor = Fn;
    }
    if (VD->getType().isDestructedType() != QualType::DK_none) {
      // Generate function that emits destructor call for the threadprivate copy
      // of the variable VD
      CodeGenFunction DtorCGF(CGM);
      FunctionArgList Args;
      ImplicitParamDecl Dst(CGM.getContext(), CGM.getContext().VoidPtrTy,
                            ImplicitParamDecl::Other);
      Args.push_back(&Dst);

      auto &FI = CGM.getTypes().arrangeBuiltinFunctionDeclaration(
          CGM.getContext().VoidTy, Args);
      auto FTy = CGM.getTypes().GetFunctionType(FI);
      auto Fn = CGM.CreateGlobalInitOrDestructFunction(
          FTy, ".__kmpc_global_dtor_.", FI, Loc);
      auto NL = ApplyDebugLocation::CreateEmpty(DtorCGF);
      DtorCGF.StartFunction(GlobalDecl(), CGM.getContext().VoidTy, Fn, FI, Args,
                            SourceLocation());
      // Create a scope with an artificial location for the body of this function.
      auto AL = ApplyDebugLocation::CreateArtificial(DtorCGF);
      auto ArgVal = DtorCGF.EmitLoadOfScalar(
          DtorCGF.GetAddrOfLocalVar(&Dst),
          /*Volatile=*/false, CGM.getContext().VoidPtrTy, Dst.getLocation());
      DtorCGF.emitDestroy(Address(ArgVal, VDAddr.getAlignment()), ASTTy,
                          DtorCGF.getDestroyer(ASTTy.isDestructedType()),
                          DtorCGF.needsEHCleanup(ASTTy.isDestructedType()));
      DtorCGF.FinishFunction();
      Dtor = Fn;
    }
    // Do not emit init function if it is not required.
    if (!Ctor && !Dtor)
      return nullptr;

    llvm::Type *CopyCtorTyArgs[] = {CGM.VoidPtrTy, CGM.VoidPtrTy};
    auto CopyCtorTy =
        llvm::FunctionType::get(CGM.VoidPtrTy, CopyCtorTyArgs,
                                /*isVarArg=*/false)->getPointerTo();
    // Copying constructor for the threadprivate variable.
    // Must be NULL - reserved by runtime, but currently it requires that this
    // parameter is always NULL. Otherwise it fires assertion.
    CopyCtor = llvm::Constant::getNullValue(CopyCtorTy);
    if (Ctor == nullptr) {
      auto CtorTy = llvm::FunctionType::get(CGM.VoidPtrTy, CGM.VoidPtrTy,
                                            /*isVarArg=*/false)->getPointerTo();
      Ctor = llvm::Constant::getNullValue(CtorTy);
    }
    if (Dtor == nullptr) {
      auto DtorTy = llvm::FunctionType::get(CGM.VoidTy, CGM.VoidPtrTy,
                                            /*isVarArg=*/false)->getPointerTo();
      Dtor = llvm::Constant::getNullValue(DtorTy);
    }
    if (!CGF) {
      auto InitFunctionTy =
          llvm::FunctionType::get(CGM.VoidTy, /*isVarArg*/ false);
      auto InitFunction = CGM.CreateGlobalInitOrDestructFunction(
          InitFunctionTy, ".__omp_threadprivate_init_.",
          CGM.getTypes().arrangeNullaryFunction());
      CodeGenFunction InitCGF(CGM);
      FunctionArgList ArgList;
      InitCGF.StartFunction(GlobalDecl(), CGM.getContext().VoidTy, InitFunction,
                            CGM.getTypes().arrangeNullaryFunction(), ArgList,
                            Loc);
      emitThreadPrivateVarInit(InitCGF, VDAddr, Ctor, CopyCtor, Dtor, Loc);
      InitCGF.FinishFunction();
      return InitFunction;
    }
    emitThreadPrivateVarInit(*CGF, VDAddr, Ctor, CopyCtor, Dtor, Loc);
  }
  return nullptr;
}

Address CGOpenMPRuntime::getAddrOfArtificialThreadPrivate(CodeGenFunction &CGF,
                                                          QualType VarType,
                                                          StringRef Name) {
  llvm::Twine VarName(Name, ".artificial.");
  llvm::Type *VarLVType = CGF.ConvertTypeForMem(VarType);
  llvm::Value *GAddr = getOrCreateInternalVariable(VarLVType, VarName);
  llvm::Value *Args[] = {
      emitUpdateLocation(CGF, SourceLocation()),
      getThreadID(CGF, SourceLocation()),
      CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(GAddr, CGM.VoidPtrTy),
      CGF.Builder.CreateIntCast(CGF.getTypeSize(VarType), CGM.SizeTy,
                                /*IsSigned=*/false),
      getOrCreateInternalVariable(CGM.VoidPtrPtrTy, VarName + ".cache.")};
  return Address(
      CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
          CGF.EmitRuntimeCall(
              createRuntimeFunction(OMPRTL__kmpc_threadprivate_cached), Args),
          VarLVType->getPointerTo(/*AddrSpace=*/0)),
      CGM.getPointerAlign());
}

/// \brief Emits code for OpenMP 'if' clause using specified \a CodeGen
/// function. Here is the logic:
/// if (Cond) {
///   ThenGen();
/// } else {
///   ElseGen();
/// }
void CGOpenMPRuntime::emitOMPIfClause(CodeGenFunction &CGF, const Expr *Cond,
                                      const RegionCodeGenTy &ThenGen,
                                      const RegionCodeGenTy &ElseGen) {
  CodeGenFunction::LexicalScope ConditionScope(CGF, Cond->getSourceRange());

  // If the condition constant folds and can be elided, try to avoid emitting
  // the condition and the dead arm of the if/else.
  bool CondConstant;
  if (CGF.ConstantFoldsToSimpleInteger(Cond, CondConstant)) {
    if (CondConstant)
      ThenGen(CGF);
    else
      ElseGen(CGF);
    return;
  }

  // Otherwise, the condition did not fold, or we couldn't elide it.  Just
  // emit the conditional branch.
  auto ThenBlock = CGF.createBasicBlock("omp_if.then");
  auto ElseBlock = CGF.createBasicBlock("omp_if.else");
  auto ContBlock = CGF.createBasicBlock("omp_if.end");
  CGF.EmitBranchOnBoolExpr(Cond, ThenBlock, ElseBlock, /*TrueCount=*/0);

  // Emit the 'then' code.
  CGF.EmitBlock(ThenBlock);
  ThenGen(CGF);
  CGF.EmitBranch(ContBlock);
  // Emit the 'else' code if present.
  // There is no need to emit line number for unconditional branch.
  (void)ApplyDebugLocation::CreateEmpty(CGF);
  CGF.EmitBlock(ElseBlock);
  ElseGen(CGF);
  // There is no need to emit line number for unconditional branch.
  (void)ApplyDebugLocation::CreateEmpty(CGF);
  CGF.EmitBranch(ContBlock);
  // Emit the continuation block for code after the if.
  CGF.EmitBlock(ContBlock, /*IsFinished=*/true);
}

void CGOpenMPRuntime::emitParallelCall(CodeGenFunction &CGF, SourceLocation Loc,
                                       llvm::Value *OutlinedFn,
                                       ArrayRef<llvm::Value *> CapturedVars,
                                       const Expr *IfCond) {
  if (!CGF.HaveInsertPoint())
    return;
  auto *RTLoc = emitUpdateLocation(CGF, Loc);
  auto &&ThenGen = [OutlinedFn, CapturedVars, RTLoc](CodeGenFunction &CGF,
                                                     PrePostActionTy &) {
    // Build call __kmpc_fork_call(loc, n, microtask, var1, .., varn);
    auto &RT = CGF.CGM.getOpenMPRuntime();
    llvm::Value *Args[] = {
        RTLoc,
        CGF.Builder.getInt32(CapturedVars.size()), // Number of captured vars
        CGF.Builder.CreateBitCast(OutlinedFn, RT.getKmpc_MicroPointerTy())};
    llvm::SmallVector<llvm::Value *, 16> RealArgs;
    RealArgs.append(std::begin(Args), std::end(Args));
    RealArgs.append(CapturedVars.begin(), CapturedVars.end());

    auto RTLFn = RT.createRuntimeFunction(OMPRTL__kmpc_fork_call);
    CGF.EmitRuntimeCall(RTLFn, RealArgs);
  };
  auto &&ElseGen = [OutlinedFn, CapturedVars, RTLoc, Loc](CodeGenFunction &CGF,
                                                          PrePostActionTy &) {
    auto &RT = CGF.CGM.getOpenMPRuntime();
    auto ThreadID = RT.getThreadID(CGF, Loc);
    // Build calls:
    // __kmpc_serialized_parallel(&Loc, GTid);
    llvm::Value *Args[] = {RTLoc, ThreadID};
    CGF.EmitRuntimeCall(
        RT.createRuntimeFunction(OMPRTL__kmpc_serialized_parallel), Args);

    // OutlinedFn(&GTid, &zero, CapturedStruct);
    auto ThreadIDAddr = RT.emitThreadIDAddress(CGF, Loc);
    Address ZeroAddr =
        CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4),
                             /*Name*/ ".zero.addr");
    CGF.InitTempAlloca(ZeroAddr, CGF.Builder.getInt32(/*C*/ 0));
    llvm::SmallVector<llvm::Value *, 16> OutlinedFnArgs;
    OutlinedFnArgs.push_back(ThreadIDAddr.getPointer());
    OutlinedFnArgs.push_back(ZeroAddr.getPointer());
    OutlinedFnArgs.append(CapturedVars.begin(), CapturedVars.end());
    RT.emitOutlinedFunctionCall(CGF, Loc, OutlinedFn, OutlinedFnArgs);

    // __kmpc_end_serialized_parallel(&Loc, GTid);
    llvm::Value *EndArgs[] = {RT.emitUpdateLocation(CGF, Loc), ThreadID};
    CGF.EmitRuntimeCall(
        RT.createRuntimeFunction(OMPRTL__kmpc_end_serialized_parallel),
        EndArgs);
  };
  if (IfCond)
    emitOMPIfClause(CGF, IfCond, ThenGen, ElseGen);
  else {
    RegionCodeGenTy ThenRCG(ThenGen);
    ThenRCG(CGF);
  }
}

// If we're inside an (outlined) parallel region, use the region info's
// thread-ID variable (it is passed in a first argument of the outlined function
// as "kmp_int32 *gtid"). Otherwise, if we're not inside parallel region, but in
// regular serial code region, get thread ID by calling kmp_int32
// kmpc_global_thread_num(ident_t *loc), stash this thread ID in a temporary and
// return the address of that temp.
Address CGOpenMPRuntime::emitThreadIDAddress(CodeGenFunction &CGF,
                                             SourceLocation Loc) {
  if (auto *OMPRegionInfo =
          dyn_cast_or_null<CGOpenMPRegionInfo>(CGF.CapturedStmtInfo))
    if (OMPRegionInfo->getThreadIDVariable())
      return OMPRegionInfo->getThreadIDVariableLValue(CGF).getAddress();

  auto ThreadID = getThreadID(CGF, Loc);
  auto Int32Ty =
      CGF.getContext().getIntTypeForBitwidth(/*DestWidth*/ 32, /*Signed*/ true);
  auto ThreadIDTemp = CGF.CreateMemTemp(Int32Ty, /*Name*/ ".threadid_temp.");
  CGF.EmitStoreOfScalar(ThreadID,
                        CGF.MakeAddrLValue(ThreadIDTemp, Int32Ty));

  return ThreadIDTemp;
}

llvm::Constant *
CGOpenMPRuntime::getOrCreateInternalVariable(llvm::Type *Ty,
                                             const llvm::Twine &Name) {
  SmallString<256> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  Out << Name;
  auto RuntimeName = Out.str();
  auto &Elem = *InternalVars.insert(std::make_pair(RuntimeName, nullptr)).first;
  if (Elem.second) {
    assert(Elem.second->getType()->getPointerElementType() == Ty &&
           "OMP internal variable has different type than requested");
    return &*Elem.second;
  }

  return Elem.second = new llvm::GlobalVariable(
             CGM.getModule(), Ty, /*IsConstant*/ false,
             llvm::GlobalValue::CommonLinkage, llvm::Constant::getNullValue(Ty),
             Elem.first());
}

llvm::Value *CGOpenMPRuntime::getCriticalRegionLock(StringRef CriticalName) {
  llvm::Twine Name(".gomp_critical_user_", CriticalName);
  return getOrCreateInternalVariable(KmpCriticalNameTy, Name.concat(".var"));
}

namespace {
/// Common pre(post)-action for different OpenMP constructs.
class CommonActionTy final : public PrePostActionTy {
  llvm::Value *EnterCallee;
  ArrayRef<llvm::Value *> EnterArgs;
  llvm::Value *ExitCallee;
  ArrayRef<llvm::Value *> ExitArgs;
  bool Conditional;
  llvm::BasicBlock *ContBlock = nullptr;

public:
  CommonActionTy(llvm::Value *EnterCallee, ArrayRef<llvm::Value *> EnterArgs,
                 llvm::Value *ExitCallee, ArrayRef<llvm::Value *> ExitArgs,
                 bool Conditional = false)
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
} // anonymous namespace

void CGOpenMPRuntime::emitCriticalRegion(CodeGenFunction &CGF,
                                         StringRef CriticalName,
                                         const RegionCodeGenTy &CriticalOpGen,
                                         SourceLocation Loc, const Expr *Hint) {
  // __kmpc_critical[_with_hint](ident_t *, gtid, Lock[, hint]);
  // CriticalOpGen();
  // __kmpc_end_critical(ident_t *, gtid, Lock);
  // Prepare arguments and build a call to __kmpc_critical
  if (!CGF.HaveInsertPoint())
    return;
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc),
                         getCriticalRegionLock(CriticalName)};
  llvm::SmallVector<llvm::Value *, 4> EnterArgs(std::begin(Args),
                                                std::end(Args));
  if (Hint) {
    EnterArgs.push_back(CGF.Builder.CreateIntCast(
        CGF.EmitScalarExpr(Hint), CGM.IntPtrTy, /*isSigned=*/false));
  }
  CommonActionTy Action(
      createRuntimeFunction(Hint ? OMPRTL__kmpc_critical_with_hint
                                 : OMPRTL__kmpc_critical),
      EnterArgs, createRuntimeFunction(OMPRTL__kmpc_end_critical), Args);
  CriticalOpGen.setAction(Action);
  emitInlinedDirective(CGF, OMPD_critical, CriticalOpGen);
}

void CGOpenMPRuntime::emitMasterRegion(CodeGenFunction &CGF,
                                       const RegionCodeGenTy &MasterOpGen,
                                       SourceLocation Loc) {
  if (!CGF.HaveInsertPoint())
    return;
  // if(__kmpc_master(ident_t *, gtid)) {
  //   MasterOpGen();
  //   __kmpc_end_master(ident_t *, gtid);
  // }
  // Prepare arguments and build a call to __kmpc_master
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc)};
  CommonActionTy Action(createRuntimeFunction(OMPRTL__kmpc_master), Args,
                        createRuntimeFunction(OMPRTL__kmpc_end_master), Args,
                        /*Conditional=*/true);
  MasterOpGen.setAction(Action);
  emitInlinedDirective(CGF, OMPD_master, MasterOpGen);
  Action.Done(CGF);
}

void CGOpenMPRuntime::emitTaskyieldCall(CodeGenFunction &CGF,
                                        SourceLocation Loc) {
  if (!CGF.HaveInsertPoint())
    return;
  // Build call __kmpc_omp_taskyield(loc, thread_id, 0);
  llvm::Value *Args[] = {
      emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc),
      llvm::ConstantInt::get(CGM.IntTy, /*V=*/0, /*isSigned=*/true)};
  CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_omp_taskyield), Args);
  if (auto *Region = dyn_cast_or_null<CGOpenMPRegionInfo>(CGF.CapturedStmtInfo))
    Region->emitUntiedSwitch(CGF);
}

void CGOpenMPRuntime::emitTaskgroupRegion(CodeGenFunction &CGF,
                                          const RegionCodeGenTy &TaskgroupOpGen,
                                          SourceLocation Loc) {
  if (!CGF.HaveInsertPoint())
    return;
  // __kmpc_taskgroup(ident_t *, gtid);
  // TaskgroupOpGen();
  // __kmpc_end_taskgroup(ident_t *, gtid);
  // Prepare arguments and build a call to __kmpc_taskgroup
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc)};
  CommonActionTy Action(createRuntimeFunction(OMPRTL__kmpc_taskgroup), Args,
                        createRuntimeFunction(OMPRTL__kmpc_end_taskgroup),
                        Args);
  TaskgroupOpGen.setAction(Action);
  emitInlinedDirective(CGF, OMPD_taskgroup, TaskgroupOpGen);
}

/// Given an array of pointers to variables, project the address of a
/// given variable.
static Address emitAddrOfVarFromArray(CodeGenFunction &CGF, Address Array,
                                      unsigned Index, const VarDecl *Var) {
  // Pull out the pointer to the variable.
  Address PtrAddr =
      CGF.Builder.CreateConstArrayGEP(Array, Index, CGF.getPointerSize());
  llvm::Value *Ptr = CGF.Builder.CreateLoad(PtrAddr);

  Address Addr = Address(Ptr, CGF.getContext().getDeclAlign(Var));
  Addr = CGF.Builder.CreateElementBitCast(
      Addr, CGF.ConvertTypeForMem(Var->getType()));
  return Addr;
}

static llvm::Value *emitCopyprivateCopyFunction(
    CodeGenModule &CGM, llvm::Type *ArgsType,
    ArrayRef<const Expr *> CopyprivateVars, ArrayRef<const Expr *> DestExprs,
    ArrayRef<const Expr *> SrcExprs, ArrayRef<const Expr *> AssignmentOps) {
  auto &C = CGM.getContext();
  // void copy_func(void *LHSArg, void *RHSArg);
  FunctionArgList Args;
  ImplicitParamDecl LHSArg(C, C.VoidPtrTy, ImplicitParamDecl::Other);
  ImplicitParamDecl RHSArg(C, C.VoidPtrTy, ImplicitParamDecl::Other);
  Args.push_back(&LHSArg);
  Args.push_back(&RHSArg);
  auto &CGFI = CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  auto *Fn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(CGFI), llvm::GlobalValue::InternalLinkage,
      ".omp.copyprivate.copy_func", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*D=*/nullptr, Fn, CGFI);
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, CGFI, Args);
  // Dest = (void*[n])(LHSArg);
  // Src = (void*[n])(RHSArg);
  Address LHS(CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      CGF.Builder.CreateLoad(CGF.GetAddrOfLocalVar(&LHSArg)),
      ArgsType), CGF.getPointerAlign());
  Address RHS(CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      CGF.Builder.CreateLoad(CGF.GetAddrOfLocalVar(&RHSArg)),
      ArgsType), CGF.getPointerAlign());
  // *(Type0*)Dst[0] = *(Type0*)Src[0];
  // *(Type1*)Dst[1] = *(Type1*)Src[1];
  // ...
  // *(Typen*)Dst[n] = *(Typen*)Src[n];
  for (unsigned I = 0, E = AssignmentOps.size(); I < E; ++I) {
    auto DestVar = cast<VarDecl>(cast<DeclRefExpr>(DestExprs[I])->getDecl());
    Address DestAddr = emitAddrOfVarFromArray(CGF, LHS, I, DestVar);

    auto SrcVar = cast<VarDecl>(cast<DeclRefExpr>(SrcExprs[I])->getDecl());
    Address SrcAddr = emitAddrOfVarFromArray(CGF, RHS, I, SrcVar);

    auto *VD = cast<DeclRefExpr>(CopyprivateVars[I])->getDecl();
    QualType Type = VD->getType();
    CGF.EmitOMPCopy(Type, DestAddr, SrcAddr, DestVar, SrcVar, AssignmentOps[I]);
  }
  CGF.FinishFunction();
  return Fn;
}

void CGOpenMPRuntime::emitSingleRegion(CodeGenFunction &CGF,
                                       const RegionCodeGenTy &SingleOpGen,
                                       SourceLocation Loc,
                                       ArrayRef<const Expr *> CopyprivateVars,
                                       ArrayRef<const Expr *> SrcExprs,
                                       ArrayRef<const Expr *> DstExprs,
                                       ArrayRef<const Expr *> AssignmentOps) {
  if (!CGF.HaveInsertPoint())
    return;
  assert(CopyprivateVars.size() == SrcExprs.size() &&
         CopyprivateVars.size() == DstExprs.size() &&
         CopyprivateVars.size() == AssignmentOps.size());
  auto &C = CGM.getContext();
  // int32 did_it = 0;
  // if(__kmpc_single(ident_t *, gtid)) {
  //   SingleOpGen();
  //   __kmpc_end_single(ident_t *, gtid);
  //   did_it = 1;
  // }
  // call __kmpc_copyprivate(ident_t *, gtid, <buf_size>, <copyprivate list>,
  // <copy_func>, did_it);

  Address DidIt = Address::invalid();
  if (!CopyprivateVars.empty()) {
    // int32 did_it = 0;
    auto KmpInt32Ty = C.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/1);
    DidIt = CGF.CreateMemTemp(KmpInt32Ty, ".omp.copyprivate.did_it");
    CGF.Builder.CreateStore(CGF.Builder.getInt32(0), DidIt);
  }
  // Prepare arguments and build a call to __kmpc_single
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc)};
  CommonActionTy Action(createRuntimeFunction(OMPRTL__kmpc_single), Args,
                        createRuntimeFunction(OMPRTL__kmpc_end_single), Args,
                        /*Conditional=*/true);
  SingleOpGen.setAction(Action);
  emitInlinedDirective(CGF, OMPD_single, SingleOpGen);
  if (DidIt.isValid()) {
    // did_it = 1;
    CGF.Builder.CreateStore(CGF.Builder.getInt32(1), DidIt);
  }
  Action.Done(CGF);
  // call __kmpc_copyprivate(ident_t *, gtid, <buf_size>, <copyprivate list>,
  // <copy_func>, did_it);
  if (DidIt.isValid()) {
    llvm::APInt ArraySize(/*unsigned int numBits=*/32, CopyprivateVars.size());
    auto CopyprivateArrayTy =
        C.getConstantArrayType(C.VoidPtrTy, ArraySize, ArrayType::Normal,
                               /*IndexTypeQuals=*/0);
    // Create a list of all private variables for copyprivate.
    Address CopyprivateList =
        CGF.CreateMemTemp(CopyprivateArrayTy, ".omp.copyprivate.cpr_list");
    for (unsigned I = 0, E = CopyprivateVars.size(); I < E; ++I) {
      Address Elem = CGF.Builder.CreateConstArrayGEP(
          CopyprivateList, I, CGF.getPointerSize());
      CGF.Builder.CreateStore(
          CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
              CGF.EmitLValue(CopyprivateVars[I]).getPointer(), CGF.VoidPtrTy),
          Elem);
    }
    // Build function that copies private values from single region to all other
    // threads in the corresponding parallel region.
    auto *CpyFn = emitCopyprivateCopyFunction(
        CGM, CGF.ConvertTypeForMem(CopyprivateArrayTy)->getPointerTo(),
        CopyprivateVars, SrcExprs, DstExprs, AssignmentOps);
    auto *BufSize = CGF.getTypeSize(CopyprivateArrayTy);
    Address CL =
      CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(CopyprivateList,
                                                      CGF.VoidPtrTy);
    auto *DidItVal = CGF.Builder.CreateLoad(DidIt);
    llvm::Value *Args[] = {
        emitUpdateLocation(CGF, Loc), // ident_t *<loc>
        getThreadID(CGF, Loc),        // i32 <gtid>
        BufSize,                      // size_t <buf_size>
        CL.getPointer(),              // void *<copyprivate list>
        CpyFn,                        // void (*) (void *, void *) <copy_func>
        DidItVal                      // i32 did_it
    };
    CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_copyprivate), Args);
  }
}

void CGOpenMPRuntime::emitOrderedRegion(CodeGenFunction &CGF,
                                        const RegionCodeGenTy &OrderedOpGen,
                                        SourceLocation Loc, bool IsThreads) {
  if (!CGF.HaveInsertPoint())
    return;
  // __kmpc_ordered(ident_t *, gtid);
  // OrderedOpGen();
  // __kmpc_end_ordered(ident_t *, gtid);
  // Prepare arguments and build a call to __kmpc_ordered
  if (IsThreads) {
    llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc)};
    CommonActionTy Action(createRuntimeFunction(OMPRTL__kmpc_ordered), Args,
                          createRuntimeFunction(OMPRTL__kmpc_end_ordered),
                          Args);
    OrderedOpGen.setAction(Action);
    emitInlinedDirective(CGF, OMPD_ordered, OrderedOpGen);
    return;
  }
  emitInlinedDirective(CGF, OMPD_ordered, OrderedOpGen);
}

void CGOpenMPRuntime::emitBarrierCall(CodeGenFunction &CGF, SourceLocation Loc,
                                      OpenMPDirectiveKind Kind, bool EmitChecks,
                                      bool ForceSimpleCall) {
  if (!CGF.HaveInsertPoint())
    return;
  // Build call __kmpc_cancel_barrier(loc, thread_id);
  // Build call __kmpc_barrier(loc, thread_id);
  unsigned Flags;
  if (Kind == OMPD_for)
    Flags = OMP_IDENT_BARRIER_IMPL_FOR;
  else if (Kind == OMPD_sections)
    Flags = OMP_IDENT_BARRIER_IMPL_SECTIONS;
  else if (Kind == OMPD_single)
    Flags = OMP_IDENT_BARRIER_IMPL_SINGLE;
  else if (Kind == OMPD_barrier)
    Flags = OMP_IDENT_BARRIER_EXPL;
  else
    Flags = OMP_IDENT_BARRIER_IMPL;
  // Build call __kmpc_cancel_barrier(loc, thread_id) or __kmpc_barrier(loc,
  // thread_id);
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc, Flags),
                         getThreadID(CGF, Loc)};
  if (auto *OMPRegionInfo =
          dyn_cast_or_null<CGOpenMPRegionInfo>(CGF.CapturedStmtInfo)) {
    if (!ForceSimpleCall && OMPRegionInfo->hasCancel()) {
      auto *Result = CGF.EmitRuntimeCall(
          createRuntimeFunction(OMPRTL__kmpc_cancel_barrier), Args);
      if (EmitChecks) {
        // if (__kmpc_cancel_barrier()) {
        //   exit from construct;
        // }
        auto *ExitBB = CGF.createBasicBlock(".cancel.exit");
        auto *ContBB = CGF.createBasicBlock(".cancel.continue");
        auto *Cmp = CGF.Builder.CreateIsNotNull(Result);
        CGF.Builder.CreateCondBr(Cmp, ExitBB, ContBB);
        CGF.EmitBlock(ExitBB);
        //   exit from construct;
        auto CancelDestination =
            CGF.getOMPCancelDestination(OMPRegionInfo->getDirectiveKind());
        CGF.EmitBranchThroughCleanup(CancelDestination);
        CGF.EmitBlock(ContBB, /*IsFinished=*/true);
      }
      return;
    }
  }
  CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_barrier), Args);
}

/// \brief Map the OpenMP loop schedule to the runtime enumeration.
static OpenMPSchedType getRuntimeSchedule(OpenMPScheduleClauseKind ScheduleKind,
                                          bool Chunked, bool Ordered) {
  switch (ScheduleKind) {
  case OMPC_SCHEDULE_static:
    return Chunked ? (Ordered ? OMP_ord_static_chunked : OMP_sch_static_chunked)
                   : (Ordered ? OMP_ord_static : OMP_sch_static);
  case OMPC_SCHEDULE_dynamic:
    return Ordered ? OMP_ord_dynamic_chunked : OMP_sch_dynamic_chunked;
  case OMPC_SCHEDULE_guided:
    return Ordered ? OMP_ord_guided_chunked : OMP_sch_guided_chunked;
  case OMPC_SCHEDULE_runtime:
    return Ordered ? OMP_ord_runtime : OMP_sch_runtime;
  case OMPC_SCHEDULE_auto:
    return Ordered ? OMP_ord_auto : OMP_sch_auto;
  case OMPC_SCHEDULE_unknown:
    assert(!Chunked && "chunk was specified but schedule kind not known");
    return Ordered ? OMP_ord_static : OMP_sch_static;
  }
  llvm_unreachable("Unexpected runtime schedule");
}

/// \brief Map the OpenMP distribute schedule to the runtime enumeration.
static OpenMPSchedType
getRuntimeSchedule(OpenMPDistScheduleClauseKind ScheduleKind, bool Chunked) {
  // only static is allowed for dist_schedule
  return Chunked ? OMP_dist_sch_static_chunked : OMP_dist_sch_static;
}

bool CGOpenMPRuntime::isStaticNonchunked(OpenMPScheduleClauseKind ScheduleKind,
                                         bool Chunked) const {
  auto Schedule = getRuntimeSchedule(ScheduleKind, Chunked, /*Ordered=*/false);
  return Schedule == OMP_sch_static;
}

bool CGOpenMPRuntime::isStaticNonchunked(
    OpenMPDistScheduleClauseKind ScheduleKind, bool Chunked) const {
  auto Schedule = getRuntimeSchedule(ScheduleKind, Chunked);
  return Schedule == OMP_dist_sch_static;
}


bool CGOpenMPRuntime::isDynamic(OpenMPScheduleClauseKind ScheduleKind) const {
  auto Schedule =
      getRuntimeSchedule(ScheduleKind, /*Chunked=*/false, /*Ordered=*/false);
  assert(Schedule != OMP_sch_static_chunked && "cannot be chunked here");
  return Schedule != OMP_sch_static;
}

static int addMonoNonMonoModifier(OpenMPSchedType Schedule,
                                  OpenMPScheduleClauseModifier M1,
                                  OpenMPScheduleClauseModifier M2) {
  int Modifier = 0;
  switch (M1) {
  case OMPC_SCHEDULE_MODIFIER_monotonic:
    Modifier = OMP_sch_modifier_monotonic;
    break;
  case OMPC_SCHEDULE_MODIFIER_nonmonotonic:
    Modifier = OMP_sch_modifier_nonmonotonic;
    break;
  case OMPC_SCHEDULE_MODIFIER_simd:
    if (Schedule == OMP_sch_static_chunked)
      Schedule = OMP_sch_static_balanced_chunked;
    break;
  case OMPC_SCHEDULE_MODIFIER_last:
  case OMPC_SCHEDULE_MODIFIER_unknown:
    break;
  }
  switch (M2) {
  case OMPC_SCHEDULE_MODIFIER_monotonic:
    Modifier = OMP_sch_modifier_monotonic;
    break;
  case OMPC_SCHEDULE_MODIFIER_nonmonotonic:
    Modifier = OMP_sch_modifier_nonmonotonic;
    break;
  case OMPC_SCHEDULE_MODIFIER_simd:
    if (Schedule == OMP_sch_static_chunked)
      Schedule = OMP_sch_static_balanced_chunked;
    break;
  case OMPC_SCHEDULE_MODIFIER_last:
  case OMPC_SCHEDULE_MODIFIER_unknown:
    break;
  }
  return Schedule | Modifier;
}

void CGOpenMPRuntime::emitForDispatchInit(
    CodeGenFunction &CGF, SourceLocation Loc,
    const OpenMPScheduleTy &ScheduleKind, unsigned IVSize, bool IVSigned,
    bool Ordered, const DispatchRTInput &DispatchValues) {
  if (!CGF.HaveInsertPoint())
    return;
  OpenMPSchedType Schedule = getRuntimeSchedule(
      ScheduleKind.Schedule, DispatchValues.Chunk != nullptr, Ordered);
  assert(Ordered ||
         (Schedule != OMP_sch_static && Schedule != OMP_sch_static_chunked &&
          Schedule != OMP_ord_static && Schedule != OMP_ord_static_chunked &&
          Schedule != OMP_sch_static_balanced_chunked));
  // Call __kmpc_dispatch_init(
  //          ident_t *loc, kmp_int32 tid, kmp_int32 schedule,
  //          kmp_int[32|64] lower, kmp_int[32|64] upper,
  //          kmp_int[32|64] stride, kmp_int[32|64] chunk);

  // If the Chunk was not specified in the clause - use default value 1.
  llvm::Value *Chunk = DispatchValues.Chunk ? DispatchValues.Chunk
                                            : CGF.Builder.getIntN(IVSize, 1);
  llvm::Value *Args[] = {
      emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc),
      CGF.Builder.getInt32(addMonoNonMonoModifier(
          Schedule, ScheduleKind.M1, ScheduleKind.M2)), // Schedule type
      DispatchValues.LB,                                // Lower
      DispatchValues.UB,                                // Upper
      CGF.Builder.getIntN(IVSize, 1),                   // Stride
      Chunk                                             // Chunk
  };
  CGF.EmitRuntimeCall(createDispatchInitFunction(IVSize, IVSigned), Args);
}

static void emitForStaticInitCall(
    CodeGenFunction &CGF, llvm::Value *UpdateLocation, llvm::Value *ThreadId,
    llvm::Constant *ForStaticInitFunction, OpenMPSchedType Schedule,
    OpenMPScheduleClauseModifier M1, OpenMPScheduleClauseModifier M2,
    const CGOpenMPRuntime::StaticRTInput &Values) {
  if (!CGF.HaveInsertPoint())
    return;

  assert(!Values.Ordered);
  assert(Schedule == OMP_sch_static || Schedule == OMP_sch_static_chunked ||
         Schedule == OMP_sch_static_balanced_chunked ||
         Schedule == OMP_ord_static || Schedule == OMP_ord_static_chunked ||
         Schedule == OMP_dist_sch_static ||
         Schedule == OMP_dist_sch_static_chunked);

  // Call __kmpc_for_static_init(
  //          ident_t *loc, kmp_int32 tid, kmp_int32 schedtype,
  //          kmp_int32 *p_lastiter, kmp_int[32|64] *p_lower,
  //          kmp_int[32|64] *p_upper, kmp_int[32|64] *p_stride,
  //          kmp_int[32|64] incr, kmp_int[32|64] chunk);
  llvm::Value *Chunk = Values.Chunk;
  if (Chunk == nullptr) {
    assert((Schedule == OMP_sch_static || Schedule == OMP_ord_static ||
            Schedule == OMP_dist_sch_static) &&
           "expected static non-chunked schedule");
    // If the Chunk was not specified in the clause - use default value 1.
    Chunk = CGF.Builder.getIntN(Values.IVSize, 1);
  } else {
    assert((Schedule == OMP_sch_static_chunked ||
            Schedule == OMP_sch_static_balanced_chunked ||
            Schedule == OMP_ord_static_chunked ||
            Schedule == OMP_dist_sch_static_chunked) &&
           "expected static chunked schedule");
  }
  llvm::Value *Args[] = {
      UpdateLocation,
      ThreadId,
      CGF.Builder.getInt32(addMonoNonMonoModifier(Schedule, M1,
                                                  M2)), // Schedule type
      Values.IL.getPointer(),                           // &isLastIter
      Values.LB.getPointer(),                           // &LB
      Values.UB.getPointer(),                           // &UB
      Values.ST.getPointer(),                           // &Stride
      CGF.Builder.getIntN(Values.IVSize, 1),            // Incr
      Chunk                                             // Chunk
  };
  CGF.EmitRuntimeCall(ForStaticInitFunction, Args);
}

void CGOpenMPRuntime::emitForStaticInit(CodeGenFunction &CGF,
                                        SourceLocation Loc,
                                        OpenMPDirectiveKind DKind,
                                        const OpenMPScheduleTy &ScheduleKind,
                                        const StaticRTInput &Values) {
  OpenMPSchedType ScheduleNum = getRuntimeSchedule(
      ScheduleKind.Schedule, Values.Chunk != nullptr, Values.Ordered);
  assert(isOpenMPWorksharingDirective(DKind) &&
         "Expected loop-based or sections-based directive.");
  auto *UpdatedLocation = emitUpdateLocation(CGF, Loc,
                                             isOpenMPLoopDirective(DKind)
                                                 ? OMP_IDENT_WORK_LOOP
                                                 : OMP_IDENT_WORK_SECTIONS);
  auto *ThreadId = getThreadID(CGF, Loc);
  auto *StaticInitFunction =
      createForStaticInitFunction(Values.IVSize, Values.IVSigned);
  emitForStaticInitCall(CGF, UpdatedLocation, ThreadId, StaticInitFunction,
                        ScheduleNum, ScheduleKind.M1, ScheduleKind.M2, Values);
}

void CGOpenMPRuntime::emitDistributeStaticInit(
    CodeGenFunction &CGF, SourceLocation Loc,
    OpenMPDistScheduleClauseKind SchedKind,
    const CGOpenMPRuntime::StaticRTInput &Values) {
  OpenMPSchedType ScheduleNum =
      getRuntimeSchedule(SchedKind, Values.Chunk != nullptr);
  auto *UpdatedLocation =
      emitUpdateLocation(CGF, Loc, OMP_IDENT_WORK_DISTRIBUTE);
  auto *ThreadId = getThreadID(CGF, Loc);
  auto *StaticInitFunction =
      createForStaticInitFunction(Values.IVSize, Values.IVSigned);
  emitForStaticInitCall(CGF, UpdatedLocation, ThreadId, StaticInitFunction,
                        ScheduleNum, OMPC_SCHEDULE_MODIFIER_unknown,
                        OMPC_SCHEDULE_MODIFIER_unknown, Values);
}

void CGOpenMPRuntime::emitForStaticFinish(CodeGenFunction &CGF,
                                          SourceLocation Loc,
                                          OpenMPDirectiveKind DKind) {
  if (!CGF.HaveInsertPoint())
    return;
  // Call __kmpc_for_static_fini(ident_t *loc, kmp_int32 tid);
  llvm::Value *Args[] = {
      emitUpdateLocation(CGF, Loc,
                         isOpenMPDistributeDirective(DKind)
                             ? OMP_IDENT_WORK_DISTRIBUTE
                             : isOpenMPLoopDirective(DKind)
                                   ? OMP_IDENT_WORK_LOOP
                                   : OMP_IDENT_WORK_SECTIONS),
      getThreadID(CGF, Loc)};
  CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_for_static_fini),
                      Args);
}

void CGOpenMPRuntime::emitForOrderedIterationEnd(CodeGenFunction &CGF,
                                                 SourceLocation Loc,
                                                 unsigned IVSize,
                                                 bool IVSigned) {
  if (!CGF.HaveInsertPoint())
    return;
  // Call __kmpc_for_dynamic_fini_(4|8)[u](ident_t *loc, kmp_int32 tid);
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc)};
  CGF.EmitRuntimeCall(createDispatchFiniFunction(IVSize, IVSigned), Args);
}

llvm::Value *CGOpenMPRuntime::emitForNext(CodeGenFunction &CGF,
                                          SourceLocation Loc, unsigned IVSize,
                                          bool IVSigned, Address IL,
                                          Address LB, Address UB,
                                          Address ST) {
  // Call __kmpc_dispatch_next(
  //          ident_t *loc, kmp_int32 tid, kmp_int32 *p_lastiter,
  //          kmp_int[32|64] *p_lower, kmp_int[32|64] *p_upper,
  //          kmp_int[32|64] *p_stride);
  llvm::Value *Args[] = {
      emitUpdateLocation(CGF, Loc),
      getThreadID(CGF, Loc),
      IL.getPointer(), // &isLastIter
      LB.getPointer(), // &Lower
      UB.getPointer(), // &Upper
      ST.getPointer()  // &Stride
  };
  llvm::Value *Call =
      CGF.EmitRuntimeCall(createDispatchNextFunction(IVSize, IVSigned), Args);
  return CGF.EmitScalarConversion(
      Call, CGF.getContext().getIntTypeForBitwidth(32, /* Signed */ true),
      CGF.getContext().BoolTy, Loc);
}

void CGOpenMPRuntime::emitNumThreadsClause(CodeGenFunction &CGF,
                                           llvm::Value *NumThreads,
                                           SourceLocation Loc) {
  if (!CGF.HaveInsertPoint())
    return;
  // Build call __kmpc_push_num_threads(&loc, global_tid, num_threads)
  llvm::Value *Args[] = {
      emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc),
      CGF.Builder.CreateIntCast(NumThreads, CGF.Int32Ty, /*isSigned*/ true)};
  CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_push_num_threads),
                      Args);
}

void CGOpenMPRuntime::emitProcBindClause(CodeGenFunction &CGF,
                                         OpenMPProcBindClauseKind ProcBind,
                                         SourceLocation Loc) {
  if (!CGF.HaveInsertPoint())
    return;
  // Constants for proc bind value accepted by the runtime.
  enum ProcBindTy {
    ProcBindFalse = 0,
    ProcBindTrue,
    ProcBindMaster,
    ProcBindClose,
    ProcBindSpread,
    ProcBindIntel,
    ProcBindDefault
  } RuntimeProcBind;
  switch (ProcBind) {
  case OMPC_PROC_BIND_master:
    RuntimeProcBind = ProcBindMaster;
    break;
  case OMPC_PROC_BIND_close:
    RuntimeProcBind = ProcBindClose;
    break;
  case OMPC_PROC_BIND_spread:
    RuntimeProcBind = ProcBindSpread;
    break;
  case OMPC_PROC_BIND_unknown:
    llvm_unreachable("Unsupported proc_bind value.");
  }
  // Build call __kmpc_push_proc_bind(&loc, global_tid, proc_bind)
  llvm::Value *Args[] = {
      emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc),
      llvm::ConstantInt::get(CGM.IntTy, RuntimeProcBind, /*isSigned=*/true)};
  CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_push_proc_bind), Args);
}

void CGOpenMPRuntime::emitFlush(CodeGenFunction &CGF, ArrayRef<const Expr *>,
                                SourceLocation Loc) {
  if (!CGF.HaveInsertPoint())
    return;
  // Build call void __kmpc_flush(ident_t *loc)
  CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_flush),
                      emitUpdateLocation(CGF, Loc));
}

namespace {
/// \brief Indexes of fields for type kmp_task_t.
enum KmpTaskTFields {
  /// \brief List of shared variables.
  KmpTaskTShareds,
  /// \brief Task routine.
  KmpTaskTRoutine,
  /// \brief Partition id for the untied tasks.
  KmpTaskTPartId,
  /// Function with call of destructors for private variables.
  Data1,
  /// Task priority.
  Data2,
  /// (Taskloops only) Lower bound.
  KmpTaskTLowerBound,
  /// (Taskloops only) Upper bound.
  KmpTaskTUpperBound,
  /// (Taskloops only) Stride.
  KmpTaskTStride,
  /// (Taskloops only) Is last iteration flag.
  KmpTaskTLastIter,
  /// (Taskloops only) Reduction data.
  KmpTaskTReductions,
};
} // anonymous namespace

bool CGOpenMPRuntime::OffloadEntriesInfoManagerTy::empty() const {
  // FIXME: Add other entries type when they become supported.
  return OffloadEntriesTargetRegion.empty();
}

/// \brief Initialize target region entry.
void CGOpenMPRuntime::OffloadEntriesInfoManagerTy::
    initializeTargetRegionEntryInfo(unsigned DeviceID, unsigned FileID,
                                    StringRef ParentName, unsigned LineNum,
                                    unsigned Order) {
  assert(CGM.getLangOpts().OpenMPIsDevice && "Initialization of entries is "
                                             "only required for the device "
                                             "code generation.");
  OffloadEntriesTargetRegion[DeviceID][FileID][ParentName][LineNum] =
      OffloadEntryInfoTargetRegion(Order, /*Addr=*/nullptr, /*ID=*/nullptr,
                                   /*Flags=*/0);
  ++OffloadingEntriesNum;
}

void CGOpenMPRuntime::OffloadEntriesInfoManagerTy::
    registerTargetRegionEntryInfo(unsigned DeviceID, unsigned FileID,
                                  StringRef ParentName, unsigned LineNum,
                                  llvm::Constant *Addr, llvm::Constant *ID,
                                  int32_t Flags) {
  // If we are emitting code for a target, the entry is already initialized,
  // only has to be registered.
  if (CGM.getLangOpts().OpenMPIsDevice) {
    assert(hasTargetRegionEntryInfo(DeviceID, FileID, ParentName, LineNum) &&
           "Entry must exist.");
    auto &Entry =
        OffloadEntriesTargetRegion[DeviceID][FileID][ParentName][LineNum];
    assert(Entry.isValid() && "Entry not initialized!");
    Entry.setAddress(Addr);
    Entry.setID(ID);
    Entry.setFlags(Flags);
    return;
  } else {
    OffloadEntryInfoTargetRegion Entry(OffloadingEntriesNum++, Addr, ID, Flags);
    OffloadEntriesTargetRegion[DeviceID][FileID][ParentName][LineNum] = Entry;
  }
}

bool CGOpenMPRuntime::OffloadEntriesInfoManagerTy::hasTargetRegionEntryInfo(
    unsigned DeviceID, unsigned FileID, StringRef ParentName,
    unsigned LineNum) const {
  auto PerDevice = OffloadEntriesTargetRegion.find(DeviceID);
  if (PerDevice == OffloadEntriesTargetRegion.end())
    return false;
  auto PerFile = PerDevice->second.find(FileID);
  if (PerFile == PerDevice->second.end())
    return false;
  auto PerParentName = PerFile->second.find(ParentName);
  if (PerParentName == PerFile->second.end())
    return false;
  auto PerLine = PerParentName->second.find(LineNum);
  if (PerLine == PerParentName->second.end())
    return false;
  // Fail if this entry is already registered.
  if (PerLine->second.getAddress() || PerLine->second.getID())
    return false;
  return true;
}

void CGOpenMPRuntime::OffloadEntriesInfoManagerTy::actOnTargetRegionEntriesInfo(
    const OffloadTargetRegionEntryInfoActTy &Action) {
  // Scan all target region entries and perform the provided action.
  for (auto &D : OffloadEntriesTargetRegion)
    for (auto &F : D.second)
      for (auto &P : F.second)
        for (auto &L : P.second)
          Action(D.first, F.first, P.first(), L.first, L.second);
}

/// \brief Create a Ctor/Dtor-like function whose body is emitted through
/// \a Codegen. This is used to emit the two functions that register and
/// unregister the descriptor of the current compilation unit.
static llvm::Function *
createOffloadingBinaryDescriptorFunction(CodeGenModule &CGM, StringRef Name,
                                         const RegionCodeGenTy &Codegen) {
  auto &C = CGM.getContext();
  FunctionArgList Args;
  ImplicitParamDecl DummyPtr(C, C.VoidPtrTy, ImplicitParamDecl::Other);
  Args.push_back(&DummyPtr);

  CodeGenFunction CGF(CGM);
  auto &FI = CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  auto FTy = CGM.getTypes().GetFunctionType(FI);
  auto *Fn =
      CGM.CreateGlobalInitOrDestructFunction(FTy, Name, FI, SourceLocation());
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, FI, Args, SourceLocation());
  Codegen(CGF);
  CGF.FinishFunction();
  return Fn;
}

llvm::Function *
CGOpenMPRuntime::createOffloadingBinaryDescriptorRegistration() {

  // If we don't have entries or if we are emitting code for the device, we
  // don't need to do anything.
  if (CGM.getLangOpts().OpenMPIsDevice || OffloadEntriesInfoManager.empty())
    return nullptr;

  auto &M = CGM.getModule();
  auto &C = CGM.getContext();

  // Get list of devices we care about
  auto &Devices = CGM.getLangOpts().OMPTargetTriples;

  // We should be creating an offloading descriptor only if there are devices
  // specified.
  assert(!Devices.empty() && "No OpenMP offloading devices??");

  // Create the external variables that will point to the begin and end of the
  // host entries section. These will be defined by the linker.
  auto *OffloadEntryTy =
      CGM.getTypes().ConvertTypeForMem(getTgtOffloadEntryQTy());
  llvm::GlobalVariable *HostEntriesBegin = new llvm::GlobalVariable(
      M, OffloadEntryTy, /*isConstant=*/true,
      llvm::GlobalValue::ExternalLinkage, /*Initializer=*/nullptr,
      ".omp_offloading.entries_begin");
  llvm::GlobalVariable *HostEntriesEnd = new llvm::GlobalVariable(
      M, OffloadEntryTy, /*isConstant=*/true,
      llvm::GlobalValue::ExternalLinkage, /*Initializer=*/nullptr,
      ".omp_offloading.entries_end");

  // Create all device images
  auto *DeviceImageTy = cast<llvm::StructType>(
      CGM.getTypes().ConvertTypeForMem(getTgtDeviceImageQTy()));
  ConstantInitBuilder DeviceImagesBuilder(CGM);
  auto DeviceImagesEntries = DeviceImagesBuilder.beginArray(DeviceImageTy);

  for (unsigned i = 0; i < Devices.size(); ++i) {
    StringRef T = Devices[i].getTriple();
    auto *ImgBegin = new llvm::GlobalVariable(
        M, CGM.Int8Ty, /*isConstant=*/true, llvm::GlobalValue::ExternalLinkage,
        /*Initializer=*/nullptr,
        Twine(".omp_offloading.img_start.") + Twine(T));
    auto *ImgEnd = new llvm::GlobalVariable(
        M, CGM.Int8Ty, /*isConstant=*/true, llvm::GlobalValue::ExternalLinkage,
        /*Initializer=*/nullptr, Twine(".omp_offloading.img_end.") + Twine(T));

    auto Dev = DeviceImagesEntries.beginStruct(DeviceImageTy);
    Dev.add(ImgBegin);
    Dev.add(ImgEnd);
    Dev.add(HostEntriesBegin);
    Dev.add(HostEntriesEnd);
    Dev.finishAndAddTo(DeviceImagesEntries);
  }

  // Create device images global array.
  llvm::GlobalVariable *DeviceImages =
    DeviceImagesEntries.finishAndCreateGlobal(".omp_offloading.device_images",
                                              CGM.getPointerAlign(),
                                              /*isConstant=*/true);
  DeviceImages->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);

  // This is a Zero array to be used in the creation of the constant expressions
  llvm::Constant *Index[] = {llvm::Constant::getNullValue(CGM.Int32Ty),
                             llvm::Constant::getNullValue(CGM.Int32Ty)};

  // Create the target region descriptor.
  auto *BinaryDescriptorTy = cast<llvm::StructType>(
      CGM.getTypes().ConvertTypeForMem(getTgtBinaryDescriptorQTy()));
  ConstantInitBuilder DescBuilder(CGM);
  auto DescInit = DescBuilder.beginStruct(BinaryDescriptorTy);
  DescInit.addInt(CGM.Int32Ty, Devices.size());
  DescInit.add(llvm::ConstantExpr::getGetElementPtr(DeviceImages->getValueType(),
                                                    DeviceImages,
                                                    Index));
  DescInit.add(HostEntriesBegin);
  DescInit.add(HostEntriesEnd);

  auto *Desc = DescInit.finishAndCreateGlobal(".omp_offloading.descriptor",
                                              CGM.getPointerAlign(),
                                              /*isConstant=*/true);

  // Emit code to register or unregister the descriptor at execution
  // startup or closing, respectively.

  // Create a variable to drive the registration and unregistration of the
  // descriptor, so we can reuse the logic that emits Ctors and Dtors.
  auto *IdentInfo = &C.Idents.get(".omp_offloading.reg_unreg_var");
  ImplicitParamDecl RegUnregVar(C, C.getTranslationUnitDecl(), SourceLocation(),
                                IdentInfo, C.CharTy, ImplicitParamDecl::Other);

  auto *UnRegFn = createOffloadingBinaryDescriptorFunction(
      CGM, ".omp_offloading.descriptor_unreg",
      [&](CodeGenFunction &CGF, PrePostActionTy &) {
        CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__tgt_unregister_lib),
                            Desc);
      });
  auto *RegFn = createOffloadingBinaryDescriptorFunction(
      CGM, ".omp_offloading.descriptor_reg",
      [&](CodeGenFunction &CGF, PrePostActionTy &) {
        CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__tgt_register_lib),
                            Desc);
        CGM.getCXXABI().registerGlobalDtor(CGF, RegUnregVar, UnRegFn, Desc);
      });
  if (CGM.supportsCOMDAT()) {
    // It is sufficient to call registration function only once, so create a
    // COMDAT group for registration/unregistration functions and associated
    // data. That would reduce startup time and code size. Registration
    // function serves as a COMDAT group key.
    auto ComdatKey = M.getOrInsertComdat(RegFn->getName());
    RegFn->setLinkage(llvm::GlobalValue::LinkOnceAnyLinkage);
    RegFn->setVisibility(llvm::GlobalValue::HiddenVisibility);
    RegFn->setComdat(ComdatKey);
    UnRegFn->setComdat(ComdatKey);
    DeviceImages->setComdat(ComdatKey);
    Desc->setComdat(ComdatKey);
  }
  return RegFn;
}

void CGOpenMPRuntime::createOffloadEntry(llvm::Constant *ID,
                                         llvm::Constant *Addr, uint64_t Size,
                                         int32_t Flags) {
  StringRef Name = Addr->getName();
  auto *TgtOffloadEntryType = cast<llvm::StructType>(
      CGM.getTypes().ConvertTypeForMem(getTgtOffloadEntryQTy()));
  llvm::LLVMContext &C = CGM.getModule().getContext();
  llvm::Module &M = CGM.getModule();

  // Make sure the address has the right type.
  llvm::Constant *AddrPtr = llvm::ConstantExpr::getBitCast(ID, CGM.VoidPtrTy);

  // Create constant string with the name.
  llvm::Constant *StrPtrInit = llvm::ConstantDataArray::getString(C, Name);

  llvm::GlobalVariable *Str =
      new llvm::GlobalVariable(M, StrPtrInit->getType(), /*isConstant=*/true,
                               llvm::GlobalValue::InternalLinkage, StrPtrInit,
                               ".omp_offloading.entry_name");
  Str->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
  llvm::Constant *StrPtr = llvm::ConstantExpr::getBitCast(Str, CGM.Int8PtrTy);

  // We can't have any padding between symbols, so we need to have 1-byte
  // alignment.
  auto Align = CharUnits::fromQuantity(1);

  // Create the entry struct.
  ConstantInitBuilder EntryBuilder(CGM);
  auto EntryInit = EntryBuilder.beginStruct(TgtOffloadEntryType);
  EntryInit.add(AddrPtr);
  EntryInit.add(StrPtr);
  EntryInit.addInt(CGM.SizeTy, Size);
  EntryInit.addInt(CGM.Int32Ty, Flags);
  EntryInit.addInt(CGM.Int32Ty, 0);
  llvm::GlobalVariable *Entry =
    EntryInit.finishAndCreateGlobal(".omp_offloading.entry",
                                    Align,
                                    /*constant*/ true,
                                    llvm::GlobalValue::ExternalLinkage);

  // The entry has to be created in the section the linker expects it to be.
  Entry->setSection(".omp_offloading.entries");
}

void CGOpenMPRuntime::createOffloadEntriesAndInfoMetadata() {
  // Emit the offloading entries and metadata so that the device codegen side
  // can easily figure out what to emit. The produced metadata looks like
  // this:
  //
  // !omp_offload.info = !{!1, ...}
  //
  // Right now we only generate metadata for function that contain target
  // regions.

  // If we do not have entries, we dont need to do anything.
  if (OffloadEntriesInfoManager.empty())
    return;

  llvm::Module &M = CGM.getModule();
  llvm::LLVMContext &C = M.getContext();
  SmallVector<OffloadEntriesInfoManagerTy::OffloadEntryInfo *, 16>
      OrderedEntries(OffloadEntriesInfoManager.size());

  // Create the offloading info metadata node.
  llvm::NamedMDNode *MD = M.getOrInsertNamedMetadata("omp_offload.info");

  // Auxiliary methods to create metadata values and strings.
  auto getMDInt = [&](unsigned v) {
    return llvm::ConstantAsMetadata::get(
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), v));
  };

  auto getMDString = [&](StringRef v) { return llvm::MDString::get(C, v); };

  // Create function that emits metadata for each target region entry;
  auto &&TargetRegionMetadataEmitter = [&](
      unsigned DeviceID, unsigned FileID, StringRef ParentName, unsigned Line,
      OffloadEntriesInfoManagerTy::OffloadEntryInfoTargetRegion &E) {
    llvm::SmallVector<llvm::Metadata *, 32> Ops;
    // Generate metadata for target regions. Each entry of this metadata
    // contains:
    // - Entry 0 -> Kind of this type of metadata (0).
    // - Entry 1 -> Device ID of the file where the entry was identified.
    // - Entry 2 -> File ID of the file where the entry was identified.
    // - Entry 3 -> Mangled name of the function where the entry was identified.
    // - Entry 4 -> Line in the file where the entry was identified.
    // - Entry 5 -> Order the entry was created.
    // The first element of the metadata node is the kind.
    Ops.push_back(getMDInt(E.getKind()));
    Ops.push_back(getMDInt(DeviceID));
    Ops.push_back(getMDInt(FileID));
    Ops.push_back(getMDString(ParentName));
    Ops.push_back(getMDInt(Line));
    Ops.push_back(getMDInt(E.getOrder()));

    // Save this entry in the right position of the ordered entries array.
    OrderedEntries[E.getOrder()] = &E;

    // Add metadata to the named metadata node.
    MD->addOperand(llvm::MDNode::get(C, Ops));
  };

  OffloadEntriesInfoManager.actOnTargetRegionEntriesInfo(
      TargetRegionMetadataEmitter);

  for (auto *E : OrderedEntries) {
    assert(E && "All ordered entries must exist!");
    if (auto *CE =
            dyn_cast<OffloadEntriesInfoManagerTy::OffloadEntryInfoTargetRegion>(
                E)) {
      assert(CE->getID() && CE->getAddress() &&
             "Entry ID and Addr are invalid!");
      createOffloadEntry(CE->getID(), CE->getAddress(), /*Size=*/0);
    } else
      llvm_unreachable("Unsupported entry kind.");
  }
}

/// \brief Loads all the offload entries information from the host IR
/// metadata.
void CGOpenMPRuntime::loadOffloadInfoMetadata() {
  // If we are in target mode, load the metadata from the host IR. This code has
  // to match the metadaata creation in createOffloadEntriesAndInfoMetadata().

  if (!CGM.getLangOpts().OpenMPIsDevice)
    return;

  if (CGM.getLangOpts().OMPHostIRFile.empty())
    return;

  auto Buf = llvm::MemoryBuffer::getFile(CGM.getLangOpts().OMPHostIRFile);
  if (Buf.getError())
    return;

  llvm::LLVMContext C;
  auto ME = expectedToErrorOrAndEmitErrors(
      C, llvm::parseBitcodeFile(Buf.get()->getMemBufferRef(), C));

  if (ME.getError())
    return;

  llvm::NamedMDNode *MD = ME.get()->getNamedMetadata("omp_offload.info");
  if (!MD)
    return;

  for (auto I : MD->operands()) {
    llvm::MDNode *MN = cast<llvm::MDNode>(I);

    auto getMDInt = [&](unsigned Idx) {
      llvm::ConstantAsMetadata *V =
          cast<llvm::ConstantAsMetadata>(MN->getOperand(Idx));
      return cast<llvm::ConstantInt>(V->getValue())->getZExtValue();
    };

    auto getMDString = [&](unsigned Idx) {
      llvm::MDString *V = cast<llvm::MDString>(MN->getOperand(Idx));
      return V->getString();
    };

    switch (getMDInt(0)) {
    default:
      llvm_unreachable("Unexpected metadata!");
      break;
    case OffloadEntriesInfoManagerTy::OffloadEntryInfo::
        OFFLOAD_ENTRY_INFO_TARGET_REGION:
      OffloadEntriesInfoManager.initializeTargetRegionEntryInfo(
          /*DeviceID=*/getMDInt(1), /*FileID=*/getMDInt(2),
          /*ParentName=*/getMDString(3), /*Line=*/getMDInt(4),
          /*Order=*/getMDInt(5));
      break;
    }
  }
}

void CGOpenMPRuntime::emitKmpRoutineEntryT(QualType KmpInt32Ty) {
  if (!KmpRoutineEntryPtrTy) {
    // Build typedef kmp_int32 (* kmp_routine_entry_t)(kmp_int32, void *); type.
    auto &C = CGM.getContext();
    QualType KmpRoutineEntryTyArgs[] = {KmpInt32Ty, C.VoidPtrTy};
    FunctionProtoType::ExtProtoInfo EPI;
    KmpRoutineEntryPtrQTy = C.getPointerType(
        C.getFunctionType(KmpInt32Ty, KmpRoutineEntryTyArgs, EPI));
    KmpRoutineEntryPtrTy = CGM.getTypes().ConvertType(KmpRoutineEntryPtrQTy);
  }
}

static FieldDecl *addFieldToRecordDecl(ASTContext &C, DeclContext *DC,
                                       QualType FieldTy) {
  auto *Field = FieldDecl::Create(
      C, DC, SourceLocation(), SourceLocation(), /*Id=*/nullptr, FieldTy,
      C.getTrivialTypeSourceInfo(FieldTy, SourceLocation()),
      /*BW=*/nullptr, /*Mutable=*/false, /*InitStyle=*/ICIS_NoInit);
  Field->setAccess(AS_public);
  DC->addDecl(Field);
  return Field;
}

QualType CGOpenMPRuntime::getTgtOffloadEntryQTy() {

  // Make sure the type of the entry is already created. This is the type we
  // have to create:
  // struct __tgt_offload_entry{
  //   void      *addr;       // Pointer to the offload entry info.
  //                          // (function or global)
  //   char      *name;       // Name of the function or global.
  //   size_t     size;       // Size of the entry info (0 if it a function).
  //   int32_t    flags;      // Flags associated with the entry, e.g. 'link'.
  //   int32_t    reserved;   // Reserved, to use by the runtime library.
  // };
  if (TgtOffloadEntryQTy.isNull()) {
    ASTContext &C = CGM.getContext();
    auto *RD = C.buildImplicitRecord("__tgt_offload_entry");
    RD->startDefinition();
    addFieldToRecordDecl(C, RD, C.VoidPtrTy);
    addFieldToRecordDecl(C, RD, C.getPointerType(C.CharTy));
    addFieldToRecordDecl(C, RD, C.getSizeType());
    addFieldToRecordDecl(
        C, RD, C.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/true));
    addFieldToRecordDecl(
        C, RD, C.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/true));
    RD->completeDefinition();
    TgtOffloadEntryQTy = C.getRecordType(RD);
  }
  return TgtOffloadEntryQTy;
}

QualType CGOpenMPRuntime::getTgtDeviceImageQTy() {
  // These are the types we need to build:
  // struct __tgt_device_image{
  // void   *ImageStart;       // Pointer to the target code start.
  // void   *ImageEnd;         // Pointer to the target code end.
  // // We also add the host entries to the device image, as it may be useful
  // // for the target runtime to have access to that information.
  // __tgt_offload_entry  *EntriesBegin;   // Begin of the table with all
  //                                       // the entries.
  // __tgt_offload_entry  *EntriesEnd;     // End of the table with all the
  //                                       // entries (non inclusive).
  // };
  if (TgtDeviceImageQTy.isNull()) {
    ASTContext &C = CGM.getContext();
    auto *RD = C.buildImplicitRecord("__tgt_device_image");
    RD->startDefinition();
    addFieldToRecordDecl(C, RD, C.VoidPtrTy);
    addFieldToRecordDecl(C, RD, C.VoidPtrTy);
    addFieldToRecordDecl(C, RD, C.getPointerType(getTgtOffloadEntryQTy()));
    addFieldToRecordDecl(C, RD, C.getPointerType(getTgtOffloadEntryQTy()));
    RD->completeDefinition();
    TgtDeviceImageQTy = C.getRecordType(RD);
  }
  return TgtDeviceImageQTy;
}

QualType CGOpenMPRuntime::getTgtBinaryDescriptorQTy() {
  // struct __tgt_bin_desc{
  //   int32_t              NumDevices;      // Number of devices supported.
  //   __tgt_device_image   *DeviceImages;   // Arrays of device images
  //                                         // (one per device).
  //   __tgt_offload_entry  *EntriesBegin;   // Begin of the table with all the
  //                                         // entries.
  //   __tgt_offload_entry  *EntriesEnd;     // End of the table with all the
  //                                         // entries (non inclusive).
  // };
  if (TgtBinaryDescriptorQTy.isNull()) {
    ASTContext &C = CGM.getContext();
    auto *RD = C.buildImplicitRecord("__tgt_bin_desc");
    RD->startDefinition();
    addFieldToRecordDecl(
        C, RD, C.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/true));
    addFieldToRecordDecl(C, RD, C.getPointerType(getTgtDeviceImageQTy()));
    addFieldToRecordDecl(C, RD, C.getPointerType(getTgtOffloadEntryQTy()));
    addFieldToRecordDecl(C, RD, C.getPointerType(getTgtOffloadEntryQTy()));
    RD->completeDefinition();
    TgtBinaryDescriptorQTy = C.getRecordType(RD);
  }
  return TgtBinaryDescriptorQTy;
}

namespace {
struct PrivateHelpersTy {
  PrivateHelpersTy(const VarDecl *Original, const VarDecl *PrivateCopy,
                   const VarDecl *PrivateElemInit)
      : Original(Original), PrivateCopy(PrivateCopy),
        PrivateElemInit(PrivateElemInit) {}
  const VarDecl *Original;
  const VarDecl *PrivateCopy;
  const VarDecl *PrivateElemInit;
};
typedef std::pair<CharUnits /*Align*/, PrivateHelpersTy> PrivateDataTy;
} // anonymous namespace

static RecordDecl *
createPrivatesRecordDecl(CodeGenModule &CGM, ArrayRef<PrivateDataTy> Privates) {
  if (!Privates.empty()) {
    auto &C = CGM.getContext();
    // Build struct .kmp_privates_t. {
    //         /*  private vars  */
    //       };
    auto *RD = C.buildImplicitRecord(".kmp_privates.t");
    RD->startDefinition();
    for (auto &&Pair : Privates) {
      auto *VD = Pair.second.Original;
      auto Type = VD->getType();
      Type = Type.getNonReferenceType();
      auto *FD = addFieldToRecordDecl(C, RD, Type);
      if (VD->hasAttrs()) {
        for (specific_attr_iterator<AlignedAttr> I(VD->getAttrs().begin()),
             E(VD->getAttrs().end());
             I != E; ++I)
          FD->addAttr(*I);
      }
    }
    RD->completeDefinition();
    return RD;
  }
  return nullptr;
}

static RecordDecl *
createKmpTaskTRecordDecl(CodeGenModule &CGM, OpenMPDirectiveKind Kind,
                         QualType KmpInt32Ty,
                         QualType KmpRoutineEntryPointerQTy) {
  auto &C = CGM.getContext();
  // Build struct kmp_task_t {
  //         void *              shareds;
  //         kmp_routine_entry_t routine;
  //         kmp_int32           part_id;
  //         kmp_cmplrdata_t data1;
  //         kmp_cmplrdata_t data2;
  // For taskloops additional fields:
  //         kmp_uint64          lb;
  //         kmp_uint64          ub;
  //         kmp_int64           st;
  //         kmp_int32           liter;
  //         void *              reductions;
  //       };
  auto *UD = C.buildImplicitRecord("kmp_cmplrdata_t", TTK_Union);
  UD->startDefinition();
  addFieldToRecordDecl(C, UD, KmpInt32Ty);
  addFieldToRecordDecl(C, UD, KmpRoutineEntryPointerQTy);
  UD->completeDefinition();
  QualType KmpCmplrdataTy = C.getRecordType(UD);
  auto *RD = C.buildImplicitRecord("kmp_task_t");
  RD->startDefinition();
  addFieldToRecordDecl(C, RD, C.VoidPtrTy);
  addFieldToRecordDecl(C, RD, KmpRoutineEntryPointerQTy);
  addFieldToRecordDecl(C, RD, KmpInt32Ty);
  addFieldToRecordDecl(C, RD, KmpCmplrdataTy);
  addFieldToRecordDecl(C, RD, KmpCmplrdataTy);
  if (isOpenMPTaskLoopDirective(Kind)) {
    QualType KmpUInt64Ty =
        CGM.getContext().getIntTypeForBitwidth(/*DestWidth=*/64, /*Signed=*/0);
    QualType KmpInt64Ty =
        CGM.getContext().getIntTypeForBitwidth(/*DestWidth=*/64, /*Signed=*/1);
    addFieldToRecordDecl(C, RD, KmpUInt64Ty);
    addFieldToRecordDecl(C, RD, KmpUInt64Ty);
    addFieldToRecordDecl(C, RD, KmpInt64Ty);
    addFieldToRecordDecl(C, RD, KmpInt32Ty);
    addFieldToRecordDecl(C, RD, C.VoidPtrTy);
  }
  RD->completeDefinition();
  return RD;
}

static RecordDecl *
createKmpTaskTWithPrivatesRecordDecl(CodeGenModule &CGM, QualType KmpTaskTQTy,
                                     ArrayRef<PrivateDataTy> Privates) {
  auto &C = CGM.getContext();
  // Build struct kmp_task_t_with_privates {
  //         kmp_task_t task_data;
  //         .kmp_privates_t. privates;
  //       };
  auto *RD = C.buildImplicitRecord("kmp_task_t_with_privates");
  RD->startDefinition();
  addFieldToRecordDecl(C, RD, KmpTaskTQTy);
  if (auto *PrivateRD = createPrivatesRecordDecl(CGM, Privates)) {
    addFieldToRecordDecl(C, RD, C.getRecordType(PrivateRD));
  }
  RD->completeDefinition();
  return RD;
}

/// \brief Emit a proxy function which accepts kmp_task_t as the second
/// argument.
/// \code
/// kmp_int32 .omp_task_entry.(kmp_int32 gtid, kmp_task_t *tt) {
///   TaskFunction(gtid, tt->part_id, &tt->privates, task_privates_map, tt,
///   For taskloops:
///   tt->task_data.lb, tt->task_data.ub, tt->task_data.st, tt->task_data.liter,
///   tt->reductions, tt->shareds);
///   return 0;
/// }
/// \endcode
static llvm::Value *
emitProxyTaskFunction(CodeGenModule &CGM, SourceLocation Loc,
                      OpenMPDirectiveKind Kind, QualType KmpInt32Ty,
                      QualType KmpTaskTWithPrivatesPtrQTy,
                      QualType KmpTaskTWithPrivatesQTy, QualType KmpTaskTQTy,
                      QualType SharedsPtrTy, llvm::Value *TaskFunction,
                      llvm::Value *TaskPrivatesMap) {
  auto &C = CGM.getContext();
  FunctionArgList Args;
  ImplicitParamDecl GtidArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, KmpInt32Ty,
                            ImplicitParamDecl::Other);
  ImplicitParamDecl TaskTypeArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                KmpTaskTWithPrivatesPtrQTy.withRestrict(),
                                ImplicitParamDecl::Other);
  Args.push_back(&GtidArg);
  Args.push_back(&TaskTypeArg);
  auto &TaskEntryFnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(KmpInt32Ty, Args);
  auto *TaskEntryTy = CGM.getTypes().GetFunctionType(TaskEntryFnInfo);
  auto *TaskEntry =
      llvm::Function::Create(TaskEntryTy, llvm::GlobalValue::InternalLinkage,
                             ".omp_task_entry.", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*D=*/nullptr, TaskEntry, TaskEntryFnInfo);
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), KmpInt32Ty, TaskEntry, TaskEntryFnInfo, Args);

  // TaskFunction(gtid, tt->task_data.part_id, &tt->privates, task_privates_map,
  // tt,
  // For taskloops:
  // tt->task_data.lb, tt->task_data.ub, tt->task_data.st, tt->task_data.liter,
  // tt->task_data.shareds);
  auto *GtidParam = CGF.EmitLoadOfScalar(
      CGF.GetAddrOfLocalVar(&GtidArg), /*Volatile=*/false, KmpInt32Ty, Loc);
  LValue TDBase = CGF.EmitLoadOfPointerLValue(
      CGF.GetAddrOfLocalVar(&TaskTypeArg),
      KmpTaskTWithPrivatesPtrQTy->castAs<PointerType>());
  auto *KmpTaskTWithPrivatesQTyRD =
      cast<RecordDecl>(KmpTaskTWithPrivatesQTy->getAsTagDecl());
  LValue Base =
      CGF.EmitLValueForField(TDBase, *KmpTaskTWithPrivatesQTyRD->field_begin());
  auto *KmpTaskTQTyRD = cast<RecordDecl>(KmpTaskTQTy->getAsTagDecl());
  auto PartIdFI = std::next(KmpTaskTQTyRD->field_begin(), KmpTaskTPartId);
  auto PartIdLVal = CGF.EmitLValueForField(Base, *PartIdFI);
  auto *PartidParam = PartIdLVal.getPointer();

  auto SharedsFI = std::next(KmpTaskTQTyRD->field_begin(), KmpTaskTShareds);
  auto SharedsLVal = CGF.EmitLValueForField(Base, *SharedsFI);
  auto *SharedsParam = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      CGF.EmitLoadOfLValue(SharedsLVal, Loc).getScalarVal(),
      CGF.ConvertTypeForMem(SharedsPtrTy));

  auto PrivatesFI = std::next(KmpTaskTWithPrivatesQTyRD->field_begin(), 1);
  llvm::Value *PrivatesParam;
  if (PrivatesFI != KmpTaskTWithPrivatesQTyRD->field_end()) {
    auto PrivatesLVal = CGF.EmitLValueForField(TDBase, *PrivatesFI);
    PrivatesParam = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
        PrivatesLVal.getPointer(), CGF.VoidPtrTy);
  } else
    PrivatesParam = llvm::ConstantPointerNull::get(CGF.VoidPtrTy);

  llvm::Value *CommonArgs[] = {GtidParam, PartidParam, PrivatesParam,
                               TaskPrivatesMap,
                               CGF.Builder
                                   .CreatePointerBitCastOrAddrSpaceCast(
                                       TDBase.getAddress(), CGF.VoidPtrTy)
                                   .getPointer()};
  SmallVector<llvm::Value *, 16> CallArgs(std::begin(CommonArgs),
                                          std::end(CommonArgs));
  if (isOpenMPTaskLoopDirective(Kind)) {
    auto LBFI = std::next(KmpTaskTQTyRD->field_begin(), KmpTaskTLowerBound);
    auto LBLVal = CGF.EmitLValueForField(Base, *LBFI);
    auto *LBParam = CGF.EmitLoadOfLValue(LBLVal, Loc).getScalarVal();
    auto UBFI = std::next(KmpTaskTQTyRD->field_begin(), KmpTaskTUpperBound);
    auto UBLVal = CGF.EmitLValueForField(Base, *UBFI);
    auto *UBParam = CGF.EmitLoadOfLValue(UBLVal, Loc).getScalarVal();
    auto StFI = std::next(KmpTaskTQTyRD->field_begin(), KmpTaskTStride);
    auto StLVal = CGF.EmitLValueForField(Base, *StFI);
    auto *StParam = CGF.EmitLoadOfLValue(StLVal, Loc).getScalarVal();
    auto LIFI = std::next(KmpTaskTQTyRD->field_begin(), KmpTaskTLastIter);
    auto LILVal = CGF.EmitLValueForField(Base, *LIFI);
    auto *LIParam = CGF.EmitLoadOfLValue(LILVal, Loc).getScalarVal();
    auto RFI = std::next(KmpTaskTQTyRD->field_begin(), KmpTaskTReductions);
    auto RLVal = CGF.EmitLValueForField(Base, *RFI);
    auto *RParam = CGF.EmitLoadOfLValue(RLVal, Loc).getScalarVal();
    CallArgs.push_back(LBParam);
    CallArgs.push_back(UBParam);
    CallArgs.push_back(StParam);
    CallArgs.push_back(LIParam);
    CallArgs.push_back(RParam);
  }
  CallArgs.push_back(SharedsParam);

  CGM.getOpenMPRuntime().emitOutlinedFunctionCall(CGF, Loc, TaskFunction,
                                                  CallArgs);
  CGF.EmitStoreThroughLValue(
      RValue::get(CGF.Builder.getInt32(/*C=*/0)),
      CGF.MakeAddrLValue(CGF.ReturnValue, KmpInt32Ty));
  CGF.FinishFunction();
  return TaskEntry;
}

static llvm::Value *emitDestructorsFunction(CodeGenModule &CGM,
                                            SourceLocation Loc,
                                            QualType KmpInt32Ty,
                                            QualType KmpTaskTWithPrivatesPtrQTy,
                                            QualType KmpTaskTWithPrivatesQTy) {
  auto &C = CGM.getContext();
  FunctionArgList Args;
  ImplicitParamDecl GtidArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, KmpInt32Ty,
                            ImplicitParamDecl::Other);
  ImplicitParamDecl TaskTypeArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                KmpTaskTWithPrivatesPtrQTy.withRestrict(),
                                ImplicitParamDecl::Other);
  Args.push_back(&GtidArg);
  Args.push_back(&TaskTypeArg);
  FunctionType::ExtInfo Info;
  auto &DestructorFnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(KmpInt32Ty, Args);
  auto *DestructorFnTy = CGM.getTypes().GetFunctionType(DestructorFnInfo);
  auto *DestructorFn =
      llvm::Function::Create(DestructorFnTy, llvm::GlobalValue::InternalLinkage,
                             ".omp_task_destructor.", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*D=*/nullptr, DestructorFn,
                                    DestructorFnInfo);
  CodeGenFunction CGF(CGM);
  CGF.disableDebugInfo();
  CGF.StartFunction(GlobalDecl(), KmpInt32Ty, DestructorFn, DestructorFnInfo,
                    Args);

  LValue Base = CGF.EmitLoadOfPointerLValue(
      CGF.GetAddrOfLocalVar(&TaskTypeArg),
      KmpTaskTWithPrivatesPtrQTy->castAs<PointerType>());
  auto *KmpTaskTWithPrivatesQTyRD =
      cast<RecordDecl>(KmpTaskTWithPrivatesQTy->getAsTagDecl());
  auto FI = std::next(KmpTaskTWithPrivatesQTyRD->field_begin());
  Base = CGF.EmitLValueForField(Base, *FI);
  for (auto *Field :
       cast<RecordDecl>(FI->getType()->getAsTagDecl())->fields()) {
    if (auto DtorKind = Field->getType().isDestructedType()) {
      auto FieldLValue = CGF.EmitLValueForField(Base, Field);
      CGF.pushDestroy(DtorKind, FieldLValue.getAddress(), Field->getType());
    }
  }
  CGF.FinishFunction();
  return DestructorFn;
}

/// \brief Emit a privates mapping function for correct handling of private and
/// firstprivate variables.
/// \code
/// void .omp_task_privates_map.(const .privates. *noalias privs, <ty1>
/// **noalias priv1,...,  <tyn> **noalias privn) {
///   *priv1 = &.privates.priv1;
///   ...;
///   *privn = &.privates.privn;
/// }
/// \endcode
static llvm::Value *
emitTaskPrivateMappingFunction(CodeGenModule &CGM, SourceLocation Loc,
                               ArrayRef<const Expr *> PrivateVars,
                               ArrayRef<const Expr *> FirstprivateVars,
                               ArrayRef<const Expr *> LastprivateVars,
                               QualType PrivatesQTy,
                               ArrayRef<PrivateDataTy> Privates) {
  auto &C = CGM.getContext();
  FunctionArgList Args;
  ImplicitParamDecl TaskPrivatesArg(
      C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
      C.getPointerType(PrivatesQTy).withConst().withRestrict(),
      ImplicitParamDecl::Other);
  Args.push_back(&TaskPrivatesArg);
  llvm::DenseMap<const VarDecl *, unsigned> PrivateVarsPos;
  unsigned Counter = 1;
  for (auto *E: PrivateVars) {
    Args.push_back(ImplicitParamDecl::Create(
        C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
        C.getPointerType(C.getPointerType(E->getType()))
            .withConst()
            .withRestrict(),
        ImplicitParamDecl::Other));
    auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
    PrivateVarsPos[VD] = Counter;
    ++Counter;
  }
  for (auto *E : FirstprivateVars) {
    Args.push_back(ImplicitParamDecl::Create(
        C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
        C.getPointerType(C.getPointerType(E->getType()))
            .withConst()
            .withRestrict(),
        ImplicitParamDecl::Other));
    auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
    PrivateVarsPos[VD] = Counter;
    ++Counter;
  }
  for (auto *E: LastprivateVars) {
    Args.push_back(ImplicitParamDecl::Create(
        C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
        C.getPointerType(C.getPointerType(E->getType()))
            .withConst()
            .withRestrict(),
        ImplicitParamDecl::Other));
    auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
    PrivateVarsPos[VD] = Counter;
    ++Counter;
  }
  auto &TaskPrivatesMapFnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  auto *TaskPrivatesMapTy =
      CGM.getTypes().GetFunctionType(TaskPrivatesMapFnInfo);
  auto *TaskPrivatesMap = llvm::Function::Create(
      TaskPrivatesMapTy, llvm::GlobalValue::InternalLinkage,
      ".omp_task_privates_map.", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*D=*/nullptr, TaskPrivatesMap,
                                    TaskPrivatesMapFnInfo);
  TaskPrivatesMap->removeFnAttr(llvm::Attribute::NoInline);
  TaskPrivatesMap->removeFnAttr(llvm::Attribute::OptimizeNone);
  TaskPrivatesMap->addFnAttr(llvm::Attribute::AlwaysInline);
  CodeGenFunction CGF(CGM);
  CGF.disableDebugInfo();
  CGF.StartFunction(GlobalDecl(), C.VoidTy, TaskPrivatesMap,
                    TaskPrivatesMapFnInfo, Args);

  // *privi = &.privates.privi;
  LValue Base = CGF.EmitLoadOfPointerLValue(
      CGF.GetAddrOfLocalVar(&TaskPrivatesArg),
      TaskPrivatesArg.getType()->castAs<PointerType>());
  auto *PrivatesQTyRD = cast<RecordDecl>(PrivatesQTy->getAsTagDecl());
  Counter = 0;
  for (auto *Field : PrivatesQTyRD->fields()) {
    auto FieldLVal = CGF.EmitLValueForField(Base, Field);
    auto *VD = Args[PrivateVarsPos[Privates[Counter].second.Original]];
    auto RefLVal = CGF.MakeAddrLValue(CGF.GetAddrOfLocalVar(VD), VD->getType());
    auto RefLoadLVal = CGF.EmitLoadOfPointerLValue(
        RefLVal.getAddress(), RefLVal.getType()->castAs<PointerType>());
    CGF.EmitStoreOfScalar(FieldLVal.getPointer(), RefLoadLVal);
    ++Counter;
  }
  CGF.FinishFunction();
  return TaskPrivatesMap;
}

static int array_pod_sort_comparator(const PrivateDataTy *P1,
                                     const PrivateDataTy *P2) {
  return P1->first < P2->first ? 1 : (P2->first < P1->first ? -1 : 0);
}

/// Emit initialization for private variables in task-based directives.
static void emitPrivatesInit(CodeGenFunction &CGF,
                             const OMPExecutableDirective &D,
                             Address KmpTaskSharedsPtr, LValue TDBase,
                             const RecordDecl *KmpTaskTWithPrivatesQTyRD,
                             QualType SharedsTy, QualType SharedsPtrTy,
                             const OMPTaskDataTy &Data,
                             ArrayRef<PrivateDataTy> Privates, bool ForDup) {
  auto &C = CGF.getContext();
  auto FI = std::next(KmpTaskTWithPrivatesQTyRD->field_begin());
  LValue PrivatesBase = CGF.EmitLValueForField(TDBase, *FI);
  LValue SrcBase;
  if (!Data.FirstprivateVars.empty()) {
    SrcBase = CGF.MakeAddrLValue(
        CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
            KmpTaskSharedsPtr, CGF.ConvertTypeForMem(SharedsPtrTy)),
        SharedsTy);
  }
  CodeGenFunction::CGCapturedStmtInfo CapturesInfo(
      cast<CapturedStmt>(*D.getAssociatedStmt()));
  FI = cast<RecordDecl>(FI->getType()->getAsTagDecl())->field_begin();
  for (auto &&Pair : Privates) {
    auto *VD = Pair.second.PrivateCopy;
    auto *Init = VD->getAnyInitializer();
    if (Init && (!ForDup || (isa<CXXConstructExpr>(Init) &&
                             !CGF.isTrivialInitializer(Init)))) {
      LValue PrivateLValue = CGF.EmitLValueForField(PrivatesBase, *FI);
      if (auto *Elem = Pair.second.PrivateElemInit) {
        auto *OriginalVD = Pair.second.Original;
        auto *SharedField = CapturesInfo.lookup(OriginalVD);
        auto SharedRefLValue = CGF.EmitLValueForField(SrcBase, SharedField);
        SharedRefLValue = CGF.MakeAddrLValue(
            Address(SharedRefLValue.getPointer(), C.getDeclAlign(OriginalVD)),
            SharedRefLValue.getType(),
            LValueBaseInfo(AlignmentSource::Decl,
                           SharedRefLValue.getBaseInfo().getMayAlias()));
        QualType Type = OriginalVD->getType();
        if (Type->isArrayType()) {
          // Initialize firstprivate array.
          if (!isa<CXXConstructExpr>(Init) || CGF.isTrivialInitializer(Init)) {
            // Perform simple memcpy.
            CGF.EmitAggregateAssign(PrivateLValue.getAddress(),
                                    SharedRefLValue.getAddress(), Type);
          } else {
            // Initialize firstprivate array using element-by-element
            // initialization.
            CGF.EmitOMPAggregateAssign(
                PrivateLValue.getAddress(), SharedRefLValue.getAddress(), Type,
                [&CGF, Elem, Init, &CapturesInfo](Address DestElement,
                                                  Address SrcElement) {
                  // Clean up any temporaries needed by the initialization.
                  CodeGenFunction::OMPPrivateScope InitScope(CGF);
                  InitScope.addPrivate(
                      Elem, [SrcElement]() -> Address { return SrcElement; });
                  (void)InitScope.Privatize();
                  // Emit initialization for single element.
                  CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(
                      CGF, &CapturesInfo);
                  CGF.EmitAnyExprToMem(Init, DestElement,
                                       Init->getType().getQualifiers(),
                                       /*IsInitializer=*/false);
                });
          }
        } else {
          CodeGenFunction::OMPPrivateScope InitScope(CGF);
          InitScope.addPrivate(Elem, [SharedRefLValue]() -> Address {
            return SharedRefLValue.getAddress();
          });
          (void)InitScope.Privatize();
          CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CapturesInfo);
          CGF.EmitExprAsInit(Init, VD, PrivateLValue,
                             /*capturedByInit=*/false);
        }
      } else
        CGF.EmitExprAsInit(Init, VD, PrivateLValue, /*capturedByInit=*/false);
    }
    ++FI;
  }
}

/// Check if duplication function is required for taskloops.
static bool checkInitIsRequired(CodeGenFunction &CGF,
                                ArrayRef<PrivateDataTy> Privates) {
  bool InitRequired = false;
  for (auto &&Pair : Privates) {
    auto *VD = Pair.second.PrivateCopy;
    auto *Init = VD->getAnyInitializer();
    InitRequired = InitRequired || (Init && isa<CXXConstructExpr>(Init) &&
                                    !CGF.isTrivialInitializer(Init));
  }
  return InitRequired;
}


/// Emit task_dup function (for initialization of
/// private/firstprivate/lastprivate vars and last_iter flag)
/// \code
/// void __task_dup_entry(kmp_task_t *task_dst, const kmp_task_t *task_src, int
/// lastpriv) {
/// // setup lastprivate flag
///    task_dst->last = lastpriv;
/// // could be constructor calls here...
/// }
/// \endcode
static llvm::Value *
emitTaskDupFunction(CodeGenModule &CGM, SourceLocation Loc,
                    const OMPExecutableDirective &D,
                    QualType KmpTaskTWithPrivatesPtrQTy,
                    const RecordDecl *KmpTaskTWithPrivatesQTyRD,
                    const RecordDecl *KmpTaskTQTyRD, QualType SharedsTy,
                    QualType SharedsPtrTy, const OMPTaskDataTy &Data,
                    ArrayRef<PrivateDataTy> Privates, bool WithLastIter) {
  auto &C = CGM.getContext();
  FunctionArgList Args;
  ImplicitParamDecl DstArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                           KmpTaskTWithPrivatesPtrQTy,
                           ImplicitParamDecl::Other);
  ImplicitParamDecl SrcArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                           KmpTaskTWithPrivatesPtrQTy,
                           ImplicitParamDecl::Other);
  ImplicitParamDecl LastprivArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, C.IntTy,
                                ImplicitParamDecl::Other);
  Args.push_back(&DstArg);
  Args.push_back(&SrcArg);
  Args.push_back(&LastprivArg);
  auto &TaskDupFnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  auto *TaskDupTy = CGM.getTypes().GetFunctionType(TaskDupFnInfo);
  auto *TaskDup =
      llvm::Function::Create(TaskDupTy, llvm::GlobalValue::InternalLinkage,
                             ".omp_task_dup.", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*D=*/nullptr, TaskDup, TaskDupFnInfo);
  CodeGenFunction CGF(CGM);
  CGF.disableDebugInfo();
  CGF.StartFunction(GlobalDecl(), C.VoidTy, TaskDup, TaskDupFnInfo, Args);

  LValue TDBase = CGF.EmitLoadOfPointerLValue(
      CGF.GetAddrOfLocalVar(&DstArg),
      KmpTaskTWithPrivatesPtrQTy->castAs<PointerType>());
  // task_dst->liter = lastpriv;
  if (WithLastIter) {
    auto LIFI = std::next(KmpTaskTQTyRD->field_begin(), KmpTaskTLastIter);
    LValue Base = CGF.EmitLValueForField(
        TDBase, *KmpTaskTWithPrivatesQTyRD->field_begin());
    LValue LILVal = CGF.EmitLValueForField(Base, *LIFI);
    llvm::Value *Lastpriv = CGF.EmitLoadOfScalar(
        CGF.GetAddrOfLocalVar(&LastprivArg), /*Volatile=*/false, C.IntTy, Loc);
    CGF.EmitStoreOfScalar(Lastpriv, LILVal);
  }

  // Emit initial values for private copies (if any).
  assert(!Privates.empty());
  Address KmpTaskSharedsPtr = Address::invalid();
  if (!Data.FirstprivateVars.empty()) {
    LValue TDBase = CGF.EmitLoadOfPointerLValue(
        CGF.GetAddrOfLocalVar(&SrcArg),
        KmpTaskTWithPrivatesPtrQTy->castAs<PointerType>());
    LValue Base = CGF.EmitLValueForField(
        TDBase, *KmpTaskTWithPrivatesQTyRD->field_begin());
    KmpTaskSharedsPtr = Address(
        CGF.EmitLoadOfScalar(CGF.EmitLValueForField(
                                 Base, *std::next(KmpTaskTQTyRD->field_begin(),
                                                  KmpTaskTShareds)),
                             Loc),
        CGF.getNaturalTypeAlignment(SharedsTy));
  }
  emitPrivatesInit(CGF, D, KmpTaskSharedsPtr, TDBase, KmpTaskTWithPrivatesQTyRD,
                   SharedsTy, SharedsPtrTy, Data, Privates, /*ForDup=*/true);
  CGF.FinishFunction();
  return TaskDup;
}

/// Checks if destructor function is required to be generated.
/// \return true if cleanups are required, false otherwise.
static bool
checkDestructorsRequired(const RecordDecl *KmpTaskTWithPrivatesQTyRD) {
  bool NeedsCleanup = false;
  auto FI = std::next(KmpTaskTWithPrivatesQTyRD->field_begin());
  auto *PrivateRD = cast<RecordDecl>(FI->getType()->getAsTagDecl());
  for (auto *FD : PrivateRD->fields()) {
    NeedsCleanup = NeedsCleanup || FD->getType().isDestructedType();
    if (NeedsCleanup)
      break;
  }
  return NeedsCleanup;
}

CGOpenMPRuntime::TaskResultTy
CGOpenMPRuntime::emitTaskInit(CodeGenFunction &CGF, SourceLocation Loc,
                              const OMPExecutableDirective &D,
                              llvm::Value *TaskFunction, QualType SharedsTy,
                              Address Shareds, const OMPTaskDataTy &Data) {
  auto &C = CGM.getContext();
  llvm::SmallVector<PrivateDataTy, 4> Privates;
  // Aggregate privates and sort them by the alignment.
  auto I = Data.PrivateCopies.begin();
  for (auto *E : Data.PrivateVars) {
    auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
    Privates.push_back(std::make_pair(
        C.getDeclAlign(VD),
        PrivateHelpersTy(VD, cast<VarDecl>(cast<DeclRefExpr>(*I)->getDecl()),
                         /*PrivateElemInit=*/nullptr)));
    ++I;
  }
  I = Data.FirstprivateCopies.begin();
  auto IElemInitRef = Data.FirstprivateInits.begin();
  for (auto *E : Data.FirstprivateVars) {
    auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
    Privates.push_back(std::make_pair(
        C.getDeclAlign(VD),
        PrivateHelpersTy(
            VD, cast<VarDecl>(cast<DeclRefExpr>(*I)->getDecl()),
            cast<VarDecl>(cast<DeclRefExpr>(*IElemInitRef)->getDecl()))));
    ++I;
    ++IElemInitRef;
  }
  I = Data.LastprivateCopies.begin();
  for (auto *E : Data.LastprivateVars) {
    auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
    Privates.push_back(std::make_pair(
        C.getDeclAlign(VD),
        PrivateHelpersTy(VD, cast<VarDecl>(cast<DeclRefExpr>(*I)->getDecl()),
                         /*PrivateElemInit=*/nullptr)));
    ++I;
  }
  llvm::array_pod_sort(Privates.begin(), Privates.end(),
                       array_pod_sort_comparator);
  auto KmpInt32Ty = C.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/1);
  // Build type kmp_routine_entry_t (if not built yet).
  emitKmpRoutineEntryT(KmpInt32Ty);
  // Build type kmp_task_t (if not built yet).
  if (KmpTaskTQTy.isNull()) {
    KmpTaskTQTy = C.getRecordType(createKmpTaskTRecordDecl(
        CGM, D.getDirectiveKind(), KmpInt32Ty, KmpRoutineEntryPtrQTy));
  }
  auto *KmpTaskTQTyRD = cast<RecordDecl>(KmpTaskTQTy->getAsTagDecl());
  // Build particular struct kmp_task_t for the given task.
  auto *KmpTaskTWithPrivatesQTyRD =
      createKmpTaskTWithPrivatesRecordDecl(CGM, KmpTaskTQTy, Privates);
  auto KmpTaskTWithPrivatesQTy = C.getRecordType(KmpTaskTWithPrivatesQTyRD);
  QualType KmpTaskTWithPrivatesPtrQTy =
      C.getPointerType(KmpTaskTWithPrivatesQTy);
  auto *KmpTaskTWithPrivatesTy = CGF.ConvertType(KmpTaskTWithPrivatesQTy);
  auto *KmpTaskTWithPrivatesPtrTy = KmpTaskTWithPrivatesTy->getPointerTo();
  auto *KmpTaskTWithPrivatesTySize = CGF.getTypeSize(KmpTaskTWithPrivatesQTy);
  QualType SharedsPtrTy = C.getPointerType(SharedsTy);

  // Emit initial values for private copies (if any).
  llvm::Value *TaskPrivatesMap = nullptr;
  auto *TaskPrivatesMapTy =
      std::next(cast<llvm::Function>(TaskFunction)->arg_begin(), 3)->getType();
  if (!Privates.empty()) {
    auto FI = std::next(KmpTaskTWithPrivatesQTyRD->field_begin());
    TaskPrivatesMap = emitTaskPrivateMappingFunction(
        CGM, Loc, Data.PrivateVars, Data.FirstprivateVars, Data.LastprivateVars,
        FI->getType(), Privates);
    TaskPrivatesMap = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
        TaskPrivatesMap, TaskPrivatesMapTy);
  } else {
    TaskPrivatesMap = llvm::ConstantPointerNull::get(
        cast<llvm::PointerType>(TaskPrivatesMapTy));
  }
  // Build a proxy function kmp_int32 .omp_task_entry.(kmp_int32 gtid,
  // kmp_task_t *tt);
  auto *TaskEntry = emitProxyTaskFunction(
      CGM, Loc, D.getDirectiveKind(), KmpInt32Ty, KmpTaskTWithPrivatesPtrQTy,
      KmpTaskTWithPrivatesQTy, KmpTaskTQTy, SharedsPtrTy, TaskFunction,
      TaskPrivatesMap);

  // Build call kmp_task_t * __kmpc_omp_task_alloc(ident_t *, kmp_int32 gtid,
  // kmp_int32 flags, size_t sizeof_kmp_task_t, size_t sizeof_shareds,
  // kmp_routine_entry_t *task_entry);
  // Task flags. Format is taken from
  // http://llvm.org/svn/llvm-project/openmp/trunk/runtime/src/kmp.h,
  // description of kmp_tasking_flags struct.
  enum {
    TiedFlag = 0x1,
    FinalFlag = 0x2,
    DestructorsFlag = 0x8,
    PriorityFlag = 0x20
  };
  unsigned Flags = Data.Tied ? TiedFlag : 0;
  bool NeedsCleanup = false;
  if (!Privates.empty()) {
    NeedsCleanup = checkDestructorsRequired(KmpTaskTWithPrivatesQTyRD);
    if (NeedsCleanup)
      Flags = Flags | DestructorsFlag;
  }
  if (Data.Priority.getInt())
    Flags = Flags | PriorityFlag;
  auto *TaskFlags =
      Data.Final.getPointer()
          ? CGF.Builder.CreateSelect(Data.Final.getPointer(),
                                     CGF.Builder.getInt32(FinalFlag),
                                     CGF.Builder.getInt32(/*C=*/0))
          : CGF.Builder.getInt32(Data.Final.getInt() ? FinalFlag : 0);
  TaskFlags = CGF.Builder.CreateOr(TaskFlags, CGF.Builder.getInt32(Flags));
  auto *SharedsSize = CGM.getSize(C.getTypeSizeInChars(SharedsTy));
  llvm::Value *AllocArgs[] = {emitUpdateLocation(CGF, Loc),
                              getThreadID(CGF, Loc), TaskFlags,
                              KmpTaskTWithPrivatesTySize, SharedsSize,
                              CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
                                  TaskEntry, KmpRoutineEntryPtrTy)};
  auto *NewTask = CGF.EmitRuntimeCall(
      createRuntimeFunction(OMPRTL__kmpc_omp_task_alloc), AllocArgs);
  auto *NewTaskNewTaskTTy = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      NewTask, KmpTaskTWithPrivatesPtrTy);
  LValue Base = CGF.MakeNaturalAlignAddrLValue(NewTaskNewTaskTTy,
                                               KmpTaskTWithPrivatesQTy);
  LValue TDBase =
      CGF.EmitLValueForField(Base, *KmpTaskTWithPrivatesQTyRD->field_begin());
  // Fill the data in the resulting kmp_task_t record.
  // Copy shareds if there are any.
  Address KmpTaskSharedsPtr = Address::invalid();
  if (!SharedsTy->getAsStructureType()->getDecl()->field_empty()) {
    KmpTaskSharedsPtr =
        Address(CGF.EmitLoadOfScalar(
                    CGF.EmitLValueForField(
                        TDBase, *std::next(KmpTaskTQTyRD->field_begin(),
                                           KmpTaskTShareds)),
                    Loc),
                CGF.getNaturalTypeAlignment(SharedsTy));
    CGF.EmitAggregateCopy(KmpTaskSharedsPtr, Shareds, SharedsTy);
  }
  // Emit initial values for private copies (if any).
  TaskResultTy Result;
  if (!Privates.empty()) {
    emitPrivatesInit(CGF, D, KmpTaskSharedsPtr, Base, KmpTaskTWithPrivatesQTyRD,
                     SharedsTy, SharedsPtrTy, Data, Privates,
                     /*ForDup=*/false);
    if (isOpenMPTaskLoopDirective(D.getDirectiveKind()) &&
        (!Data.LastprivateVars.empty() || checkInitIsRequired(CGF, Privates))) {
      Result.TaskDupFn = emitTaskDupFunction(
          CGM, Loc, D, KmpTaskTWithPrivatesPtrQTy, KmpTaskTWithPrivatesQTyRD,
          KmpTaskTQTyRD, SharedsTy, SharedsPtrTy, Data, Privates,
          /*WithLastIter=*/!Data.LastprivateVars.empty());
    }
  }
  // Fields of union "kmp_cmplrdata_t" for destructors and priority.
  enum { Priority = 0, Destructors = 1 };
  // Provide pointer to function with destructors for privates.
  auto FI = std::next(KmpTaskTQTyRD->field_begin(), Data1);
  auto *KmpCmplrdataUD = (*FI)->getType()->getAsUnionType()->getDecl();
  if (NeedsCleanup) {
    llvm::Value *DestructorFn = emitDestructorsFunction(
        CGM, Loc, KmpInt32Ty, KmpTaskTWithPrivatesPtrQTy,
        KmpTaskTWithPrivatesQTy);
    LValue Data1LV = CGF.EmitLValueForField(TDBase, *FI);
    LValue DestructorsLV = CGF.EmitLValueForField(
        Data1LV, *std::next(KmpCmplrdataUD->field_begin(), Destructors));
    CGF.EmitStoreOfScalar(CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
                              DestructorFn, KmpRoutineEntryPtrTy),
                          DestructorsLV);
  }
  // Set priority.
  if (Data.Priority.getInt()) {
    LValue Data2LV = CGF.EmitLValueForField(
        TDBase, *std::next(KmpTaskTQTyRD->field_begin(), Data2));
    LValue PriorityLV = CGF.EmitLValueForField(
        Data2LV, *std::next(KmpCmplrdataUD->field_begin(), Priority));
    CGF.EmitStoreOfScalar(Data.Priority.getPointer(), PriorityLV);
  }
  Result.NewTask = NewTask;
  Result.TaskEntry = TaskEntry;
  Result.NewTaskNewTaskTTy = NewTaskNewTaskTTy;
  Result.TDBase = TDBase;
  Result.KmpTaskTQTyRD = KmpTaskTQTyRD;
  return Result;
}

void CGOpenMPRuntime::emitTaskCall(CodeGenFunction &CGF, SourceLocation Loc,
                                   const OMPExecutableDirective &D,
                                   llvm::Value *TaskFunction,
                                   QualType SharedsTy, Address Shareds,
                                   const Expr *IfCond,
                                   const OMPTaskDataTy &Data) {
  if (!CGF.HaveInsertPoint())
    return;

  TaskResultTy Result =
      emitTaskInit(CGF, Loc, D, TaskFunction, SharedsTy, Shareds, Data);
  llvm::Value *NewTask = Result.NewTask;
  llvm::Value *TaskEntry = Result.TaskEntry;
  llvm::Value *NewTaskNewTaskTTy = Result.NewTaskNewTaskTTy;
  LValue TDBase = Result.TDBase;
  RecordDecl *KmpTaskTQTyRD = Result.KmpTaskTQTyRD;
  auto &C = CGM.getContext();
  // Process list of dependences.
  Address DependenciesArray = Address::invalid();
  unsigned NumDependencies = Data.Dependences.size();
  if (NumDependencies) {
    // Dependence kind for RTL.
    enum RTLDependenceKindTy { DepIn = 0x01, DepInOut = 0x3 };
    enum RTLDependInfoFieldsTy { BaseAddr, Len, Flags };
    RecordDecl *KmpDependInfoRD;
    QualType FlagsTy =
        C.getIntTypeForBitwidth(C.getTypeSize(C.BoolTy), /*Signed=*/false);
    llvm::Type *LLVMFlagsTy = CGF.ConvertTypeForMem(FlagsTy);
    if (KmpDependInfoTy.isNull()) {
      KmpDependInfoRD = C.buildImplicitRecord("kmp_depend_info");
      KmpDependInfoRD->startDefinition();
      addFieldToRecordDecl(C, KmpDependInfoRD, C.getIntPtrType());
      addFieldToRecordDecl(C, KmpDependInfoRD, C.getSizeType());
      addFieldToRecordDecl(C, KmpDependInfoRD, FlagsTy);
      KmpDependInfoRD->completeDefinition();
      KmpDependInfoTy = C.getRecordType(KmpDependInfoRD);
    } else
      KmpDependInfoRD = cast<RecordDecl>(KmpDependInfoTy->getAsTagDecl());
    CharUnits DependencySize = C.getTypeSizeInChars(KmpDependInfoTy);
    // Define type kmp_depend_info[<Dependences.size()>];
    QualType KmpDependInfoArrayTy = C.getConstantArrayType(
        KmpDependInfoTy, llvm::APInt(/*numBits=*/64, NumDependencies),
        ArrayType::Normal, /*IndexTypeQuals=*/0);
    // kmp_depend_info[<Dependences.size()>] deps;
    DependenciesArray =
        CGF.CreateMemTemp(KmpDependInfoArrayTy, ".dep.arr.addr");
    for (unsigned i = 0; i < NumDependencies; ++i) {
      const Expr *E = Data.Dependences[i].second;
      auto Addr = CGF.EmitLValue(E);
      llvm::Value *Size;
      QualType Ty = E->getType();
      if (auto *ASE = dyn_cast<OMPArraySectionExpr>(E->IgnoreParenImpCasts())) {
        LValue UpAddrLVal =
            CGF.EmitOMPArraySectionExpr(ASE, /*LowerBound=*/false);
        llvm::Value *UpAddr =
            CGF.Builder.CreateConstGEP1_32(UpAddrLVal.getPointer(), /*Idx0=*/1);
        llvm::Value *LowIntPtr =
            CGF.Builder.CreatePtrToInt(Addr.getPointer(), CGM.SizeTy);
        llvm::Value *UpIntPtr = CGF.Builder.CreatePtrToInt(UpAddr, CGM.SizeTy);
        Size = CGF.Builder.CreateNUWSub(UpIntPtr, LowIntPtr);
      } else
        Size = CGF.getTypeSize(Ty);
      auto Base = CGF.MakeAddrLValue(
          CGF.Builder.CreateConstArrayGEP(DependenciesArray, i, DependencySize),
          KmpDependInfoTy);
      // deps[i].base_addr = &<Dependences[i].second>;
      auto BaseAddrLVal = CGF.EmitLValueForField(
          Base, *std::next(KmpDependInfoRD->field_begin(), BaseAddr));
      CGF.EmitStoreOfScalar(
          CGF.Builder.CreatePtrToInt(Addr.getPointer(), CGF.IntPtrTy),
          BaseAddrLVal);
      // deps[i].len = sizeof(<Dependences[i].second>);
      auto LenLVal = CGF.EmitLValueForField(
          Base, *std::next(KmpDependInfoRD->field_begin(), Len));
      CGF.EmitStoreOfScalar(Size, LenLVal);
      // deps[i].flags = <Dependences[i].first>;
      RTLDependenceKindTy DepKind;
      switch (Data.Dependences[i].first) {
      case OMPC_DEPEND_in:
        DepKind = DepIn;
        break;
      // Out and InOut dependencies must use the same code.
      case OMPC_DEPEND_out:
      case OMPC_DEPEND_inout:
        DepKind = DepInOut;
        break;
      case OMPC_DEPEND_source:
      case OMPC_DEPEND_sink:
      case OMPC_DEPEND_unknown:
        llvm_unreachable("Unknown task dependence type");
      }
      auto FlagsLVal = CGF.EmitLValueForField(
          Base, *std::next(KmpDependInfoRD->field_begin(), Flags));
      CGF.EmitStoreOfScalar(llvm::ConstantInt::get(LLVMFlagsTy, DepKind),
                            FlagsLVal);
    }
    DependenciesArray = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
        CGF.Builder.CreateStructGEP(DependenciesArray, 0, CharUnits::Zero()),
        CGF.VoidPtrTy);
  }

  // NOTE: routine and part_id fields are intialized by __kmpc_omp_task_alloc()
  // libcall.
  // Build kmp_int32 __kmpc_omp_task_with_deps(ident_t *, kmp_int32 gtid,
  // kmp_task_t *new_task, kmp_int32 ndeps, kmp_depend_info_t *dep_list,
  // kmp_int32 ndeps_noalias, kmp_depend_info_t *noalias_dep_list) if dependence
  // list is not empty
  auto *ThreadID = getThreadID(CGF, Loc);
  auto *UpLoc = emitUpdateLocation(CGF, Loc);
  llvm::Value *TaskArgs[] = { UpLoc, ThreadID, NewTask };
  llvm::Value *DepTaskArgs[7];
  if (NumDependencies) {
    DepTaskArgs[0] = UpLoc;
    DepTaskArgs[1] = ThreadID;
    DepTaskArgs[2] = NewTask;
    DepTaskArgs[3] = CGF.Builder.getInt32(NumDependencies);
    DepTaskArgs[4] = DependenciesArray.getPointer();
    DepTaskArgs[5] = CGF.Builder.getInt32(0);
    DepTaskArgs[6] = llvm::ConstantPointerNull::get(CGF.VoidPtrTy);
  }
  auto &&ThenCodeGen = [this, &Data, TDBase, KmpTaskTQTyRD, NumDependencies,
                        &TaskArgs,
                        &DepTaskArgs](CodeGenFunction &CGF, PrePostActionTy &) {
    if (!Data.Tied) {
      auto PartIdFI = std::next(KmpTaskTQTyRD->field_begin(), KmpTaskTPartId);
      auto PartIdLVal = CGF.EmitLValueForField(TDBase, *PartIdFI);
      CGF.EmitStoreOfScalar(CGF.Builder.getInt32(0), PartIdLVal);
    }
    if (NumDependencies) {
      CGF.EmitRuntimeCall(
          createRuntimeFunction(OMPRTL__kmpc_omp_task_with_deps), DepTaskArgs);
    } else {
      CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_omp_task),
                          TaskArgs);
    }
    // Check if parent region is untied and build return for untied task;
    if (auto *Region =
            dyn_cast_or_null<CGOpenMPRegionInfo>(CGF.CapturedStmtInfo))
      Region->emitUntiedSwitch(CGF);
  };

  llvm::Value *DepWaitTaskArgs[6];
  if (NumDependencies) {
    DepWaitTaskArgs[0] = UpLoc;
    DepWaitTaskArgs[1] = ThreadID;
    DepWaitTaskArgs[2] = CGF.Builder.getInt32(NumDependencies);
    DepWaitTaskArgs[3] = DependenciesArray.getPointer();
    DepWaitTaskArgs[4] = CGF.Builder.getInt32(0);
    DepWaitTaskArgs[5] = llvm::ConstantPointerNull::get(CGF.VoidPtrTy);
  }
  auto &&ElseCodeGen = [&TaskArgs, ThreadID, NewTaskNewTaskTTy, TaskEntry,
                        NumDependencies, &DepWaitTaskArgs,
                        Loc](CodeGenFunction &CGF, PrePostActionTy &) {
    auto &RT = CGF.CGM.getOpenMPRuntime();
    CodeGenFunction::RunCleanupsScope LocalScope(CGF);
    // Build void __kmpc_omp_wait_deps(ident_t *, kmp_int32 gtid,
    // kmp_int32 ndeps, kmp_depend_info_t *dep_list, kmp_int32
    // ndeps_noalias, kmp_depend_info_t *noalias_dep_list); if dependence info
    // is specified.
    if (NumDependencies)
      CGF.EmitRuntimeCall(RT.createRuntimeFunction(OMPRTL__kmpc_omp_wait_deps),
                          DepWaitTaskArgs);
    // Call proxy_task_entry(gtid, new_task);
    auto &&CodeGen = [TaskEntry, ThreadID, NewTaskNewTaskTTy,
                      Loc](CodeGenFunction &CGF, PrePostActionTy &Action) {
      Action.Enter(CGF);
      llvm::Value *OutlinedFnArgs[] = {ThreadID, NewTaskNewTaskTTy};
      CGF.CGM.getOpenMPRuntime().emitOutlinedFunctionCall(CGF, Loc, TaskEntry,
                                                          OutlinedFnArgs);
    };

    // Build void __kmpc_omp_task_begin_if0(ident_t *, kmp_int32 gtid,
    // kmp_task_t *new_task);
    // Build void __kmpc_omp_task_complete_if0(ident_t *, kmp_int32 gtid,
    // kmp_task_t *new_task);
    RegionCodeGenTy RCG(CodeGen);
    CommonActionTy Action(
        RT.createRuntimeFunction(OMPRTL__kmpc_omp_task_begin_if0), TaskArgs,
        RT.createRuntimeFunction(OMPRTL__kmpc_omp_task_complete_if0), TaskArgs);
    RCG.setAction(Action);
    RCG(CGF);
  };

  if (IfCond)
    emitOMPIfClause(CGF, IfCond, ThenCodeGen, ElseCodeGen);
  else {
    RegionCodeGenTy ThenRCG(ThenCodeGen);
    ThenRCG(CGF);
  }
}

void CGOpenMPRuntime::emitTaskLoopCall(CodeGenFunction &CGF, SourceLocation Loc,
                                       const OMPLoopDirective &D,
                                       llvm::Value *TaskFunction,
                                       QualType SharedsTy, Address Shareds,
                                       const Expr *IfCond,
                                       const OMPTaskDataTy &Data) {
  if (!CGF.HaveInsertPoint())
    return;
  TaskResultTy Result =
      emitTaskInit(CGF, Loc, D, TaskFunction, SharedsTy, Shareds, Data);
  // NOTE: routine and part_id fields are intialized by __kmpc_omp_task_alloc()
  // libcall.
  // Call to void __kmpc_taskloop(ident_t *loc, int gtid, kmp_task_t *task, int
  // if_val, kmp_uint64 *lb, kmp_uint64 *ub, kmp_int64 st, int nogroup, int
  // sched, kmp_uint64 grainsize, void *task_dup);
  llvm::Value *ThreadID = getThreadID(CGF, Loc);
  llvm::Value *UpLoc = emitUpdateLocation(CGF, Loc);
  llvm::Value *IfVal;
  if (IfCond) {
    IfVal = CGF.Builder.CreateIntCast(CGF.EvaluateExprAsBool(IfCond), CGF.IntTy,
                                      /*isSigned=*/true);
  } else
    IfVal = llvm::ConstantInt::getSigned(CGF.IntTy, /*V=*/1);

  LValue LBLVal = CGF.EmitLValueForField(
      Result.TDBase,
      *std::next(Result.KmpTaskTQTyRD->field_begin(), KmpTaskTLowerBound));
  auto *LBVar =
      cast<VarDecl>(cast<DeclRefExpr>(D.getLowerBoundVariable())->getDecl());
  CGF.EmitAnyExprToMem(LBVar->getInit(), LBLVal.getAddress(), LBLVal.getQuals(),
                       /*IsInitializer=*/true);
  LValue UBLVal = CGF.EmitLValueForField(
      Result.TDBase,
      *std::next(Result.KmpTaskTQTyRD->field_begin(), KmpTaskTUpperBound));
  auto *UBVar =
      cast<VarDecl>(cast<DeclRefExpr>(D.getUpperBoundVariable())->getDecl());
  CGF.EmitAnyExprToMem(UBVar->getInit(), UBLVal.getAddress(), UBLVal.getQuals(),
                       /*IsInitializer=*/true);
  LValue StLVal = CGF.EmitLValueForField(
      Result.TDBase,
      *std::next(Result.KmpTaskTQTyRD->field_begin(), KmpTaskTStride));
  auto *StVar =
      cast<VarDecl>(cast<DeclRefExpr>(D.getStrideVariable())->getDecl());
  CGF.EmitAnyExprToMem(StVar->getInit(), StLVal.getAddress(), StLVal.getQuals(),
                       /*IsInitializer=*/true);
  // Store reductions address.
  LValue RedLVal = CGF.EmitLValueForField(
      Result.TDBase,
      *std::next(Result.KmpTaskTQTyRD->field_begin(), KmpTaskTReductions));
  if (Data.Reductions)
    CGF.EmitStoreOfScalar(Data.Reductions, RedLVal);
  else {
    CGF.EmitNullInitialization(RedLVal.getAddress(),
                               CGF.getContext().VoidPtrTy);
  }
  enum { NoSchedule = 0, Grainsize = 1, NumTasks = 2 };
  llvm::Value *TaskArgs[] = {
      UpLoc,
      ThreadID,
      Result.NewTask,
      IfVal,
      LBLVal.getPointer(),
      UBLVal.getPointer(),
      CGF.EmitLoadOfScalar(StLVal, SourceLocation()),
      llvm::ConstantInt::getNullValue(
          CGF.IntTy), // Always 0 because taskgroup emitted by the compiler
      llvm::ConstantInt::getSigned(
          CGF.IntTy, Data.Schedule.getPointer()
                         ? Data.Schedule.getInt() ? NumTasks : Grainsize
                         : NoSchedule),
      Data.Schedule.getPointer()
          ? CGF.Builder.CreateIntCast(Data.Schedule.getPointer(), CGF.Int64Ty,
                                      /*isSigned=*/false)
          : llvm::ConstantInt::get(CGF.Int64Ty, /*V=*/0),
      Result.TaskDupFn ? CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
                             Result.TaskDupFn, CGF.VoidPtrTy)
                       : llvm::ConstantPointerNull::get(CGF.VoidPtrTy)};
  CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_taskloop), TaskArgs);
}

/// \brief Emit reduction operation for each element of array (required for
/// array sections) LHS op = RHS.
/// \param Type Type of array.
/// \param LHSVar Variable on the left side of the reduction operation
/// (references element of array in original variable).
/// \param RHSVar Variable on the right side of the reduction operation
/// (references element of array in original variable).
/// \param RedOpGen Generator of reduction operation with use of LHSVar and
/// RHSVar.
static void EmitOMPAggregateReduction(
    CodeGenFunction &CGF, QualType Type, const VarDecl *LHSVar,
    const VarDecl *RHSVar,
    const llvm::function_ref<void(CodeGenFunction &CGF, const Expr *,
                                  const Expr *, const Expr *)> &RedOpGen,
    const Expr *XExpr = nullptr, const Expr *EExpr = nullptr,
    const Expr *UpExpr = nullptr) {
  // Perform element-by-element initialization.
  QualType ElementTy;
  Address LHSAddr = CGF.GetAddrOfLocalVar(LHSVar);
  Address RHSAddr = CGF.GetAddrOfLocalVar(RHSVar);

  // Drill down to the base element type on both arrays.
  auto ArrayTy = Type->getAsArrayTypeUnsafe();
  auto NumElements = CGF.emitArrayLength(ArrayTy, ElementTy, LHSAddr);

  auto RHSBegin = RHSAddr.getPointer();
  auto LHSBegin = LHSAddr.getPointer();
  // Cast from pointer to array type to pointer to single element.
  auto LHSEnd = CGF.Builder.CreateGEP(LHSBegin, NumElements);
  // The basic structure here is a while-do loop.
  auto BodyBB = CGF.createBasicBlock("omp.arraycpy.body");
  auto DoneBB = CGF.createBasicBlock("omp.arraycpy.done");
  auto IsEmpty =
      CGF.Builder.CreateICmpEQ(LHSBegin, LHSEnd, "omp.arraycpy.isempty");
  CGF.Builder.CreateCondBr(IsEmpty, DoneBB, BodyBB);

  // Enter the loop body, making that address the current address.
  auto EntryBB = CGF.Builder.GetInsertBlock();
  CGF.EmitBlock(BodyBB);

  CharUnits ElementSize = CGF.getContext().getTypeSizeInChars(ElementTy);

  llvm::PHINode *RHSElementPHI = CGF.Builder.CreatePHI(
      RHSBegin->getType(), 2, "omp.arraycpy.srcElementPast");
  RHSElementPHI->addIncoming(RHSBegin, EntryBB);
  Address RHSElementCurrent =
      Address(RHSElementPHI,
              RHSAddr.getAlignment().alignmentOfArrayElement(ElementSize));

  llvm::PHINode *LHSElementPHI = CGF.Builder.CreatePHI(
      LHSBegin->getType(), 2, "omp.arraycpy.destElementPast");
  LHSElementPHI->addIncoming(LHSBegin, EntryBB);
  Address LHSElementCurrent =
      Address(LHSElementPHI,
              LHSAddr.getAlignment().alignmentOfArrayElement(ElementSize));

  // Emit copy.
  CodeGenFunction::OMPPrivateScope Scope(CGF);
  Scope.addPrivate(LHSVar, [=]() -> Address { return LHSElementCurrent; });
  Scope.addPrivate(RHSVar, [=]() -> Address { return RHSElementCurrent; });
  Scope.Privatize();
  RedOpGen(CGF, XExpr, EExpr, UpExpr);
  Scope.ForceCleanup();

  // Shift the address forward by one element.
  auto LHSElementNext = CGF.Builder.CreateConstGEP1_32(
      LHSElementPHI, /*Idx0=*/1, "omp.arraycpy.dest.element");
  auto RHSElementNext = CGF.Builder.CreateConstGEP1_32(
      RHSElementPHI, /*Idx0=*/1, "omp.arraycpy.src.element");
  // Check whether we've reached the end.
  auto Done =
      CGF.Builder.CreateICmpEQ(LHSElementNext, LHSEnd, "omp.arraycpy.done");
  CGF.Builder.CreateCondBr(Done, DoneBB, BodyBB);
  LHSElementPHI->addIncoming(LHSElementNext, CGF.Builder.GetInsertBlock());
  RHSElementPHI->addIncoming(RHSElementNext, CGF.Builder.GetInsertBlock());

  // Done.
  CGF.EmitBlock(DoneBB, /*IsFinished=*/true);
}

/// Emit reduction combiner. If the combiner is a simple expression emit it as
/// is, otherwise consider it as combiner of UDR decl and emit it as a call of
/// UDR combiner function.
static void emitReductionCombiner(CodeGenFunction &CGF,
                                  const Expr *ReductionOp) {
  if (auto *CE = dyn_cast<CallExpr>(ReductionOp))
    if (auto *OVE = dyn_cast<OpaqueValueExpr>(CE->getCallee()))
      if (auto *DRE =
              dyn_cast<DeclRefExpr>(OVE->getSourceExpr()->IgnoreImpCasts()))
        if (auto *DRD = dyn_cast<OMPDeclareReductionDecl>(DRE->getDecl())) {
          std::pair<llvm::Function *, llvm::Function *> Reduction =
              CGF.CGM.getOpenMPRuntime().getUserDefinedReduction(DRD);
          RValue Func = RValue::get(Reduction.first);
          CodeGenFunction::OpaqueValueMapping Map(CGF, OVE, Func);
          CGF.EmitIgnoredExpr(ReductionOp);
          return;
        }
  CGF.EmitIgnoredExpr(ReductionOp);
}

llvm::Value *CGOpenMPRuntime::emitReductionFunction(
    CodeGenModule &CGM, llvm::Type *ArgsType, ArrayRef<const Expr *> Privates,
    ArrayRef<const Expr *> LHSExprs, ArrayRef<const Expr *> RHSExprs,
    ArrayRef<const Expr *> ReductionOps) {
  auto &C = CGM.getContext();

  // void reduction_func(void *LHSArg, void *RHSArg);
  FunctionArgList Args;
  ImplicitParamDecl LHSArg(C, C.VoidPtrTy, ImplicitParamDecl::Other);
  ImplicitParamDecl RHSArg(C, C.VoidPtrTy, ImplicitParamDecl::Other);
  Args.push_back(&LHSArg);
  Args.push_back(&RHSArg);
  auto &CGFI = CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  auto *Fn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(CGFI), llvm::GlobalValue::InternalLinkage,
      ".omp.reduction.reduction_func", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*D=*/nullptr, Fn, CGFI);
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, CGFI, Args);

  // Dst = (void*[n])(LHSArg);
  // Src = (void*[n])(RHSArg);
  Address LHS(CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      CGF.Builder.CreateLoad(CGF.GetAddrOfLocalVar(&LHSArg)),
      ArgsType), CGF.getPointerAlign());
  Address RHS(CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      CGF.Builder.CreateLoad(CGF.GetAddrOfLocalVar(&RHSArg)),
      ArgsType), CGF.getPointerAlign());

  //  ...
  //  *(Type<i>*)lhs[i] = RedOp<i>(*(Type<i>*)lhs[i], *(Type<i>*)rhs[i]);
  //  ...
  CodeGenFunction::OMPPrivateScope Scope(CGF);
  auto IPriv = Privates.begin();
  unsigned Idx = 0;
  for (unsigned I = 0, E = ReductionOps.size(); I < E; ++I, ++IPriv, ++Idx) {
    auto RHSVar = cast<VarDecl>(cast<DeclRefExpr>(RHSExprs[I])->getDecl());
    Scope.addPrivate(RHSVar, [&]() -> Address {
      return emitAddrOfVarFromArray(CGF, RHS, Idx, RHSVar);
    });
    auto LHSVar = cast<VarDecl>(cast<DeclRefExpr>(LHSExprs[I])->getDecl());
    Scope.addPrivate(LHSVar, [&]() -> Address {
      return emitAddrOfVarFromArray(CGF, LHS, Idx, LHSVar);
    });
    QualType PrivTy = (*IPriv)->getType();
    if (PrivTy->isVariablyModifiedType()) {
      // Get array size and emit VLA type.
      ++Idx;
      Address Elem =
          CGF.Builder.CreateConstArrayGEP(LHS, Idx, CGF.getPointerSize());
      llvm::Value *Ptr = CGF.Builder.CreateLoad(Elem);
      auto *VLA = CGF.getContext().getAsVariableArrayType(PrivTy);
      auto *OVE = cast<OpaqueValueExpr>(VLA->getSizeExpr());
      CodeGenFunction::OpaqueValueMapping OpaqueMap(
          CGF, OVE, RValue::get(CGF.Builder.CreatePtrToInt(Ptr, CGF.SizeTy)));
      CGF.EmitVariablyModifiedType(PrivTy);
    }
  }
  Scope.Privatize();
  IPriv = Privates.begin();
  auto ILHS = LHSExprs.begin();
  auto IRHS = RHSExprs.begin();
  for (auto *E : ReductionOps) {
    if ((*IPriv)->getType()->isArrayType()) {
      // Emit reduction for array section.
      auto *LHSVar = cast<VarDecl>(cast<DeclRefExpr>(*ILHS)->getDecl());
      auto *RHSVar = cast<VarDecl>(cast<DeclRefExpr>(*IRHS)->getDecl());
      EmitOMPAggregateReduction(
          CGF, (*IPriv)->getType(), LHSVar, RHSVar,
          [=](CodeGenFunction &CGF, const Expr *, const Expr *, const Expr *) {
            emitReductionCombiner(CGF, E);
          });
    } else
      // Emit reduction for array subscript or single variable.
      emitReductionCombiner(CGF, E);
    ++IPriv;
    ++ILHS;
    ++IRHS;
  }
  Scope.ForceCleanup();
  CGF.FinishFunction();
  return Fn;
}

void CGOpenMPRuntime::emitSingleReductionCombiner(CodeGenFunction &CGF,
                                                  const Expr *ReductionOp,
                                                  const Expr *PrivateRef,
                                                  const DeclRefExpr *LHS,
                                                  const DeclRefExpr *RHS) {
  if (PrivateRef->getType()->isArrayType()) {
    // Emit reduction for array section.
    auto *LHSVar = cast<VarDecl>(LHS->getDecl());
    auto *RHSVar = cast<VarDecl>(RHS->getDecl());
    EmitOMPAggregateReduction(
        CGF, PrivateRef->getType(), LHSVar, RHSVar,
        [=](CodeGenFunction &CGF, const Expr *, const Expr *, const Expr *) {
          emitReductionCombiner(CGF, ReductionOp);
        });
  } else
    // Emit reduction for array subscript or single variable.
    emitReductionCombiner(CGF, ReductionOp);
}

void CGOpenMPRuntime::emitReduction(CodeGenFunction &CGF, SourceLocation Loc,
                                    ArrayRef<const Expr *> Privates,
                                    ArrayRef<const Expr *> LHSExprs,
                                    ArrayRef<const Expr *> RHSExprs,
                                    ArrayRef<const Expr *> ReductionOps,
                                    ReductionOptionsTy Options) {
  if (!CGF.HaveInsertPoint())
    return;

  bool WithNowait = Options.WithNowait;
  bool SimpleReduction = Options.SimpleReduction;

  // Next code should be emitted for reduction:
  //
  // static kmp_critical_name lock = { 0 };
  //
  // void reduce_func(void *lhs[<n>], void *rhs[<n>]) {
  //  *(Type0*)lhs[0] = ReductionOperation0(*(Type0*)lhs[0], *(Type0*)rhs[0]);
  //  ...
  //  *(Type<n>-1*)lhs[<n>-1] = ReductionOperation<n>-1(*(Type<n>-1*)lhs[<n>-1],
  //  *(Type<n>-1*)rhs[<n>-1]);
  // }
  //
  // ...
  // void *RedList[<n>] = {&<RHSExprs>[0], ..., &<RHSExprs>[<n>-1]};
  // switch (__kmpc_reduce{_nowait}(<loc>, <gtid>, <n>, sizeof(RedList),
  // RedList, reduce_func, &<lock>)) {
  // case 1:
  //  ...
  //  <LHSExprs>[i] = RedOp<i>(*<LHSExprs>[i], *<RHSExprs>[i]);
  //  ...
  // __kmpc_end_reduce{_nowait}(<loc>, <gtid>, &<lock>);
  // break;
  // case 2:
  //  ...
  //  Atomic(<LHSExprs>[i] = RedOp<i>(*<LHSExprs>[i], *<RHSExprs>[i]));
  //  ...
  // [__kmpc_end_reduce(<loc>, <gtid>, &<lock>);]
  // break;
  // default:;
  // }
  //
  // if SimpleReduction is true, only the next code is generated:
  //  ...
  //  <LHSExprs>[i] = RedOp<i>(*<LHSExprs>[i], *<RHSExprs>[i]);
  //  ...

  auto &C = CGM.getContext();

  if (SimpleReduction) {
    CodeGenFunction::RunCleanupsScope Scope(CGF);
    auto IPriv = Privates.begin();
    auto ILHS = LHSExprs.begin();
    auto IRHS = RHSExprs.begin();
    for (auto *E : ReductionOps) {
      emitSingleReductionCombiner(CGF, E, *IPriv, cast<DeclRefExpr>(*ILHS),
                                  cast<DeclRefExpr>(*IRHS));
      ++IPriv;
      ++ILHS;
      ++IRHS;
    }
    return;
  }

  // 1. Build a list of reduction variables.
  // void *RedList[<n>] = {<ReductionVars>[0], ..., <ReductionVars>[<n>-1]};
  auto Size = RHSExprs.size();
  for (auto *E : Privates) {
    if (E->getType()->isVariablyModifiedType())
      // Reserve place for array size.
      ++Size;
  }
  llvm::APInt ArraySize(/*unsigned int numBits=*/32, Size);
  QualType ReductionArrayTy =
      C.getConstantArrayType(C.VoidPtrTy, ArraySize, ArrayType::Normal,
                             /*IndexTypeQuals=*/0);
  Address ReductionList =
      CGF.CreateMemTemp(ReductionArrayTy, ".omp.reduction.red_list");
  auto IPriv = Privates.begin();
  unsigned Idx = 0;
  for (unsigned I = 0, E = RHSExprs.size(); I < E; ++I, ++IPriv, ++Idx) {
    Address Elem =
      CGF.Builder.CreateConstArrayGEP(ReductionList, Idx, CGF.getPointerSize());
    CGF.Builder.CreateStore(
        CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
            CGF.EmitLValue(RHSExprs[I]).getPointer(), CGF.VoidPtrTy),
        Elem);
    if ((*IPriv)->getType()->isVariablyModifiedType()) {
      // Store array size.
      ++Idx;
      Elem = CGF.Builder.CreateConstArrayGEP(ReductionList, Idx,
                                             CGF.getPointerSize());
      llvm::Value *Size = CGF.Builder.CreateIntCast(
          CGF.getVLASize(
                 CGF.getContext().getAsVariableArrayType((*IPriv)->getType()))
              .first,
          CGF.SizeTy, /*isSigned=*/false);
      CGF.Builder.CreateStore(CGF.Builder.CreateIntToPtr(Size, CGF.VoidPtrTy),
                              Elem);
    }
  }

  // 2. Emit reduce_func().
  auto *ReductionFn = emitReductionFunction(
      CGM, CGF.ConvertTypeForMem(ReductionArrayTy)->getPointerTo(), Privates,
      LHSExprs, RHSExprs, ReductionOps);

  // 3. Create static kmp_critical_name lock = { 0 };
  auto *Lock = getCriticalRegionLock(".reduction");

  // 4. Build res = __kmpc_reduce{_nowait}(<loc>, <gtid>, <n>, sizeof(RedList),
  // RedList, reduce_func, &<lock>);
  auto *IdentTLoc = emitUpdateLocation(CGF, Loc, OMP_ATOMIC_REDUCE);
  auto *ThreadId = getThreadID(CGF, Loc);
  auto *ReductionArrayTySize = CGF.getTypeSize(ReductionArrayTy);
  auto *RL = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      ReductionList.getPointer(), CGF.VoidPtrTy);
  llvm::Value *Args[] = {
      IdentTLoc,                             // ident_t *<loc>
      ThreadId,                              // i32 <gtid>
      CGF.Builder.getInt32(RHSExprs.size()), // i32 <n>
      ReductionArrayTySize,                  // size_type sizeof(RedList)
      RL,                                    // void *RedList
      ReductionFn, // void (*) (void *, void *) <reduce_func>
      Lock         // kmp_critical_name *&<lock>
  };
  auto Res = CGF.EmitRuntimeCall(
      createRuntimeFunction(WithNowait ? OMPRTL__kmpc_reduce_nowait
                                       : OMPRTL__kmpc_reduce),
      Args);

  // 5. Build switch(res)
  auto *DefaultBB = CGF.createBasicBlock(".omp.reduction.default");
  auto *SwInst = CGF.Builder.CreateSwitch(Res, DefaultBB, /*NumCases=*/2);

  // 6. Build case 1:
  //  ...
  //  <LHSExprs>[i] = RedOp<i>(*<LHSExprs>[i], *<RHSExprs>[i]);
  //  ...
  // __kmpc_end_reduce{_nowait}(<loc>, <gtid>, &<lock>);
  // break;
  auto *Case1BB = CGF.createBasicBlock(".omp.reduction.case1");
  SwInst->addCase(CGF.Builder.getInt32(1), Case1BB);
  CGF.EmitBlock(Case1BB);

  // Add emission of __kmpc_end_reduce{_nowait}(<loc>, <gtid>, &<lock>);
  llvm::Value *EndArgs[] = {
      IdentTLoc, // ident_t *<loc>
      ThreadId,  // i32 <gtid>
      Lock       // kmp_critical_name *&<lock>
  };
  auto &&CodeGen = [&Privates, &LHSExprs, &RHSExprs, &ReductionOps](
      CodeGenFunction &CGF, PrePostActionTy &Action) {
    auto &RT = CGF.CGM.getOpenMPRuntime();
    auto IPriv = Privates.begin();
    auto ILHS = LHSExprs.begin();
    auto IRHS = RHSExprs.begin();
    for (auto *E : ReductionOps) {
      RT.emitSingleReductionCombiner(CGF, E, *IPriv, cast<DeclRefExpr>(*ILHS),
                                     cast<DeclRefExpr>(*IRHS));
      ++IPriv;
      ++ILHS;
      ++IRHS;
    }
  };
  RegionCodeGenTy RCG(CodeGen);
  CommonActionTy Action(
      nullptr, llvm::None,
      createRuntimeFunction(WithNowait ? OMPRTL__kmpc_end_reduce_nowait
                                       : OMPRTL__kmpc_end_reduce),
      EndArgs);
  RCG.setAction(Action);
  RCG(CGF);

  CGF.EmitBranch(DefaultBB);

  // 7. Build case 2:
  //  ...
  //  Atomic(<LHSExprs>[i] = RedOp<i>(*<LHSExprs>[i], *<RHSExprs>[i]));
  //  ...
  // break;
  auto *Case2BB = CGF.createBasicBlock(".omp.reduction.case2");
  SwInst->addCase(CGF.Builder.getInt32(2), Case2BB);
  CGF.EmitBlock(Case2BB);

  auto &&AtomicCodeGen = [Loc, &Privates, &LHSExprs, &RHSExprs, &ReductionOps](
      CodeGenFunction &CGF, PrePostActionTy &Action) {
    auto ILHS = LHSExprs.begin();
    auto IRHS = RHSExprs.begin();
    auto IPriv = Privates.begin();
    for (auto *E : ReductionOps) {
      const Expr *XExpr = nullptr;
      const Expr *EExpr = nullptr;
      const Expr *UpExpr = nullptr;
      BinaryOperatorKind BO = BO_Comma;
      if (auto *BO = dyn_cast<BinaryOperator>(E)) {
        if (BO->getOpcode() == BO_Assign) {
          XExpr = BO->getLHS();
          UpExpr = BO->getRHS();
        }
      }
      // Try to emit update expression as a simple atomic.
      auto *RHSExpr = UpExpr;
      if (RHSExpr) {
        // Analyze RHS part of the whole expression.
        if (auto *ACO = dyn_cast<AbstractConditionalOperator>(
                RHSExpr->IgnoreParenImpCasts())) {
          // If this is a conditional operator, analyze its condition for
          // min/max reduction operator.
          RHSExpr = ACO->getCond();
        }
        if (auto *BORHS =
                dyn_cast<BinaryOperator>(RHSExpr->IgnoreParenImpCasts())) {
          EExpr = BORHS->getRHS();
          BO = BORHS->getOpcode();
        }
      }
      if (XExpr) {
        auto *VD = cast<VarDecl>(cast<DeclRefExpr>(*ILHS)->getDecl());
        auto &&AtomicRedGen = [BO, VD,
                               Loc](CodeGenFunction &CGF, const Expr *XExpr,
                                    const Expr *EExpr, const Expr *UpExpr) {
          LValue X = CGF.EmitLValue(XExpr);
          RValue E;
          if (EExpr)
            E = CGF.EmitAnyExpr(EExpr);
          CGF.EmitOMPAtomicSimpleUpdateExpr(
              X, E, BO, /*IsXLHSInRHSPart=*/true,
              llvm::AtomicOrdering::Monotonic, Loc,
              [&CGF, UpExpr, VD, Loc](RValue XRValue) {
                CodeGenFunction::OMPPrivateScope PrivateScope(CGF);
                PrivateScope.addPrivate(
                    VD, [&CGF, VD, XRValue, Loc]() -> Address {
                      Address LHSTemp = CGF.CreateMemTemp(VD->getType());
                      CGF.emitOMPSimpleStore(
                          CGF.MakeAddrLValue(LHSTemp, VD->getType()), XRValue,
                          VD->getType().getNonReferenceType(), Loc);
                      return LHSTemp;
                    });
                (void)PrivateScope.Privatize();
                return CGF.EmitAnyExpr(UpExpr);
              });
        };
        if ((*IPriv)->getType()->isArrayType()) {
          // Emit atomic reduction for array section.
          auto *RHSVar = cast<VarDecl>(cast<DeclRefExpr>(*IRHS)->getDecl());
          EmitOMPAggregateReduction(CGF, (*IPriv)->getType(), VD, RHSVar,
                                    AtomicRedGen, XExpr, EExpr, UpExpr);
        } else
          // Emit atomic reduction for array subscript or single variable.
          AtomicRedGen(CGF, XExpr, EExpr, UpExpr);
      } else {
        // Emit as a critical region.
        auto &&CritRedGen = [E, Loc](CodeGenFunction &CGF, const Expr *,
                                     const Expr *, const Expr *) {
          auto &RT = CGF.CGM.getOpenMPRuntime();
          RT.emitCriticalRegion(
              CGF, ".atomic_reduction",
              [=](CodeGenFunction &CGF, PrePostActionTy &Action) {
                Action.Enter(CGF);
                emitReductionCombiner(CGF, E);
              },
              Loc);
        };
        if ((*IPriv)->getType()->isArrayType()) {
          auto *LHSVar = cast<VarDecl>(cast<DeclRefExpr>(*ILHS)->getDecl());
          auto *RHSVar = cast<VarDecl>(cast<DeclRefExpr>(*IRHS)->getDecl());
          EmitOMPAggregateReduction(CGF, (*IPriv)->getType(), LHSVar, RHSVar,
                                    CritRedGen);
        } else
          CritRedGen(CGF, nullptr, nullptr, nullptr);
      }
      ++ILHS;
      ++IRHS;
      ++IPriv;
    }
  };
  RegionCodeGenTy AtomicRCG(AtomicCodeGen);
  if (!WithNowait) {
    // Add emission of __kmpc_end_reduce(<loc>, <gtid>, &<lock>);
    llvm::Value *EndArgs[] = {
        IdentTLoc, // ident_t *<loc>
        ThreadId,  // i32 <gtid>
        Lock       // kmp_critical_name *&<lock>
    };
    CommonActionTy Action(nullptr, llvm::None,
                          createRuntimeFunction(OMPRTL__kmpc_end_reduce),
                          EndArgs);
    AtomicRCG.setAction(Action);
    AtomicRCG(CGF);
  } else
    AtomicRCG(CGF);

  CGF.EmitBranch(DefaultBB);
  CGF.EmitBlock(DefaultBB, /*IsFinished=*/true);
}

/// Generates unique name for artificial threadprivate variables.
/// Format is: <Prefix> "." <Loc_raw_encoding> "_" <N>
static std::string generateUniqueName(StringRef Prefix, SourceLocation Loc,
                                      unsigned N) {
  SmallString<256> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  Out << Prefix << "." << Loc.getRawEncoding() << "_" << N;
  return Out.str();
}

/// Emits reduction initializer function:
/// \code
/// void @.red_init(void* %arg) {
/// %0 = bitcast void* %arg to <type>*
/// store <type> <init>, <type>* %0
/// ret void
/// }
/// \endcode
static llvm::Value *emitReduceInitFunction(CodeGenModule &CGM,
                                           SourceLocation Loc,
                                           ReductionCodeGen &RCG, unsigned N) {
  auto &C = CGM.getContext();
  FunctionArgList Args;
  ImplicitParamDecl Param(C, C.VoidPtrTy, ImplicitParamDecl::Other);
  Args.emplace_back(&Param);
  auto &FnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  auto *FnTy = CGM.getTypes().GetFunctionType(FnInfo);
  auto *Fn = llvm::Function::Create(FnTy, llvm::GlobalValue::InternalLinkage,
                                    ".red_init.", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*D=*/nullptr, Fn, FnInfo);
  CodeGenFunction CGF(CGM);
  CGF.disableDebugInfo();
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, FnInfo, Args);
  Address PrivateAddr = CGF.EmitLoadOfPointer(
      CGF.GetAddrOfLocalVar(&Param),
      C.getPointerType(C.VoidPtrTy).castAs<PointerType>());
  llvm::Value *Size = nullptr;
  // If the size of the reduction item is non-constant, load it from global
  // threadprivate variable.
  if (RCG.getSizes(N).second) {
    Address SizeAddr = CGM.getOpenMPRuntime().getAddrOfArtificialThreadPrivate(
        CGF, CGM.getContext().getSizeType(),
        generateUniqueName("reduction_size", Loc, N));
    Size =
        CGF.EmitLoadOfScalar(SizeAddr, /*Volatile=*/false,
                             CGM.getContext().getSizeType(), SourceLocation());
  }
  RCG.emitAggregateType(CGF, N, Size);
  LValue SharedLVal;
  // If initializer uses initializer from declare reduction construct, emit a
  // pointer to the address of the original reduction item (reuired by reduction
  // initializer)
  if (RCG.usesReductionInitializer(N)) {
    Address SharedAddr =
        CGM.getOpenMPRuntime().getAddrOfArtificialThreadPrivate(
            CGF, CGM.getContext().VoidPtrTy,
            generateUniqueName("reduction", Loc, N));
    SharedLVal = CGF.MakeAddrLValue(SharedAddr, CGM.getContext().VoidPtrTy);
  } else {
    SharedLVal = CGF.MakeNaturalAlignAddrLValue(
        llvm::ConstantPointerNull::get(CGM.VoidPtrTy),
        CGM.getContext().VoidPtrTy);
  }
  // Emit the initializer:
  // %0 = bitcast void* %arg to <type>*
  // store <type> <init>, <type>* %0
  RCG.emitInitialization(CGF, N, PrivateAddr, SharedLVal,
                         [](CodeGenFunction &) { return false; });
  CGF.FinishFunction();
  return Fn;
}

/// Emits reduction combiner function:
/// \code
/// void @.red_comb(void* %arg0, void* %arg1) {
/// %lhs = bitcast void* %arg0 to <type>*
/// %rhs = bitcast void* %arg1 to <type>*
/// %2 = <ReductionOp>(<type>* %lhs, <type>* %rhs)
/// store <type> %2, <type>* %lhs
/// ret void
/// }
/// \endcode
static llvm::Value *emitReduceCombFunction(CodeGenModule &CGM,
                                           SourceLocation Loc,
                                           ReductionCodeGen &RCG, unsigned N,
                                           const Expr *ReductionOp,
                                           const Expr *LHS, const Expr *RHS,
                                           const Expr *PrivateRef) {
  auto &C = CGM.getContext();
  auto *LHSVD = cast<VarDecl>(cast<DeclRefExpr>(LHS)->getDecl());
  auto *RHSVD = cast<VarDecl>(cast<DeclRefExpr>(RHS)->getDecl());
  FunctionArgList Args;
  ImplicitParamDecl ParamInOut(C, C.VoidPtrTy, ImplicitParamDecl::Other);
  ImplicitParamDecl ParamIn(C, C.VoidPtrTy, ImplicitParamDecl::Other);
  Args.emplace_back(&ParamInOut);
  Args.emplace_back(&ParamIn);
  auto &FnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  auto *FnTy = CGM.getTypes().GetFunctionType(FnInfo);
  auto *Fn = llvm::Function::Create(FnTy, llvm::GlobalValue::InternalLinkage,
                                    ".red_comb.", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*D=*/nullptr, Fn, FnInfo);
  CodeGenFunction CGF(CGM);
  CGF.disableDebugInfo();
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, FnInfo, Args);
  llvm::Value *Size = nullptr;
  // If the size of the reduction item is non-constant, load it from global
  // threadprivate variable.
  if (RCG.getSizes(N).second) {
    Address SizeAddr = CGM.getOpenMPRuntime().getAddrOfArtificialThreadPrivate(
        CGF, CGM.getContext().getSizeType(),
        generateUniqueName("reduction_size", Loc, N));
    Size =
        CGF.EmitLoadOfScalar(SizeAddr, /*Volatile=*/false,
                             CGM.getContext().getSizeType(), SourceLocation());
  }
  RCG.emitAggregateType(CGF, N, Size);
  // Remap lhs and rhs variables to the addresses of the function arguments.
  // %lhs = bitcast void* %arg0 to <type>*
  // %rhs = bitcast void* %arg1 to <type>*
  CodeGenFunction::OMPPrivateScope PrivateScope(CGF);
  PrivateScope.addPrivate(LHSVD, [&C, &CGF, &ParamInOut, LHSVD]() -> Address {
    // Pull out the pointer to the variable.
    Address PtrAddr = CGF.EmitLoadOfPointer(
        CGF.GetAddrOfLocalVar(&ParamInOut),
        C.getPointerType(C.VoidPtrTy).castAs<PointerType>());
    return CGF.Builder.CreateElementBitCast(
        PtrAddr, CGF.ConvertTypeForMem(LHSVD->getType()));
  });
  PrivateScope.addPrivate(RHSVD, [&C, &CGF, &ParamIn, RHSVD]() -> Address {
    // Pull out the pointer to the variable.
    Address PtrAddr = CGF.EmitLoadOfPointer(
        CGF.GetAddrOfLocalVar(&ParamIn),
        C.getPointerType(C.VoidPtrTy).castAs<PointerType>());
    return CGF.Builder.CreateElementBitCast(
        PtrAddr, CGF.ConvertTypeForMem(RHSVD->getType()));
  });
  PrivateScope.Privatize();
  // Emit the combiner body:
  // %2 = <ReductionOp>(<type> *%lhs, <type> *%rhs)
  // store <type> %2, <type>* %lhs
  CGM.getOpenMPRuntime().emitSingleReductionCombiner(
      CGF, ReductionOp, PrivateRef, cast<DeclRefExpr>(LHS),
      cast<DeclRefExpr>(RHS));
  CGF.FinishFunction();
  return Fn;
}

/// Emits reduction finalizer function:
/// \code
/// void @.red_fini(void* %arg) {
/// %0 = bitcast void* %arg to <type>*
/// <destroy>(<type>* %0)
/// ret void
/// }
/// \endcode
static llvm::Value *emitReduceFiniFunction(CodeGenModule &CGM,
                                           SourceLocation Loc,
                                           ReductionCodeGen &RCG, unsigned N) {
  if (!RCG.needCleanups(N))
    return nullptr;
  auto &C = CGM.getContext();
  FunctionArgList Args;
  ImplicitParamDecl Param(C, C.VoidPtrTy, ImplicitParamDecl::Other);
  Args.emplace_back(&Param);
  auto &FnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  auto *FnTy = CGM.getTypes().GetFunctionType(FnInfo);
  auto *Fn = llvm::Function::Create(FnTy, llvm::GlobalValue::InternalLinkage,
                                    ".red_fini.", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*D=*/nullptr, Fn, FnInfo);
  CodeGenFunction CGF(CGM);
  CGF.disableDebugInfo();
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, FnInfo, Args);
  Address PrivateAddr = CGF.EmitLoadOfPointer(
      CGF.GetAddrOfLocalVar(&Param),
      C.getPointerType(C.VoidPtrTy).castAs<PointerType>());
  llvm::Value *Size = nullptr;
  // If the size of the reduction item is non-constant, load it from global
  // threadprivate variable.
  if (RCG.getSizes(N).second) {
    Address SizeAddr = CGM.getOpenMPRuntime().getAddrOfArtificialThreadPrivate(
        CGF, CGM.getContext().getSizeType(),
        generateUniqueName("reduction_size", Loc, N));
    Size =
        CGF.EmitLoadOfScalar(SizeAddr, /*Volatile=*/false,
                             CGM.getContext().getSizeType(), SourceLocation());
  }
  RCG.emitAggregateType(CGF, N, Size);
  // Emit the finalizer body:
  // <destroy>(<type>* %0)
  RCG.emitCleanups(CGF, N, PrivateAddr);
  CGF.FinishFunction();
  return Fn;
}

llvm::Value *CGOpenMPRuntime::emitTaskReductionInit(
    CodeGenFunction &CGF, SourceLocation Loc, ArrayRef<const Expr *> LHSExprs,
    ArrayRef<const Expr *> RHSExprs, const OMPTaskDataTy &Data) {
  if (!CGF.HaveInsertPoint() || Data.ReductionVars.empty())
    return nullptr;

  // Build typedef struct:
  // kmp_task_red_input {
  //   void *reduce_shar; // shared reduction item
  //   size_t reduce_size; // size of data item
  //   void *reduce_init; // data initialization routine
  //   void *reduce_fini; // data finalization routine
  //   void *reduce_comb; // data combiner routine
  //   kmp_task_red_flags_t flags; // flags for additional info from compiler
  // } kmp_task_red_input_t;
  ASTContext &C = CGM.getContext();
  auto *RD = C.buildImplicitRecord("kmp_task_red_input_t");
  RD->startDefinition();
  const FieldDecl *SharedFD = addFieldToRecordDecl(C, RD, C.VoidPtrTy);
  const FieldDecl *SizeFD = addFieldToRecordDecl(C, RD, C.getSizeType());
  const FieldDecl *InitFD  = addFieldToRecordDecl(C, RD, C.VoidPtrTy);
  const FieldDecl *FiniFD = addFieldToRecordDecl(C, RD, C.VoidPtrTy);
  const FieldDecl *CombFD = addFieldToRecordDecl(C, RD, C.VoidPtrTy);
  const FieldDecl *FlagsFD = addFieldToRecordDecl(
      C, RD, C.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/false));
  RD->completeDefinition();
  QualType RDType = C.getRecordType(RD);
  unsigned Size = Data.ReductionVars.size();
  llvm::APInt ArraySize(/*numBits=*/64, Size);
  QualType ArrayRDType = C.getConstantArrayType(
      RDType, ArraySize, ArrayType::Normal, /*IndexTypeQuals=*/0);
  // kmp_task_red_input_t .rd_input.[Size];
  Address TaskRedInput = CGF.CreateMemTemp(ArrayRDType, ".rd_input.");
  ReductionCodeGen RCG(Data.ReductionVars, Data.ReductionCopies,
                       Data.ReductionOps);
  for (unsigned Cnt = 0; Cnt < Size; ++Cnt) {
    // kmp_task_red_input_t &ElemLVal = .rd_input.[Cnt];
    llvm::Value *Idxs[] = {llvm::ConstantInt::get(CGM.SizeTy, /*V=*/0),
                           llvm::ConstantInt::get(CGM.SizeTy, Cnt)};
    llvm::Value *GEP = CGF.EmitCheckedInBoundsGEP(
        TaskRedInput.getPointer(), Idxs,
        /*SignedIndices=*/false, /*IsSubtraction=*/false, Loc,
        ".rd_input.gep.");
    LValue ElemLVal = CGF.MakeNaturalAlignAddrLValue(GEP, RDType);
    // ElemLVal.reduce_shar = &Shareds[Cnt];
    LValue SharedLVal = CGF.EmitLValueForField(ElemLVal, SharedFD);
    RCG.emitSharedLValue(CGF, Cnt);
    llvm::Value *CastedShared =
        CGF.EmitCastToVoidPtr(RCG.getSharedLValue(Cnt).getPointer());
    CGF.EmitStoreOfScalar(CastedShared, SharedLVal);
    RCG.emitAggregateType(CGF, Cnt);
    llvm::Value *SizeValInChars;
    llvm::Value *SizeVal;
    std::tie(SizeValInChars, SizeVal) = RCG.getSizes(Cnt);
    // We use delayed creation/initialization for VLAs, array sections and
    // custom reduction initializations. It is required because runtime does not
    // provide the way to pass the sizes of VLAs/array sections to
    // initializer/combiner/finalizer functions and does not pass the pointer to
    // original reduction item to the initializer. Instead threadprivate global
    // variables are used to store these values and use them in the functions.
    bool DelayedCreation = !!SizeVal;
    SizeValInChars = CGF.Builder.CreateIntCast(SizeValInChars, CGM.SizeTy,
                                               /*isSigned=*/false);
    LValue SizeLVal = CGF.EmitLValueForField(ElemLVal, SizeFD);
    CGF.EmitStoreOfScalar(SizeValInChars, SizeLVal);
    // ElemLVal.reduce_init = init;
    LValue InitLVal = CGF.EmitLValueForField(ElemLVal, InitFD);
    llvm::Value *InitAddr =
        CGF.EmitCastToVoidPtr(emitReduceInitFunction(CGM, Loc, RCG, Cnt));
    CGF.EmitStoreOfScalar(InitAddr, InitLVal);
    DelayedCreation = DelayedCreation || RCG.usesReductionInitializer(Cnt);
    // ElemLVal.reduce_fini = fini;
    LValue FiniLVal = CGF.EmitLValueForField(ElemLVal, FiniFD);
    llvm::Value *Fini = emitReduceFiniFunction(CGM, Loc, RCG, Cnt);
    llvm::Value *FiniAddr = Fini
                                ? CGF.EmitCastToVoidPtr(Fini)
                                : llvm::ConstantPointerNull::get(CGM.VoidPtrTy);
    CGF.EmitStoreOfScalar(FiniAddr, FiniLVal);
    // ElemLVal.reduce_comb = comb;
    LValue CombLVal = CGF.EmitLValueForField(ElemLVal, CombFD);
    llvm::Value *CombAddr = CGF.EmitCastToVoidPtr(emitReduceCombFunction(
        CGM, Loc, RCG, Cnt, Data.ReductionOps[Cnt], LHSExprs[Cnt],
        RHSExprs[Cnt], Data.ReductionCopies[Cnt]));
    CGF.EmitStoreOfScalar(CombAddr, CombLVal);
    // ElemLVal.flags = 0;
    LValue FlagsLVal = CGF.EmitLValueForField(ElemLVal, FlagsFD);
    if (DelayedCreation) {
      CGF.EmitStoreOfScalar(
          llvm::ConstantInt::get(CGM.Int32Ty, /*V=*/1, /*IsSigned=*/true),
          FlagsLVal);
    } else
      CGF.EmitNullInitialization(FlagsLVal.getAddress(), FlagsLVal.getType());
  }
  // Build call void *__kmpc_task_reduction_init(int gtid, int num_data, void
  // *data);
  llvm::Value *Args[] = {
      CGF.Builder.CreateIntCast(getThreadID(CGF, Loc), CGM.IntTy,
                                /*isSigned=*/true),
      llvm::ConstantInt::get(CGM.IntTy, Size, /*isSigned=*/true),
      CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(TaskRedInput.getPointer(),
                                                      CGM.VoidPtrTy)};
  return CGF.EmitRuntimeCall(
      createRuntimeFunction(OMPRTL__kmpc_task_reduction_init), Args);
}

void CGOpenMPRuntime::emitTaskReductionFixups(CodeGenFunction &CGF,
                                              SourceLocation Loc,
                                              ReductionCodeGen &RCG,
                                              unsigned N) {
  auto Sizes = RCG.getSizes(N);
  // Emit threadprivate global variable if the type is non-constant
  // (Sizes.second = nullptr).
  if (Sizes.second) {
    llvm::Value *SizeVal = CGF.Builder.CreateIntCast(Sizes.second, CGM.SizeTy,
                                                     /*isSigned=*/false);
    Address SizeAddr = getAddrOfArtificialThreadPrivate(
        CGF, CGM.getContext().getSizeType(),
        generateUniqueName("reduction_size", Loc, N));
    CGF.Builder.CreateStore(SizeVal, SizeAddr, /*IsVolatile=*/false);
  }
  // Store address of the original reduction item if custom initializer is used.
  if (RCG.usesReductionInitializer(N)) {
    Address SharedAddr = getAddrOfArtificialThreadPrivate(
        CGF, CGM.getContext().VoidPtrTy,
        generateUniqueName("reduction", Loc, N));
    CGF.Builder.CreateStore(
        CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
            RCG.getSharedLValue(N).getPointer(), CGM.VoidPtrTy),
        SharedAddr, /*IsVolatile=*/false);
  }
}

Address CGOpenMPRuntime::getTaskReductionItem(CodeGenFunction &CGF,
                                              SourceLocation Loc,
                                              llvm::Value *ReductionsPtr,
                                              LValue SharedLVal) {
  // Build call void *__kmpc_task_reduction_get_th_data(int gtid, void *tg, void
  // *d);
  llvm::Value *Args[] = {
      CGF.Builder.CreateIntCast(getThreadID(CGF, Loc), CGM.IntTy,
                                /*isSigned=*/true),
      ReductionsPtr,
      CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(SharedLVal.getPointer(),
                                                      CGM.VoidPtrTy)};
  return Address(
      CGF.EmitRuntimeCall(
          createRuntimeFunction(OMPRTL__kmpc_task_reduction_get_th_data), Args),
      SharedLVal.getAlignment());
}

void CGOpenMPRuntime::emitTaskwaitCall(CodeGenFunction &CGF,
                                       SourceLocation Loc) {
  if (!CGF.HaveInsertPoint())
    return;
  // Build call kmp_int32 __kmpc_omp_taskwait(ident_t *loc, kmp_int32
  // global_tid);
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc)};
  // Ignore return result until untied tasks are supported.
  CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_omp_taskwait), Args);
  if (auto *Region = dyn_cast_or_null<CGOpenMPRegionInfo>(CGF.CapturedStmtInfo))
    Region->emitUntiedSwitch(CGF);
}

void CGOpenMPRuntime::emitInlinedDirective(CodeGenFunction &CGF,
                                           OpenMPDirectiveKind InnerKind,
                                           const RegionCodeGenTy &CodeGen,
                                           bool HasCancel) {
  if (!CGF.HaveInsertPoint())
    return;
  InlinedOpenMPRegionRAII Region(CGF, CodeGen, InnerKind, HasCancel);
  CGF.CapturedStmtInfo->EmitBody(CGF, /*S=*/nullptr);
}

namespace {
enum RTCancelKind {
  CancelNoreq = 0,
  CancelParallel = 1,
  CancelLoop = 2,
  CancelSections = 3,
  CancelTaskgroup = 4
};
} // anonymous namespace

static RTCancelKind getCancellationKind(OpenMPDirectiveKind CancelRegion) {
  RTCancelKind CancelKind = CancelNoreq;
  if (CancelRegion == OMPD_parallel)
    CancelKind = CancelParallel;
  else if (CancelRegion == OMPD_for)
    CancelKind = CancelLoop;
  else if (CancelRegion == OMPD_sections)
    CancelKind = CancelSections;
  else {
    assert(CancelRegion == OMPD_taskgroup);
    CancelKind = CancelTaskgroup;
  }
  return CancelKind;
}

void CGOpenMPRuntime::emitCancellationPointCall(
    CodeGenFunction &CGF, SourceLocation Loc,
    OpenMPDirectiveKind CancelRegion) {
  if (!CGF.HaveInsertPoint())
    return;
  // Build call kmp_int32 __kmpc_cancellationpoint(ident_t *loc, kmp_int32
  // global_tid, kmp_int32 cncl_kind);
  if (auto *OMPRegionInfo =
          dyn_cast_or_null<CGOpenMPRegionInfo>(CGF.CapturedStmtInfo)) {
    // For 'cancellation point taskgroup', the task region info may not have a
    // cancel. This may instead happen in another adjacent task.
    if (CancelRegion == OMPD_taskgroup || OMPRegionInfo->hasCancel()) {
      llvm::Value *Args[] = {
          emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc),
          CGF.Builder.getInt32(getCancellationKind(CancelRegion))};
      // Ignore return result until untied tasks are supported.
      auto *Result = CGF.EmitRuntimeCall(
          createRuntimeFunction(OMPRTL__kmpc_cancellationpoint), Args);
      // if (__kmpc_cancellationpoint()) {
      //   exit from construct;
      // }
      auto *ExitBB = CGF.createBasicBlock(".cancel.exit");
      auto *ContBB = CGF.createBasicBlock(".cancel.continue");
      auto *Cmp = CGF.Builder.CreateIsNotNull(Result);
      CGF.Builder.CreateCondBr(Cmp, ExitBB, ContBB);
      CGF.EmitBlock(ExitBB);
      // exit from construct;
      auto CancelDest =
          CGF.getOMPCancelDestination(OMPRegionInfo->getDirectiveKind());
      CGF.EmitBranchThroughCleanup(CancelDest);
      CGF.EmitBlock(ContBB, /*IsFinished=*/true);
    }
  }
}

void CGOpenMPRuntime::emitCancelCall(CodeGenFunction &CGF, SourceLocation Loc,
                                     const Expr *IfCond,
                                     OpenMPDirectiveKind CancelRegion) {
  if (!CGF.HaveInsertPoint())
    return;
  // Build call kmp_int32 __kmpc_cancel(ident_t *loc, kmp_int32 global_tid,
  // kmp_int32 cncl_kind);
  if (auto *OMPRegionInfo =
          dyn_cast_or_null<CGOpenMPRegionInfo>(CGF.CapturedStmtInfo)) {
    auto &&ThenGen = [Loc, CancelRegion, OMPRegionInfo](CodeGenFunction &CGF,
                                                        PrePostActionTy &) {
      auto &RT = CGF.CGM.getOpenMPRuntime();
      llvm::Value *Args[] = {
          RT.emitUpdateLocation(CGF, Loc), RT.getThreadID(CGF, Loc),
          CGF.Builder.getInt32(getCancellationKind(CancelRegion))};
      // Ignore return result until untied tasks are supported.
      auto *Result = CGF.EmitRuntimeCall(
          RT.createRuntimeFunction(OMPRTL__kmpc_cancel), Args);
      // if (__kmpc_cancel()) {
      //   exit from construct;
      // }
      auto *ExitBB = CGF.createBasicBlock(".cancel.exit");
      auto *ContBB = CGF.createBasicBlock(".cancel.continue");
      auto *Cmp = CGF.Builder.CreateIsNotNull(Result);
      CGF.Builder.CreateCondBr(Cmp, ExitBB, ContBB);
      CGF.EmitBlock(ExitBB);
      // exit from construct;
      auto CancelDest =
          CGF.getOMPCancelDestination(OMPRegionInfo->getDirectiveKind());
      CGF.EmitBranchThroughCleanup(CancelDest);
      CGF.EmitBlock(ContBB, /*IsFinished=*/true);
    };
    if (IfCond)
      emitOMPIfClause(CGF, IfCond, ThenGen,
                      [](CodeGenFunction &, PrePostActionTy &) {});
    else {
      RegionCodeGenTy ThenRCG(ThenGen);
      ThenRCG(CGF);
    }
  }
}

/// \brief Obtain information that uniquely identifies a target entry. This
/// consists of the file and device IDs as well as line number associated with
/// the relevant entry source location.
static void getTargetEntryUniqueInfo(ASTContext &C, SourceLocation Loc,
                                     unsigned &DeviceID, unsigned &FileID,
                                     unsigned &LineNum) {

  auto &SM = C.getSourceManager();

  // The loc should be always valid and have a file ID (the user cannot use
  // #pragma directives in macros)

  assert(Loc.isValid() && "Source location is expected to be always valid.");
  assert(Loc.isFileID() && "Source location is expected to refer to a file.");

  PresumedLoc PLoc = SM.getPresumedLoc(Loc);
  assert(PLoc.isValid() && "Source location is expected to be always valid.");

  llvm::sys::fs::UniqueID ID;
  if (llvm::sys::fs::getUniqueID(PLoc.getFilename(), ID))
    llvm_unreachable("Source file with target region no longer exists!");

  DeviceID = ID.getDevice();
  FileID = ID.getFile();
  LineNum = PLoc.getLine();
}

void CGOpenMPRuntime::emitTargetOutlinedFunction(
    const OMPExecutableDirective &D, StringRef ParentName,
    llvm::Function *&OutlinedFn, llvm::Constant *&OutlinedFnID,
    bool IsOffloadEntry, const RegionCodeGenTy &CodeGen) {
  assert(!ParentName.empty() && "Invalid target region parent name!");

  emitTargetOutlinedFunctionHelper(D, ParentName, OutlinedFn, OutlinedFnID,
                                   IsOffloadEntry, CodeGen);
}

void CGOpenMPRuntime::emitTargetOutlinedFunctionHelper(
    const OMPExecutableDirective &D, StringRef ParentName,
    llvm::Function *&OutlinedFn, llvm::Constant *&OutlinedFnID,
    bool IsOffloadEntry, const RegionCodeGenTy &CodeGen) {
  // Create a unique name for the entry function using the source location
  // information of the current target region. The name will be something like:
  //
  // __omp_offloading_DD_FFFF_PP_lBB
  //
  // where DD_FFFF is an ID unique to the file (device and file IDs), PP is the
  // mangled name of the function that encloses the target region and BB is the
  // line number of the target region.

  unsigned DeviceID;
  unsigned FileID;
  unsigned Line;
  getTargetEntryUniqueInfo(CGM.getContext(), D.getLocStart(), DeviceID, FileID,
                           Line);
  SmallString<64> EntryFnName;
  {
    llvm::raw_svector_ostream OS(EntryFnName);
    OS << "__omp_offloading" << llvm::format("_%x", DeviceID)
       << llvm::format("_%x_", FileID) << ParentName << "_l" << Line;
  }

  const CapturedStmt &CS = *cast<CapturedStmt>(D.getAssociatedStmt());

  CodeGenFunction CGF(CGM, true);
  CGOpenMPTargetRegionInfo CGInfo(CS, CodeGen, EntryFnName);
  CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CGInfo);

  OutlinedFn = CGF.GenerateOpenMPCapturedStmtFunction(CS);

  // If this target outline function is not an offload entry, we don't need to
  // register it.
  if (!IsOffloadEntry)
    return;

  // The target region ID is used by the runtime library to identify the current
  // target region, so it only has to be unique and not necessarily point to
  // anything. It could be the pointer to the outlined function that implements
  // the target region, but we aren't using that so that the compiler doesn't
  // need to keep that, and could therefore inline the host function if proven
  // worthwhile during optimization. In the other hand, if emitting code for the
  // device, the ID has to be the function address so that it can retrieved from
  // the offloading entry and launched by the runtime library. We also mark the
  // outlined function to have external linkage in case we are emitting code for
  // the device, because these functions will be entry points to the device.

  if (CGM.getLangOpts().OpenMPIsDevice) {
    OutlinedFnID = llvm::ConstantExpr::getBitCast(OutlinedFn, CGM.Int8PtrTy);
    OutlinedFn->setLinkage(llvm::GlobalValue::ExternalLinkage);
  } else
    OutlinedFnID = new llvm::GlobalVariable(
        CGM.getModule(), CGM.Int8Ty, /*isConstant=*/true,
        llvm::GlobalValue::PrivateLinkage,
        llvm::Constant::getNullValue(CGM.Int8Ty), ".omp_offload.region_id");

  // Register the information for the entry associated with this target region.
  OffloadEntriesInfoManager.registerTargetRegionEntryInfo(
      DeviceID, FileID, ParentName, Line, OutlinedFn, OutlinedFnID,
      /*Flags=*/0);
}

/// discard all CompoundStmts intervening between two constructs
static const Stmt *ignoreCompoundStmts(const Stmt *Body) {
  while (auto *CS = dyn_cast_or_null<CompoundStmt>(Body))
    Body = CS->body_front();

  return Body;
}

/// Emit the number of teams for a target directive.  Inspect the num_teams
/// clause associated with a teams construct combined or closely nested
/// with the target directive.
///
/// Emit a team of size one for directives such as 'target parallel' that
/// have no associated teams construct.
///
/// Otherwise, return nullptr.
static llvm::Value *
emitNumTeamsForTargetDirective(CGOpenMPRuntime &OMPRuntime,
                               CodeGenFunction &CGF,
                               const OMPExecutableDirective &D) {

  assert(!CGF.getLangOpts().OpenMPIsDevice && "Clauses associated with the "
                                              "teams directive expected to be "
                                              "emitted only for the host!");

  auto &Bld = CGF.Builder;

  // If the target directive is combined with a teams directive:
  //   Return the value in the num_teams clause, if any.
  //   Otherwise, return 0 to denote the runtime default.
  if (isOpenMPTeamsDirective(D.getDirectiveKind())) {
    if (const auto *NumTeamsClause = D.getSingleClause<OMPNumTeamsClause>()) {
      CodeGenFunction::RunCleanupsScope NumTeamsScope(CGF);
      auto NumTeams = CGF.EmitScalarExpr(NumTeamsClause->getNumTeams(),
                                         /*IgnoreResultAssign*/ true);
      return Bld.CreateIntCast(NumTeams, CGF.Int32Ty,
                               /*IsSigned=*/true);
    }

    // The default value is 0.
    return Bld.getInt32(0);
  }

  // If the target directive is combined with a parallel directive but not a
  // teams directive, start one team.
  if (isOpenMPParallelDirective(D.getDirectiveKind()))
    return Bld.getInt32(1);

  // If the current target region has a teams region enclosed, we need to get
  // the number of teams to pass to the runtime function call. This is done
  // by generating the expression in a inlined region. This is required because
  // the expression is captured in the enclosing target environment when the
  // teams directive is not combined with target.

  const CapturedStmt &CS = *cast<CapturedStmt>(D.getAssociatedStmt());

  // FIXME: Accommodate other combined directives with teams when they become
  // available.
  if (auto *TeamsDir = dyn_cast_or_null<OMPTeamsDirective>(
          ignoreCompoundStmts(CS.getCapturedStmt()))) {
    if (auto *NTE = TeamsDir->getSingleClause<OMPNumTeamsClause>()) {
      CGOpenMPInnerExprInfo CGInfo(CGF, CS);
      CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CGInfo);
      llvm::Value *NumTeams = CGF.EmitScalarExpr(NTE->getNumTeams());
      return Bld.CreateIntCast(NumTeams, CGF.Int32Ty,
                               /*IsSigned=*/true);
    }

    // If we have an enclosed teams directive but no num_teams clause we use
    // the default value 0.
    return Bld.getInt32(0);
  }

  // No teams associated with the directive.
  return nullptr;
}

/// Emit the number of threads for a target directive.  Inspect the
/// thread_limit clause associated with a teams construct combined or closely
/// nested with the target directive.
///
/// Emit the num_threads clause for directives such as 'target parallel' that
/// have no associated teams construct.
///
/// Otherwise, return nullptr.
static llvm::Value *
emitNumThreadsForTargetDirective(CGOpenMPRuntime &OMPRuntime,
                                 CodeGenFunction &CGF,
                                 const OMPExecutableDirective &D) {

  assert(!CGF.getLangOpts().OpenMPIsDevice && "Clauses associated with the "
                                              "teams directive expected to be "
                                              "emitted only for the host!");

  auto &Bld = CGF.Builder;

  //
  // If the target directive is combined with a teams directive:
  //   Return the value in the thread_limit clause, if any.
  //
  // If the target directive is combined with a parallel directive:
  //   Return the value in the num_threads clause, if any.
  //
  // If both clauses are set, select the minimum of the two.
  //
  // If neither teams or parallel combined directives set the number of threads
  // in a team, return 0 to denote the runtime default.
  //
  // If this is not a teams directive return nullptr.

  if (isOpenMPTeamsDirective(D.getDirectiveKind()) ||
      isOpenMPParallelDirective(D.getDirectiveKind())) {
    llvm::Value *DefaultThreadLimitVal = Bld.getInt32(0);
    llvm::Value *NumThreadsVal = nullptr;
    llvm::Value *ThreadLimitVal = nullptr;

    if (const auto *ThreadLimitClause =
            D.getSingleClause<OMPThreadLimitClause>()) {
      CodeGenFunction::RunCleanupsScope ThreadLimitScope(CGF);
      auto ThreadLimit = CGF.EmitScalarExpr(ThreadLimitClause->getThreadLimit(),
                                            /*IgnoreResultAssign*/ true);
      ThreadLimitVal = Bld.CreateIntCast(ThreadLimit, CGF.Int32Ty,
                                         /*IsSigned=*/true);
    }

    if (const auto *NumThreadsClause =
            D.getSingleClause<OMPNumThreadsClause>()) {
      CodeGenFunction::RunCleanupsScope NumThreadsScope(CGF);
      llvm::Value *NumThreads =
          CGF.EmitScalarExpr(NumThreadsClause->getNumThreads(),
                             /*IgnoreResultAssign*/ true);
      NumThreadsVal =
          Bld.CreateIntCast(NumThreads, CGF.Int32Ty, /*IsSigned=*/true);
    }

    // Select the lesser of thread_limit and num_threads.
    if (NumThreadsVal)
      ThreadLimitVal = ThreadLimitVal
                           ? Bld.CreateSelect(Bld.CreateICmpSLT(NumThreadsVal,
                                                                ThreadLimitVal),
                                              NumThreadsVal, ThreadLimitVal)
                           : NumThreadsVal;

    // Set default value passed to the runtime if either teams or a target
    // parallel type directive is found but no clause is specified.
    if (!ThreadLimitVal)
      ThreadLimitVal = DefaultThreadLimitVal;

    return ThreadLimitVal;
  }

  // If the current target region has a teams region enclosed, we need to get
  // the thread limit to pass to the runtime function call. This is done
  // by generating the expression in a inlined region. This is required because
  // the expression is captured in the enclosing target environment when the
  // teams directive is not combined with target.

  const CapturedStmt &CS = *cast<CapturedStmt>(D.getAssociatedStmt());

  // FIXME: Accommodate other combined directives with teams when they become
  // available.
  if (auto *TeamsDir = dyn_cast_or_null<OMPTeamsDirective>(
          ignoreCompoundStmts(CS.getCapturedStmt()))) {
    if (auto *TLE = TeamsDir->getSingleClause<OMPThreadLimitClause>()) {
      CGOpenMPInnerExprInfo CGInfo(CGF, CS);
      CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CGInfo);
      llvm::Value *ThreadLimit = CGF.EmitScalarExpr(TLE->getThreadLimit());
      return CGF.Builder.CreateIntCast(ThreadLimit, CGF.Int32Ty,
                                       /*IsSigned=*/true);
    }

    // If we have an enclosed teams directive but no thread_limit clause we use
    // the default value 0.
    return CGF.Builder.getInt32(0);
  }

  // No teams associated with the directive.
  return nullptr;
}

namespace {
// \brief Utility to handle information from clauses associated with a given
// construct that use mappable expressions (e.g. 'map' clause, 'to' clause).
// It provides a convenient interface to obtain the information and generate
// code for that information.
class MappableExprsHandler {
public:
  /// \brief Values for bit flags used to specify the mapping type for
  /// offloading.
  enum OpenMPOffloadMappingFlags {
    /// \brief Allocate memory on the device and move data from host to device.
    OMP_MAP_TO = 0x01,
    /// \brief Allocate memory on the device and move data from device to host.
    OMP_MAP_FROM = 0x02,
    /// \brief Always perform the requested mapping action on the element, even
    /// if it was already mapped before.
    OMP_MAP_ALWAYS = 0x04,
    /// \brief Delete the element from the device environment, ignoring the
    /// current reference count associated with the element.
    OMP_MAP_DELETE = 0x08,
    /// \brief The element being mapped is a pointer, therefore the pointee
    /// should be mapped as well.
    OMP_MAP_IS_PTR = 0x10,
    /// \brief This flags signals that an argument is the first one relating to
    /// a map/private clause expression. For some cases a single
    /// map/privatization results in multiple arguments passed to the runtime
    /// library.
    OMP_MAP_FIRST_REF = 0x20,
    /// \brief Signal that the runtime library has to return the device pointer
    /// in the current position for the data being mapped.
    OMP_MAP_RETURN_PTR = 0x40,
    /// \brief This flag signals that the reference being passed is a pointer to
    /// private data.
    OMP_MAP_PRIVATE_PTR = 0x80,
    /// \brief Pass the element to the device by value.
    OMP_MAP_PRIVATE_VAL = 0x100,
  };

  /// Class that associates information with a base pointer to be passed to the
  /// runtime library.
  class BasePointerInfo {
    /// The base pointer.
    llvm::Value *Ptr = nullptr;
    /// The base declaration that refers to this device pointer, or null if
    /// there is none.
    const ValueDecl *DevPtrDecl = nullptr;

  public:
    BasePointerInfo(llvm::Value *Ptr, const ValueDecl *DevPtrDecl = nullptr)
        : Ptr(Ptr), DevPtrDecl(DevPtrDecl) {}
    llvm::Value *operator*() const { return Ptr; }
    const ValueDecl *getDevicePtrDecl() const { return DevPtrDecl; }
    void setDevicePtrDecl(const ValueDecl *D) { DevPtrDecl = D; }
  };

  typedef SmallVector<BasePointerInfo, 16> MapBaseValuesArrayTy;
  typedef SmallVector<llvm::Value *, 16> MapValuesArrayTy;
  typedef SmallVector<unsigned, 16> MapFlagsArrayTy;

private:
  /// \brief Directive from where the map clauses were extracted.
  const OMPExecutableDirective &CurDir;

  /// \brief Function the directive is being generated for.
  CodeGenFunction &CGF;

  /// \brief Set of all first private variables in the current directive.
  llvm::SmallPtrSet<const VarDecl *, 8> FirstPrivateDecls;

  /// Map between device pointer declarations and their expression components.
  /// The key value for declarations in 'this' is null.
  llvm::DenseMap<
      const ValueDecl *,
      SmallVector<OMPClauseMappableExprCommon::MappableExprComponentListRef, 4>>
      DevPointersMap;

  llvm::Value *getExprTypeSize(const Expr *E) const {
    auto ExprTy = E->getType().getCanonicalType();

    // Reference types are ignored for mapping purposes.
    if (auto *RefTy = ExprTy->getAs<ReferenceType>())
      ExprTy = RefTy->getPointeeType().getCanonicalType();

    // Given that an array section is considered a built-in type, we need to
    // do the calculation based on the length of the section instead of relying
    // on CGF.getTypeSize(E->getType()).
    if (const auto *OAE = dyn_cast<OMPArraySectionExpr>(E)) {
      QualType BaseTy = OMPArraySectionExpr::getBaseOriginalType(
                            OAE->getBase()->IgnoreParenImpCasts())
                            .getCanonicalType();

      // If there is no length associated with the expression, that means we
      // are using the whole length of the base.
      if (!OAE->getLength() && OAE->getColonLoc().isValid())
        return CGF.getTypeSize(BaseTy);

      llvm::Value *ElemSize;
      if (auto *PTy = BaseTy->getAs<PointerType>())
        ElemSize = CGF.getTypeSize(PTy->getPointeeType().getCanonicalType());
      else {
        auto *ATy = cast<ArrayType>(BaseTy.getTypePtr());
        assert(ATy && "Expecting array type if not a pointer type.");
        ElemSize = CGF.getTypeSize(ATy->getElementType().getCanonicalType());
      }

      // If we don't have a length at this point, that is because we have an
      // array section with a single element.
      if (!OAE->getLength())
        return ElemSize;

      auto *LengthVal = CGF.EmitScalarExpr(OAE->getLength());
      LengthVal =
          CGF.Builder.CreateIntCast(LengthVal, CGF.SizeTy, /*isSigned=*/false);
      return CGF.Builder.CreateNUWMul(LengthVal, ElemSize);
    }
    return CGF.getTypeSize(ExprTy);
  }

  /// \brief Return the corresponding bits for a given map clause modifier. Add
  /// a flag marking the map as a pointer if requested. Add a flag marking the
  /// map as the first one of a series of maps that relate to the same map
  /// expression.
  unsigned getMapTypeBits(OpenMPMapClauseKind MapType,
                          OpenMPMapClauseKind MapTypeModifier, bool AddPtrFlag,
                          bool AddIsFirstFlag) const {
    unsigned Bits = 0u;
    switch (MapType) {
    case OMPC_MAP_alloc:
    case OMPC_MAP_release:
      // alloc and release is the default behavior in the runtime library,  i.e.
      // if we don't pass any bits alloc/release that is what the runtime is
      // going to do. Therefore, we don't need to signal anything for these two
      // type modifiers.
      break;
    case OMPC_MAP_to:
      Bits = OMP_MAP_TO;
      break;
    case OMPC_MAP_from:
      Bits = OMP_MAP_FROM;
      break;
    case OMPC_MAP_tofrom:
      Bits = OMP_MAP_TO | OMP_MAP_FROM;
      break;
    case OMPC_MAP_delete:
      Bits = OMP_MAP_DELETE;
      break;
    default:
      llvm_unreachable("Unexpected map type!");
      break;
    }
    if (AddPtrFlag)
      Bits |= OMP_MAP_IS_PTR;
    if (AddIsFirstFlag)
      Bits |= OMP_MAP_FIRST_REF;
    if (MapTypeModifier == OMPC_MAP_always)
      Bits |= OMP_MAP_ALWAYS;
    return Bits;
  }

  /// \brief Return true if the provided expression is a final array section. A
  /// final array section, is one whose length can't be proved to be one.
  bool isFinalArraySectionExpression(const Expr *E) const {
    auto *OASE = dyn_cast<OMPArraySectionExpr>(E);

    // It is not an array section and therefore not a unity-size one.
    if (!OASE)
      return false;

    // An array section with no colon always refer to a single element.
    if (OASE->getColonLoc().isInvalid())
      return false;

    auto *Length = OASE->getLength();

    // If we don't have a length we have to check if the array has size 1
    // for this dimension. Also, we should always expect a length if the
    // base type is pointer.
    if (!Length) {
      auto BaseQTy = OMPArraySectionExpr::getBaseOriginalType(
                         OASE->getBase()->IgnoreParenImpCasts())
                         .getCanonicalType();
      if (auto *ATy = dyn_cast<ConstantArrayType>(BaseQTy.getTypePtr()))
        return ATy->getSize().getSExtValue() != 1;
      // If we don't have a constant dimension length, we have to consider
      // the current section as having any size, so it is not necessarily
      // unitary. If it happen to be unity size, that's user fault.
      return true;
    }

    // Check if the length evaluates to 1.
    llvm::APSInt ConstLength;
    if (!Length->EvaluateAsInt(ConstLength, CGF.getContext()))
      return true; // Can have more that size 1.

    return ConstLength.getSExtValue() != 1;
  }

  /// \brief Generate the base pointers, section pointers, sizes and map type
  /// bits for the provided map type, map modifier, and expression components.
  /// \a IsFirstComponent should be set to true if the provided set of
  /// components is the first associated with a capture.
  void generateInfoForComponentList(
      OpenMPMapClauseKind MapType, OpenMPMapClauseKind MapTypeModifier,
      OMPClauseMappableExprCommon::MappableExprComponentListRef Components,
      MapBaseValuesArrayTy &BasePointers, MapValuesArrayTy &Pointers,
      MapValuesArrayTy &Sizes, MapFlagsArrayTy &Types,
      bool IsFirstComponentList) const {

    // The following summarizes what has to be generated for each map and the
    // types bellow. The generated information is expressed in this order:
    // base pointer, section pointer, size, flags
    // (to add to the ones that come from the map type and modifier).
    //
    // double d;
    // int i[100];
    // float *p;
    //
    // struct S1 {
    //   int i;
    //   float f[50];
    // }
    // struct S2 {
    //   int i;
    //   float f[50];
    //   S1 s;
    //   double *p;
    //   struct S2 *ps;
    // }
    // S2 s;
    // S2 *ps;
    //
    // map(d)
    // &d, &d, sizeof(double), noflags
    //
    // map(i)
    // &i, &i, 100*sizeof(int), noflags
    //
    // map(i[1:23])
    // &i(=&i[0]), &i[1], 23*sizeof(int), noflags
    //
    // map(p)
    // &p, &p, sizeof(float*), noflags
    //
    // map(p[1:24])
    // p, &p[1], 24*sizeof(float), noflags
    //
    // map(s)
    // &s, &s, sizeof(S2), noflags
    //
    // map(s.i)
    // &s, &(s.i), sizeof(int), noflags
    //
    // map(s.s.f)
    // &s, &(s.i.f), 50*sizeof(int), noflags
    //
    // map(s.p)
    // &s, &(s.p), sizeof(double*), noflags
    //
    // map(s.p[:22], s.a s.b)
    // &s, &(s.p), sizeof(double*), noflags
    // &(s.p), &(s.p[0]), 22*sizeof(double), ptr_flag + extra_flag
    //
    // map(s.ps)
    // &s, &(s.ps), sizeof(S2*), noflags
    //
    // map(s.ps->s.i)
    // &s, &(s.ps), sizeof(S2*), noflags
    // &(s.ps), &(s.ps->s.i), sizeof(int), ptr_flag + extra_flag
    //
    // map(s.ps->ps)
    // &s, &(s.ps), sizeof(S2*), noflags
    // &(s.ps), &(s.ps->ps), sizeof(S2*), ptr_flag + extra_flag
    //
    // map(s.ps->ps->ps)
    // &s, &(s.ps), sizeof(S2*), noflags
    // &(s.ps), &(s.ps->ps), sizeof(S2*), ptr_flag + extra_flag
    // &(s.ps->ps), &(s.ps->ps->ps), sizeof(S2*), ptr_flag + extra_flag
    //
    // map(s.ps->ps->s.f[:22])
    // &s, &(s.ps), sizeof(S2*), noflags
    // &(s.ps), &(s.ps->ps), sizeof(S2*), ptr_flag + extra_flag
    // &(s.ps->ps), &(s.ps->ps->s.f[0]), 22*sizeof(float), ptr_flag + extra_flag
    //
    // map(ps)
    // &ps, &ps, sizeof(S2*), noflags
    //
    // map(ps->i)
    // ps, &(ps->i), sizeof(int), noflags
    //
    // map(ps->s.f)
    // ps, &(ps->s.f[0]), 50*sizeof(float), noflags
    //
    // map(ps->p)
    // ps, &(ps->p), sizeof(double*), noflags
    //
    // map(ps->p[:22])
    // ps, &(ps->p), sizeof(double*), noflags
    // &(ps->p), &(ps->p[0]), 22*sizeof(double), ptr_flag + extra_flag
    //
    // map(ps->ps)
    // ps, &(ps->ps), sizeof(S2*), noflags
    //
    // map(ps->ps->s.i)
    // ps, &(ps->ps), sizeof(S2*), noflags
    // &(ps->ps), &(ps->ps->s.i), sizeof(int), ptr_flag + extra_flag
    //
    // map(ps->ps->ps)
    // ps, &(ps->ps), sizeof(S2*), noflags
    // &(ps->ps), &(ps->ps->ps), sizeof(S2*), ptr_flag + extra_flag
    //
    // map(ps->ps->ps->ps)
    // ps, &(ps->ps), sizeof(S2*), noflags
    // &(ps->ps), &(ps->ps->ps), sizeof(S2*), ptr_flag + extra_flag
    // &(ps->ps->ps), &(ps->ps->ps->ps), sizeof(S2*), ptr_flag + extra_flag
    //
    // map(ps->ps->ps->s.f[:22])
    // ps, &(ps->ps), sizeof(S2*), noflags
    // &(ps->ps), &(ps->ps->ps), sizeof(S2*), ptr_flag + extra_flag
    // &(ps->ps->ps), &(ps->ps->ps->s.f[0]), 22*sizeof(float), ptr_flag +
    // extra_flag

    // Track if the map information being generated is the first for a capture.
    bool IsCaptureFirstInfo = IsFirstComponentList;

    // Scan the components from the base to the complete expression.
    auto CI = Components.rbegin();
    auto CE = Components.rend();
    auto I = CI;

    // Track if the map information being generated is the first for a list of
    // components.
    bool IsExpressionFirstInfo = true;
    llvm::Value *BP = nullptr;

    if (auto *ME = dyn_cast<MemberExpr>(I->getAssociatedExpression())) {
      // The base is the 'this' pointer. The content of the pointer is going
      // to be the base of the field being mapped.
      BP = CGF.EmitScalarExpr(ME->getBase());
    } else {
      // The base is the reference to the variable.
      // BP = &Var.
      BP = CGF.EmitLValue(cast<DeclRefExpr>(I->getAssociatedExpression()))
               .getPointer();

      // If the variable is a pointer and is being dereferenced (i.e. is not
      // the last component), the base has to be the pointer itself, not its
      // reference. References are ignored for mapping purposes.
      QualType Ty =
          I->getAssociatedDeclaration()->getType().getNonReferenceType();
      if (Ty->isAnyPointerType() && std::next(I) != CE) {
        auto PtrAddr = CGF.MakeNaturalAlignAddrLValue(BP, Ty);
        BP = CGF.EmitLoadOfPointerLValue(PtrAddr.getAddress(),
                                         Ty->castAs<PointerType>())
                 .getPointer();

        // We do not need to generate individual map information for the
        // pointer, it can be associated with the combined storage.
        ++I;
      }
    }

    for (; I != CE; ++I) {
      auto Next = std::next(I);

      // We need to generate the addresses and sizes if this is the last
      // component, if the component is a pointer or if it is an array section
      // whose length can't be proved to be one. If this is a pointer, it
      // becomes the base address for the following components.

      // A final array section, is one whose length can't be proved to be one.
      bool IsFinalArraySection =
          isFinalArraySectionExpression(I->getAssociatedExpression());

      // Get information on whether the element is a pointer. Have to do a
      // special treatment for array sections given that they are built-in
      // types.
      const auto *OASE =
          dyn_cast<OMPArraySectionExpr>(I->getAssociatedExpression());
      bool IsPointer =
          (OASE &&
           OMPArraySectionExpr::getBaseOriginalType(OASE)
               .getCanonicalType()
               ->isAnyPointerType()) ||
          I->getAssociatedExpression()->getType()->isAnyPointerType();

      if (Next == CE || IsPointer || IsFinalArraySection) {

        // If this is not the last component, we expect the pointer to be
        // associated with an array expression or member expression.
        assert((Next == CE ||
                isa<MemberExpr>(Next->getAssociatedExpression()) ||
                isa<ArraySubscriptExpr>(Next->getAssociatedExpression()) ||
                isa<OMPArraySectionExpr>(Next->getAssociatedExpression())) &&
               "Unexpected expression");

        auto *LB = CGF.EmitLValue(I->getAssociatedExpression()).getPointer();
        auto *Size = getExprTypeSize(I->getAssociatedExpression());

        // If we have a member expression and the current component is a
        // reference, we have to map the reference too. Whenever we have a
        // reference, the section that reference refers to is going to be a
        // load instruction from the storage assigned to the reference.
        if (isa<MemberExpr>(I->getAssociatedExpression()) &&
            I->getAssociatedDeclaration()->getType()->isReferenceType()) {
          auto *LI = cast<llvm::LoadInst>(LB);
          auto *RefAddr = LI->getPointerOperand();

          BasePointers.push_back(BP);
          Pointers.push_back(RefAddr);
          Sizes.push_back(CGF.getTypeSize(CGF.getContext().VoidPtrTy));
          Types.push_back(getMapTypeBits(
              /*MapType*/ OMPC_MAP_alloc, /*MapTypeModifier=*/OMPC_MAP_unknown,
              !IsExpressionFirstInfo, IsCaptureFirstInfo));
          IsExpressionFirstInfo = false;
          IsCaptureFirstInfo = false;
          // The reference will be the next base address.
          BP = RefAddr;
        }

        BasePointers.push_back(BP);
        Pointers.push_back(LB);
        Sizes.push_back(Size);

        // We need to add a pointer flag for each map that comes from the
        // same expression except for the first one. We also need to signal
        // this map is the first one that relates with the current capture
        // (there is a set of entries for each capture).
        Types.push_back(getMapTypeBits(MapType, MapTypeModifier,
                                       !IsExpressionFirstInfo,
                                       IsCaptureFirstInfo));

        // If we have a final array section, we are done with this expression.
        if (IsFinalArraySection)
          break;

        // The pointer becomes the base for the next element.
        if (Next != CE)
          BP = LB;

        IsExpressionFirstInfo = false;
        IsCaptureFirstInfo = false;
        continue;
      }
    }
  }

  /// \brief Return the adjusted map modifiers if the declaration a capture
  /// refers to appears in a first-private clause. This is expected to be used
  /// only with directives that start with 'target'.
  unsigned adjustMapModifiersForPrivateClauses(const CapturedStmt::Capture &Cap,
                                               unsigned CurrentModifiers) {
    assert(Cap.capturesVariable() && "Expected capture by reference only!");

    // A first private variable captured by reference will use only the
    // 'private ptr' and 'map to' flag. Return the right flags if the captured
    // declaration is known as first-private in this handler.
    if (FirstPrivateDecls.count(Cap.getCapturedVar()))
      return MappableExprsHandler::OMP_MAP_PRIVATE_PTR |
             MappableExprsHandler::OMP_MAP_TO;

    // We didn't modify anything.
    return CurrentModifiers;
  }

public:
  MappableExprsHandler(const OMPExecutableDirective &Dir, CodeGenFunction &CGF)
      : CurDir(Dir), CGF(CGF) {
    // Extract firstprivate clause information.
    for (const auto *C : Dir.getClausesOfKind<OMPFirstprivateClause>())
      for (const auto *D : C->varlists())
        FirstPrivateDecls.insert(
            cast<VarDecl>(cast<DeclRefExpr>(D)->getDecl())->getCanonicalDecl());
    // Extract device pointer clause information.
    for (const auto *C : Dir.getClausesOfKind<OMPIsDevicePtrClause>())
      for (auto L : C->component_lists())
        DevPointersMap[L.first].push_back(L.second);
  }

  /// \brief Generate all the base pointers, section pointers, sizes and map
  /// types for the extracted mappable expressions. Also, for each item that
  /// relates with a device pointer, a pair of the relevant declaration and
  /// index where it occurs is appended to the device pointers info array.
  void generateAllInfo(MapBaseValuesArrayTy &BasePointers,
                       MapValuesArrayTy &Pointers, MapValuesArrayTy &Sizes,
                       MapFlagsArrayTy &Types) const {
    BasePointers.clear();
    Pointers.clear();
    Sizes.clear();
    Types.clear();

    struct MapInfo {
      /// Kind that defines how a device pointer has to be returned.
      enum ReturnPointerKind {
        // Don't have to return any pointer.
        RPK_None,
        // Pointer is the base of the declaration.
        RPK_Base,
        // Pointer is a member of the base declaration - 'this'
        RPK_Member,
        // Pointer is a reference and a member of the base declaration - 'this'
        RPK_MemberReference,
      };
      OMPClauseMappableExprCommon::MappableExprComponentListRef Components;
      OpenMPMapClauseKind MapType;
      OpenMPMapClauseKind MapTypeModifier;
      ReturnPointerKind ReturnDevicePointer;

      MapInfo()
          : MapType(OMPC_MAP_unknown), MapTypeModifier(OMPC_MAP_unknown),
            ReturnDevicePointer(RPK_None) {}
      MapInfo(
          OMPClauseMappableExprCommon::MappableExprComponentListRef Components,
          OpenMPMapClauseKind MapType, OpenMPMapClauseKind MapTypeModifier,
          ReturnPointerKind ReturnDevicePointer)
          : Components(Components), MapType(MapType),
            MapTypeModifier(MapTypeModifier),
            ReturnDevicePointer(ReturnDevicePointer) {}
    };

    // We have to process the component lists that relate with the same
    // declaration in a single chunk so that we can generate the map flags
    // correctly. Therefore, we organize all lists in a map.
    llvm::MapVector<const ValueDecl *, SmallVector<MapInfo, 8>> Info;

    // Helper function to fill the information map for the different supported
    // clauses.
    auto &&InfoGen = [&Info](
        const ValueDecl *D,
        OMPClauseMappableExprCommon::MappableExprComponentListRef L,
        OpenMPMapClauseKind MapType, OpenMPMapClauseKind MapModifier,
        MapInfo::ReturnPointerKind ReturnDevicePointer) {
      const ValueDecl *VD =
          D ? cast<ValueDecl>(D->getCanonicalDecl()) : nullptr;
      Info[VD].push_back({L, MapType, MapModifier, ReturnDevicePointer});
    };

    // FIXME: MSVC 2013 seems to require this-> to find member CurDir.
    for (auto *C : this->CurDir.getClausesOfKind<OMPMapClause>())
      for (auto L : C->component_lists())
        InfoGen(L.first, L.second, C->getMapType(), C->getMapTypeModifier(),
                MapInfo::RPK_None);
    for (auto *C : this->CurDir.getClausesOfKind<OMPToClause>())
      for (auto L : C->component_lists())
        InfoGen(L.first, L.second, OMPC_MAP_to, OMPC_MAP_unknown,
                MapInfo::RPK_None);
    for (auto *C : this->CurDir.getClausesOfKind<OMPFromClause>())
      for (auto L : C->component_lists())
        InfoGen(L.first, L.second, OMPC_MAP_from, OMPC_MAP_unknown,
                MapInfo::RPK_None);

    // Look at the use_device_ptr clause information and mark the existing map
    // entries as such. If there is no map information for an entry in the
    // use_device_ptr list, we create one with map type 'alloc' and zero size
    // section. It is the user fault if that was not mapped before.
    // FIXME: MSVC 2013 seems to require this-> to find member CurDir.
    for (auto *C : this->CurDir.getClausesOfKind<OMPUseDevicePtrClause>())
      for (auto L : C->component_lists()) {
        assert(!L.second.empty() && "Not expecting empty list of components!");
        const ValueDecl *VD = L.second.back().getAssociatedDeclaration();
        VD = cast<ValueDecl>(VD->getCanonicalDecl());
        auto *IE = L.second.back().getAssociatedExpression();
        // If the first component is a member expression, we have to look into
        // 'this', which maps to null in the map of map information. Otherwise
        // look directly for the information.
        auto It = Info.find(isa<MemberExpr>(IE) ? nullptr : VD);

        // We potentially have map information for this declaration already.
        // Look for the first set of components that refer to it.
        if (It != Info.end()) {
          auto CI = std::find_if(
              It->second.begin(), It->second.end(), [VD](const MapInfo &MI) {
                return MI.Components.back().getAssociatedDeclaration() == VD;
              });
          // If we found a map entry, signal that the pointer has to be returned
          // and move on to the next declaration.
          if (CI != It->second.end()) {
            CI->ReturnDevicePointer = isa<MemberExpr>(IE)
                                          ? (VD->getType()->isReferenceType()
                                                 ? MapInfo::RPK_MemberReference
                                                 : MapInfo::RPK_Member)
                                          : MapInfo::RPK_Base;
            continue;
          }
        }

        // We didn't find any match in our map information - generate a zero
        // size array section.
        // FIXME: MSVC 2013 seems to require this-> to find member CGF.
        llvm::Value *Ptr =
            this->CGF
                .EmitLoadOfLValue(this->CGF.EmitLValue(IE), SourceLocation())
                .getScalarVal();
        BasePointers.push_back({Ptr, VD});
        Pointers.push_back(Ptr);
        Sizes.push_back(llvm::Constant::getNullValue(this->CGF.SizeTy));
        Types.push_back(OMP_MAP_RETURN_PTR | OMP_MAP_FIRST_REF);
      }

    for (auto &M : Info) {
      // We need to know when we generate information for the first component
      // associated with a capture, because the mapping flags depend on it.
      bool IsFirstComponentList = true;
      for (MapInfo &L : M.second) {
        assert(!L.Components.empty() &&
               "Not expecting declaration with no component lists.");

        // Remember the current base pointer index.
        unsigned CurrentBasePointersIdx = BasePointers.size();
        // FIXME: MSVC 2013 seems to require this-> to find the member method.
        this->generateInfoForComponentList(L.MapType, L.MapTypeModifier,
                                           L.Components, BasePointers, Pointers,
                                           Sizes, Types, IsFirstComponentList);

        // If this entry relates with a device pointer, set the relevant
        // declaration and add the 'return pointer' flag.
        if (IsFirstComponentList &&
            L.ReturnDevicePointer != MapInfo::RPK_None) {
          // If the pointer is not the base of the map, we need to skip the
          // base. If it is a reference in a member field, we also need to skip
          // the map of the reference.
          if (L.ReturnDevicePointer != MapInfo::RPK_Base) {
            ++CurrentBasePointersIdx;
            if (L.ReturnDevicePointer == MapInfo::RPK_MemberReference)
              ++CurrentBasePointersIdx;
          }
          assert(BasePointers.size() > CurrentBasePointersIdx &&
                 "Unexpected number of mapped base pointers.");

          auto *RelevantVD = L.Components.back().getAssociatedDeclaration();
          assert(RelevantVD &&
                 "No relevant declaration related with device pointer??");

          BasePointers[CurrentBasePointersIdx].setDevicePtrDecl(RelevantVD);
          Types[CurrentBasePointersIdx] |= OMP_MAP_RETURN_PTR;
        }
        IsFirstComponentList = false;
      }
    }
  }

  /// \brief Generate the base pointers, section pointers, sizes and map types
  /// associated to a given capture.
  void generateInfoForCapture(const CapturedStmt::Capture *Cap,
                              llvm::Value *Arg,
                              MapBaseValuesArrayTy &BasePointers,
                              MapValuesArrayTy &Pointers,
                              MapValuesArrayTy &Sizes,
                              MapFlagsArrayTy &Types) const {
    assert(!Cap->capturesVariableArrayType() &&
           "Not expecting to generate map info for a variable array type!");

    BasePointers.clear();
    Pointers.clear();
    Sizes.clear();
    Types.clear();

    // We need to know when we generating information for the first component
    // associated with a capture, because the mapping flags depend on it.
    bool IsFirstComponentList = true;

    const ValueDecl *VD =
        Cap->capturesThis()
            ? nullptr
            : cast<ValueDecl>(Cap->getCapturedVar()->getCanonicalDecl());

    // If this declaration appears in a is_device_ptr clause we just have to
    // pass the pointer by value. If it is a reference to a declaration, we just
    // pass its value, otherwise, if it is a member expression, we need to map
    // 'to' the field.
    if (!VD) {
      auto It = DevPointersMap.find(VD);
      if (It != DevPointersMap.end()) {
        for (auto L : It->second) {
          generateInfoForComponentList(
              /*MapType=*/OMPC_MAP_to, /*MapTypeModifier=*/OMPC_MAP_unknown, L,
              BasePointers, Pointers, Sizes, Types, IsFirstComponentList);
          IsFirstComponentList = false;
        }
        return;
      }
    } else if (DevPointersMap.count(VD)) {
      BasePointers.push_back({Arg, VD});
      Pointers.push_back(Arg);
      Sizes.push_back(CGF.getTypeSize(CGF.getContext().VoidPtrTy));
      Types.push_back(OMP_MAP_PRIVATE_VAL | OMP_MAP_FIRST_REF);
      return;
    }

    // FIXME: MSVC 2013 seems to require this-> to find member CurDir.
    for (auto *C : this->CurDir.getClausesOfKind<OMPMapClause>())
      for (auto L : C->decl_component_lists(VD)) {
        assert(L.first == VD &&
               "We got information for the wrong declaration??");
        assert(!L.second.empty() &&
               "Not expecting declaration with no component lists.");
        generateInfoForComponentList(C->getMapType(), C->getMapTypeModifier(),
                                     L.second, BasePointers, Pointers, Sizes,
                                     Types, IsFirstComponentList);
        IsFirstComponentList = false;
      }

    return;
  }

  /// \brief Generate the default map information for a given capture \a CI,
  /// record field declaration \a RI and captured value \a CV.
  void generateDefaultMapInfo(const CapturedStmt::Capture &CI,
                              const FieldDecl &RI, llvm::Value *CV,
                              MapBaseValuesArrayTy &CurBasePointers,
                              MapValuesArrayTy &CurPointers,
                              MapValuesArrayTy &CurSizes,
                              MapFlagsArrayTy &CurMapTypes) {

    // Do the default mapping.
    if (CI.capturesThis()) {
      CurBasePointers.push_back(CV);
      CurPointers.push_back(CV);
      const PointerType *PtrTy = cast<PointerType>(RI.getType().getTypePtr());
      CurSizes.push_back(CGF.getTypeSize(PtrTy->getPointeeType()));
      // Default map type.
      CurMapTypes.push_back(OMP_MAP_TO | OMP_MAP_FROM);
    } else if (CI.capturesVariableByCopy()) {
      CurBasePointers.push_back(CV);
      CurPointers.push_back(CV);
      if (!RI.getType()->isAnyPointerType()) {
        // We have to signal to the runtime captures passed by value that are
        // not pointers.
        CurMapTypes.push_back(OMP_MAP_PRIVATE_VAL);
        CurSizes.push_back(CGF.getTypeSize(RI.getType()));
      } else {
        // Pointers are implicitly mapped with a zero size and no flags
        // (other than first map that is added for all implicit maps).
        CurMapTypes.push_back(0u);
        CurSizes.push_back(llvm::Constant::getNullValue(CGF.SizeTy));
      }
    } else {
      assert(CI.capturesVariable() && "Expected captured reference.");
      CurBasePointers.push_back(CV);
      CurPointers.push_back(CV);

      const ReferenceType *PtrTy =
          cast<ReferenceType>(RI.getType().getTypePtr());
      QualType ElementType = PtrTy->getPointeeType();
      CurSizes.push_back(CGF.getTypeSize(ElementType));
      // The default map type for a scalar/complex type is 'to' because by
      // default the value doesn't have to be retrieved. For an aggregate
      // type, the default is 'tofrom'.
      CurMapTypes.push_back(ElementType->isAggregateType()
                                ? (OMP_MAP_TO | OMP_MAP_FROM)
                                : OMP_MAP_TO);

      // If we have a capture by reference we may need to add the private
      // pointer flag if the base declaration shows in some first-private
      // clause.
      CurMapTypes.back() =
          adjustMapModifiersForPrivateClauses(CI, CurMapTypes.back());
    }
    // Every default map produces a single argument, so, it is always the
    // first one.
    CurMapTypes.back() |= OMP_MAP_FIRST_REF;
  }
};

enum OpenMPOffloadingReservedDeviceIDs {
  /// \brief Device ID if the device was not defined, runtime should get it
  /// from environment variables in the spec.
  OMP_DEVICEID_UNDEF = -1,
};
} // anonymous namespace

/// \brief Emit the arrays used to pass the captures and map information to the
/// offloading runtime library. If there is no map or capture information,
/// return nullptr by reference.
static void
emitOffloadingArrays(CodeGenFunction &CGF,
                     MappableExprsHandler::MapBaseValuesArrayTy &BasePointers,
                     MappableExprsHandler::MapValuesArrayTy &Pointers,
                     MappableExprsHandler::MapValuesArrayTy &Sizes,
                     MappableExprsHandler::MapFlagsArrayTy &MapTypes,
                     CGOpenMPRuntime::TargetDataInfo &Info) {
  auto &CGM = CGF.CGM;
  auto &Ctx = CGF.getContext();

  // Reset the array information.
  Info.clearArrayInfo();
  Info.NumberOfPtrs = BasePointers.size();

  if (Info.NumberOfPtrs) {
    // Detect if we have any capture size requiring runtime evaluation of the
    // size so that a constant array could be eventually used.
    bool hasRuntimeEvaluationCaptureSize = false;
    for (auto *S : Sizes)
      if (!isa<llvm::Constant>(S)) {
        hasRuntimeEvaluationCaptureSize = true;
        break;
      }

    llvm::APInt PointerNumAP(32, Info.NumberOfPtrs, /*isSigned=*/true);
    QualType PointerArrayType =
        Ctx.getConstantArrayType(Ctx.VoidPtrTy, PointerNumAP, ArrayType::Normal,
                                 /*IndexTypeQuals=*/0);

    Info.BasePointersArray =
        CGF.CreateMemTemp(PointerArrayType, ".offload_baseptrs").getPointer();
    Info.PointersArray =
        CGF.CreateMemTemp(PointerArrayType, ".offload_ptrs").getPointer();

    // If we don't have any VLA types or other types that require runtime
    // evaluation, we can use a constant array for the map sizes, otherwise we
    // need to fill up the arrays as we do for the pointers.
    if (hasRuntimeEvaluationCaptureSize) {
      QualType SizeArrayType = Ctx.getConstantArrayType(
          Ctx.getSizeType(), PointerNumAP, ArrayType::Normal,
          /*IndexTypeQuals=*/0);
      Info.SizesArray =
          CGF.CreateMemTemp(SizeArrayType, ".offload_sizes").getPointer();
    } else {
      // We expect all the sizes to be constant, so we collect them to create
      // a constant array.
      SmallVector<llvm::Constant *, 16> ConstSizes;
      for (auto S : Sizes)
        ConstSizes.push_back(cast<llvm::Constant>(S));

      auto *SizesArrayInit = llvm::ConstantArray::get(
          llvm::ArrayType::get(CGM.SizeTy, ConstSizes.size()), ConstSizes);
      auto *SizesArrayGbl = new llvm::GlobalVariable(
          CGM.getModule(), SizesArrayInit->getType(),
          /*isConstant=*/true, llvm::GlobalValue::PrivateLinkage,
          SizesArrayInit, ".offload_sizes");
      SizesArrayGbl->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
      Info.SizesArray = SizesArrayGbl;
    }

    // The map types are always constant so we don't need to generate code to
    // fill arrays. Instead, we create an array constant.
    llvm::Constant *MapTypesArrayInit =
        llvm::ConstantDataArray::get(CGF.Builder.getContext(), MapTypes);
    auto *MapTypesArrayGbl = new llvm::GlobalVariable(
        CGM.getModule(), MapTypesArrayInit->getType(),
        /*isConstant=*/true, llvm::GlobalValue::PrivateLinkage,
        MapTypesArrayInit, ".offload_maptypes");
    MapTypesArrayGbl->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
    Info.MapTypesArray = MapTypesArrayGbl;

    for (unsigned i = 0; i < Info.NumberOfPtrs; ++i) {
      llvm::Value *BPVal = *BasePointers[i];
      llvm::Value *BP = CGF.Builder.CreateConstInBoundsGEP2_32(
          llvm::ArrayType::get(CGM.VoidPtrTy, Info.NumberOfPtrs),
          Info.BasePointersArray, 0, i);
      BP = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
          BP, BPVal->getType()->getPointerTo(/*AddrSpace=*/0));
      Address BPAddr(BP, Ctx.getTypeAlignInChars(Ctx.VoidPtrTy));
      CGF.Builder.CreateStore(BPVal, BPAddr);

      if (Info.requiresDevicePointerInfo())
        if (auto *DevVD = BasePointers[i].getDevicePtrDecl())
          Info.CaptureDeviceAddrMap.insert(std::make_pair(DevVD, BPAddr));

      llvm::Value *PVal = Pointers[i];
      llvm::Value *P = CGF.Builder.CreateConstInBoundsGEP2_32(
          llvm::ArrayType::get(CGM.VoidPtrTy, Info.NumberOfPtrs),
          Info.PointersArray, 0, i);
      P = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
          P, PVal->getType()->getPointerTo(/*AddrSpace=*/0));
      Address PAddr(P, Ctx.getTypeAlignInChars(Ctx.VoidPtrTy));
      CGF.Builder.CreateStore(PVal, PAddr);

      if (hasRuntimeEvaluationCaptureSize) {
        llvm::Value *S = CGF.Builder.CreateConstInBoundsGEP2_32(
            llvm::ArrayType::get(CGM.SizeTy, Info.NumberOfPtrs),
            Info.SizesArray,
            /*Idx0=*/0,
            /*Idx1=*/i);
        Address SAddr(S, Ctx.getTypeAlignInChars(Ctx.getSizeType()));
        CGF.Builder.CreateStore(
            CGF.Builder.CreateIntCast(Sizes[i], CGM.SizeTy, /*isSigned=*/true),
            SAddr);
      }
    }
  }
}
/// \brief Emit the arguments to be passed to the runtime library based on the
/// arrays of pointers, sizes and map types.
static void emitOffloadingArraysArgument(
    CodeGenFunction &CGF, llvm::Value *&BasePointersArrayArg,
    llvm::Value *&PointersArrayArg, llvm::Value *&SizesArrayArg,
    llvm::Value *&MapTypesArrayArg, CGOpenMPRuntime::TargetDataInfo &Info) {
  auto &CGM = CGF.CGM;
  if (Info.NumberOfPtrs) {
    BasePointersArrayArg = CGF.Builder.CreateConstInBoundsGEP2_32(
        llvm::ArrayType::get(CGM.VoidPtrTy, Info.NumberOfPtrs),
        Info.BasePointersArray,
        /*Idx0=*/0, /*Idx1=*/0);
    PointersArrayArg = CGF.Builder.CreateConstInBoundsGEP2_32(
        llvm::ArrayType::get(CGM.VoidPtrTy, Info.NumberOfPtrs),
        Info.PointersArray,
        /*Idx0=*/0,
        /*Idx1=*/0);
    SizesArrayArg = CGF.Builder.CreateConstInBoundsGEP2_32(
        llvm::ArrayType::get(CGM.SizeTy, Info.NumberOfPtrs), Info.SizesArray,
        /*Idx0=*/0, /*Idx1=*/0);
    MapTypesArrayArg = CGF.Builder.CreateConstInBoundsGEP2_32(
        llvm::ArrayType::get(CGM.Int32Ty, Info.NumberOfPtrs),
        Info.MapTypesArray,
        /*Idx0=*/0,
        /*Idx1=*/0);
  } else {
    BasePointersArrayArg = llvm::ConstantPointerNull::get(CGM.VoidPtrPtrTy);
    PointersArrayArg = llvm::ConstantPointerNull::get(CGM.VoidPtrPtrTy);
    SizesArrayArg = llvm::ConstantPointerNull::get(CGM.SizeTy->getPointerTo());
    MapTypesArrayArg =
        llvm::ConstantPointerNull::get(CGM.Int32Ty->getPointerTo());
  }
}

void CGOpenMPRuntime::emitTargetCall(CodeGenFunction &CGF,
                                     const OMPExecutableDirective &D,
                                     llvm::Value *OutlinedFn,
                                     llvm::Value *OutlinedFnID,
                                     const Expr *IfCond, const Expr *Device,
                                     ArrayRef<llvm::Value *> CapturedVars) {
  if (!CGF.HaveInsertPoint())
    return;

  assert(OutlinedFn && "Invalid outlined function!");

  auto &Ctx = CGF.getContext();

  // Fill up the arrays with all the captured variables.
  MappableExprsHandler::MapValuesArrayTy KernelArgs;
  MappableExprsHandler::MapBaseValuesArrayTy BasePointers;
  MappableExprsHandler::MapValuesArrayTy Pointers;
  MappableExprsHandler::MapValuesArrayTy Sizes;
  MappableExprsHandler::MapFlagsArrayTy MapTypes;

  MappableExprsHandler::MapBaseValuesArrayTy CurBasePointers;
  MappableExprsHandler::MapValuesArrayTy CurPointers;
  MappableExprsHandler::MapValuesArrayTy CurSizes;
  MappableExprsHandler::MapFlagsArrayTy CurMapTypes;

  // Get mappable expression information.
  MappableExprsHandler MEHandler(D, CGF);

  const CapturedStmt &CS = *cast<CapturedStmt>(D.getAssociatedStmt());
  auto RI = CS.getCapturedRecordDecl()->field_begin();
  auto CV = CapturedVars.begin();
  for (CapturedStmt::const_capture_iterator CI = CS.capture_begin(),
                                            CE = CS.capture_end();
       CI != CE; ++CI, ++RI, ++CV) {
    StringRef Name;
    QualType Ty;

    CurBasePointers.clear();
    CurPointers.clear();
    CurSizes.clear();
    CurMapTypes.clear();

    // VLA sizes are passed to the outlined region by copy and do not have map
    // information associated.
    if (CI->capturesVariableArrayType()) {
      CurBasePointers.push_back(*CV);
      CurPointers.push_back(*CV);
      CurSizes.push_back(CGF.getTypeSize(RI->getType()));
      // Copy to the device as an argument. No need to retrieve it.
      CurMapTypes.push_back(MappableExprsHandler::OMP_MAP_PRIVATE_VAL |
                            MappableExprsHandler::OMP_MAP_FIRST_REF);
    } else {
      // If we have any information in the map clause, we use it, otherwise we
      // just do a default mapping.
      MEHandler.generateInfoForCapture(CI, *CV, CurBasePointers, CurPointers,
                                       CurSizes, CurMapTypes);
      if (CurBasePointers.empty())
        MEHandler.generateDefaultMapInfo(*CI, **RI, *CV, CurBasePointers,
                                         CurPointers, CurSizes, CurMapTypes);
    }
    // We expect to have at least an element of information for this capture.
    assert(!CurBasePointers.empty() && "Non-existing map pointer for capture!");
    assert(CurBasePointers.size() == CurPointers.size() &&
           CurBasePointers.size() == CurSizes.size() &&
           CurBasePointers.size() == CurMapTypes.size() &&
           "Inconsistent map information sizes!");

    // The kernel args are always the first elements of the base pointers
    // associated with a capture.
    KernelArgs.push_back(*CurBasePointers.front());
    // We need to append the results of this capture to what we already have.
    BasePointers.append(CurBasePointers.begin(), CurBasePointers.end());
    Pointers.append(CurPointers.begin(), CurPointers.end());
    Sizes.append(CurSizes.begin(), CurSizes.end());
    MapTypes.append(CurMapTypes.begin(), CurMapTypes.end());
  }

  // Keep track on whether the host function has to be executed.
  auto OffloadErrorQType =
      Ctx.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/true);
  auto OffloadError = CGF.MakeAddrLValue(
      CGF.CreateMemTemp(OffloadErrorQType, ".run_host_version"),
      OffloadErrorQType);
  CGF.EmitStoreOfScalar(llvm::Constant::getNullValue(CGM.Int32Ty),
                        OffloadError);

  // Fill up the pointer arrays and transfer execution to the device.
  auto &&ThenGen = [&BasePointers, &Pointers, &Sizes, &MapTypes, Device,
                    OutlinedFnID, OffloadError,
                    &D](CodeGenFunction &CGF, PrePostActionTy &) {
    auto &RT = CGF.CGM.getOpenMPRuntime();
    // Emit the offloading arrays.
    TargetDataInfo Info;
    emitOffloadingArrays(CGF, BasePointers, Pointers, Sizes, MapTypes, Info);
    emitOffloadingArraysArgument(CGF, Info.BasePointersArray,
                                 Info.PointersArray, Info.SizesArray,
                                 Info.MapTypesArray, Info);

    // On top of the arrays that were filled up, the target offloading call
    // takes as arguments the device id as well as the host pointer. The host
    // pointer is used by the runtime library to identify the current target
    // region, so it only has to be unique and not necessarily point to
    // anything. It could be the pointer to the outlined function that
    // implements the target region, but we aren't using that so that the
    // compiler doesn't need to keep that, and could therefore inline the host
    // function if proven worthwhile during optimization.

    // From this point on, we need to have an ID of the target region defined.
    assert(OutlinedFnID && "Invalid outlined function ID!");

    // Emit device ID if any.
    llvm::Value *DeviceID;
    if (Device)
      DeviceID = CGF.Builder.CreateIntCast(CGF.EmitScalarExpr(Device),
                                           CGF.Int32Ty, /*isSigned=*/true);
    else
      DeviceID = CGF.Builder.getInt32(OMP_DEVICEID_UNDEF);

    // Emit the number of elements in the offloading arrays.
    llvm::Value *PointerNum = CGF.Builder.getInt32(BasePointers.size());

    // Return value of the runtime offloading call.
    llvm::Value *Return;

    auto *NumTeams = emitNumTeamsForTargetDirective(RT, CGF, D);
    auto *NumThreads = emitNumThreadsForTargetDirective(RT, CGF, D);

    // The target region is an outlined function launched by the runtime
    // via calls __tgt_target() or __tgt_target_teams().
    //
    // __tgt_target() launches a target region with one team and one thread,
    // executing a serial region.  This master thread may in turn launch
    // more threads within its team upon encountering a parallel region,
    // however, no additional teams can be launched on the device.
    //
    // __tgt_target_teams() launches a target region with one or more teams,
    // each with one or more threads.  This call is required for target
    // constructs such as:
    //  'target teams'
    //  'target' / 'teams'
    //  'target teams distribute parallel for'
    //  'target parallel'
    // and so on.
    //
    // Note that on the host and CPU targets, the runtime implementation of
    // these calls simply call the outlined function without forking threads.
    // The outlined functions themselves have runtime calls to
    // __kmpc_fork_teams() and __kmpc_fork() for this purpose, codegen'd by
    // the compiler in emitTeamsCall() and emitParallelCall().
    //
    // In contrast, on the NVPTX target, the implementation of
    // __tgt_target_teams() launches a GPU kernel with the requested number
    // of teams and threads so no additional calls to the runtime are required.
    if (NumTeams) {
      // If we have NumTeams defined this means that we have an enclosed teams
      // region. Therefore we also expect to have NumThreads defined. These two
      // values should be defined in the presence of a teams directive,
      // regardless of having any clauses associated. If the user is using teams
      // but no clauses, these two values will be the default that should be
      // passed to the runtime library - a 32-bit integer with the value zero.
      assert(NumThreads && "Thread limit expression should be available along "
                           "with number of teams.");
      llvm::Value *OffloadingArgs[] = {
          DeviceID,           OutlinedFnID,
          PointerNum,         Info.BasePointersArray,
          Info.PointersArray, Info.SizesArray,
          Info.MapTypesArray, NumTeams,
          NumThreads};
      Return = CGF.EmitRuntimeCall(
          RT.createRuntimeFunction(OMPRTL__tgt_target_teams), OffloadingArgs);
    } else {
      llvm::Value *OffloadingArgs[] = {
          DeviceID,           OutlinedFnID,
          PointerNum,         Info.BasePointersArray,
          Info.PointersArray, Info.SizesArray,
          Info.MapTypesArray};
      Return = CGF.EmitRuntimeCall(RT.createRuntimeFunction(OMPRTL__tgt_target),
                                   OffloadingArgs);
    }

    CGF.EmitStoreOfScalar(Return, OffloadError);
  };

  // Notify that the host version must be executed.
  auto &&ElseGen = [OffloadError](CodeGenFunction &CGF, PrePostActionTy &) {
    CGF.EmitStoreOfScalar(llvm::ConstantInt::get(CGF.Int32Ty, /*V=*/-1u),
                          OffloadError);
  };

  // If we have a target function ID it means that we need to support
  // offloading, otherwise, just execute on the host. We need to execute on host
  // regardless of the conditional in the if clause if, e.g., the user do not
  // specify target triples.
  if (OutlinedFnID) {
    if (IfCond)
      emitOMPIfClause(CGF, IfCond, ThenGen, ElseGen);
    else {
      RegionCodeGenTy ThenRCG(ThenGen);
      ThenRCG(CGF);
    }
  } else {
    RegionCodeGenTy ElseRCG(ElseGen);
    ElseRCG(CGF);
  }

  // Check the error code and execute the host version if required.
  auto OffloadFailedBlock = CGF.createBasicBlock("omp_offload.failed");
  auto OffloadContBlock = CGF.createBasicBlock("omp_offload.cont");
  auto OffloadErrorVal = CGF.EmitLoadOfScalar(OffloadError, SourceLocation());
  auto Failed = CGF.Builder.CreateIsNotNull(OffloadErrorVal);
  CGF.Builder.CreateCondBr(Failed, OffloadFailedBlock, OffloadContBlock);

  CGF.EmitBlock(OffloadFailedBlock);
  emitOutlinedFunctionCall(CGF, D.getLocStart(), OutlinedFn, KernelArgs);
  CGF.EmitBranch(OffloadContBlock);

  CGF.EmitBlock(OffloadContBlock, /*IsFinished=*/true);
}

void CGOpenMPRuntime::scanForTargetRegionsFunctions(const Stmt *S,
                                                    StringRef ParentName) {
  if (!S)
    return;

  // Codegen OMP target directives that offload compute to the device.
  bool requiresDeviceCodegen =
      isa<OMPExecutableDirective>(S) &&
      isOpenMPTargetExecutionDirective(
          cast<OMPExecutableDirective>(S)->getDirectiveKind());

  if (requiresDeviceCodegen) {
    auto &E = *cast<OMPExecutableDirective>(S);
    unsigned DeviceID;
    unsigned FileID;
    unsigned Line;
    getTargetEntryUniqueInfo(CGM.getContext(), E.getLocStart(), DeviceID,
                             FileID, Line);

    // Is this a target region that should not be emitted as an entry point? If
    // so just signal we are done with this target region.
    if (!OffloadEntriesInfoManager.hasTargetRegionEntryInfo(DeviceID, FileID,
                                                            ParentName, Line))
      return;

    switch (S->getStmtClass()) {
    case Stmt::OMPTargetDirectiveClass:
      CodeGenFunction::EmitOMPTargetDeviceFunction(
          CGM, ParentName, cast<OMPTargetDirective>(*S));
      break;
    case Stmt::OMPTargetParallelDirectiveClass:
      CodeGenFunction::EmitOMPTargetParallelDeviceFunction(
          CGM, ParentName, cast<OMPTargetParallelDirective>(*S));
      break;
    case Stmt::OMPTargetTeamsDirectiveClass:
      CodeGenFunction::EmitOMPTargetTeamsDeviceFunction(
          CGM, ParentName, cast<OMPTargetTeamsDirective>(*S));
      break;
    default:
      llvm_unreachable("Unknown target directive for OpenMP device codegen.");
    }
    return;
  }

  if (const OMPExecutableDirective *E = dyn_cast<OMPExecutableDirective>(S)) {
    if (!E->hasAssociatedStmt())
      return;

    scanForTargetRegionsFunctions(
        cast<CapturedStmt>(E->getAssociatedStmt())->getCapturedStmt(),
        ParentName);
    return;
  }

  // If this is a lambda function, look into its body.
  if (auto *L = dyn_cast<LambdaExpr>(S))
    S = L->getBody();

  // Keep looking for target regions recursively.
  for (auto *II : S->children())
    scanForTargetRegionsFunctions(II, ParentName);
}

bool CGOpenMPRuntime::emitTargetFunctions(GlobalDecl GD) {
  auto &FD = *cast<FunctionDecl>(GD.getDecl());

  // If emitting code for the host, we do not process FD here. Instead we do
  // the normal code generation.
  if (!CGM.getLangOpts().OpenMPIsDevice)
    return false;

  // Try to detect target regions in the function.
  scanForTargetRegionsFunctions(FD.getBody(), CGM.getMangledName(GD));

  // We should not emit any function other that the ones created during the
  // scanning. Therefore, we signal that this function is completely dealt
  // with.
  return true;
}

bool CGOpenMPRuntime::emitTargetGlobalVariable(GlobalDecl GD) {
  if (!CGM.getLangOpts().OpenMPIsDevice)
    return false;

  // Check if there are Ctors/Dtors in this declaration and look for target
  // regions in it. We use the complete variant to produce the kernel name
  // mangling.
  QualType RDTy = cast<VarDecl>(GD.getDecl())->getType();
  if (auto *RD = RDTy->getBaseElementTypeUnsafe()->getAsCXXRecordDecl()) {
    for (auto *Ctor : RD->ctors()) {
      StringRef ParentName =
          CGM.getMangledName(GlobalDecl(Ctor, Ctor_Complete));
      scanForTargetRegionsFunctions(Ctor->getBody(), ParentName);
    }
    auto *Dtor = RD->getDestructor();
    if (Dtor) {
      StringRef ParentName =
          CGM.getMangledName(GlobalDecl(Dtor, Dtor_Complete));
      scanForTargetRegionsFunctions(Dtor->getBody(), ParentName);
    }
  }

  // If we are in target mode, we do not emit any global (declare target is not
  // implemented yet). Therefore we signal that GD was processed in this case.
  return true;
}

bool CGOpenMPRuntime::emitTargetGlobal(GlobalDecl GD) {
  auto *VD = GD.getDecl();
  if (isa<FunctionDecl>(VD))
    return emitTargetFunctions(GD);

  return emitTargetGlobalVariable(GD);
}

llvm::Function *CGOpenMPRuntime::emitRegistrationFunction() {
  // If we have offloading in the current module, we need to emit the entries
  // now and register the offloading descriptor.
  createOffloadEntriesAndInfoMetadata();

  // Create and register the offloading binary descriptors. This is the main
  // entity that captures all the information about offloading in the current
  // compilation unit.
  return createOffloadingBinaryDescriptorRegistration();
}

void CGOpenMPRuntime::emitTeamsCall(CodeGenFunction &CGF,
                                    const OMPExecutableDirective &D,
                                    SourceLocation Loc,
                                    llvm::Value *OutlinedFn,
                                    ArrayRef<llvm::Value *> CapturedVars) {
  if (!CGF.HaveInsertPoint())
    return;

  auto *RTLoc = emitUpdateLocation(CGF, Loc);
  CodeGenFunction::RunCleanupsScope Scope(CGF);

  // Build call __kmpc_fork_teams(loc, n, microtask, var1, .., varn);
  llvm::Value *Args[] = {
      RTLoc,
      CGF.Builder.getInt32(CapturedVars.size()), // Number of captured vars
      CGF.Builder.CreateBitCast(OutlinedFn, getKmpc_MicroPointerTy())};
  llvm::SmallVector<llvm::Value *, 16> RealArgs;
  RealArgs.append(std::begin(Args), std::end(Args));
  RealArgs.append(CapturedVars.begin(), CapturedVars.end());

  auto RTLFn = createRuntimeFunction(OMPRTL__kmpc_fork_teams);
  CGF.EmitRuntimeCall(RTLFn, RealArgs);
}

void CGOpenMPRuntime::emitNumTeamsClause(CodeGenFunction &CGF,
                                         const Expr *NumTeams,
                                         const Expr *ThreadLimit,
                                         SourceLocation Loc) {
  if (!CGF.HaveInsertPoint())
    return;

  auto *RTLoc = emitUpdateLocation(CGF, Loc);

  llvm::Value *NumTeamsVal =
      (NumTeams)
          ? CGF.Builder.CreateIntCast(CGF.EmitScalarExpr(NumTeams),
                                      CGF.CGM.Int32Ty, /* isSigned = */ true)
          : CGF.Builder.getInt32(0);

  llvm::Value *ThreadLimitVal =
      (ThreadLimit)
          ? CGF.Builder.CreateIntCast(CGF.EmitScalarExpr(ThreadLimit),
                                      CGF.CGM.Int32Ty, /* isSigned = */ true)
          : CGF.Builder.getInt32(0);

  // Build call __kmpc_push_num_teamss(&loc, global_tid, num_teams, thread_limit)
  llvm::Value *PushNumTeamsArgs[] = {RTLoc, getThreadID(CGF, Loc), NumTeamsVal,
                                     ThreadLimitVal};
  CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_push_num_teams),
                      PushNumTeamsArgs);
}

void CGOpenMPRuntime::emitTargetDataCalls(
    CodeGenFunction &CGF, const OMPExecutableDirective &D, const Expr *IfCond,
    const Expr *Device, const RegionCodeGenTy &CodeGen, TargetDataInfo &Info) {
  if (!CGF.HaveInsertPoint())
    return;

  // Action used to replace the default codegen action and turn privatization
  // off.
  PrePostActionTy NoPrivAction;

  // Generate the code for the opening of the data environment. Capture all the
  // arguments of the runtime call by reference because they are used in the
  // closing of the region.
  auto &&BeginThenGen = [&D, Device, &Info, &CodeGen](CodeGenFunction &CGF,
                                                      PrePostActionTy &) {
    // Fill up the arrays with all the mapped variables.
    MappableExprsHandler::MapBaseValuesArrayTy BasePointers;
    MappableExprsHandler::MapValuesArrayTy Pointers;
    MappableExprsHandler::MapValuesArrayTy Sizes;
    MappableExprsHandler::MapFlagsArrayTy MapTypes;

    // Get map clause information.
    MappableExprsHandler MCHandler(D, CGF);
    MCHandler.generateAllInfo(BasePointers, Pointers, Sizes, MapTypes);

    // Fill up the arrays and create the arguments.
    emitOffloadingArrays(CGF, BasePointers, Pointers, Sizes, MapTypes, Info);

    llvm::Value *BasePointersArrayArg = nullptr;
    llvm::Value *PointersArrayArg = nullptr;
    llvm::Value *SizesArrayArg = nullptr;
    llvm::Value *MapTypesArrayArg = nullptr;
    emitOffloadingArraysArgument(CGF, BasePointersArrayArg, PointersArrayArg,
                                 SizesArrayArg, MapTypesArrayArg, Info);

    // Emit device ID if any.
    llvm::Value *DeviceID = nullptr;
    if (Device)
      DeviceID = CGF.Builder.CreateIntCast(CGF.EmitScalarExpr(Device),
                                           CGF.Int32Ty, /*isSigned=*/true);
    else
      DeviceID = CGF.Builder.getInt32(OMP_DEVICEID_UNDEF);

    // Emit the number of elements in the offloading arrays.
    auto *PointerNum = CGF.Builder.getInt32(Info.NumberOfPtrs);

    llvm::Value *OffloadingArgs[] = {
        DeviceID,         PointerNum,    BasePointersArrayArg,
        PointersArrayArg, SizesArrayArg, MapTypesArrayArg};
    auto &RT = CGF.CGM.getOpenMPRuntime();
    CGF.EmitRuntimeCall(RT.createRuntimeFunction(OMPRTL__tgt_target_data_begin),
                        OffloadingArgs);

    // If device pointer privatization is required, emit the body of the region
    // here. It will have to be duplicated: with and without privatization.
    if (!Info.CaptureDeviceAddrMap.empty())
      CodeGen(CGF);
  };

  // Generate code for the closing of the data region.
  auto &&EndThenGen = [Device, &Info](CodeGenFunction &CGF, PrePostActionTy &) {
    assert(Info.isValid() && "Invalid data environment closing arguments.");

    llvm::Value *BasePointersArrayArg = nullptr;
    llvm::Value *PointersArrayArg = nullptr;
    llvm::Value *SizesArrayArg = nullptr;
    llvm::Value *MapTypesArrayArg = nullptr;
    emitOffloadingArraysArgument(CGF, BasePointersArrayArg, PointersArrayArg,
                                 SizesArrayArg, MapTypesArrayArg, Info);

    // Emit device ID if any.
    llvm::Value *DeviceID = nullptr;
    if (Device)
      DeviceID = CGF.Builder.CreateIntCast(CGF.EmitScalarExpr(Device),
                                           CGF.Int32Ty, /*isSigned=*/true);
    else
      DeviceID = CGF.Builder.getInt32(OMP_DEVICEID_UNDEF);

    // Emit the number of elements in the offloading arrays.
    auto *PointerNum = CGF.Builder.getInt32(Info.NumberOfPtrs);

    llvm::Value *OffloadingArgs[] = {
        DeviceID,         PointerNum,    BasePointersArrayArg,
        PointersArrayArg, SizesArrayArg, MapTypesArrayArg};
    auto &RT = CGF.CGM.getOpenMPRuntime();
    CGF.EmitRuntimeCall(RT.createRuntimeFunction(OMPRTL__tgt_target_data_end),
                        OffloadingArgs);
  };

  // If we need device pointer privatization, we need to emit the body of the
  // region with no privatization in the 'else' branch of the conditional.
  // Otherwise, we don't have to do anything.
  auto &&BeginElseGen = [&Info, &CodeGen, &NoPrivAction](CodeGenFunction &CGF,
                                                         PrePostActionTy &) {
    if (!Info.CaptureDeviceAddrMap.empty()) {
      CodeGen.setAction(NoPrivAction);
      CodeGen(CGF);
    }
  };

  // We don't have to do anything to close the region if the if clause evaluates
  // to false.
  auto &&EndElseGen = [](CodeGenFunction &CGF, PrePostActionTy &) {};

  if (IfCond) {
    emitOMPIfClause(CGF, IfCond, BeginThenGen, BeginElseGen);
  } else {
    RegionCodeGenTy RCG(BeginThenGen);
    RCG(CGF);
  }

  // If we don't require privatization of device pointers, we emit the body in
  // between the runtime calls. This avoids duplicating the body code.
  if (Info.CaptureDeviceAddrMap.empty()) {
    CodeGen.setAction(NoPrivAction);
    CodeGen(CGF);
  }

  if (IfCond) {
    emitOMPIfClause(CGF, IfCond, EndThenGen, EndElseGen);
  } else {
    RegionCodeGenTy RCG(EndThenGen);
    RCG(CGF);
  }
}

void CGOpenMPRuntime::emitTargetDataStandAloneCall(
    CodeGenFunction &CGF, const OMPExecutableDirective &D, const Expr *IfCond,
    const Expr *Device) {
  if (!CGF.HaveInsertPoint())
    return;

  assert((isa<OMPTargetEnterDataDirective>(D) ||
          isa<OMPTargetExitDataDirective>(D) ||
          isa<OMPTargetUpdateDirective>(D)) &&
         "Expecting either target enter, exit data, or update directives.");

  // Generate the code for the opening of the data environment.
  auto &&ThenGen = [&D, Device](CodeGenFunction &CGF, PrePostActionTy &) {
    // Fill up the arrays with all the mapped variables.
    MappableExprsHandler::MapBaseValuesArrayTy BasePointers;
    MappableExprsHandler::MapValuesArrayTy Pointers;
    MappableExprsHandler::MapValuesArrayTy Sizes;
    MappableExprsHandler::MapFlagsArrayTy MapTypes;

    // Get map clause information.
    MappableExprsHandler MEHandler(D, CGF);
    MEHandler.generateAllInfo(BasePointers, Pointers, Sizes, MapTypes);

    // Fill up the arrays and create the arguments.
    TargetDataInfo Info;
    emitOffloadingArrays(CGF, BasePointers, Pointers, Sizes, MapTypes, Info);
    emitOffloadingArraysArgument(CGF, Info.BasePointersArray,
                                 Info.PointersArray, Info.SizesArray,
                                 Info.MapTypesArray, Info);

    // Emit device ID if any.
    llvm::Value *DeviceID = nullptr;
    if (Device)
      DeviceID = CGF.Builder.CreateIntCast(CGF.EmitScalarExpr(Device),
                                           CGF.Int32Ty, /*isSigned=*/true);
    else
      DeviceID = CGF.Builder.getInt32(OMP_DEVICEID_UNDEF);

    // Emit the number of elements in the offloading arrays.
    auto *PointerNum = CGF.Builder.getInt32(BasePointers.size());

    llvm::Value *OffloadingArgs[] = {
        DeviceID,           PointerNum,      Info.BasePointersArray,
        Info.PointersArray, Info.SizesArray, Info.MapTypesArray};

    auto &RT = CGF.CGM.getOpenMPRuntime();
    // Select the right runtime function call for each expected standalone
    // directive.
    OpenMPRTLFunction RTLFn;
    switch (D.getDirectiveKind()) {
    default:
      llvm_unreachable("Unexpected standalone target data directive.");
      break;
    case OMPD_target_enter_data:
      RTLFn = OMPRTL__tgt_target_data_begin;
      break;
    case OMPD_target_exit_data:
      RTLFn = OMPRTL__tgt_target_data_end;
      break;
    case OMPD_target_update:
      RTLFn = OMPRTL__tgt_target_data_update;
      break;
    }
    CGF.EmitRuntimeCall(RT.createRuntimeFunction(RTLFn), OffloadingArgs);
  };

  // In the event we get an if clause, we don't have to take any action on the
  // else side.
  auto &&ElseGen = [](CodeGenFunction &CGF, PrePostActionTy &) {};

  if (IfCond) {
    emitOMPIfClause(CGF, IfCond, ThenGen, ElseGen);
  } else {
    RegionCodeGenTy ThenGenRCG(ThenGen);
    ThenGenRCG(CGF);
  }
}

namespace {
  /// Kind of parameter in a function with 'declare simd' directive.
  enum ParamKindTy { LinearWithVarStride, Linear, Uniform, Vector };
  /// Attribute set of the parameter.
  struct ParamAttrTy {
    ParamKindTy Kind = Vector;
    llvm::APSInt StrideOrArg;
    llvm::APSInt Alignment;
  };
} // namespace

static unsigned evaluateCDTSize(const FunctionDecl *FD,
                                ArrayRef<ParamAttrTy> ParamAttrs) {
  // Every vector variant of a SIMD-enabled function has a vector length (VLEN).
  // If OpenMP clause "simdlen" is used, the VLEN is the value of the argument
  // of that clause. The VLEN value must be power of 2.
  // In other case the notion of the function`s "characteristic data type" (CDT)
  // is used to compute the vector length.
  // CDT is defined in the following order:
  //   a) For non-void function, the CDT is the return type.
  //   b) If the function has any non-uniform, non-linear parameters, then the
  //   CDT is the type of the first such parameter.
  //   c) If the CDT determined by a) or b) above is struct, union, or class
  //   type which is pass-by-value (except for the type that maps to the
  //   built-in complex data type), the characteristic data type is int.
  //   d) If none of the above three cases is applicable, the CDT is int.
  // The VLEN is then determined based on the CDT and the size of vector
  // register of that ISA for which current vector version is generated. The
  // VLEN is computed using the formula below:
  //   VLEN  = sizeof(vector_register) / sizeof(CDT),
  // where vector register size specified in section 3.2.1 Registers and the
  // Stack Frame of original AMD64 ABI document.
  QualType RetType = FD->getReturnType();
  if (RetType.isNull())
    return 0;
  ASTContext &C = FD->getASTContext();
  QualType CDT;
  if (!RetType.isNull() && !RetType->isVoidType())
    CDT = RetType;
  else {
    unsigned Offset = 0;
    if (auto *MD = dyn_cast<CXXMethodDecl>(FD)) {
      if (ParamAttrs[Offset].Kind == Vector)
        CDT = C.getPointerType(C.getRecordType(MD->getParent()));
      ++Offset;
    }
    if (CDT.isNull()) {
      for (unsigned I = 0, E = FD->getNumParams(); I < E; ++I) {
        if (ParamAttrs[I + Offset].Kind == Vector) {
          CDT = FD->getParamDecl(I)->getType();
          break;
        }
      }
    }
  }
  if (CDT.isNull())
    CDT = C.IntTy;
  CDT = CDT->getCanonicalTypeUnqualified();
  if (CDT->isRecordType() || CDT->isUnionType())
    CDT = C.IntTy;
  return C.getTypeSize(CDT);
}

static void
emitX86DeclareSimdFunction(const FunctionDecl *FD, llvm::Function *Fn,
                           const llvm::APSInt &VLENVal,
                           ArrayRef<ParamAttrTy> ParamAttrs,
                           OMPDeclareSimdDeclAttr::BranchStateTy State) {
  struct ISADataTy {
    char ISA;
    unsigned VecRegSize;
  };
  ISADataTy ISAData[] = {
      {
          'b', 128
      }, // SSE
      {
          'c', 256
      }, // AVX
      {
          'd', 256
      }, // AVX2
      {
          'e', 512
      }, // AVX512
  };
  llvm::SmallVector<char, 2> Masked;
  switch (State) {
  case OMPDeclareSimdDeclAttr::BS_Undefined:
    Masked.push_back('N');
    Masked.push_back('M');
    break;
  case OMPDeclareSimdDeclAttr::BS_Notinbranch:
    Masked.push_back('N');
    break;
  case OMPDeclareSimdDeclAttr::BS_Inbranch:
    Masked.push_back('M');
    break;
  }
  for (auto Mask : Masked) {
    for (auto &Data : ISAData) {
      SmallString<256> Buffer;
      llvm::raw_svector_ostream Out(Buffer);
      Out << "_ZGV" << Data.ISA << Mask;
      if (!VLENVal) {
        Out << llvm::APSInt::getUnsigned(Data.VecRegSize /
                                         evaluateCDTSize(FD, ParamAttrs));
      } else
        Out << VLENVal;
      for (auto &ParamAttr : ParamAttrs) {
        switch (ParamAttr.Kind){
        case LinearWithVarStride:
          Out << 's' << ParamAttr.StrideOrArg;
          break;
        case Linear:
          Out << 'l';
          if (!!ParamAttr.StrideOrArg)
            Out << ParamAttr.StrideOrArg;
          break;
        case Uniform:
          Out << 'u';
          break;
        case Vector:
          Out << 'v';
          break;
        }
        if (!!ParamAttr.Alignment)
          Out << 'a' << ParamAttr.Alignment;
      }
      Out << '_' << Fn->getName();
      Fn->addFnAttr(Out.str());
    }
  }
}

void CGOpenMPRuntime::emitDeclareSimdFunction(const FunctionDecl *FD,
                                              llvm::Function *Fn) {
  ASTContext &C = CGM.getContext();
  FD = FD->getCanonicalDecl();
  // Map params to their positions in function decl.
  llvm::DenseMap<const Decl *, unsigned> ParamPositions;
  if (isa<CXXMethodDecl>(FD))
    ParamPositions.insert({FD, 0});
  unsigned ParamPos = ParamPositions.size();
  for (auto *P : FD->parameters()) {
    ParamPositions.insert({P->getCanonicalDecl(), ParamPos});
    ++ParamPos;
  }
  for (auto *Attr : FD->specific_attrs<OMPDeclareSimdDeclAttr>()) {
    llvm::SmallVector<ParamAttrTy, 8> ParamAttrs(ParamPositions.size());
    // Mark uniform parameters.
    for (auto *E : Attr->uniforms()) {
      E = E->IgnoreParenImpCasts();
      unsigned Pos;
      if (isa<CXXThisExpr>(E))
        Pos = ParamPositions[FD];
      else {
        auto *PVD = cast<ParmVarDecl>(cast<DeclRefExpr>(E)->getDecl())
                        ->getCanonicalDecl();
        Pos = ParamPositions[PVD];
      }
      ParamAttrs[Pos].Kind = Uniform;
    }
    // Get alignment info.
    auto NI = Attr->alignments_begin();
    for (auto *E : Attr->aligneds()) {
      E = E->IgnoreParenImpCasts();
      unsigned Pos;
      QualType ParmTy;
      if (isa<CXXThisExpr>(E)) {
        Pos = ParamPositions[FD];
        ParmTy = E->getType();
      } else {
        auto *PVD = cast<ParmVarDecl>(cast<DeclRefExpr>(E)->getDecl())
                        ->getCanonicalDecl();
        Pos = ParamPositions[PVD];
        ParmTy = PVD->getType();
      }
      ParamAttrs[Pos].Alignment =
          (*NI) ? (*NI)->EvaluateKnownConstInt(C)
                : llvm::APSInt::getUnsigned(
                      C.toCharUnitsFromBits(C.getOpenMPDefaultSimdAlign(ParmTy))
                          .getQuantity());
      ++NI;
    }
    // Mark linear parameters.
    auto SI = Attr->steps_begin();
    auto MI = Attr->modifiers_begin();
    for (auto *E : Attr->linears()) {
      E = E->IgnoreParenImpCasts();
      unsigned Pos;
      if (isa<CXXThisExpr>(E))
        Pos = ParamPositions[FD];
      else {
        auto *PVD = cast<ParmVarDecl>(cast<DeclRefExpr>(E)->getDecl())
                        ->getCanonicalDecl();
        Pos = ParamPositions[PVD];
      }
      auto &ParamAttr = ParamAttrs[Pos];
      ParamAttr.Kind = Linear;
      if (*SI) {
        if (!(*SI)->EvaluateAsInt(ParamAttr.StrideOrArg, C,
                                  Expr::SE_AllowSideEffects)) {
          if (auto *DRE = cast<DeclRefExpr>((*SI)->IgnoreParenImpCasts())) {
            if (auto *StridePVD = cast<ParmVarDecl>(DRE->getDecl())) {
              ParamAttr.Kind = LinearWithVarStride;
              ParamAttr.StrideOrArg = llvm::APSInt::getUnsigned(
                  ParamPositions[StridePVD->getCanonicalDecl()]);
            }
          }
        }
      }
      ++SI;
      ++MI;
    }
    llvm::APSInt VLENVal;
    if (const Expr *VLEN = Attr->getSimdlen())
      VLENVal = VLEN->EvaluateKnownConstInt(C);
    OMPDeclareSimdDeclAttr::BranchStateTy State = Attr->getBranchState();
    if (CGM.getTriple().getArch() == llvm::Triple::x86 ||
        CGM.getTriple().getArch() == llvm::Triple::x86_64)
      emitX86DeclareSimdFunction(FD, Fn, VLENVal, ParamAttrs, State);
  }
}

namespace {
/// Cleanup action for doacross support.
class DoacrossCleanupTy final : public EHScopeStack::Cleanup {
public:
  static const int DoacrossFinArgs = 2;

private:
  llvm::Value *RTLFn;
  llvm::Value *Args[DoacrossFinArgs];

public:
  DoacrossCleanupTy(llvm::Value *RTLFn, ArrayRef<llvm::Value *> CallArgs)
      : RTLFn(RTLFn) {
    assert(CallArgs.size() == DoacrossFinArgs);
    std::copy(CallArgs.begin(), CallArgs.end(), std::begin(Args));
  }
  void Emit(CodeGenFunction &CGF, Flags /*flags*/) override {
    if (!CGF.HaveInsertPoint())
      return;
    CGF.EmitRuntimeCall(RTLFn, Args);
  }
};
} // namespace

void CGOpenMPRuntime::emitDoacrossInit(CodeGenFunction &CGF,
                                       const OMPLoopDirective &D) {
  if (!CGF.HaveInsertPoint())
    return;

  ASTContext &C = CGM.getContext();
  QualType Int64Ty = C.getIntTypeForBitwidth(/*DestWidth=*/64, /*Signed=*/true);
  RecordDecl *RD;
  if (KmpDimTy.isNull()) {
    // Build struct kmp_dim {  // loop bounds info casted to kmp_int64
    //  kmp_int64 lo; // lower
    //  kmp_int64 up; // upper
    //  kmp_int64 st; // stride
    // };
    RD = C.buildImplicitRecord("kmp_dim");
    RD->startDefinition();
    addFieldToRecordDecl(C, RD, Int64Ty);
    addFieldToRecordDecl(C, RD, Int64Ty);
    addFieldToRecordDecl(C, RD, Int64Ty);
    RD->completeDefinition();
    KmpDimTy = C.getRecordType(RD);
  } else
    RD = cast<RecordDecl>(KmpDimTy->getAsTagDecl());

  Address DimsAddr = CGF.CreateMemTemp(KmpDimTy, "dims");
  CGF.EmitNullInitialization(DimsAddr, KmpDimTy);
  enum { LowerFD = 0, UpperFD, StrideFD };
  // Fill dims with data.
  LValue DimsLVal = CGF.MakeAddrLValue(DimsAddr, KmpDimTy);
  // dims.upper = num_iterations;
  LValue UpperLVal =
      CGF.EmitLValueForField(DimsLVal, *std::next(RD->field_begin(), UpperFD));
  llvm::Value *NumIterVal = CGF.EmitScalarConversion(
      CGF.EmitScalarExpr(D.getNumIterations()), D.getNumIterations()->getType(),
      Int64Ty, D.getNumIterations()->getExprLoc());
  CGF.EmitStoreOfScalar(NumIterVal, UpperLVal);
  // dims.stride = 1;
  LValue StrideLVal =
      CGF.EmitLValueForField(DimsLVal, *std::next(RD->field_begin(), StrideFD));
  CGF.EmitStoreOfScalar(llvm::ConstantInt::getSigned(CGM.Int64Ty, /*V=*/1),
                        StrideLVal);

  // Build call void __kmpc_doacross_init(ident_t *loc, kmp_int32 gtid,
  // kmp_int32 num_dims, struct kmp_dim * dims);
  llvm::Value *Args[] = {emitUpdateLocation(CGF, D.getLocStart()),
                         getThreadID(CGF, D.getLocStart()),
                         llvm::ConstantInt::getSigned(CGM.Int32Ty, 1),
                         CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
                             DimsAddr.getPointer(), CGM.VoidPtrTy)};

  llvm::Value *RTLFn = createRuntimeFunction(OMPRTL__kmpc_doacross_init);
  CGF.EmitRuntimeCall(RTLFn, Args);
  llvm::Value *FiniArgs[DoacrossCleanupTy::DoacrossFinArgs] = {
      emitUpdateLocation(CGF, D.getLocEnd()), getThreadID(CGF, D.getLocEnd())};
  llvm::Value *FiniRTLFn = createRuntimeFunction(OMPRTL__kmpc_doacross_fini);
  CGF.EHStack.pushCleanup<DoacrossCleanupTy>(NormalAndEHCleanup, FiniRTLFn,
                                             llvm::makeArrayRef(FiniArgs));
}

void CGOpenMPRuntime::emitDoacrossOrdered(CodeGenFunction &CGF,
                                          const OMPDependClause *C) {
  QualType Int64Ty =
      CGM.getContext().getIntTypeForBitwidth(/*DestWidth=*/64, /*Signed=*/1);
  const Expr *CounterVal = C->getCounterValue();
  assert(CounterVal);
  llvm::Value *CntVal = CGF.EmitScalarConversion(CGF.EmitScalarExpr(CounterVal),
                                                 CounterVal->getType(), Int64Ty,
                                                 CounterVal->getExprLoc());
  Address CntAddr = CGF.CreateMemTemp(Int64Ty, ".cnt.addr");
  CGF.EmitStoreOfScalar(CntVal, CntAddr, /*Volatile=*/false, Int64Ty);
  llvm::Value *Args[] = {emitUpdateLocation(CGF, C->getLocStart()),
                         getThreadID(CGF, C->getLocStart()),
                         CntAddr.getPointer()};
  llvm::Value *RTLFn;
  if (C->getDependencyKind() == OMPC_DEPEND_source)
    RTLFn = createRuntimeFunction(OMPRTL__kmpc_doacross_post);
  else {
    assert(C->getDependencyKind() == OMPC_DEPEND_sink);
    RTLFn = createRuntimeFunction(OMPRTL__kmpc_doacross_wait);
  }
  CGF.EmitRuntimeCall(RTLFn, Args);
}

void CGOpenMPRuntime::emitCall(CodeGenFunction &CGF, llvm::Value *Callee,
                               ArrayRef<llvm::Value *> Args,
                               SourceLocation Loc) const {
  auto DL = ApplyDebugLocation::CreateDefaultArtificial(CGF, Loc);

  if (auto *Fn = dyn_cast<llvm::Function>(Callee)) {
    if (Fn->doesNotThrow()) {
      CGF.EmitNounwindRuntimeCall(Fn, Args);
      return;
    }
  }
  CGF.EmitRuntimeCall(Callee, Args);
}

void CGOpenMPRuntime::emitOutlinedFunctionCall(
    CodeGenFunction &CGF, SourceLocation Loc, llvm::Value *OutlinedFn,
    ArrayRef<llvm::Value *> Args) const {
  assert(Loc.isValid() && "Outlined function call location must be valid.");
  emitCall(CGF, OutlinedFn, Args, Loc);
}

Address CGOpenMPRuntime::getParameterAddress(CodeGenFunction &CGF,
                                             const VarDecl *NativeParam,
                                             const VarDecl *TargetParam) const {
  return CGF.GetAddrOfLocalVar(NativeParam);
}
