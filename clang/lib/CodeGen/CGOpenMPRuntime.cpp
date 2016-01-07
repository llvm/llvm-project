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
#include "clang/AST/Decl.h"
#include "clang/AST/StmtOpenMP.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Bitcode/ReaderWriter.h"
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

  CGOpenMPRegionKind getRegionKind() const { return RegionKind; }

  OpenMPDirectiveKind getDirectiveKind() const { return Kind; }

  bool hasCancel() const { return HasCancel; }

  static bool classof(const CGCapturedStmtInfo *Info) {
    return Info->getKind() == CR_OpenMP;
  }

protected:
  CGOpenMPRegionKind RegionKind;
  const RegionCodeGenTy &CodeGen;
  OpenMPDirectiveKind Kind;
  bool HasCancel;
};

/// \brief API for captured statement code generation in OpenMP constructs.
class CGOpenMPOutlinedRegionInfo : public CGOpenMPRegionInfo {
public:
  CGOpenMPOutlinedRegionInfo(const CapturedStmt &CS, const VarDecl *ThreadIDVar,
                             const RegionCodeGenTy &CodeGen,
                             OpenMPDirectiveKind Kind, bool HasCancel)
      : CGOpenMPRegionInfo(CS, ParallelOutlinedRegion, CodeGen, Kind,
                           HasCancel),
        ThreadIDVar(ThreadIDVar) {
    assert(ThreadIDVar != nullptr && "No ThreadID in OpenMP region.");
  }
  /// \brief Get a variable or parameter for storing global thread id
  /// inside OpenMP construct.
  const VarDecl *getThreadIDVariable() const override { return ThreadIDVar; }

  /// \brief Get the name of the capture helper.
  StringRef getHelperName() const override { return ".omp_outlined."; }

  static bool classof(const CGCapturedStmtInfo *Info) {
    return CGOpenMPRegionInfo::classof(Info) &&
           cast<CGOpenMPRegionInfo>(Info)->getRegionKind() ==
               ParallelOutlinedRegion;
  }

private:
  /// \brief A variable or parameter storing global thread id for OpenMP
  /// constructs.
  const VarDecl *ThreadIDVar;
};

/// \brief API for captured statement code generation in OpenMP constructs.
class CGOpenMPTaskOutlinedRegionInfo : public CGOpenMPRegionInfo {
public:
  CGOpenMPTaskOutlinedRegionInfo(const CapturedStmt &CS,
                                 const VarDecl *ThreadIDVar,
                                 const RegionCodeGenTy &CodeGen,
                                 OpenMPDirectiveKind Kind, bool HasCancel)
      : CGOpenMPRegionInfo(CS, TaskOutlinedRegion, CodeGen, Kind, HasCancel),
        ThreadIDVar(ThreadIDVar) {
    assert(ThreadIDVar != nullptr && "No ThreadID in OpenMP region.");
  }
  /// \brief Get a variable or parameter for storing global thread id
  /// inside OpenMP construct.
  const VarDecl *getThreadIDVariable() const override { return ThreadIDVar; }

  /// \brief Get an LValue for the current ThreadID variable.
  LValue getThreadIDVariableLValue(CodeGenFunction &CGF) override;

  /// \brief Get the name of the capture helper.
  StringRef getHelperName() const override { return ".omp_outlined."; }

  static bool classof(const CGCapturedStmtInfo *Info) {
    return CGOpenMPRegionInfo::classof(Info) &&
           cast<CGOpenMPRegionInfo>(Info)->getRegionKind() ==
               TaskOutlinedRegion;
  }

private:
  /// \brief A variable or parameter storing global thread id for OpenMP
  /// constructs.
  const VarDecl *ThreadIDVar;
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

  CodeGenFunction::CGCapturedStmtInfo *getOldCSI() const { return OldCSI; }

  static bool classof(const CGCapturedStmtInfo *Info) {
    return CGOpenMPRegionInfo::classof(Info) &&
           cast<CGOpenMPRegionInfo>(Info)->getRegionKind() == InlinedRegion;
  }

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
class CGOpenMPTargetRegionInfo : public CGOpenMPRegionInfo {
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

/// \brief RAII for emitting code of OpenMP constructs.
class InlinedOpenMPRegionRAII {
  CodeGenFunction &CGF;

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
  }
  ~InlinedOpenMPRegionRAII() {
    // Restore original CapturedStmtInfo only if we're done with code emission.
    auto *OldCSI =
        cast<CGOpenMPInlinedRegionInfo>(CGF.CapturedStmtInfo)->getOldCSI();
    delete CGF.CapturedStmtInfo;
    CGF.CapturedStmtInfo = OldCSI;
  }
};

} // anonymous namespace

static LValue emitLoadOfPointerLValue(CodeGenFunction &CGF, Address PtrAddr,
                                      QualType Ty) {
  AlignmentSource Source;
  CharUnits Align = CGF.getNaturalPointeeTypeAlignment(Ty, &Source);
  return CGF.MakeAddrLValue(Address(CGF.Builder.CreateLoad(PtrAddr), Align),
                            Ty->getPointeeType(), Source);
}

LValue CGOpenMPRegionInfo::getThreadIDVariableLValue(CodeGenFunction &CGF) {
  return emitLoadOfPointerLValue(CGF,
                                 CGF.GetAddrOfLocalVar(getThreadIDVariable()),
                                 getThreadIDVariable()->getType());
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
  {
    CodeGenFunction::RunCleanupsScope Scope(CGF);
    CodeGen(CGF);
  }
  CGF.EHStack.popTerminate();
}

LValue CGOpenMPTaskOutlinedRegionInfo::getThreadIDVariableLValue(
    CodeGenFunction &CGF) {
  return CGF.MakeAddrLValue(CGF.GetAddrOfLocalVar(getThreadIDVariable()),
                            getThreadIDVariable()->getType(),
                            AlignmentSource::Decl);
}

CGOpenMPRuntime::CGOpenMPRuntime(CodeGenModule &CGM)
    : CGM(CGM), DefaultOpenMPPSource(nullptr), KmpRoutineEntryPtrTy(nullptr),
      OffloadEntriesInfoManager(CGM) {
  IdentTy = llvm::StructType::create(
      "ident_t", CGM.Int32Ty /* reserved_1 */, CGM.Int32Ty /* flags */,
      CGM.Int32Ty /* reserved_2 */, CGM.Int32Ty /* reserved_3 */,
      CGM.Int8PtrTy /* psource */, nullptr);
  // Build void (*kmpc_micro)(kmp_int32 *global_tid, kmp_int32 *bound_tid,...)
  llvm::Type *MicroParams[] = {llvm::PointerType::getUnqual(CGM.Int32Ty),
                               llvm::PointerType::getUnqual(CGM.Int32Ty)};
  Kmpc_MicroTy = llvm::FunctionType::get(CGM.VoidTy, MicroParams, true);
  KmpCriticalNameTy = llvm::ArrayType::get(CGM.Int32Ty, /*NumElements*/ 8);

  loadOffloadInfoMetadata();
}

void CGOpenMPRuntime::clear() {
  InternalVars.clear();
}

// Layout information for ident_t.
static CharUnits getIdentAlign(CodeGenModule &CGM) {
  return CGM.getPointerAlign();
}
static CharUnits getIdentSize(CodeGenModule &CGM) {
  assert((4 * CGM.getPointerSize()).isMultipleOf(CGM.getPointerAlign()));
  return CharUnits::fromQuantity(16) + CGM.getPointerSize();
}
static CharUnits getOffsetOfIdentField(CGOpenMPRuntime::IdentFieldIndex Field) {
  // All the fields except the last are i32, so this works beautifully.
  return unsigned(Field) * CharUnits::fromQuantity(4);
}
static Address createIdentFieldGEP(CodeGenFunction &CGF, Address Addr,
                                   CGOpenMPRuntime::IdentFieldIndex Field,
                                   const llvm::Twine &Name = "") {
  auto Offset = getOffsetOfIdentField(Field);
  return CGF.Builder.CreateStructGEP(Addr, Field, Offset, Name);
}

llvm::Value *CGOpenMPRuntime::emitParallelOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
    OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen) {
  assert(ThreadIDVar->getType()->isPointerType() &&
         "thread id variable must be of type kmp_int32 *");
  const CapturedStmt *CS = cast<CapturedStmt>(D.getAssociatedStmt());
  CodeGenFunction CGF(CGM, true);
  bool HasCancel = false;
  if (auto *OPD = dyn_cast<OMPParallelDirective>(&D))
    HasCancel = OPD->hasCancel();
  else if (auto *OPSD = dyn_cast<OMPParallelSectionsDirective>(&D))
    HasCancel = OPSD->hasCancel();
  else if (auto *OPFD = dyn_cast<OMPParallelForDirective>(&D))
    HasCancel = OPFD->hasCancel();
  CGOpenMPOutlinedRegionInfo CGInfo(*CS, ThreadIDVar, CodeGen, InnermostKind,
                                    HasCancel);
  CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CGInfo);
  return CGF.GenerateOpenMPCapturedStmtFunction(*CS);
}

llvm::Value *CGOpenMPRuntime::emitTaskOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
    OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen) {
  assert(!ThreadIDVar->getType()->isPointerType() &&
         "thread id variable must be of type kmp_int32 for tasks");
  auto *CS = cast<CapturedStmt>(D.getAssociatedStmt());
  CodeGenFunction CGF(CGM, true);
  CGOpenMPTaskOutlinedRegionInfo CGInfo(*CS, ThreadIDVar, CodeGen,
                                        InnermostKind,
                                        cast<OMPTaskDirective>(D).hasCancel());
  CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CGInfo);
  return CGF.GenerateCapturedStmtFunction(*CS);
}

Address CGOpenMPRuntime::getOrCreateDefaultLocation(OpenMPLocationFlags Flags) {
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
    auto DefaultOpenMPLocation = new llvm::GlobalVariable(
        CGM.getModule(), IdentTy, /*isConstant*/ true,
        llvm::GlobalValue::PrivateLinkage, /*Initializer*/ nullptr);
    DefaultOpenMPLocation->setUnnamedAddr(true);
    DefaultOpenMPLocation->setAlignment(Align.getQuantity());

    llvm::Constant *Zero = llvm::ConstantInt::get(CGM.Int32Ty, 0, true);
    llvm::Constant *Values[] = {Zero,
                                llvm::ConstantInt::get(CGM.Int32Ty, Flags),
                                Zero, Zero, DefaultOpenMPPSource};
    llvm::Constant *Init = llvm::ConstantStruct::get(IdentTy, Values);
    DefaultOpenMPLocation->setInitializer(Init);
    OpenMPDefaultLocMap[Flags] = Entry = DefaultOpenMPLocation;
  }
  return Address(Entry, Align);
}

llvm::Value *CGOpenMPRuntime::emitUpdateLocation(CodeGenFunction &CGF,
                                                 SourceLocation Loc,
                                                 OpenMPLocationFlags Flags) {
  // If no debug info is generated - return global default location.
  if (CGM.getCodeGenOpts().getDebugInfo() == CodeGenOptions::NoDebugInfo ||
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
  if (auto OMPRegionInfo =
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
}

llvm::Type *CGOpenMPRuntime::getIdentTyPointerTy() {
  return llvm::PointerType::getUnqual(IdentTy);
}

llvm::Type *CGOpenMPRuntime::getKmpc_MicroPointerTy() {
  return llvm::PointerType::getUnqual(Kmpc_MicroTy);
}

llvm::Constant *
CGOpenMPRuntime::createRuntimeFunction(OpenMPRTLFunction Function) {
  llvm::Constant *RTLFn = nullptr;
  switch (Function) {
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
  }
  return RTLFn;
}

static llvm::Value *getTypeSize(CodeGenFunction &CGF, QualType Ty) {
  auto &C = CGF.getContext();
  llvm::Value *Size = nullptr;
  auto SizeInChars = C.getTypeSizeInChars(Ty);
  if (SizeInChars.isZero()) {
    // getTypeSizeInChars() returns 0 for a VLA.
    while (auto *VAT = C.getAsVariableArrayType(Ty)) {
      llvm::Value *ArraySize;
      std::tie(ArraySize, Ty) = CGF.getVLASize(VAT);
      Size = Size ? CGF.Builder.CreateNUWMul(Size, ArraySize) : ArraySize;
    }
    SizeInChars = C.getTypeSizeInChars(Ty);
    assert(!SizeInChars.isZero());
    Size = CGF.Builder.CreateNUWMul(
        Size, llvm::ConstantInt::get(CGF.SizeTy, SizeInChars.getQuantity()));
  } else
    Size = llvm::ConstantInt::get(CGF.SizeTy, SizeInChars.getQuantity());
  return Size;
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
      ImplicitParamDecl Dst(CGM.getContext(), /*DC=*/nullptr, SourceLocation(),
                            /*Id=*/nullptr, CGM.getContext().VoidPtrTy);
      Args.push_back(&Dst);

      auto &FI = CGM.getTypes().arrangeFreeFunctionDeclaration(
          CGM.getContext().VoidPtrTy, Args, FunctionType::ExtInfo(),
          /*isVariadic=*/false);
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
      ImplicitParamDecl Dst(CGM.getContext(), /*DC=*/nullptr, SourceLocation(),
                            /*Id=*/nullptr, CGM.getContext().VoidPtrTy);
      Args.push_back(&Dst);

      auto &FI = CGM.getTypes().arrangeFreeFunctionDeclaration(
          CGM.getContext().VoidTy, Args, FunctionType::ExtInfo(),
          /*isVariadic=*/false);
      auto FTy = CGM.getTypes().GetFunctionType(FI);
      auto Fn = CGM.CreateGlobalInitOrDestructFunction(
          FTy, ".__kmpc_global_dtor_.", FI, Loc);
      DtorCGF.StartFunction(GlobalDecl(), CGM.getContext().VoidTy, Fn, FI, Args,
                            SourceLocation());
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

/// \brief Emits code for OpenMP 'if' clause using specified \a CodeGen
/// function. Here is the logic:
/// if (Cond) {
///   ThenGen();
/// } else {
///   ElseGen();
/// }
static void emitOMPIfClause(CodeGenFunction &CGF, const Expr *Cond,
                            const RegionCodeGenTy &ThenGen,
                            const RegionCodeGenTy &ElseGen) {
  CodeGenFunction::LexicalScope ConditionScope(CGF, Cond->getSourceRange());

  // If the condition constant folds and can be elided, try to avoid emitting
  // the condition and the dead arm of the if/else.
  bool CondConstant;
  if (CGF.ConstantFoldsToSimpleInteger(Cond, CondConstant)) {
    CodeGenFunction::RunCleanupsScope Scope(CGF);
    if (CondConstant) {
      ThenGen(CGF);
    } else {
      ElseGen(CGF);
    }
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
  {
    CodeGenFunction::RunCleanupsScope ThenScope(CGF);
    ThenGen(CGF);
  }
  CGF.EmitBranch(ContBlock);
  // Emit the 'else' code if present.
  {
    // There is no need to emit line number for unconditional branch.
    auto NL = ApplyDebugLocation::CreateEmpty(CGF);
    CGF.EmitBlock(ElseBlock);
  }
  {
    CodeGenFunction::RunCleanupsScope ThenScope(CGF);
    ElseGen(CGF);
  }
  {
    // There is no need to emit line number for unconditional branch.
    auto NL = ApplyDebugLocation::CreateEmpty(CGF);
    CGF.EmitBranch(ContBlock);
  }
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
  auto &&ThenGen = [this, OutlinedFn, CapturedVars,
                    RTLoc](CodeGenFunction &CGF) {
    // Build call __kmpc_fork_call(loc, n, microtask, var1, .., varn);
    llvm::Value *Args[] = {
        RTLoc,
        CGF.Builder.getInt32(CapturedVars.size()), // Number of captured vars
        CGF.Builder.CreateBitCast(OutlinedFn, getKmpc_MicroPointerTy())};
    llvm::SmallVector<llvm::Value *, 16> RealArgs;
    RealArgs.append(std::begin(Args), std::end(Args));
    RealArgs.append(CapturedVars.begin(), CapturedVars.end());

    auto RTLFn = createRuntimeFunction(OMPRTL__kmpc_fork_call);
    CGF.EmitRuntimeCall(RTLFn, RealArgs);
  };
  auto &&ElseGen = [this, OutlinedFn, CapturedVars, RTLoc,
                    Loc](CodeGenFunction &CGF) {
    auto ThreadID = getThreadID(CGF, Loc);
    // Build calls:
    // __kmpc_serialized_parallel(&Loc, GTid);
    llvm::Value *Args[] = {RTLoc, ThreadID};
    CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_serialized_parallel),
                        Args);

    // OutlinedFn(&GTid, &zero, CapturedStruct);
    auto ThreadIDAddr = emitThreadIDAddress(CGF, Loc);
    Address ZeroAddr =
      CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4),
                           /*Name*/ ".zero.addr");
    CGF.InitTempAlloca(ZeroAddr, CGF.Builder.getInt32(/*C*/ 0));
    llvm::SmallVector<llvm::Value *, 16> OutlinedFnArgs;
    OutlinedFnArgs.push_back(ThreadIDAddr.getPointer());
    OutlinedFnArgs.push_back(ZeroAddr.getPointer());
    OutlinedFnArgs.append(CapturedVars.begin(), CapturedVars.end());
    CGF.EmitCallOrInvoke(OutlinedFn, OutlinedFnArgs);

    // __kmpc_end_serialized_parallel(&Loc, GTid);
    llvm::Value *EndArgs[] = {emitUpdateLocation(CGF, Loc), ThreadID};
    CGF.EmitRuntimeCall(
        createRuntimeFunction(OMPRTL__kmpc_end_serialized_parallel), EndArgs);
  };
  if (IfCond) {
    emitOMPIfClause(CGF, IfCond, ThenGen, ElseGen);
  } else {
    CodeGenFunction::RunCleanupsScope Scope(CGF);
    ThenGen(CGF);
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
  if (auto OMPRegionInfo =
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
template <size_t N> class CallEndCleanup final : public EHScopeStack::Cleanup {
  llvm::Value *Callee;
  llvm::Value *Args[N];

public:
  CallEndCleanup(llvm::Value *Callee, ArrayRef<llvm::Value *> CleanupArgs)
      : Callee(Callee) {
    assert(CleanupArgs.size() == N);
    std::copy(CleanupArgs.begin(), CleanupArgs.end(), std::begin(Args));
  }
  void Emit(CodeGenFunction &CGF, Flags /*flags*/) override {
    if (!CGF.HaveInsertPoint())
      return;
    CGF.EmitRuntimeCall(Callee, Args);
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
  CodeGenFunction::RunCleanupsScope Scope(CGF);
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc),
                         getCriticalRegionLock(CriticalName)};
  if (Hint) {
    llvm::SmallVector<llvm::Value *, 8> ArgsWithHint(std::begin(Args),
                                                     std::end(Args));
    auto *HintVal = CGF.EmitScalarExpr(Hint);
    ArgsWithHint.push_back(
        CGF.Builder.CreateIntCast(HintVal, CGM.IntPtrTy, /*isSigned=*/false));
    CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_critical_with_hint),
                        ArgsWithHint);
  } else
    CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_critical), Args);
  // Build a call to __kmpc_end_critical
  CGF.EHStack.pushCleanup<CallEndCleanup<std::extent<decltype(Args)>::value>>(
      NormalAndEHCleanup, createRuntimeFunction(OMPRTL__kmpc_end_critical),
      llvm::makeArrayRef(Args));
  emitInlinedDirective(CGF, OMPD_critical, CriticalOpGen);
}

static void emitIfStmt(CodeGenFunction &CGF, llvm::Value *IfCond,
                       OpenMPDirectiveKind Kind, SourceLocation Loc,
                       const RegionCodeGenTy &BodyOpGen) {
  llvm::Value *CallBool = CGF.EmitScalarConversion(
      IfCond,
      CGF.getContext().getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/true),
      CGF.getContext().BoolTy, Loc);

  auto *ThenBlock = CGF.createBasicBlock("omp_if.then");
  auto *ContBlock = CGF.createBasicBlock("omp_if.end");
  // Generate the branch (If-stmt)
  CGF.Builder.CreateCondBr(CallBool, ThenBlock, ContBlock);
  CGF.EmitBlock(ThenBlock);
  CGF.CGM.getOpenMPRuntime().emitInlinedDirective(CGF, Kind, BodyOpGen);
  // Emit the rest of bblocks/branches
  CGF.EmitBranch(ContBlock);
  CGF.EmitBlock(ContBlock, true);
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
  auto *IsMaster =
      CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_master), Args);
  typedef CallEndCleanup<std::extent<decltype(Args)>::value>
      MasterCallEndCleanup;
  emitIfStmt(
      CGF, IsMaster, OMPD_master, Loc, [&](CodeGenFunction &CGF) -> void {
        CodeGenFunction::RunCleanupsScope Scope(CGF);
        CGF.EHStack.pushCleanup<MasterCallEndCleanup>(
            NormalAndEHCleanup, createRuntimeFunction(OMPRTL__kmpc_end_master),
            llvm::makeArrayRef(Args));
        MasterOpGen(CGF);
      });
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
  {
    CodeGenFunction::RunCleanupsScope Scope(CGF);
    llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc)};
    CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_taskgroup), Args);
    // Build a call to __kmpc_end_taskgroup
    CGF.EHStack.pushCleanup<CallEndCleanup<std::extent<decltype(Args)>::value>>(
        NormalAndEHCleanup, createRuntimeFunction(OMPRTL__kmpc_end_taskgroup),
        llvm::makeArrayRef(Args));
    emitInlinedDirective(CGF, OMPD_taskgroup, TaskgroupOpGen);
  }
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
  ImplicitParamDecl LHSArg(C, /*DC=*/nullptr, SourceLocation(), /*Id=*/nullptr,
                           C.VoidPtrTy);
  ImplicitParamDecl RHSArg(C, /*DC=*/nullptr, SourceLocation(), /*Id=*/nullptr,
                           C.VoidPtrTy);
  Args.push_back(&LHSArg);
  Args.push_back(&RHSArg);
  FunctionType::ExtInfo EI;
  auto &CGFI = CGM.getTypes().arrangeFreeFunctionDeclaration(
      C.VoidTy, Args, EI, /*isVariadic=*/false);
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
  auto *IsSingle =
      CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_single), Args);
  typedef CallEndCleanup<std::extent<decltype(Args)>::value>
      SingleCallEndCleanup;
  emitIfStmt(
      CGF, IsSingle, OMPD_single, Loc, [&](CodeGenFunction &CGF) -> void {
        CodeGenFunction::RunCleanupsScope Scope(CGF);
        CGF.EHStack.pushCleanup<SingleCallEndCleanup>(
            NormalAndEHCleanup, createRuntimeFunction(OMPRTL__kmpc_end_single),
            llvm::makeArrayRef(Args));
        SingleOpGen(CGF);
        if (DidIt.isValid()) {
          // did_it = 1;
          CGF.Builder.CreateStore(CGF.Builder.getInt32(1), DidIt);
        }
      });
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
    auto *BufSize = getTypeSize(CGF, CopyprivateArrayTy);
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
  CodeGenFunction::RunCleanupsScope Scope(CGF);
  if (IsThreads) {
    llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc)};
    CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_ordered), Args);
    // Build a call to __kmpc_end_ordered
    CGF.EHStack.pushCleanup<CallEndCleanup<std::extent<decltype(Args)>::value>>(
        NormalAndEHCleanup, createRuntimeFunction(OMPRTL__kmpc_end_ordered),
        llvm::makeArrayRef(Args));
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
  OpenMPLocationFlags Flags = OMP_IDENT_KMPC;
  if (Kind == OMPD_for) {
    Flags =
        static_cast<OpenMPLocationFlags>(Flags | OMP_IDENT_BARRIER_IMPL_FOR);
  } else if (Kind == OMPD_sections) {
    Flags = static_cast<OpenMPLocationFlags>(Flags |
                                             OMP_IDENT_BARRIER_IMPL_SECTIONS);
  } else if (Kind == OMPD_single) {
    Flags =
        static_cast<OpenMPLocationFlags>(Flags | OMP_IDENT_BARRIER_IMPL_SINGLE);
  } else if (Kind == OMPD_barrier) {
    Flags = static_cast<OpenMPLocationFlags>(Flags | OMP_IDENT_BARRIER_EXPL);
  } else {
    Flags = static_cast<OpenMPLocationFlags>(Flags | OMP_IDENT_BARRIER_IMPL);
  }
  // Build call __kmpc_cancel_barrier(loc, thread_id) or __kmpc_barrier(loc,
  // thread_id);
  auto *OMPRegionInfo =
      dyn_cast_or_null<CGOpenMPRegionInfo>(CGF.CapturedStmtInfo);
  // Do not emit barrier call in the single directive emitted in some rare cases
  // for sections directives.
  if (OMPRegionInfo && OMPRegionInfo->getDirectiveKind() == OMPD_single)
    return;
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc, Flags),
                         getThreadID(CGF, Loc)};
  if (OMPRegionInfo) {
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
  /// \brief Lower bound for 'ordered' versions.
  OMP_ord_lower = 64,
  OMP_ord_static_chunked = 65,
  OMP_ord_static = 66,
  OMP_ord_dynamic_chunked = 67,
  OMP_ord_guided_chunked = 68,
  OMP_ord_runtime = 69,
  OMP_ord_auto = 70,
  OMP_sch_default = OMP_sch_static,
};

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

bool CGOpenMPRuntime::isStaticNonchunked(OpenMPScheduleClauseKind ScheduleKind,
                                         bool Chunked) const {
  auto Schedule = getRuntimeSchedule(ScheduleKind, Chunked, /*Ordered=*/false);
  return Schedule == OMP_sch_static;
}

bool CGOpenMPRuntime::isDynamic(OpenMPScheduleClauseKind ScheduleKind) const {
  auto Schedule =
      getRuntimeSchedule(ScheduleKind, /*Chunked=*/false, /*Ordered=*/false);
  assert(Schedule != OMP_sch_static_chunked && "cannot be chunked here");
  return Schedule != OMP_sch_static;
}

void CGOpenMPRuntime::emitForDispatchInit(CodeGenFunction &CGF,
                                          SourceLocation Loc,
                                          OpenMPScheduleClauseKind ScheduleKind,
                                          unsigned IVSize, bool IVSigned,
                                          bool Ordered, llvm::Value *UB,
                                          llvm::Value *Chunk) {
  if (!CGF.HaveInsertPoint())
    return;
  OpenMPSchedType Schedule =
      getRuntimeSchedule(ScheduleKind, Chunk != nullptr, Ordered);
  assert(Ordered ||
         (Schedule != OMP_sch_static && Schedule != OMP_sch_static_chunked &&
          Schedule != OMP_ord_static && Schedule != OMP_ord_static_chunked));
  // Call __kmpc_dispatch_init(
  //          ident_t *loc, kmp_int32 tid, kmp_int32 schedule,
  //          kmp_int[32|64] lower, kmp_int[32|64] upper,
  //          kmp_int[32|64] stride, kmp_int[32|64] chunk);

  // If the Chunk was not specified in the clause - use default value 1.
  if (Chunk == nullptr)
    Chunk = CGF.Builder.getIntN(IVSize, 1);
  llvm::Value *Args[] = {
    emitUpdateLocation(CGF, Loc, OMP_IDENT_KMPC),
    getThreadID(CGF, Loc),
    CGF.Builder.getInt32(Schedule), // Schedule type
    CGF.Builder.getIntN(IVSize, 0), // Lower
    UB,                             // Upper
    CGF.Builder.getIntN(IVSize, 1), // Stride
    Chunk                           // Chunk
  };
  CGF.EmitRuntimeCall(createDispatchInitFunction(IVSize, IVSigned), Args);
}

void CGOpenMPRuntime::emitForStaticInit(CodeGenFunction &CGF,
                                        SourceLocation Loc,
                                        OpenMPScheduleClauseKind ScheduleKind,
                                        unsigned IVSize, bool IVSigned,
                                        bool Ordered, Address IL, Address LB,
                                        Address UB, Address ST,
                                        llvm::Value *Chunk) {
  if (!CGF.HaveInsertPoint())
    return;
  OpenMPSchedType Schedule =
    getRuntimeSchedule(ScheduleKind, Chunk != nullptr, Ordered);
  assert(!Ordered);
  assert(Schedule == OMP_sch_static || Schedule == OMP_sch_static_chunked ||
         Schedule == OMP_ord_static || Schedule == OMP_ord_static_chunked);

  // Call __kmpc_for_static_init(
  //          ident_t *loc, kmp_int32 tid, kmp_int32 schedtype,
  //          kmp_int32 *p_lastiter, kmp_int[32|64] *p_lower,
  //          kmp_int[32|64] *p_upper, kmp_int[32|64] *p_stride,
  //          kmp_int[32|64] incr, kmp_int[32|64] chunk);
  if (Chunk == nullptr) {
    assert((Schedule == OMP_sch_static || Schedule == OMP_ord_static) &&
           "expected static non-chunked schedule");
    // If the Chunk was not specified in the clause - use default value 1.
      Chunk = CGF.Builder.getIntN(IVSize, 1);
  } else {
    assert((Schedule == OMP_sch_static_chunked ||
            Schedule == OMP_ord_static_chunked) &&
           "expected static chunked schedule");
  }
  llvm::Value *Args[] = {
    emitUpdateLocation(CGF, Loc, OMP_IDENT_KMPC),
    getThreadID(CGF, Loc),
    CGF.Builder.getInt32(Schedule), // Schedule type
    IL.getPointer(),                // &isLastIter
    LB.getPointer(),                // &LB
    UB.getPointer(),                // &UB
    ST.getPointer(),                // &Stride
    CGF.Builder.getIntN(IVSize, 1), // Incr
    Chunk                           // Chunk
  };
  CGF.EmitRuntimeCall(createForStaticInitFunction(IVSize, IVSigned), Args);
}

void CGOpenMPRuntime::emitForStaticFinish(CodeGenFunction &CGF,
                                          SourceLocation Loc) {
  if (!CGF.HaveInsertPoint())
    return;
  // Call __kmpc_for_static_fini(ident_t *loc, kmp_int32 tid);
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc, OMP_IDENT_KMPC),
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
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc, OMP_IDENT_KMPC),
                         getThreadID(CGF, Loc)};
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
      emitUpdateLocation(CGF, Loc, OMP_IDENT_KMPC), getThreadID(CGF, Loc),
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
  /// \brief Function with call of destructors for private variables.
  KmpTaskTDestructors,
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
                                    unsigned ColNum, unsigned Order) {
  assert(CGM.getLangOpts().OpenMPIsDevice && "Initialization of entries is "
                                             "only required for the device "
                                             "code generation.");
  OffloadEntriesTargetRegion[DeviceID][FileID][ParentName][LineNum][ColNum] =
      OffloadEntryInfoTargetRegion(Order, /*Addr=*/nullptr, /*ID=*/nullptr);
  ++OffloadingEntriesNum;
}

void CGOpenMPRuntime::OffloadEntriesInfoManagerTy::
    registerTargetRegionEntryInfo(unsigned DeviceID, unsigned FileID,
                                  StringRef ParentName, unsigned LineNum,
                                  unsigned ColNum, llvm::Constant *Addr,
                                  llvm::Constant *ID) {
  // If we are emitting code for a target, the entry is already initialized,
  // only has to be registered.
  if (CGM.getLangOpts().OpenMPIsDevice) {
    assert(hasTargetRegionEntryInfo(DeviceID, FileID, ParentName, LineNum,
                                    ColNum) &&
           "Entry must exist.");
    auto &Entry = OffloadEntriesTargetRegion[DeviceID][FileID][ParentName]
                                            [LineNum][ColNum];
    assert(Entry.isValid() && "Entry not initialized!");
    Entry.setAddress(Addr);
    Entry.setID(ID);
    return;
  } else {
    OffloadEntryInfoTargetRegion Entry(OffloadingEntriesNum++, Addr, ID);
    OffloadEntriesTargetRegion[DeviceID][FileID][ParentName][LineNum][ColNum] =
        Entry;
  }
}

bool CGOpenMPRuntime::OffloadEntriesInfoManagerTy::hasTargetRegionEntryInfo(
    unsigned DeviceID, unsigned FileID, StringRef ParentName, unsigned LineNum,
    unsigned ColNum) const {
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
  auto PerColumn = PerLine->second.find(ColNum);
  if (PerColumn == PerLine->second.end())
    return false;
  // Fail if this entry is already registered.
  if (PerColumn->second.getAddress() || PerColumn->second.getID())
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
          for (auto &C : L.second)
            Action(D.first, F.first, P.first(), L.first, C.first, C.second);
}

/// \brief Create a Ctor/Dtor-like function whose body is emitted through
/// \a Codegen. This is used to emit the two functions that register and
/// unregister the descriptor of the current compilation unit.
static llvm::Function *
createOffloadingBinaryDescriptorFunction(CodeGenModule &CGM, StringRef Name,
                                         const RegionCodeGenTy &Codegen) {
  auto &C = CGM.getContext();
  FunctionArgList Args;
  ImplicitParamDecl DummyPtr(C, /*DC=*/nullptr, SourceLocation(),
                             /*Id=*/nullptr, C.VoidPtrTy);
  Args.push_back(&DummyPtr);

  CodeGenFunction CGF(CGM);
  GlobalDecl();
  auto &FI = CGM.getTypes().arrangeFreeFunctionDeclaration(
      C.VoidTy, Args, FunctionType::ExtInfo(),
      /*isVariadic=*/false);
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
      llvm::GlobalValue::ExternalLinkage, /*Initializer=*/0,
      ".omp_offloading.entries_begin");
  llvm::GlobalVariable *HostEntriesEnd = new llvm::GlobalVariable(
      M, OffloadEntryTy, /*isConstant=*/true,
      llvm::GlobalValue::ExternalLinkage, /*Initializer=*/0,
      ".omp_offloading.entries_end");

  // Create all device images
  llvm::SmallVector<llvm::Constant *, 4> DeviceImagesEntires;
  auto *DeviceImageTy = cast<llvm::StructType>(
      CGM.getTypes().ConvertTypeForMem(getTgtDeviceImageQTy()));

  for (unsigned i = 0; i < Devices.size(); ++i) {
    StringRef T = Devices[i].getTriple();
    auto *ImgBegin = new llvm::GlobalVariable(
        M, CGM.Int8Ty, /*isConstant=*/true, llvm::GlobalValue::ExternalLinkage,
        /*Initializer=*/0, Twine(".omp_offloading.img_start.") + Twine(T));
    auto *ImgEnd = new llvm::GlobalVariable(
        M, CGM.Int8Ty, /*isConstant=*/true, llvm::GlobalValue::ExternalLinkage,
        /*Initializer=*/0, Twine(".omp_offloading.img_end.") + Twine(T));

    llvm::Constant *Dev =
        llvm::ConstantStruct::get(DeviceImageTy, ImgBegin, ImgEnd,
                                  HostEntriesBegin, HostEntriesEnd, nullptr);
    DeviceImagesEntires.push_back(Dev);
  }

  // Create device images global array.
  llvm::ArrayType *DeviceImagesInitTy =
      llvm::ArrayType::get(DeviceImageTy, DeviceImagesEntires.size());
  llvm::Constant *DeviceImagesInit =
      llvm::ConstantArray::get(DeviceImagesInitTy, DeviceImagesEntires);

  llvm::GlobalVariable *DeviceImages = new llvm::GlobalVariable(
      M, DeviceImagesInitTy, /*isConstant=*/true,
      llvm::GlobalValue::InternalLinkage, DeviceImagesInit,
      ".omp_offloading.device_images");
  DeviceImages->setUnnamedAddr(true);

  // This is a Zero array to be used in the creation of the constant expressions
  llvm::Constant *Index[] = {llvm::Constant::getNullValue(CGM.Int32Ty),
                             llvm::Constant::getNullValue(CGM.Int32Ty)};

  // Create the target region descriptor.
  auto *BinaryDescriptorTy = cast<llvm::StructType>(
      CGM.getTypes().ConvertTypeForMem(getTgtBinaryDescriptorQTy()));
  llvm::Constant *TargetRegionsDescriptorInit = llvm::ConstantStruct::get(
      BinaryDescriptorTy, llvm::ConstantInt::get(CGM.Int32Ty, Devices.size()),
      llvm::ConstantExpr::getGetElementPtr(DeviceImagesInitTy, DeviceImages,
                                           Index),
      HostEntriesBegin, HostEntriesEnd, nullptr);

  auto *Desc = new llvm::GlobalVariable(
      M, BinaryDescriptorTy, /*isConstant=*/true,
      llvm::GlobalValue::InternalLinkage, TargetRegionsDescriptorInit,
      ".omp_offloading.descriptor");

  // Emit code to register or unregister the descriptor at execution
  // startup or closing, respectively.

  // Create a variable to drive the registration and unregistration of the
  // descriptor, so we can reuse the logic that emits Ctors and Dtors.
  auto *IdentInfo = &C.Idents.get(".omp_offloading.reg_unreg_var");
  ImplicitParamDecl RegUnregVar(C, C.getTranslationUnitDecl(), SourceLocation(),
                                IdentInfo, C.CharTy);

  auto *UnRegFn = createOffloadingBinaryDescriptorFunction(
      CGM, ".omp_offloading.descriptor_unreg", [&](CodeGenFunction &CGF) {
        CGF.EmitCallOrInvoke(createRuntimeFunction(OMPRTL__tgt_unregister_lib),
                             Desc);
      });
  auto *RegFn = createOffloadingBinaryDescriptorFunction(
      CGM, ".omp_offloading.descriptor_reg", [&](CodeGenFunction &CGF) {
        CGF.EmitCallOrInvoke(createRuntimeFunction(OMPRTL__tgt_register_lib),
                             Desc);
        CGM.getCXXABI().registerGlobalDtor(CGF, RegUnregVar, UnRegFn, Desc);
      });
  return RegFn;
}

void CGOpenMPRuntime::createOffloadEntry(llvm::Constant *Addr, StringRef Name,
                                         uint64_t Size) {
  auto *TgtOffloadEntryType = cast<llvm::StructType>(
      CGM.getTypes().ConvertTypeForMem(getTgtOffloadEntryQTy()));
  llvm::LLVMContext &C = CGM.getModule().getContext();
  llvm::Module &M = CGM.getModule();

  // Make sure the address has the right type.
  llvm::Constant *AddrPtr = llvm::ConstantExpr::getBitCast(Addr, CGM.VoidPtrTy);

  // Create constant string with the name.
  llvm::Constant *StrPtrInit = llvm::ConstantDataArray::getString(C, Name);

  llvm::GlobalVariable *Str =
      new llvm::GlobalVariable(M, StrPtrInit->getType(), /*isConstant=*/true,
                               llvm::GlobalValue::InternalLinkage, StrPtrInit,
                               ".omp_offloading.entry_name");
  Str->setUnnamedAddr(true);
  llvm::Constant *StrPtr = llvm::ConstantExpr::getBitCast(Str, CGM.Int8PtrTy);

  // Create the entry struct.
  llvm::Constant *EntryInit = llvm::ConstantStruct::get(
      TgtOffloadEntryType, AddrPtr, StrPtr,
      llvm::ConstantInt::get(CGM.SizeTy, Size), nullptr);
  llvm::GlobalVariable *Entry = new llvm::GlobalVariable(
      M, TgtOffloadEntryType, true, llvm::GlobalValue::ExternalLinkage,
      EntryInit, ".omp_offloading.entry");

  // The entry has to be created in the section the linker expects it to be.
  Entry->setSection(".omp_offloading.entries");
  // We can't have any padding between symbols, so we need to have 1-byte
  // alignment.
  Entry->setAlignment(1);
  return;
}

void CGOpenMPRuntime::createOffloadEntriesAndInfoMetadata() {
  // Emit the offloading entries and metadata so that the device codegen side
  // can
  // easily figure out what to emit. The produced metadata looks like this:
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

  // Auxiliar methods to create metadata values and strings.
  auto getMDInt = [&](unsigned v) {
    return llvm::ConstantAsMetadata::get(
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), v));
  };

  auto getMDString = [&](StringRef v) { return llvm::MDString::get(C, v); };

  // Create function that emits metadata for each target region entry;
  auto &&TargetRegionMetadataEmitter = [&](
      unsigned DeviceID, unsigned FileID, StringRef ParentName, unsigned Line,
      unsigned Column,
      OffloadEntriesInfoManagerTy::OffloadEntryInfoTargetRegion &E) {
    llvm::SmallVector<llvm::Metadata *, 32> Ops;
    // Generate metadata for target regions. Each entry of this metadata
    // contains:
    // - Entry 0 -> Kind of this type of metadata (0).
    // - Entry 1 -> Device ID of the file where the entry was identified.
    // - Entry 2 -> File ID of the file where the entry was identified.
    // - Entry 3 -> Mangled name of the function where the entry was identified.
    // - Entry 4 -> Line in the file where the entry was identified.
    // - Entry 5 -> Column in the file where the entry was identified.
    // - Entry 6 -> Order the entry was created.
    // The first element of the metadata node is the kind.
    Ops.push_back(getMDInt(E.getKind()));
    Ops.push_back(getMDInt(DeviceID));
    Ops.push_back(getMDInt(FileID));
    Ops.push_back(getMDString(ParentName));
    Ops.push_back(getMDInt(Line));
    Ops.push_back(getMDInt(Column));
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
      createOffloadEntry(CE->getID(), CE->getAddress()->getName(), /*Size=*/0);
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
  auto ME = llvm::parseBitcodeFile(Buf.get()->getMemBufferRef(), C);

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
          /*Column=*/getMDInt(5), /*Order=*/getMDInt(6));
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
  // };
  if (TgtOffloadEntryQTy.isNull()) {
    ASTContext &C = CGM.getContext();
    auto *RD = C.buildImplicitRecord("__tgt_offload_entry");
    RD->startDefinition();
    addFieldToRecordDecl(C, RD, C.VoidPtrTy);
    addFieldToRecordDecl(C, RD, C.getPointerType(C.CharTy));
    addFieldToRecordDecl(C, RD, C.getSizeType());
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
createKmpTaskTRecordDecl(CodeGenModule &CGM, QualType KmpInt32Ty,
                         QualType KmpRoutineEntryPointerQTy) {
  auto &C = CGM.getContext();
  // Build struct kmp_task_t {
  //         void *              shareds;
  //         kmp_routine_entry_t routine;
  //         kmp_int32           part_id;
  //         kmp_routine_entry_t destructors;
  //       };
  auto *RD = C.buildImplicitRecord("kmp_task_t");
  RD->startDefinition();
  addFieldToRecordDecl(C, RD, C.VoidPtrTy);
  addFieldToRecordDecl(C, RD, KmpRoutineEntryPointerQTy);
  addFieldToRecordDecl(C, RD, KmpInt32Ty);
  addFieldToRecordDecl(C, RD, KmpRoutineEntryPointerQTy);
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
///   TaskFunction(gtid, tt->part_id, &tt->privates, task_privates_map,
///   tt->shareds);
///   return 0;
/// }
/// \endcode
static llvm::Value *
emitProxyTaskFunction(CodeGenModule &CGM, SourceLocation Loc,
                      QualType KmpInt32Ty, QualType KmpTaskTWithPrivatesPtrQTy,
                      QualType KmpTaskTWithPrivatesQTy, QualType KmpTaskTQTy,
                      QualType SharedsPtrTy, llvm::Value *TaskFunction,
                      llvm::Value *TaskPrivatesMap) {
  auto &C = CGM.getContext();
  FunctionArgList Args;
  ImplicitParamDecl GtidArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, KmpInt32Ty);
  ImplicitParamDecl TaskTypeArg(C, /*DC=*/nullptr, Loc,
                                /*Id=*/nullptr,
                                KmpTaskTWithPrivatesPtrQTy.withRestrict());
  Args.push_back(&GtidArg);
  Args.push_back(&TaskTypeArg);
  FunctionType::ExtInfo Info;
  auto &TaskEntryFnInfo =
      CGM.getTypes().arrangeFreeFunctionDeclaration(KmpInt32Ty, Args, Info,
                                                    /*isVariadic=*/false);
  auto *TaskEntryTy = CGM.getTypes().GetFunctionType(TaskEntryFnInfo);
  auto *TaskEntry =
      llvm::Function::Create(TaskEntryTy, llvm::GlobalValue::InternalLinkage,
                             ".omp_task_entry.", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*D=*/nullptr, TaskEntry, TaskEntryFnInfo);
  CodeGenFunction CGF(CGM);
  CGF.disableDebugInfo();
  CGF.StartFunction(GlobalDecl(), KmpInt32Ty, TaskEntry, TaskEntryFnInfo, Args);

  // TaskFunction(gtid, tt->task_data.part_id, &tt->privates, task_privates_map,
  // tt->task_data.shareds);
  auto *GtidParam = CGF.EmitLoadOfScalar(
      CGF.GetAddrOfLocalVar(&GtidArg), /*Volatile=*/false, KmpInt32Ty, Loc);
  LValue TDBase = emitLoadOfPointerLValue(
      CGF, CGF.GetAddrOfLocalVar(&TaskTypeArg), KmpTaskTWithPrivatesPtrQTy);
  auto *KmpTaskTWithPrivatesQTyRD =
      cast<RecordDecl>(KmpTaskTWithPrivatesQTy->getAsTagDecl());
  LValue Base =
      CGF.EmitLValueForField(TDBase, *KmpTaskTWithPrivatesQTyRD->field_begin());
  auto *KmpTaskTQTyRD = cast<RecordDecl>(KmpTaskTQTy->getAsTagDecl());
  auto PartIdFI = std::next(KmpTaskTQTyRD->field_begin(), KmpTaskTPartId);
  auto PartIdLVal = CGF.EmitLValueForField(Base, *PartIdFI);
  auto *PartidParam = CGF.EmitLoadOfLValue(PartIdLVal, Loc).getScalarVal();

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
  } else {
    PrivatesParam = llvm::ConstantPointerNull::get(CGF.VoidPtrTy);
  }

  llvm::Value *CallArgs[] = {GtidParam, PartidParam, PrivatesParam,
                             TaskPrivatesMap, SharedsParam};
  CGF.EmitCallOrInvoke(TaskFunction, CallArgs);
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
  ImplicitParamDecl GtidArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, KmpInt32Ty);
  ImplicitParamDecl TaskTypeArg(C, /*DC=*/nullptr, Loc,
                                /*Id=*/nullptr,
                                KmpTaskTWithPrivatesPtrQTy.withRestrict());
  Args.push_back(&GtidArg);
  Args.push_back(&TaskTypeArg);
  FunctionType::ExtInfo Info;
  auto &DestructorFnInfo =
      CGM.getTypes().arrangeFreeFunctionDeclaration(KmpInt32Ty, Args, Info,
                                                    /*isVariadic=*/false);
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

  LValue Base = emitLoadOfPointerLValue(
      CGF, CGF.GetAddrOfLocalVar(&TaskTypeArg), KmpTaskTWithPrivatesPtrQTy);
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
                               QualType PrivatesQTy,
                               ArrayRef<PrivateDataTy> Privates) {
  auto &C = CGM.getContext();
  FunctionArgList Args;
  ImplicitParamDecl TaskPrivatesArg(
      C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
      C.getPointerType(PrivatesQTy).withConst().withRestrict());
  Args.push_back(&TaskPrivatesArg);
  llvm::DenseMap<const VarDecl *, unsigned> PrivateVarsPos;
  unsigned Counter = 1;
  for (auto *E: PrivateVars) {
    Args.push_back(ImplicitParamDecl::Create(
        C, /*DC=*/nullptr, Loc,
        /*Id=*/nullptr, C.getPointerType(C.getPointerType(E->getType()))
                            .withConst()
                            .withRestrict()));
    auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
    PrivateVarsPos[VD] = Counter;
    ++Counter;
  }
  for (auto *E : FirstprivateVars) {
    Args.push_back(ImplicitParamDecl::Create(
        C, /*DC=*/nullptr, Loc,
        /*Id=*/nullptr, C.getPointerType(C.getPointerType(E->getType()))
                            .withConst()
                            .withRestrict()));
    auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
    PrivateVarsPos[VD] = Counter;
    ++Counter;
  }
  FunctionType::ExtInfo Info;
  auto &TaskPrivatesMapFnInfo =
      CGM.getTypes().arrangeFreeFunctionDeclaration(C.VoidTy, Args, Info,
                                                    /*isVariadic=*/false);
  auto *TaskPrivatesMapTy =
      CGM.getTypes().GetFunctionType(TaskPrivatesMapFnInfo);
  auto *TaskPrivatesMap = llvm::Function::Create(
      TaskPrivatesMapTy, llvm::GlobalValue::InternalLinkage,
      ".omp_task_privates_map.", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*D=*/nullptr, TaskPrivatesMap,
                                    TaskPrivatesMapFnInfo);
  TaskPrivatesMap->addFnAttr(llvm::Attribute::AlwaysInline);
  CodeGenFunction CGF(CGM);
  CGF.disableDebugInfo();
  CGF.StartFunction(GlobalDecl(), C.VoidTy, TaskPrivatesMap,
                    TaskPrivatesMapFnInfo, Args);

  // *privi = &.privates.privi;
  LValue Base = emitLoadOfPointerLValue(
      CGF, CGF.GetAddrOfLocalVar(&TaskPrivatesArg), TaskPrivatesArg.getType());
  auto *PrivatesQTyRD = cast<RecordDecl>(PrivatesQTy->getAsTagDecl());
  Counter = 0;
  for (auto *Field : PrivatesQTyRD->fields()) {
    auto FieldLVal = CGF.EmitLValueForField(Base, Field);
    auto *VD = Args[PrivateVarsPos[Privates[Counter].second.Original]];
    auto RefLVal = CGF.MakeAddrLValue(CGF.GetAddrOfLocalVar(VD), VD->getType());
    auto RefLoadLVal =
        emitLoadOfPointerLValue(CGF, RefLVal.getAddress(), RefLVal.getType());
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

void CGOpenMPRuntime::emitTaskCall(
    CodeGenFunction &CGF, SourceLocation Loc, const OMPExecutableDirective &D,
    bool Tied, llvm::PointerIntPair<llvm::Value *, 1, bool> Final,
    llvm::Value *TaskFunction, QualType SharedsTy, Address Shareds,
    const Expr *IfCond, ArrayRef<const Expr *> PrivateVars,
    ArrayRef<const Expr *> PrivateCopies,
    ArrayRef<const Expr *> FirstprivateVars,
    ArrayRef<const Expr *> FirstprivateCopies,
    ArrayRef<const Expr *> FirstprivateInits,
    ArrayRef<std::pair<OpenMPDependClauseKind, const Expr *>> Dependences) {
  if (!CGF.HaveInsertPoint())
    return;
  auto &C = CGM.getContext();
  llvm::SmallVector<PrivateDataTy, 8> Privates;
  // Aggregate privates and sort them by the alignment.
  auto I = PrivateCopies.begin();
  for (auto *E : PrivateVars) {
    auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
    Privates.push_back(std::make_pair(
        C.getDeclAlign(VD),
        PrivateHelpersTy(VD, cast<VarDecl>(cast<DeclRefExpr>(*I)->getDecl()),
                         /*PrivateElemInit=*/nullptr)));
    ++I;
  }
  I = FirstprivateCopies.begin();
  auto IElemInitRef = FirstprivateInits.begin();
  for (auto *E : FirstprivateVars) {
    auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
    Privates.push_back(std::make_pair(
        C.getDeclAlign(VD),
        PrivateHelpersTy(
            VD, cast<VarDecl>(cast<DeclRefExpr>(*I)->getDecl()),
            cast<VarDecl>(cast<DeclRefExpr>(*IElemInitRef)->getDecl()))));
    ++I, ++IElemInitRef;
  }
  llvm::array_pod_sort(Privates.begin(), Privates.end(),
                       array_pod_sort_comparator);
  auto KmpInt32Ty = C.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/1);
  // Build type kmp_routine_entry_t (if not built yet).
  emitKmpRoutineEntryT(KmpInt32Ty);
  // Build type kmp_task_t (if not built yet).
  if (KmpTaskTQTy.isNull()) {
    KmpTaskTQTy = C.getRecordType(
        createKmpTaskTRecordDecl(CGM, KmpInt32Ty, KmpRoutineEntryPtrQTy));
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
  auto *KmpTaskTWithPrivatesTySize = getTypeSize(CGF, KmpTaskTWithPrivatesQTy);
  QualType SharedsPtrTy = C.getPointerType(SharedsTy);

  // Emit initial values for private copies (if any).
  llvm::Value *TaskPrivatesMap = nullptr;
  auto *TaskPrivatesMapTy =
      std::next(cast<llvm::Function>(TaskFunction)->getArgumentList().begin(),
                3)
          ->getType();
  if (!Privates.empty()) {
    auto FI = std::next(KmpTaskTWithPrivatesQTyRD->field_begin());
    TaskPrivatesMap = emitTaskPrivateMappingFunction(
        CGM, Loc, PrivateVars, FirstprivateVars, FI->getType(), Privates);
    TaskPrivatesMap = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
        TaskPrivatesMap, TaskPrivatesMapTy);
  } else {
    TaskPrivatesMap = llvm::ConstantPointerNull::get(
        cast<llvm::PointerType>(TaskPrivatesMapTy));
  }
  // Build a proxy function kmp_int32 .omp_task_entry.(kmp_int32 gtid,
  // kmp_task_t *tt);
  auto *TaskEntry = emitProxyTaskFunction(
      CGM, Loc, KmpInt32Ty, KmpTaskTWithPrivatesPtrQTy, KmpTaskTWithPrivatesQTy,
      KmpTaskTQTy, SharedsPtrTy, TaskFunction, TaskPrivatesMap);

  // Build call kmp_task_t * __kmpc_omp_task_alloc(ident_t *, kmp_int32 gtid,
  // kmp_int32 flags, size_t sizeof_kmp_task_t, size_t sizeof_shareds,
  // kmp_routine_entry_t *task_entry);
  // Task flags. Format is taken from
  // http://llvm.org/svn/llvm-project/openmp/trunk/runtime/src/kmp.h,
  // description of kmp_tasking_flags struct.
  const unsigned TiedFlag = 0x1;
  const unsigned FinalFlag = 0x2;
  unsigned Flags = Tied ? TiedFlag : 0;
  auto *TaskFlags =
      Final.getPointer()
          ? CGF.Builder.CreateSelect(Final.getPointer(),
                                     CGF.Builder.getInt32(FinalFlag),
                                     CGF.Builder.getInt32(/*C=*/0))
          : CGF.Builder.getInt32(Final.getInt() ? FinalFlag : 0);
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
  bool NeedsCleanup = false;
  if (!Privates.empty()) {
    auto FI = std::next(KmpTaskTWithPrivatesQTyRD->field_begin());
    auto PrivatesBase = CGF.EmitLValueForField(Base, *FI);
    FI = cast<RecordDecl>(FI->getType()->getAsTagDecl())->field_begin();
    LValue SharedsBase;
    if (!FirstprivateVars.empty()) {
      SharedsBase = CGF.MakeAddrLValue(
          CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
              KmpTaskSharedsPtr, CGF.ConvertTypeForMem(SharedsPtrTy)),
          SharedsTy);
    }
    CodeGenFunction::CGCapturedStmtInfo CapturesInfo(
        cast<CapturedStmt>(*D.getAssociatedStmt()));
    for (auto &&Pair : Privates) {
      auto *VD = Pair.second.PrivateCopy;
      auto *Init = VD->getAnyInitializer();
      LValue PrivateLValue = CGF.EmitLValueForField(PrivatesBase, *FI);
      if (Init) {
        if (auto *Elem = Pair.second.PrivateElemInit) {
          auto *OriginalVD = Pair.second.Original;
          auto *SharedField = CapturesInfo.lookup(OriginalVD);
          auto SharedRefLValue =
              CGF.EmitLValueForField(SharedsBase, SharedField);
          SharedRefLValue = CGF.MakeAddrLValue(
              Address(SharedRefLValue.getPointer(), C.getDeclAlign(OriginalVD)),
              SharedRefLValue.getType(), AlignmentSource::Decl);
          QualType Type = OriginalVD->getType();
          if (Type->isArrayType()) {
            // Initialize firstprivate array.
            if (!isa<CXXConstructExpr>(Init) ||
                CGF.isTrivialInitializer(Init)) {
              // Perform simple memcpy.
              CGF.EmitAggregateAssign(PrivateLValue.getAddress(),
                                      SharedRefLValue.getAddress(), Type);
            } else {
              // Initialize firstprivate array using element-by-element
              // intialization.
              CGF.EmitOMPAggregateAssign(
                  PrivateLValue.getAddress(), SharedRefLValue.getAddress(),
                  Type, [&CGF, Elem, Init, &CapturesInfo](
                            Address DestElement, Address SrcElement) {
                    // Clean up any temporaries needed by the initialization.
                    CodeGenFunction::OMPPrivateScope InitScope(CGF);
                    InitScope.addPrivate(Elem, [SrcElement]() -> Address {
                      return SrcElement;
                    });
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
        } else {
          CGF.EmitExprAsInit(Init, VD, PrivateLValue, /*capturedByInit=*/false);
        }
      }
      NeedsCleanup = NeedsCleanup || FI->getType().isDestructedType();
      ++FI;
    }
  }
  // Provide pointer to function with destructors for privates.
  llvm::Value *DestructorFn =
      NeedsCleanup ? emitDestructorsFunction(CGM, Loc, KmpInt32Ty,
                                             KmpTaskTWithPrivatesPtrQTy,
                                             KmpTaskTWithPrivatesQTy)
                   : llvm::ConstantPointerNull::get(
                         cast<llvm::PointerType>(KmpRoutineEntryPtrTy));
  LValue Destructor = CGF.EmitLValueForField(
      TDBase, *std::next(KmpTaskTQTyRD->field_begin(), KmpTaskTDestructors));
  CGF.EmitStoreOfScalar(CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
                            DestructorFn, KmpRoutineEntryPtrTy),
                        Destructor);

  // Process list of dependences.
  Address DependenciesArray = Address::invalid();
  unsigned NumDependencies = Dependences.size();
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
    } else {
      KmpDependInfoRD = cast<RecordDecl>(KmpDependInfoTy->getAsTagDecl());
    }
    CharUnits DependencySize = C.getTypeSizeInChars(KmpDependInfoTy);
    // Define type kmp_depend_info[<Dependences.size()>];
    QualType KmpDependInfoArrayTy = C.getConstantArrayType(
        KmpDependInfoTy, llvm::APInt(/*numBits=*/64, NumDependencies),
        ArrayType::Normal, /*IndexTypeQuals=*/0);
    // kmp_depend_info[<Dependences.size()>] deps;
    DependenciesArray = CGF.CreateMemTemp(KmpDependInfoArrayTy);
    for (unsigned i = 0; i < NumDependencies; ++i) {
      const Expr *E = Dependences[i].second;
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
        Size = getTypeSize(CGF, Ty);
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
      switch (Dependences[i].first) {
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
  // Build kmp_int32 __kmpc_omp_task(ident_t *, kmp_int32 gtid, kmp_task_t
  // *new_task);
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
  auto &&ThenCodeGen = [this, NumDependencies,
                        &TaskArgs, &DepTaskArgs](CodeGenFunction &CGF) {
    // TODO: add check for untied tasks.    
    if (NumDependencies) {
      CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_omp_task_with_deps),
                          DepTaskArgs);
    } else {
      CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_omp_task),
                          TaskArgs);
    }
  };
  typedef CallEndCleanup<std::extent<decltype(TaskArgs)>::value>
      IfCallEndCleanup;

  llvm::Value *DepWaitTaskArgs[6];
  if (NumDependencies) {
    DepWaitTaskArgs[0] = UpLoc;
    DepWaitTaskArgs[1] = ThreadID;
    DepWaitTaskArgs[2] = CGF.Builder.getInt32(NumDependencies);
    DepWaitTaskArgs[3] = DependenciesArray.getPointer();
    DepWaitTaskArgs[4] = CGF.Builder.getInt32(0);
    DepWaitTaskArgs[5] = llvm::ConstantPointerNull::get(CGF.VoidPtrTy);
  }
  auto &&ElseCodeGen = [this, &TaskArgs, ThreadID, NewTaskNewTaskTTy, TaskEntry,
                        NumDependencies, &DepWaitTaskArgs](CodeGenFunction &CGF) {
    CodeGenFunction::RunCleanupsScope LocalScope(CGF);
    // Build void __kmpc_omp_wait_deps(ident_t *, kmp_int32 gtid,
    // kmp_int32 ndeps, kmp_depend_info_t *dep_list, kmp_int32
    // ndeps_noalias, kmp_depend_info_t *noalias_dep_list); if dependence info
    // is specified.
    if (NumDependencies)
      CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_omp_wait_deps),
                          DepWaitTaskArgs);
    // Build void __kmpc_omp_task_begin_if0(ident_t *, kmp_int32 gtid,
    // kmp_task_t *new_task);
    CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_omp_task_begin_if0),
                        TaskArgs);
    // Build void __kmpc_omp_task_complete_if0(ident_t *, kmp_int32 gtid,
    // kmp_task_t *new_task);
    CGF.EHStack.pushCleanup<IfCallEndCleanup>(
        NormalAndEHCleanup,
        createRuntimeFunction(OMPRTL__kmpc_omp_task_complete_if0),
        llvm::makeArrayRef(TaskArgs));

    // Call proxy_task_entry(gtid, new_task);
    llvm::Value *OutlinedFnArgs[] = {ThreadID, NewTaskNewTaskTTy};
    CGF.EmitCallOrInvoke(TaskEntry, OutlinedFnArgs);
  };

  if (IfCond) {
    emitOMPIfClause(CGF, IfCond, ThenCodeGen, ElseCodeGen);
  } else {
    CodeGenFunction::RunCleanupsScope Scope(CGF);
    ThenCodeGen(CGF);
  }
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

static llvm::Value *emitReductionFunction(CodeGenModule &CGM,
                                          llvm::Type *ArgsType,
                                          ArrayRef<const Expr *> Privates,
                                          ArrayRef<const Expr *> LHSExprs,
                                          ArrayRef<const Expr *> RHSExprs,
                                          ArrayRef<const Expr *> ReductionOps) {
  auto &C = CGM.getContext();

  // void reduction_func(void *LHSArg, void *RHSArg);
  FunctionArgList Args;
  ImplicitParamDecl LHSArg(C, /*DC=*/nullptr, SourceLocation(), /*Id=*/nullptr,
                           C.VoidPtrTy);
  ImplicitParamDecl RHSArg(C, /*DC=*/nullptr, SourceLocation(), /*Id=*/nullptr,
                           C.VoidPtrTy);
  Args.push_back(&LHSArg);
  Args.push_back(&RHSArg);
  FunctionType::ExtInfo EI;
  auto &CGFI = CGM.getTypes().arrangeFreeFunctionDeclaration(
      C.VoidTy, Args, EI, /*isVariadic=*/false);
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
    if (PrivTy->isArrayType()) {
      // Get array size and emit VLA type.
      ++Idx;
      Address Elem =
          CGF.Builder.CreateConstArrayGEP(LHS, Idx, CGF.getPointerSize());
      llvm::Value *Ptr = CGF.Builder.CreateLoad(Elem);
      CodeGenFunction::OpaqueValueMapping OpaqueMap(
          CGF,
          cast<OpaqueValueExpr>(
              CGF.getContext().getAsVariableArrayType(PrivTy)->getSizeExpr()),
          RValue::get(CGF.Builder.CreatePtrToInt(Ptr, CGF.SizeTy)));
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
      EmitOMPAggregateReduction(CGF, (*IPriv)->getType(), LHSVar, RHSVar,
                                [=](CodeGenFunction &CGF, const Expr *,
                                    const Expr *,
                                    const Expr *) { CGF.EmitIgnoredExpr(E); });
    } else
      // Emit reduction for array subscript or single variable.
      CGF.EmitIgnoredExpr(E);
    ++IPriv, ++ILHS, ++IRHS;
  }
  Scope.ForceCleanup();
  CGF.FinishFunction();
  return Fn;
}

void CGOpenMPRuntime::emitReduction(CodeGenFunction &CGF, SourceLocation Loc,
                                    ArrayRef<const Expr *> Privates,
                                    ArrayRef<const Expr *> LHSExprs,
                                    ArrayRef<const Expr *> RHSExprs,
                                    ArrayRef<const Expr *> ReductionOps,
                                    bool WithNowait, bool SimpleReduction) {
  if (!CGF.HaveInsertPoint())
    return;
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
      if ((*IPriv)->getType()->isArrayType()) {
        auto *LHSVar = cast<VarDecl>(cast<DeclRefExpr>(*ILHS)->getDecl());
        auto *RHSVar = cast<VarDecl>(cast<DeclRefExpr>(*IRHS)->getDecl());
        EmitOMPAggregateReduction(
            CGF, (*IPriv)->getType(), LHSVar, RHSVar,
            [=](CodeGenFunction &CGF, const Expr *, const Expr *,
                const Expr *) { CGF.EmitIgnoredExpr(E); });
      } else
        CGF.EmitIgnoredExpr(E);
      ++IPriv, ++ILHS, ++IRHS;
    }
    return;
  }

  // 1. Build a list of reduction variables.
  // void *RedList[<n>] = {<ReductionVars>[0], ..., <ReductionVars>[<n>-1]};
  auto Size = RHSExprs.size();
  for (auto *E : Privates) {
    if (E->getType()->isArrayType())
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
    if ((*IPriv)->getType()->isArrayType()) {
      // Store array size.
      ++Idx;
      Elem = CGF.Builder.CreateConstArrayGEP(ReductionList, Idx,
                                             CGF.getPointerSize());
      CGF.Builder.CreateStore(
          CGF.Builder.CreateIntToPtr(
              CGF.Builder.CreateIntCast(
                  CGF.getVLASize(CGF.getContext().getAsVariableArrayType(
                                     (*IPriv)->getType()))
                      .first,
                  CGF.SizeTy, /*isSigned=*/false),
              CGF.VoidPtrTy),
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
  auto *IdentTLoc = emitUpdateLocation(
      CGF, Loc,
      static_cast<OpenMPLocationFlags>(OMP_IDENT_KMPC | OMP_ATOMIC_REDUCE));
  auto *ThreadId = getThreadID(CGF, Loc);
  auto *ReductionArrayTySize = getTypeSize(CGF, ReductionArrayTy);
  auto *RL =
    CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(ReductionList.getPointer(),
                                                    CGF.VoidPtrTy);
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

  {
    CodeGenFunction::RunCleanupsScope Scope(CGF);
    // Add emission of __kmpc_end_reduce{_nowait}(<loc>, <gtid>, &<lock>);
    llvm::Value *EndArgs[] = {
        IdentTLoc, // ident_t *<loc>
        ThreadId,  // i32 <gtid>
        Lock       // kmp_critical_name *&<lock>
    };
    CGF.EHStack
        .pushCleanup<CallEndCleanup<std::extent<decltype(EndArgs)>::value>>(
            NormalAndEHCleanup,
            createRuntimeFunction(WithNowait ? OMPRTL__kmpc_end_reduce_nowait
                                             : OMPRTL__kmpc_end_reduce),
            llvm::makeArrayRef(EndArgs));
    auto IPriv = Privates.begin();
    auto ILHS = LHSExprs.begin();
    auto IRHS = RHSExprs.begin();
    for (auto *E : ReductionOps) {
      if ((*IPriv)->getType()->isArrayType()) {
        // Emit reduction for array section.
        auto *LHSVar = cast<VarDecl>(cast<DeclRefExpr>(*ILHS)->getDecl());
        auto *RHSVar = cast<VarDecl>(cast<DeclRefExpr>(*IRHS)->getDecl());
        EmitOMPAggregateReduction(
            CGF, (*IPriv)->getType(), LHSVar, RHSVar,
            [=](CodeGenFunction &CGF, const Expr *, const Expr *,
                const Expr *) { CGF.EmitIgnoredExpr(E); });
      } else
        // Emit reduction for array subscript or single variable.
        CGF.EmitIgnoredExpr(E);
      ++IPriv, ++ILHS, ++IRHS;
    }
  }

  CGF.EmitBranch(DefaultBB);

  // 7. Build case 2:
  //  ...
  //  Atomic(<LHSExprs>[i] = RedOp<i>(*<LHSExprs>[i], *<RHSExprs>[i]));
  //  ...
  // break;
  auto *Case2BB = CGF.createBasicBlock(".omp.reduction.case2");
  SwInst->addCase(CGF.Builder.getInt32(2), Case2BB);
  CGF.EmitBlock(Case2BB);

  {
    CodeGenFunction::RunCleanupsScope Scope(CGF);
    if (!WithNowait) {
      // Add emission of __kmpc_end_reduce(<loc>, <gtid>, &<lock>);
      llvm::Value *EndArgs[] = {
          IdentTLoc, // ident_t *<loc>
          ThreadId,  // i32 <gtid>
          Lock       // kmp_critical_name *&<lock>
      };
      CGF.EHStack
          .pushCleanup<CallEndCleanup<std::extent<decltype(EndArgs)>::value>>(
              NormalAndEHCleanup,
              createRuntimeFunction(OMPRTL__kmpc_end_reduce),
              llvm::makeArrayRef(EndArgs));
    }
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
          auto &&AtomicRedGen = [this, BO, VD, IPriv,
                                 Loc](CodeGenFunction &CGF, const Expr *XExpr,
                                      const Expr *EExpr, const Expr *UpExpr) {
            LValue X = CGF.EmitLValue(XExpr);
            RValue E;
            if (EExpr)
              E = CGF.EmitAnyExpr(EExpr);
            CGF.EmitOMPAtomicSimpleUpdateExpr(
                X, E, BO, /*IsXLHSInRHSPart=*/true, llvm::Monotonic, Loc,
                [&CGF, UpExpr, VD, IPriv](RValue XRValue) {
                  CodeGenFunction::OMPPrivateScope PrivateScope(CGF);
                  PrivateScope.addPrivate(VD, [&CGF, VD, XRValue]() -> Address {
                    Address LHSTemp = CGF.CreateMemTemp(VD->getType());
                    CGF.EmitStoreThroughLValue(
                        XRValue, CGF.MakeAddrLValue(LHSTemp, VD->getType()));
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
          auto &&CritRedGen = [this, E, Loc](CodeGenFunction &CGF, const Expr *,
                                             const Expr *, const Expr *) {
            emitCriticalRegion(
                CGF, ".atomic_reduction",
                [E](CodeGenFunction &CGF) { CGF.EmitIgnoredExpr(E); }, Loc);
          };
          if ((*IPriv)->getType()->isArrayType()) {
            auto *LHSVar = cast<VarDecl>(cast<DeclRefExpr>(*ILHS)->getDecl());
            auto *RHSVar = cast<VarDecl>(cast<DeclRefExpr>(*IRHS)->getDecl());
            EmitOMPAggregateReduction(CGF, (*IPriv)->getType(), LHSVar, RHSVar,
                                      CritRedGen);
          } else
            CritRedGen(CGF, nullptr, nullptr, nullptr);
        }
      ++ILHS, ++IRHS, ++IPriv;
    }
  }

  CGF.EmitBranch(DefaultBB);
  CGF.EmitBlock(DefaultBB, /*IsFinished=*/true);
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
}

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
    if (OMPRegionInfo->getDirectiveKind() == OMPD_single)
      return;
    if (OMPRegionInfo->hasCancel()) {
      llvm::Value *Args[] = {
          emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc),
          CGF.Builder.getInt32(getCancellationKind(CancelRegion))};
      // Ignore return result until untied tasks are supported.
      auto *Result = CGF.EmitRuntimeCall(
          createRuntimeFunction(OMPRTL__kmpc_cancellationpoint), Args);
      // if (__kmpc_cancellationpoint()) {
      //  __kmpc_cancel_barrier();
      //   exit from construct;
      // }
      auto *ExitBB = CGF.createBasicBlock(".cancel.exit");
      auto *ContBB = CGF.createBasicBlock(".cancel.continue");
      auto *Cmp = CGF.Builder.CreateIsNotNull(Result);
      CGF.Builder.CreateCondBr(Cmp, ExitBB, ContBB);
      CGF.EmitBlock(ExitBB);
      // __kmpc_cancel_barrier();
      emitBarrierCall(CGF, Loc, OMPD_unknown, /*EmitChecks=*/false);
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
    if (OMPRegionInfo->getDirectiveKind() == OMPD_single)
      return;
    auto &&ThenGen = [this, Loc, CancelRegion,
                      OMPRegionInfo](CodeGenFunction &CGF) {
      llvm::Value *Args[] = {
          emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc),
          CGF.Builder.getInt32(getCancellationKind(CancelRegion))};
      // Ignore return result until untied tasks are supported.
      auto *Result =
          CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__kmpc_cancel), Args);
      // if (__kmpc_cancel()) {
      //  __kmpc_cancel_barrier();
      //   exit from construct;
      // }
      auto *ExitBB = CGF.createBasicBlock(".cancel.exit");
      auto *ContBB = CGF.createBasicBlock(".cancel.continue");
      auto *Cmp = CGF.Builder.CreateIsNotNull(Result);
      CGF.Builder.CreateCondBr(Cmp, ExitBB, ContBB);
      CGF.EmitBlock(ExitBB);
      // __kmpc_cancel_barrier();
      emitBarrierCall(CGF, Loc, OMPD_unknown, /*EmitChecks=*/false);
      // exit from construct;
      auto CancelDest =
          CGF.getOMPCancelDestination(OMPRegionInfo->getDirectiveKind());
      CGF.EmitBranchThroughCleanup(CancelDest);
      CGF.EmitBlock(ContBB, /*IsFinished=*/true);
    };
    if (IfCond)
      emitOMPIfClause(CGF, IfCond, ThenGen, [](CodeGenFunction &) {});
    else
      ThenGen(CGF);
  }
}

/// \brief Obtain information that uniquely identifies a target entry. This
/// consists of the file and device IDs as well as line and column numbers
/// associated with the relevant entry source location.
static void getTargetEntryUniqueInfo(ASTContext &C, SourceLocation Loc,
                                     unsigned &DeviceID, unsigned &FileID,
                                     unsigned &LineNum, unsigned &ColumnNum) {

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
  ColumnNum = PLoc.getColumn();
  return;
}

void CGOpenMPRuntime::emitTargetOutlinedFunction(
    const OMPExecutableDirective &D, StringRef ParentName,
    llvm::Function *&OutlinedFn, llvm::Constant *&OutlinedFnID,
    bool IsOffloadEntry) {

  assert(!ParentName.empty() && "Invalid target region parent name!");

  const CapturedStmt &CS = *cast<CapturedStmt>(D.getAssociatedStmt());

  // Emit target region as a standalone region.
  auto &&CodeGen = [&CS](CodeGenFunction &CGF) {
    CGF.EmitStmt(CS.getCapturedStmt());
  };

  // Create a unique name for the proxy/entry function that using the source
  // location information of the current target region. The name will be
  // something like:
  //
  // .omp_offloading.DD_FFFF.PP.lBB.cCC
  //
  // where DD_FFFF is an ID unique to the file (device and file IDs), PP is the
  // mangled name of the function that encloses the target region, BB is the
  // line number of the target region, and CC is the column number of the target
  // region.

  unsigned DeviceID;
  unsigned FileID;
  unsigned Line;
  unsigned Column;
  getTargetEntryUniqueInfo(CGM.getContext(), D.getLocStart(), DeviceID, FileID,
                           Line, Column);
  SmallString<64> EntryFnName;
  {
    llvm::raw_svector_ostream OS(EntryFnName);
    OS << ".omp_offloading" << llvm::format(".%x", DeviceID)
       << llvm::format(".%x.", FileID) << ParentName << ".l" << Line << ".c"
       << Column;
  }

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
      DeviceID, FileID, ParentName, Line, Column, OutlinedFn, OutlinedFnID);
  return;
}

void CGOpenMPRuntime::emitTargetCall(CodeGenFunction &CGF,
                                     const OMPExecutableDirective &D,
                                     llvm::Value *OutlinedFn,
                                     llvm::Value *OutlinedFnID,
                                     const Expr *IfCond, const Expr *Device,
                                     ArrayRef<llvm::Value *> CapturedVars) {
  if (!CGF.HaveInsertPoint())
    return;
  /// \brief Values for bit flags used to specify the mapping type for
  /// offloading.
  enum OpenMPOffloadMappingFlags {
    /// \brief Allocate memory on the device and move data from host to device.
    OMP_MAP_TO = 0x01,
    /// \brief Allocate memory on the device and move data from device to host.
    OMP_MAP_FROM = 0x02,
    /// \brief The element passed to the device is a pointer.
    OMP_MAP_PTR = 0x20,
    /// \brief Pass the element to the device by value.
    OMP_MAP_BYCOPY = 0x80,
  };

  enum OpenMPOffloadingReservedDeviceIDs {
    /// \brief Device ID if the device was not defined, runtime should get it
    /// from environment variables in the spec.
    OMP_DEVICEID_UNDEF = -1,
  };

  assert(OutlinedFn && "Invalid outlined function!");

  auto &Ctx = CGF.getContext();

  // Fill up the arrays with the all the captured variables.
  SmallVector<llvm::Value *, 16> BasePointers;
  SmallVector<llvm::Value *, 16> Pointers;
  SmallVector<llvm::Value *, 16> Sizes;
  SmallVector<unsigned, 16> MapTypes;

  bool hasVLACaptures = false;

  const CapturedStmt &CS = *cast<CapturedStmt>(D.getAssociatedStmt());
  auto RI = CS.getCapturedRecordDecl()->field_begin();
  // auto II = CS.capture_init_begin();
  auto CV = CapturedVars.begin();
  for (CapturedStmt::const_capture_iterator CI = CS.capture_begin(),
                                            CE = CS.capture_end();
       CI != CE; ++CI, ++RI, ++CV) {
    StringRef Name;
    QualType Ty;
    llvm::Value *BasePointer;
    llvm::Value *Pointer;
    llvm::Value *Size;
    unsigned MapType;

    // VLA sizes are passed to the outlined region by copy.
    if (CI->capturesVariableArrayType()) {
      BasePointer = Pointer = *CV;
      Size = getTypeSize(CGF, RI->getType());
      // Copy to the device as an argument. No need to retrieve it.
      MapType = OMP_MAP_BYCOPY;
      hasVLACaptures = true;
    } else if (CI->capturesThis()) {
      BasePointer = Pointer = *CV;
      const PointerType *PtrTy = cast<PointerType>(RI->getType().getTypePtr());
      Size = getTypeSize(CGF, PtrTy->getPointeeType());
      // Default map type.
      MapType = OMP_MAP_TO | OMP_MAP_FROM;
    } else if (CI->capturesVariableByCopy()) {
      MapType = OMP_MAP_BYCOPY;
      if (!RI->getType()->isAnyPointerType()) {
        // If the field is not a pointer, we need to save the actual value and
        // load it as a void pointer.
        auto DstAddr = CGF.CreateMemTemp(
            Ctx.getUIntPtrType(),
            Twine(CI->getCapturedVar()->getName()) + ".casted");
        LValue DstLV = CGF.MakeAddrLValue(DstAddr, Ctx.getUIntPtrType());

        auto *SrcAddrVal = CGF.EmitScalarConversion(
            DstAddr.getPointer(), Ctx.getPointerType(Ctx.getUIntPtrType()),
            Ctx.getPointerType(RI->getType()), SourceLocation());
        LValue SrcLV =
            CGF.MakeNaturalAlignAddrLValue(SrcAddrVal, RI->getType());

        // Store the value using the source type pointer.
        CGF.EmitStoreThroughLValue(RValue::get(*CV), SrcLV);

        // Load the value using the destination type pointer.
        BasePointer = Pointer =
            CGF.EmitLoadOfLValue(DstLV, SourceLocation()).getScalarVal();
      } else {
        MapType |= OMP_MAP_PTR;
        BasePointer = Pointer = *CV;
      }
      Size = getTypeSize(CGF, RI->getType());
    } else {
      assert(CI->capturesVariable() && "Expected captured reference.");
      BasePointer = Pointer = *CV;

      const ReferenceType *PtrTy =
          cast<ReferenceType>(RI->getType().getTypePtr());
      QualType ElementType = PtrTy->getPointeeType();
      Size = getTypeSize(CGF, ElementType);
      // The default map type for a scalar/complex type is 'to' because by
      // default the value doesn't have to be retrieved. For an aggregate type,
      // the default is 'tofrom'.
      MapType = ElementType->isAggregateType() ? (OMP_MAP_TO | OMP_MAP_FROM)
                                               : OMP_MAP_TO;
      if (ElementType->isAnyPointerType())
        MapType |= OMP_MAP_PTR;
    }

    BasePointers.push_back(BasePointer);
    Pointers.push_back(Pointer);
    Sizes.push_back(Size);
    MapTypes.push_back(MapType);
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
  auto &&ThenGen = [this, &Ctx, &BasePointers, &Pointers, &Sizes, &MapTypes,
                    hasVLACaptures, Device, OutlinedFnID, OffloadError,
                    OffloadErrorQType](CodeGenFunction &CGF) {
    unsigned PointerNumVal = BasePointers.size();
    llvm::Value *PointerNum = CGF.Builder.getInt32(PointerNumVal);
    llvm::Value *BasePointersArray;
    llvm::Value *PointersArray;
    llvm::Value *SizesArray;
    llvm::Value *MapTypesArray;

    if (PointerNumVal) {
      llvm::APInt PointerNumAP(32, PointerNumVal, /*isSigned=*/true);
      QualType PointerArrayType = Ctx.getConstantArrayType(
          Ctx.VoidPtrTy, PointerNumAP, ArrayType::Normal,
          /*IndexTypeQuals=*/0);

      BasePointersArray =
          CGF.CreateMemTemp(PointerArrayType, ".offload_baseptrs").getPointer();
      PointersArray =
          CGF.CreateMemTemp(PointerArrayType, ".offload_ptrs").getPointer();

      // If we don't have any VLA types, we can use a constant array for the map
      // sizes, otherwise we need to fill up the arrays as we do for the
      // pointers.
      if (hasVLACaptures) {
        QualType SizeArrayType = Ctx.getConstantArrayType(
            Ctx.getSizeType(), PointerNumAP, ArrayType::Normal,
            /*IndexTypeQuals=*/0);
        SizesArray =
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
        SizesArrayGbl->setUnnamedAddr(true);
        SizesArray = SizesArrayGbl;
      }

      // The map types are always constant so we don't need to generate code to
      // fill arrays. Instead, we create an array constant.
      llvm::Constant *MapTypesArrayInit =
          llvm::ConstantDataArray::get(CGF.Builder.getContext(), MapTypes);
      auto *MapTypesArrayGbl = new llvm::GlobalVariable(
          CGM.getModule(), MapTypesArrayInit->getType(),
          /*isConstant=*/true, llvm::GlobalValue::PrivateLinkage,
          MapTypesArrayInit, ".offload_maptypes");
      MapTypesArrayGbl->setUnnamedAddr(true);
      MapTypesArray = MapTypesArrayGbl;

      for (unsigned i = 0; i < PointerNumVal; ++i) {

        llvm::Value *BPVal = BasePointers[i];
        if (BPVal->getType()->isPointerTy())
          BPVal = CGF.Builder.CreateBitCast(BPVal, CGM.VoidPtrTy);
        else {
          assert(BPVal->getType()->isIntegerTy() &&
                 "If not a pointer, the value type must be an integer.");
          BPVal = CGF.Builder.CreateIntToPtr(BPVal, CGM.VoidPtrTy);
        }
        llvm::Value *BP = CGF.Builder.CreateConstInBoundsGEP2_32(
            llvm::ArrayType::get(CGM.VoidPtrTy, PointerNumVal),
            BasePointersArray, 0, i);
        Address BPAddr(BP, Ctx.getTypeAlignInChars(Ctx.VoidPtrTy));
        CGF.Builder.CreateStore(BPVal, BPAddr);

        llvm::Value *PVal = Pointers[i];
        if (PVal->getType()->isPointerTy())
          PVal = CGF.Builder.CreateBitCast(PVal, CGM.VoidPtrTy);
        else {
          assert(PVal->getType()->isIntegerTy() &&
                 "If not a pointer, the value type must be an integer.");
          PVal = CGF.Builder.CreateIntToPtr(PVal, CGM.VoidPtrTy);
        }
        llvm::Value *P = CGF.Builder.CreateConstInBoundsGEP2_32(
            llvm::ArrayType::get(CGM.VoidPtrTy, PointerNumVal), PointersArray,
            0, i);
        Address PAddr(P, Ctx.getTypeAlignInChars(Ctx.VoidPtrTy));
        CGF.Builder.CreateStore(PVal, PAddr);

        if (hasVLACaptures) {
          llvm::Value *S = CGF.Builder.CreateConstInBoundsGEP2_32(
              llvm::ArrayType::get(CGM.SizeTy, PointerNumVal), SizesArray,
              /*Idx0=*/0,
              /*Idx1=*/i);
          Address SAddr(S, Ctx.getTypeAlignInChars(Ctx.getSizeType()));
          CGF.Builder.CreateStore(CGF.Builder.CreateIntCast(
                                      Sizes[i], CGM.SizeTy, /*isSigned=*/true),
                                  SAddr);
        }
      }

      BasePointersArray = CGF.Builder.CreateConstInBoundsGEP2_32(
          llvm::ArrayType::get(CGM.VoidPtrTy, PointerNumVal), BasePointersArray,
          /*Idx0=*/0, /*Idx1=*/0);
      PointersArray = CGF.Builder.CreateConstInBoundsGEP2_32(
          llvm::ArrayType::get(CGM.VoidPtrTy, PointerNumVal), PointersArray,
          /*Idx0=*/0,
          /*Idx1=*/0);
      SizesArray = CGF.Builder.CreateConstInBoundsGEP2_32(
          llvm::ArrayType::get(CGM.SizeTy, PointerNumVal), SizesArray,
          /*Idx0=*/0, /*Idx1=*/0);
      MapTypesArray = CGF.Builder.CreateConstInBoundsGEP2_32(
          llvm::ArrayType::get(CGM.Int32Ty, PointerNumVal), MapTypesArray,
          /*Idx0=*/0,
          /*Idx1=*/0);

    } else {
      BasePointersArray = llvm::ConstantPointerNull::get(CGM.VoidPtrPtrTy);
      PointersArray = llvm::ConstantPointerNull::get(CGM.VoidPtrPtrTy);
      SizesArray = llvm::ConstantPointerNull::get(CGM.SizeTy->getPointerTo());
      MapTypesArray =
          llvm::ConstantPointerNull::get(CGM.Int32Ty->getPointerTo());
    }

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
                                           CGM.Int32Ty, /*isSigned=*/true);
    else
      DeviceID = CGF.Builder.getInt32(OMP_DEVICEID_UNDEF);

    llvm::Value *OffloadingArgs[] = {
        DeviceID,      OutlinedFnID, PointerNum,   BasePointersArray,
        PointersArray, SizesArray,   MapTypesArray};
    auto Return = CGF.EmitRuntimeCall(createRuntimeFunction(OMPRTL__tgt_target),
                                      OffloadingArgs);

    CGF.EmitStoreOfScalar(Return, OffloadError);
  };

  // Notify that the host version must be executed.
  auto &&ElseGen = [this, OffloadError,
                    OffloadErrorQType](CodeGenFunction &CGF) {
    CGF.EmitStoreOfScalar(llvm::ConstantInt::get(CGM.Int32Ty, /*V=*/-1u),
                          OffloadError);
  };

  // If we have a target function ID it means that we need to support
  // offloading, otherwise, just execute on the host. We need to execute on host
  // regardless of the conditional in the if clause if, e.g., the user do not
  // specify target triples.
  if (OutlinedFnID) {
    if (IfCond) {
      emitOMPIfClause(CGF, IfCond, ThenGen, ElseGen);
    } else {
      CodeGenFunction::RunCleanupsScope Scope(CGF);
      ThenGen(CGF);
    }
  } else {
    CodeGenFunction::RunCleanupsScope Scope(CGF);
    ElseGen(CGF);
  }

  // Check the error code and execute the host version if required.
  auto OffloadFailedBlock = CGF.createBasicBlock("omp_offload.failed");
  auto OffloadContBlock = CGF.createBasicBlock("omp_offload.cont");
  auto OffloadErrorVal = CGF.EmitLoadOfScalar(OffloadError, SourceLocation());
  auto Failed = CGF.Builder.CreateIsNotNull(OffloadErrorVal);
  CGF.Builder.CreateCondBr(Failed, OffloadFailedBlock, OffloadContBlock);

  CGF.EmitBlock(OffloadFailedBlock);
  CGF.Builder.CreateCall(OutlinedFn, BasePointers);
  CGF.EmitBranch(OffloadContBlock);

  CGF.EmitBlock(OffloadContBlock, /*IsFinished=*/true);
  return;
}

void CGOpenMPRuntime::scanForTargetRegionsFunctions(const Stmt *S,
                                                    StringRef ParentName) {
  if (!S)
    return;

  // If we find a OMP target directive, codegen the outline function and
  // register the result.
  // FIXME: Add other directives with target when they become supported.
  bool isTargetDirective = isa<OMPTargetDirective>(S);

  if (isTargetDirective) {
    auto *E = cast<OMPExecutableDirective>(S);
    unsigned DeviceID;
    unsigned FileID;
    unsigned Line;
    unsigned Column;
    getTargetEntryUniqueInfo(CGM.getContext(), E->getLocStart(), DeviceID,
                             FileID, Line, Column);

    // Is this a target region that should not be emitted as an entry point? If
    // so just signal we are done with this target region.
    if (!OffloadEntriesInfoManager.hasTargetRegionEntryInfo(
            DeviceID, FileID, ParentName, Line, Column))
      return;

    llvm::Function *Fn;
    llvm::Constant *Addr;
    emitTargetOutlinedFunction(*E, ParentName, Fn, Addr,
                               /*isOffloadEntry=*/true);
    assert(Fn && Addr && "Target region emission failed.");
    return;
  }

  if (const OMPExecutableDirective *E = dyn_cast<OMPExecutableDirective>(S)) {
    if (!E->getAssociatedStmt())
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

  return;
}

bool CGOpenMPRuntime::emitTargetFunctions(GlobalDecl GD) {
  auto &FD = *cast<FunctionDecl>(GD.getDecl());

  // If emitting code for the host, we do not process FD here. Instead we do
  // the normal code generation.
  if (!CGM.getLangOpts().OpenMPIsDevice)
    return false;

  // Try to detect target regions in the function.
  scanForTargetRegionsFunctions(FD.getBody(), CGM.getMangledName(GD));

  // We should not emit any function othen that the ones created during the
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

  // If we are in target mode we do not emit any global (declare target is not
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
