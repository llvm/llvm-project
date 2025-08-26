//===---- CGBuiltin.cpp - Emit LLVM Code for builtins ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Builtin calls as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CGBuiltin.h"
#include "ABIInfo.h"
#include "CGCUDARuntime.h"
#include "CGCXXABI.h"
#include "CGDebugInfo.h"
#include "CGObjCRuntime.h"
#include "CGOpenCLRuntime.h"
#include "CGRecordLayout.h"
#include "CGValue.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "ConstantEmitter.h"
#include "PatternInit.h"
#include "TargetInfo.h"
#include "clang/AST/OSLog.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/IR/MatrixBuilder.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/ScopedPrinter.h"
#include <optional>
#include <utility>

using namespace clang;
using namespace CodeGen;
using namespace llvm;

/// Some builtins do not have library implementation on some targets and
/// are instead emitted as LLVM IRs by some target builtin emitters.
/// FIXME: Remove this when library support is added
static bool shouldEmitBuiltinAsIR(unsigned BuiltinID,
                                  const Builtin::Context &BI,
                                  const CodeGenFunction &CGF) {
  if (!CGF.CGM.getLangOpts().MathErrno &&
      CGF.CurFPFeatures.getExceptionMode() ==
          LangOptions::FPExceptionModeKind::FPE_Ignore &&
      !CGF.CGM.getTargetCodeGenInfo().supportsLibCall()) {
    switch (BuiltinID) {
    default:
      return false;
    case Builtin::BIlogbf:
    case Builtin::BI__builtin_logbf:
    case Builtin::BIlogb:
    case Builtin::BI__builtin_logb:
    case Builtin::BIscalbnf:
    case Builtin::BI__builtin_scalbnf:
    case Builtin::BIscalbn:
    case Builtin::BI__builtin_scalbn:
      return true;
    }
  }
  return false;
}

static Value *EmitTargetArchBuiltinExpr(CodeGenFunction *CGF,
                                        unsigned BuiltinID, const CallExpr *E,
                                        ReturnValueSlot ReturnValue,
                                        llvm::Triple::ArchType Arch) {
  // When compiling in HipStdPar mode we have to be conservative in rejecting
  // target specific features in the FE, and defer the possible error to the
  // AcceleratorCodeSelection pass, wherein iff an unsupported target builtin is
  // referenced by an accelerator executable function, we emit an error.
  // Returning nullptr here leads to the builtin being handled in
  // EmitStdParUnsupportedBuiltin.
  if (CGF->getLangOpts().HIPStdPar && CGF->getLangOpts().CUDAIsDevice &&
      Arch != CGF->getTarget().getTriple().getArch())
    return nullptr;

  switch (Arch) {
  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb:
    return CGF->EmitARMBuiltinExpr(BuiltinID, E, ReturnValue, Arch);
  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_32:
  case llvm::Triple::aarch64_be:
    return CGF->EmitAArch64BuiltinExpr(BuiltinID, E, Arch);
  case llvm::Triple::bpfeb:
  case llvm::Triple::bpfel:
    return CGF->EmitBPFBuiltinExpr(BuiltinID, E);
  case llvm::Triple::dxil:
    return CGF->EmitDirectXBuiltinExpr(BuiltinID, E);
  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    return CGF->EmitX86BuiltinExpr(BuiltinID, E);
  case llvm::Triple::ppc:
  case llvm::Triple::ppcle:
  case llvm::Triple::ppc64:
  case llvm::Triple::ppc64le:
    return CGF->EmitPPCBuiltinExpr(BuiltinID, E);
  case llvm::Triple::r600:
  case llvm::Triple::amdgcn:
    return CGF->EmitAMDGPUBuiltinExpr(BuiltinID, E);
  case llvm::Triple::systemz:
    return CGF->EmitSystemZBuiltinExpr(BuiltinID, E);
  case llvm::Triple::nvptx:
  case llvm::Triple::nvptx64:
    return CGF->EmitNVPTXBuiltinExpr(BuiltinID, E);
  case llvm::Triple::wasm32:
  case llvm::Triple::wasm64:
    return CGF->EmitWebAssemblyBuiltinExpr(BuiltinID, E);
  case llvm::Triple::hexagon:
    return CGF->EmitHexagonBuiltinExpr(BuiltinID, E);
  case llvm::Triple::riscv32:
  case llvm::Triple::riscv64:
    return CGF->EmitRISCVBuiltinExpr(BuiltinID, E, ReturnValue);
  case llvm::Triple::spirv32:
  case llvm::Triple::spirv64:
    if (CGF->getTarget().getTriple().getOS() == llvm::Triple::OSType::AMDHSA)
      return CGF->EmitAMDGPUBuiltinExpr(BuiltinID, E);
    [[fallthrough]];
  case llvm::Triple::spirv:
    return CGF->EmitSPIRVBuiltinExpr(BuiltinID, E);
  default:
    return nullptr;
  }
}

Value *CodeGenFunction::EmitTargetBuiltinExpr(unsigned BuiltinID,
                                              const CallExpr *E,
                                              ReturnValueSlot ReturnValue) {
  if (getContext().BuiltinInfo.isAuxBuiltinID(BuiltinID)) {
    assert(getContext().getAuxTargetInfo() && "Missing aux target info");
    return EmitTargetArchBuiltinExpr(
        this, getContext().BuiltinInfo.getAuxBuiltinID(BuiltinID), E,
        ReturnValue, getContext().getAuxTargetInfo()->getTriple().getArch());
  }

  return EmitTargetArchBuiltinExpr(this, BuiltinID, E, ReturnValue,
                                   getTarget().getTriple().getArch());
}

static void initializeAlloca(CodeGenFunction &CGF, AllocaInst *AI, Value *Size,
                             Align AlignmentInBytes) {
  ConstantInt *Byte;
  switch (CGF.getLangOpts().getTrivialAutoVarInit()) {
  case LangOptions::TrivialAutoVarInitKind::Uninitialized:
    // Nothing to initialize.
    return;
  case LangOptions::TrivialAutoVarInitKind::Zero:
    Byte = CGF.Builder.getInt8(0x00);
    break;
  case LangOptions::TrivialAutoVarInitKind::Pattern: {
    llvm::Type *Int8 = llvm::IntegerType::getInt8Ty(CGF.CGM.getLLVMContext());
    Byte = llvm::dyn_cast<llvm::ConstantInt>(
        initializationPatternFor(CGF.CGM, Int8));
    break;
  }
  }
  if (CGF.CGM.stopAutoInit())
    return;
  auto *I = CGF.Builder.CreateMemSet(AI, Byte, Size, AlignmentInBytes);
  I->addAnnotationMetadata("auto-init");
}

/// getBuiltinLibFunction - Given a builtin id for a function like
/// "__builtin_fabsf", return a Function* for "fabsf".
llvm::Constant *CodeGenModule::getBuiltinLibFunction(const FunctionDecl *FD,
                                                     unsigned BuiltinID) {
  assert(Context.BuiltinInfo.isLibFunction(BuiltinID));

  // Get the name, skip over the __builtin_ prefix (if necessary). We may have
  // to build this up so provide a small stack buffer to handle the vast
  // majority of names.
  llvm::SmallString<64> Name;
  GlobalDecl D(FD);

  // TODO: This list should be expanded or refactored after all GCC-compatible
  // std libcall builtins are implemented.
  static SmallDenseMap<unsigned, StringRef, 64> F128Builtins{
      {Builtin::BI__builtin___fprintf_chk, "__fprintf_chkieee128"},
      {Builtin::BI__builtin___printf_chk, "__printf_chkieee128"},
      {Builtin::BI__builtin___snprintf_chk, "__snprintf_chkieee128"},
      {Builtin::BI__builtin___sprintf_chk, "__sprintf_chkieee128"},
      {Builtin::BI__builtin___vfprintf_chk, "__vfprintf_chkieee128"},
      {Builtin::BI__builtin___vprintf_chk, "__vprintf_chkieee128"},
      {Builtin::BI__builtin___vsnprintf_chk, "__vsnprintf_chkieee128"},
      {Builtin::BI__builtin___vsprintf_chk, "__vsprintf_chkieee128"},
      {Builtin::BI__builtin_fprintf, "__fprintfieee128"},
      {Builtin::BI__builtin_printf, "__printfieee128"},
      {Builtin::BI__builtin_snprintf, "__snprintfieee128"},
      {Builtin::BI__builtin_sprintf, "__sprintfieee128"},
      {Builtin::BI__builtin_vfprintf, "__vfprintfieee128"},
      {Builtin::BI__builtin_vprintf, "__vprintfieee128"},
      {Builtin::BI__builtin_vsnprintf, "__vsnprintfieee128"},
      {Builtin::BI__builtin_vsprintf, "__vsprintfieee128"},
      {Builtin::BI__builtin_fscanf, "__fscanfieee128"},
      {Builtin::BI__builtin_scanf, "__scanfieee128"},
      {Builtin::BI__builtin_sscanf, "__sscanfieee128"},
      {Builtin::BI__builtin_vfscanf, "__vfscanfieee128"},
      {Builtin::BI__builtin_vscanf, "__vscanfieee128"},
      {Builtin::BI__builtin_vsscanf, "__vsscanfieee128"},
      {Builtin::BI__builtin_nexttowardf128, "__nexttowardieee128"},
  };

  // The AIX library functions frexpl, ldexpl, and modfl are for 128-bit
  // IBM 'long double' (i.e. __ibm128). Map to the 'double' versions
  // if it is 64-bit 'long double' mode.
  static SmallDenseMap<unsigned, StringRef, 4> AIXLongDouble64Builtins{
      {Builtin::BI__builtin_frexpl, "frexp"},
      {Builtin::BI__builtin_ldexpl, "ldexp"},
      {Builtin::BI__builtin_modfl, "modf"},
  };

  // If the builtin has been declared explicitly with an assembler label,
  // use the mangled name. This differs from the plain label on platforms
  // that prefix labels.
  if (FD->hasAttr<AsmLabelAttr>())
    Name = getMangledName(D);
  else {
    // TODO: This mutation should also be applied to other targets other than
    // PPC, after backend supports IEEE 128-bit style libcalls.
    if (getTriple().isPPC64() &&
        &getTarget().getLongDoubleFormat() == &llvm::APFloat::IEEEquad() &&
        F128Builtins.contains(BuiltinID))
      Name = F128Builtins[BuiltinID];
    else if (getTriple().isOSAIX() &&
             &getTarget().getLongDoubleFormat() ==
                 &llvm::APFloat::IEEEdouble() &&
             AIXLongDouble64Builtins.contains(BuiltinID))
      Name = AIXLongDouble64Builtins[BuiltinID];
    else
      Name = Context.BuiltinInfo.getName(BuiltinID).substr(10);
  }

  llvm::FunctionType *Ty =
    cast<llvm::FunctionType>(getTypes().ConvertType(FD->getType()));

  return GetOrCreateLLVMFunction(Name, Ty, D, /*ForVTable=*/false);
}

/// Emit the conversions required to turn the given value into an
/// integer of the given size.
Value *EmitToInt(CodeGenFunction &CGF, llvm::Value *V,
                        QualType T, llvm::IntegerType *IntType) {
  V = CGF.EmitToMemory(V, T);

  if (V->getType()->isPointerTy())
    return CGF.Builder.CreatePtrToInt(V, IntType);

  assert(V->getType() == IntType);
  return V;
}

Value *EmitFromInt(CodeGenFunction &CGF, llvm::Value *V,
                          QualType T, llvm::Type *ResultType) {
  V = CGF.EmitFromMemory(V, T);

  if (ResultType->isPointerTy())
    return CGF.Builder.CreateIntToPtr(V, ResultType);

  assert(V->getType() == ResultType);
  return V;
}

Address CheckAtomicAlignment(CodeGenFunction &CGF, const CallExpr *E) {
  ASTContext &Ctx = CGF.getContext();
  Address Ptr = CGF.EmitPointerWithAlignment(E->getArg(0));
  const llvm::DataLayout &DL = CGF.CGM.getDataLayout();
  unsigned Bytes = Ptr.getElementType()->isPointerTy()
                       ? Ctx.getTypeSizeInChars(Ctx.VoidPtrTy).getQuantity()
                       : DL.getTypeStoreSize(Ptr.getElementType());
  unsigned Align = Ptr.getAlignment().getQuantity();
  if (Align % Bytes != 0) {
    DiagnosticsEngine &Diags = CGF.CGM.getDiags();
    Diags.Report(E->getBeginLoc(), diag::warn_sync_op_misaligned);
    // Force address to be at least naturally-aligned.
    return Ptr.withAlignment(CharUnits::fromQuantity(Bytes));
  }
  return Ptr;
}

/// Utility to insert an atomic instruction based on Intrinsic::ID
/// and the expression node.
Value *MakeBinaryAtomicValue(
    CodeGenFunction &CGF, llvm::AtomicRMWInst::BinOp Kind, const CallExpr *E,
    AtomicOrdering Ordering) {

  QualType T = E->getType();
  assert(E->getArg(0)->getType()->isPointerType());
  assert(CGF.getContext().hasSameUnqualifiedType(T,
                                  E->getArg(0)->getType()->getPointeeType()));
  assert(CGF.getContext().hasSameUnqualifiedType(T, E->getArg(1)->getType()));

  Address DestAddr = CheckAtomicAlignment(CGF, E);

  llvm::IntegerType *IntType = llvm::IntegerType::get(
      CGF.getLLVMContext(), CGF.getContext().getTypeSize(T));

  llvm::Value *Val = CGF.EmitScalarExpr(E->getArg(1));
  llvm::Type *ValueType = Val->getType();
  Val = EmitToInt(CGF, Val, T, IntType);

  llvm::Value *Result =
      CGF.Builder.CreateAtomicRMW(Kind, DestAddr, Val, Ordering);
  return EmitFromInt(CGF, Result, T, ValueType);
}

static Value *EmitNontemporalStore(CodeGenFunction &CGF, const CallExpr *E) {
  Value *Val = CGF.EmitScalarExpr(E->getArg(0));
  Address Addr = CGF.EmitPointerWithAlignment(E->getArg(1));

  Val = CGF.EmitToMemory(Val, E->getArg(0)->getType());
  LValue LV = CGF.MakeAddrLValue(Addr, E->getArg(0)->getType());
  LV.setNontemporal(true);
  CGF.EmitStoreOfScalar(Val, LV, false);
  return nullptr;
}

static Value *EmitNontemporalLoad(CodeGenFunction &CGF, const CallExpr *E) {
  Address Addr = CGF.EmitPointerWithAlignment(E->getArg(0));

  LValue LV = CGF.MakeAddrLValue(Addr, E->getType());
  LV.setNontemporal(true);
  return CGF.EmitLoadOfScalar(LV, E->getExprLoc());
}

static RValue EmitBinaryAtomic(CodeGenFunction &CGF,
                               llvm::AtomicRMWInst::BinOp Kind,
                               const CallExpr *E) {
  return RValue::get(MakeBinaryAtomicValue(CGF, Kind, E));
}

/// Utility to insert an atomic instruction based Intrinsic::ID and
/// the expression node, where the return value is the result of the
/// operation.
static RValue EmitBinaryAtomicPost(CodeGenFunction &CGF,
                                   llvm::AtomicRMWInst::BinOp Kind,
                                   const CallExpr *E,
                                   Instruction::BinaryOps Op,
                                   bool Invert = false) {
  QualType T = E->getType();
  assert(E->getArg(0)->getType()->isPointerType());
  assert(CGF.getContext().hasSameUnqualifiedType(T,
                                  E->getArg(0)->getType()->getPointeeType()));
  assert(CGF.getContext().hasSameUnqualifiedType(T, E->getArg(1)->getType()));

  Address DestAddr = CheckAtomicAlignment(CGF, E);

  llvm::IntegerType *IntType = llvm::IntegerType::get(
      CGF.getLLVMContext(), CGF.getContext().getTypeSize(T));

  llvm::Value *Val = CGF.EmitScalarExpr(E->getArg(1));
  llvm::Type *ValueType = Val->getType();
  Val = EmitToInt(CGF, Val, T, IntType);

  llvm::Value *Result = CGF.Builder.CreateAtomicRMW(
      Kind, DestAddr, Val, llvm::AtomicOrdering::SequentiallyConsistent);
  Result = CGF.Builder.CreateBinOp(Op, Result, Val);
  if (Invert)
    Result =
        CGF.Builder.CreateBinOp(llvm::Instruction::Xor, Result,
                                llvm::ConstantInt::getAllOnesValue(IntType));
  Result = EmitFromInt(CGF, Result, T, ValueType);
  return RValue::get(Result);
}

/// Utility to insert an atomic cmpxchg instruction.
///
/// @param CGF The current codegen function.
/// @param E   Builtin call expression to convert to cmpxchg.
///            arg0 - address to operate on
///            arg1 - value to compare with
///            arg2 - new value
/// @param ReturnBool Specifies whether to return success flag of
///                   cmpxchg result or the old value.
///
/// @returns result of cmpxchg, according to ReturnBool
///
/// Note: In order to lower Microsoft's _InterlockedCompareExchange* intrinsics
/// invoke the function EmitAtomicCmpXchgForMSIntrin.
Value *MakeAtomicCmpXchgValue(CodeGenFunction &CGF, const CallExpr *E,
                                     bool ReturnBool) {
  QualType T = ReturnBool ? E->getArg(1)->getType() : E->getType();
  Address DestAddr = CheckAtomicAlignment(CGF, E);

  llvm::IntegerType *IntType = llvm::IntegerType::get(
      CGF.getLLVMContext(), CGF.getContext().getTypeSize(T));

  Value *Cmp = CGF.EmitScalarExpr(E->getArg(1));
  llvm::Type *ValueType = Cmp->getType();
  Cmp = EmitToInt(CGF, Cmp, T, IntType);
  Value *New = EmitToInt(CGF, CGF.EmitScalarExpr(E->getArg(2)), T, IntType);

  Value *Pair = CGF.Builder.CreateAtomicCmpXchg(
      DestAddr, Cmp, New, llvm::AtomicOrdering::SequentiallyConsistent,
      llvm::AtomicOrdering::SequentiallyConsistent);
  if (ReturnBool)
    // Extract boolean success flag and zext it to int.
    return CGF.Builder.CreateZExt(CGF.Builder.CreateExtractValue(Pair, 1),
                                  CGF.ConvertType(E->getType()));
  else
    // Extract old value and emit it using the same type as compare value.
    return EmitFromInt(CGF, CGF.Builder.CreateExtractValue(Pair, 0), T,
                       ValueType);
}

/// This function should be invoked to emit atomic cmpxchg for Microsoft's
/// _InterlockedCompareExchange* intrinsics which have the following signature:
/// T _InterlockedCompareExchange(T volatile *Destination,
///                               T Exchange,
///                               T Comparand);
///
/// Whereas the llvm 'cmpxchg' instruction has the following syntax:
/// cmpxchg *Destination, Comparand, Exchange.
/// So we need to swap Comparand and Exchange when invoking
/// CreateAtomicCmpXchg. That is the reason we could not use the above utility
/// function MakeAtomicCmpXchgValue since it expects the arguments to be
/// already swapped.

static
Value *EmitAtomicCmpXchgForMSIntrin(CodeGenFunction &CGF, const CallExpr *E,
    AtomicOrdering SuccessOrdering = AtomicOrdering::SequentiallyConsistent) {
  assert(E->getArg(0)->getType()->isPointerType());
  assert(CGF.getContext().hasSameUnqualifiedType(
      E->getType(), E->getArg(0)->getType()->getPointeeType()));
  assert(CGF.getContext().hasSameUnqualifiedType(E->getType(),
                                                 E->getArg(1)->getType()));
  assert(CGF.getContext().hasSameUnqualifiedType(E->getType(),
                                                 E->getArg(2)->getType()));

  Address DestAddr = CheckAtomicAlignment(CGF, E);

  auto *Exchange = CGF.EmitScalarExpr(E->getArg(1));
  auto *RTy = Exchange->getType();

  auto *Comparand = CGF.EmitScalarExpr(E->getArg(2));

  if (RTy->isPointerTy()) {
    Exchange = CGF.Builder.CreatePtrToInt(Exchange, CGF.IntPtrTy);
    Comparand = CGF.Builder.CreatePtrToInt(Comparand, CGF.IntPtrTy);
  }

  // For Release ordering, the failure ordering should be Monotonic.
  auto FailureOrdering = SuccessOrdering == AtomicOrdering::Release ?
                         AtomicOrdering::Monotonic :
                         SuccessOrdering;

  // The atomic instruction is marked volatile for consistency with MSVC. This
  // blocks the few atomics optimizations that LLVM has. If we want to optimize
  // _Interlocked* operations in the future, we will have to remove the volatile
  // marker.
  auto *CmpXchg = CGF.Builder.CreateAtomicCmpXchg(
      DestAddr, Comparand, Exchange, SuccessOrdering, FailureOrdering);
  CmpXchg->setVolatile(true);

  auto *Result = CGF.Builder.CreateExtractValue(CmpXchg, 0);
  if (RTy->isPointerTy()) {
    Result = CGF.Builder.CreateIntToPtr(Result, RTy);
  }

  return Result;
}

// 64-bit Microsoft platforms support 128 bit cmpxchg operations. They are
// prototyped like this:
//
// unsigned char _InterlockedCompareExchange128...(
//     __int64 volatile * _Destination,
//     __int64 _ExchangeHigh,
//     __int64 _ExchangeLow,
//     __int64 * _ComparandResult);
//
// Note that Destination is assumed to be at least 16-byte aligned, despite
// being typed int64.

static Value *EmitAtomicCmpXchg128ForMSIntrin(CodeGenFunction &CGF,
                                              const CallExpr *E,
                                              AtomicOrdering SuccessOrdering) {
  assert(E->getNumArgs() == 4);
  llvm::Value *DestPtr = CGF.EmitScalarExpr(E->getArg(0));
  llvm::Value *ExchangeHigh = CGF.EmitScalarExpr(E->getArg(1));
  llvm::Value *ExchangeLow = CGF.EmitScalarExpr(E->getArg(2));
  Address ComparandAddr = CGF.EmitPointerWithAlignment(E->getArg(3));

  assert(DestPtr->getType()->isPointerTy());
  assert(!ExchangeHigh->getType()->isPointerTy());
  assert(!ExchangeLow->getType()->isPointerTy());

  // For Release ordering, the failure ordering should be Monotonic.
  auto FailureOrdering = SuccessOrdering == AtomicOrdering::Release
                             ? AtomicOrdering::Monotonic
                             : SuccessOrdering;

  // Convert to i128 pointers and values. Alignment is also overridden for
  // destination pointer.
  llvm::Type *Int128Ty = llvm::IntegerType::get(CGF.getLLVMContext(), 128);
  Address DestAddr(DestPtr, Int128Ty,
                   CGF.getContext().toCharUnitsFromBits(128));
  ComparandAddr = ComparandAddr.withElementType(Int128Ty);

  // (((i128)hi) << 64) | ((i128)lo)
  ExchangeHigh = CGF.Builder.CreateZExt(ExchangeHigh, Int128Ty);
  ExchangeLow = CGF.Builder.CreateZExt(ExchangeLow, Int128Ty);
  ExchangeHigh =
      CGF.Builder.CreateShl(ExchangeHigh, llvm::ConstantInt::get(Int128Ty, 64));
  llvm::Value *Exchange = CGF.Builder.CreateOr(ExchangeHigh, ExchangeLow);

  // Load the comparand for the instruction.
  llvm::Value *Comparand = CGF.Builder.CreateLoad(ComparandAddr);

  auto *CXI = CGF.Builder.CreateAtomicCmpXchg(DestAddr, Comparand, Exchange,
                                              SuccessOrdering, FailureOrdering);

  // The atomic instruction is marked volatile for consistency with MSVC. This
  // blocks the few atomics optimizations that LLVM has. If we want to optimize
  // _Interlocked* operations in the future, we will have to remove the volatile
  // marker.
  CXI->setVolatile(true);

  // Store the result as an outparameter.
  CGF.Builder.CreateStore(CGF.Builder.CreateExtractValue(CXI, 0),
                          ComparandAddr);

  // Get the success boolean and zero extend it to i8.
  Value *Success = CGF.Builder.CreateExtractValue(CXI, 1);
  return CGF.Builder.CreateZExt(Success, CGF.Int8Ty);
}

static Value *EmitAtomicIncrementValue(CodeGenFunction &CGF, const CallExpr *E,
    AtomicOrdering Ordering = AtomicOrdering::SequentiallyConsistent) {
  assert(E->getArg(0)->getType()->isPointerType());

  auto *IntTy = CGF.ConvertType(E->getType());
  Address DestAddr = CheckAtomicAlignment(CGF, E);
  auto *Result = CGF.Builder.CreateAtomicRMW(
      AtomicRMWInst::Add, DestAddr, ConstantInt::get(IntTy, 1), Ordering);
  return CGF.Builder.CreateAdd(Result, ConstantInt::get(IntTy, 1));
}

static Value *EmitAtomicDecrementValue(
    CodeGenFunction &CGF, const CallExpr *E,
    AtomicOrdering Ordering = AtomicOrdering::SequentiallyConsistent) {
  assert(E->getArg(0)->getType()->isPointerType());

  auto *IntTy = CGF.ConvertType(E->getType());
  Address DestAddr = CheckAtomicAlignment(CGF, E);
  auto *Result = CGF.Builder.CreateAtomicRMW(
      AtomicRMWInst::Sub, DestAddr, ConstantInt::get(IntTy, 1), Ordering);
  return CGF.Builder.CreateSub(Result, ConstantInt::get(IntTy, 1));
}

// Build a plain volatile load.
static Value *EmitISOVolatileLoad(CodeGenFunction &CGF, const CallExpr *E) {
  Value *Ptr = CGF.EmitScalarExpr(E->getArg(0));
  QualType ElTy = E->getArg(0)->getType()->getPointeeType();
  CharUnits LoadSize = CGF.getContext().getTypeSizeInChars(ElTy);
  llvm::Type *ITy =
      llvm::IntegerType::get(CGF.getLLVMContext(), LoadSize.getQuantity() * 8);
  llvm::LoadInst *Load = CGF.Builder.CreateAlignedLoad(ITy, Ptr, LoadSize);
  Load->setVolatile(true);
  return Load;
}

// Build a plain volatile store.
static Value *EmitISOVolatileStore(CodeGenFunction &CGF, const CallExpr *E) {
  Value *Ptr = CGF.EmitScalarExpr(E->getArg(0));
  Value *Value = CGF.EmitScalarExpr(E->getArg(1));
  QualType ElTy = E->getArg(0)->getType()->getPointeeType();
  CharUnits StoreSize = CGF.getContext().getTypeSizeInChars(ElTy);
  llvm::StoreInst *Store =
      CGF.Builder.CreateAlignedStore(Value, Ptr, StoreSize);
  Store->setVolatile(true);
  return Store;
}

// Emit a simple mangled intrinsic that has 1 argument and a return type
// matching the argument type. Depending on mode, this may be a constrained
// floating-point intrinsic.
Value *emitUnaryMaybeConstrainedFPBuiltin(CodeGenFunction &CGF,
                                const CallExpr *E, unsigned IntrinsicID,
                                unsigned ConstrainedIntrinsicID) {
  llvm::Value *Src0 = CGF.EmitScalarExpr(E->getArg(0));

  CodeGenFunction::CGFPOptionsRAII FPOptsRAII(CGF, E);
  if (CGF.Builder.getIsFPConstrained()) {
    Function *F = CGF.CGM.getIntrinsic(ConstrainedIntrinsicID, Src0->getType());
    return CGF.Builder.CreateConstrainedFPCall(F, { Src0 });
  } else {
    Function *F = CGF.CGM.getIntrinsic(IntrinsicID, Src0->getType());
    return CGF.Builder.CreateCall(F, Src0);
  }
}

// Emit an intrinsic that has 2 operands of the same type as its result.
// Depending on mode, this may be a constrained floating-point intrinsic.
static Value *emitBinaryMaybeConstrainedFPBuiltin(CodeGenFunction &CGF,
                                const CallExpr *E, unsigned IntrinsicID,
                                unsigned ConstrainedIntrinsicID) {
  llvm::Value *Src0 = CGF.EmitScalarExpr(E->getArg(0));
  llvm::Value *Src1 = CGF.EmitScalarExpr(E->getArg(1));

  CodeGenFunction::CGFPOptionsRAII FPOptsRAII(CGF, E);
  if (CGF.Builder.getIsFPConstrained()) {
    Function *F = CGF.CGM.getIntrinsic(ConstrainedIntrinsicID, Src0->getType());
    return CGF.Builder.CreateConstrainedFPCall(F, { Src0, Src1 });
  } else {
    Function *F = CGF.CGM.getIntrinsic(IntrinsicID, Src0->getType());
    return CGF.Builder.CreateCall(F, { Src0, Src1 });
  }
}

// Has second type mangled argument.
static Value *
emitBinaryExpMaybeConstrainedFPBuiltin(CodeGenFunction &CGF, const CallExpr *E,
                                       Intrinsic::ID IntrinsicID,
                                       Intrinsic::ID ConstrainedIntrinsicID) {
  llvm::Value *Src0 = CGF.EmitScalarExpr(E->getArg(0));
  llvm::Value *Src1 = CGF.EmitScalarExpr(E->getArg(1));

  CodeGenFunction::CGFPOptionsRAII FPOptsRAII(CGF, E);
  if (CGF.Builder.getIsFPConstrained()) {
    Function *F = CGF.CGM.getIntrinsic(ConstrainedIntrinsicID,
                                       {Src0->getType(), Src1->getType()});
    return CGF.Builder.CreateConstrainedFPCall(F, {Src0, Src1});
  }

  Function *F =
      CGF.CGM.getIntrinsic(IntrinsicID, {Src0->getType(), Src1->getType()});
  return CGF.Builder.CreateCall(F, {Src0, Src1});
}

// Emit an intrinsic that has 3 operands of the same type as its result.
// Depending on mode, this may be a constrained floating-point intrinsic.
static Value *emitTernaryMaybeConstrainedFPBuiltin(CodeGenFunction &CGF,
                                 const CallExpr *E, unsigned IntrinsicID,
                                 unsigned ConstrainedIntrinsicID) {
  llvm::Value *Src0 = CGF.EmitScalarExpr(E->getArg(0));
  llvm::Value *Src1 = CGF.EmitScalarExpr(E->getArg(1));
  llvm::Value *Src2 = CGF.EmitScalarExpr(E->getArg(2));

  CodeGenFunction::CGFPOptionsRAII FPOptsRAII(CGF, E);
  if (CGF.Builder.getIsFPConstrained()) {
    Function *F = CGF.CGM.getIntrinsic(ConstrainedIntrinsicID, Src0->getType());
    return CGF.Builder.CreateConstrainedFPCall(F, { Src0, Src1, Src2 });
  } else {
    Function *F = CGF.CGM.getIntrinsic(IntrinsicID, Src0->getType());
    return CGF.Builder.CreateCall(F, { Src0, Src1, Src2 });
  }
}

// Emit an intrinsic that has overloaded integer result and fp operand.
static Value *
emitMaybeConstrainedFPToIntRoundBuiltin(CodeGenFunction &CGF, const CallExpr *E,
                                        unsigned IntrinsicID,
                                        unsigned ConstrainedIntrinsicID) {
  llvm::Type *ResultType = CGF.ConvertType(E->getType());
  llvm::Value *Src0 = CGF.EmitScalarExpr(E->getArg(0));

  if (CGF.Builder.getIsFPConstrained()) {
    CodeGenFunction::CGFPOptionsRAII FPOptsRAII(CGF, E);
    Function *F = CGF.CGM.getIntrinsic(ConstrainedIntrinsicID,
                                       {ResultType, Src0->getType()});
    return CGF.Builder.CreateConstrainedFPCall(F, {Src0});
  } else {
    Function *F =
        CGF.CGM.getIntrinsic(IntrinsicID, {ResultType, Src0->getType()});
    return CGF.Builder.CreateCall(F, Src0);
  }
}

static Value *emitFrexpBuiltin(CodeGenFunction &CGF, const CallExpr *E,
                               Intrinsic::ID IntrinsicID) {
  llvm::Value *Src0 = CGF.EmitScalarExpr(E->getArg(0));
  llvm::Value *Src1 = CGF.EmitScalarExpr(E->getArg(1));

  QualType IntPtrTy = E->getArg(1)->getType()->getPointeeType();
  llvm::Type *IntTy = CGF.ConvertType(IntPtrTy);
  llvm::Function *F =
      CGF.CGM.getIntrinsic(IntrinsicID, {Src0->getType(), IntTy});
  llvm::Value *Call = CGF.Builder.CreateCall(F, Src0);

  llvm::Value *Exp = CGF.Builder.CreateExtractValue(Call, 1);
  LValue LV = CGF.MakeNaturalAlignAddrLValue(Src1, IntPtrTy);
  CGF.EmitStoreOfScalar(Exp, LV);

  return CGF.Builder.CreateExtractValue(Call, 0);
}

static void emitSincosBuiltin(CodeGenFunction &CGF, const CallExpr *E,
                              Intrinsic::ID IntrinsicID) {
  llvm::Value *Val = CGF.EmitScalarExpr(E->getArg(0));
  llvm::Value *Dest0 = CGF.EmitScalarExpr(E->getArg(1));
  llvm::Value *Dest1 = CGF.EmitScalarExpr(E->getArg(2));

  llvm::Function *F = CGF.CGM.getIntrinsic(IntrinsicID, {Val->getType()});
  llvm::Value *Call = CGF.Builder.CreateCall(F, Val);

  llvm::Value *SinResult = CGF.Builder.CreateExtractValue(Call, 0);
  llvm::Value *CosResult = CGF.Builder.CreateExtractValue(Call, 1);

  QualType DestPtrType = E->getArg(1)->getType()->getPointeeType();
  LValue SinLV = CGF.MakeNaturalAlignAddrLValue(Dest0, DestPtrType);
  LValue CosLV = CGF.MakeNaturalAlignAddrLValue(Dest1, DestPtrType);

  llvm::StoreInst *StoreSin =
      CGF.Builder.CreateStore(SinResult, SinLV.getAddress());
  llvm::StoreInst *StoreCos =
      CGF.Builder.CreateStore(CosResult, CosLV.getAddress());

  // Mark the two stores as non-aliasing with each other. The order of stores
  // emitted by this builtin is arbitrary, enforcing a particular order will
  // prevent optimizations later on.
  llvm::MDBuilder MDHelper(CGF.getLLVMContext());
  MDNode *Domain = MDHelper.createAnonymousAliasScopeDomain();
  MDNode *AliasScope = MDHelper.createAnonymousAliasScope(Domain);
  MDNode *AliasScopeList = MDNode::get(Call->getContext(), AliasScope);
  StoreSin->setMetadata(LLVMContext::MD_alias_scope, AliasScopeList);
  StoreCos->setMetadata(LLVMContext::MD_noalias, AliasScopeList);
}

static llvm::Value *emitModfBuiltin(CodeGenFunction &CGF, const CallExpr *E,
                                    Intrinsic::ID IntrinsicID) {
  llvm::Value *Val = CGF.EmitScalarExpr(E->getArg(0));
  llvm::Value *IntPartDest = CGF.EmitScalarExpr(E->getArg(1));

  llvm::Value *Call =
      CGF.Builder.CreateIntrinsic(IntrinsicID, {Val->getType()}, Val);

  llvm::Value *FractionalResult = CGF.Builder.CreateExtractValue(Call, 0);
  llvm::Value *IntegralResult = CGF.Builder.CreateExtractValue(Call, 1);

  QualType DestPtrType = E->getArg(1)->getType()->getPointeeType();
  LValue IntegralLV = CGF.MakeNaturalAlignAddrLValue(IntPartDest, DestPtrType);
  CGF.EmitStoreOfScalar(IntegralResult, IntegralLV);

  return FractionalResult;
}

/// EmitFAbs - Emit a call to @llvm.fabs().
static Value *EmitFAbs(CodeGenFunction &CGF, Value *V) {
  Function *F = CGF.CGM.getIntrinsic(Intrinsic::fabs, V->getType());
  llvm::CallInst *Call = CGF.Builder.CreateCall(F, V);
  Call->setDoesNotAccessMemory();
  return Call;
}

/// Emit the computation of the sign bit for a floating point value. Returns
/// the i1 sign bit value.
static Value *EmitSignBit(CodeGenFunction &CGF, Value *V) {
  LLVMContext &C = CGF.CGM.getLLVMContext();

  llvm::Type *Ty = V->getType();
  int Width = Ty->getPrimitiveSizeInBits();
  llvm::Type *IntTy = llvm::IntegerType::get(C, Width);
  V = CGF.Builder.CreateBitCast(V, IntTy);
  if (Ty->isPPC_FP128Ty()) {
    // We want the sign bit of the higher-order double. The bitcast we just
    // did works as if the double-double was stored to memory and then
    // read as an i128. The "store" will put the higher-order double in the
    // lower address in both little- and big-Endian modes, but the "load"
    // will treat those bits as a different part of the i128: the low bits in
    // little-Endian, the high bits in big-Endian. Therefore, on big-Endian
    // we need to shift the high bits down to the low before truncating.
    Width >>= 1;
    if (CGF.getTarget().isBigEndian()) {
      Value *ShiftCst = llvm::ConstantInt::get(IntTy, Width);
      V = CGF.Builder.CreateLShr(V, ShiftCst);
    }
    // We are truncating value in order to extract the higher-order
    // double, which we will be using to extract the sign from.
    IntTy = llvm::IntegerType::get(C, Width);
    V = CGF.Builder.CreateTrunc(V, IntTy);
  }
  Value *Zero = llvm::Constant::getNullValue(IntTy);
  return CGF.Builder.CreateICmpSLT(V, Zero);
}

/// Checks no arguments or results are passed indirectly in the ABI (i.e. via a
/// hidden pointer). This is used to check annotating FP libcalls (that could
/// set `errno`) with "int" TBAA metadata is safe. If any floating-point
/// arguments are passed indirectly, setup for the call could be incorrectly
/// optimized out.
static bool HasNoIndirectArgumentsOrResults(CGFunctionInfo const &FnInfo) {
  auto IsIndirect = [&](ABIArgInfo const &info) {
    return info.isIndirect() || info.isIndirectAliased() || info.isInAlloca();
  };
  return !IsIndirect(FnInfo.getReturnInfo()) &&
         llvm::none_of(FnInfo.arguments(),
                       [&](CGFunctionInfoArgInfo const &ArgInfo) {
                         return IsIndirect(ArgInfo.info);
                       });
}

static RValue emitLibraryCall(CodeGenFunction &CGF, const FunctionDecl *FD,
                              const CallExpr *E, llvm::Constant *calleeValue) {
  CodeGenFunction::CGFPOptionsRAII FPOptsRAII(CGF, E);
  CGCallee callee = CGCallee::forDirect(calleeValue, GlobalDecl(FD));
  llvm::CallBase *callOrInvoke = nullptr;
  CGFunctionInfo const *FnInfo = nullptr;
  RValue Call =
      CGF.EmitCall(E->getCallee()->getType(), callee, E, ReturnValueSlot(),
                   /*Chain=*/nullptr, &callOrInvoke, &FnInfo);

  if (unsigned BuiltinID = FD->getBuiltinID()) {
    // Check whether a FP math builtin function, such as BI__builtin_expf
    ASTContext &Context = CGF.getContext();
    bool ConstWithoutErrnoAndExceptions =
        Context.BuiltinInfo.isConstWithoutErrnoAndExceptions(BuiltinID);
    // Restrict to target with errno, for example, MacOS doesn't set errno.
    // TODO: Support builtin function with complex type returned, eg: cacosh
    if (ConstWithoutErrnoAndExceptions && CGF.CGM.getLangOpts().MathErrno &&
        !CGF.Builder.getIsFPConstrained() && Call.isScalar() &&
        HasNoIndirectArgumentsOrResults(*FnInfo)) {
      // Emit "int" TBAA metadata on FP math libcalls.
      clang::QualType IntTy = Context.IntTy;
      TBAAAccessInfo TBAAInfo = CGF.CGM.getTBAAAccessInfo(IntTy);
      CGF.CGM.DecorateInstructionWithTBAA(callOrInvoke, TBAAInfo);
    }
  }
  return Call;
}

/// Emit a call to llvm.{sadd,uadd,ssub,usub,smul,umul}.with.overflow.*
/// depending on IntrinsicID.
///
/// \arg CGF The current codegen function.
/// \arg IntrinsicID The ID for the Intrinsic we wish to generate.
/// \arg X The first argument to the llvm.*.with.overflow.*.
/// \arg Y The second argument to the llvm.*.with.overflow.*.
/// \arg Carry The carry returned by the llvm.*.with.overflow.*.
/// \returns The result (i.e. sum/product) returned by the intrinsic.
llvm::Value *EmitOverflowIntrinsic(CodeGenFunction &CGF,
                                   const Intrinsic::ID IntrinsicID,
                                   llvm::Value *X, llvm::Value *Y,
                                   llvm::Value *&Carry) {
  // Make sure we have integers of the same width.
  assert(X->getType() == Y->getType() &&
         "Arguments must be the same type. (Did you forget to make sure both "
         "arguments have the same integer width?)");

  Function *Callee = CGF.CGM.getIntrinsic(IntrinsicID, X->getType());
  llvm::Value *Tmp = CGF.Builder.CreateCall(Callee, {X, Y});
  Carry = CGF.Builder.CreateExtractValue(Tmp, 1);
  return CGF.Builder.CreateExtractValue(Tmp, 0);
}

namespace {
  struct WidthAndSignedness {
    unsigned Width;
    bool Signed;
  };
}

static WidthAndSignedness
getIntegerWidthAndSignedness(const clang::ASTContext &context,
                             const clang::QualType Type) {
  assert(Type->isIntegerType() && "Given type is not an integer.");
  unsigned Width = context.getIntWidth(Type);
  bool Signed = Type->isSignedIntegerType();
  return {Width, Signed};
}

// Given one or more integer types, this function produces an integer type that
// encompasses them: any value in one of the given types could be expressed in
// the encompassing type.
static struct WidthAndSignedness
EncompassingIntegerType(ArrayRef<struct WidthAndSignedness> Types) {
  assert(Types.size() > 0 && "Empty list of types.");

  // If any of the given types is signed, we must return a signed type.
  bool Signed = false;
  for (const auto &Type : Types) {
    Signed |= Type.Signed;
  }

  // The encompassing type must have a width greater than or equal to the width
  // of the specified types.  Additionally, if the encompassing type is signed,
  // its width must be strictly greater than the width of any unsigned types
  // given.
  unsigned Width = 0;
  for (const auto &Type : Types) {
    unsigned MinWidth = Type.Width + (Signed && !Type.Signed);
    if (Width < MinWidth) {
      Width = MinWidth;
    }
  }

  return {Width, Signed};
}

Value *CodeGenFunction::EmitVAStartEnd(Value *ArgValue, bool IsStart) {
  Intrinsic::ID inst = IsStart ? Intrinsic::vastart : Intrinsic::vaend;
  return Builder.CreateCall(CGM.getIntrinsic(inst, {ArgValue->getType()}),
                            ArgValue);
}

/// Checks if using the result of __builtin_object_size(p, @p From) in place of
/// __builtin_object_size(p, @p To) is correct
static bool areBOSTypesCompatible(int From, int To) {
  // Note: Our __builtin_object_size implementation currently treats Type=0 and
  // Type=2 identically. Encoding this implementation detail here may make
  // improving __builtin_object_size difficult in the future, so it's omitted.
  return From == To || (From == 0 && To == 1) || (From == 3 && To == 2);
}

static llvm::Value *
getDefaultBuiltinObjectSizeResult(unsigned Type, llvm::IntegerType *ResType) {
  return ConstantInt::get(ResType, (Type & 2) ? 0 : -1, /*isSigned=*/true);
}

llvm::Value *
CodeGenFunction::evaluateOrEmitBuiltinObjectSize(const Expr *E, unsigned Type,
                                                 llvm::IntegerType *ResType,
                                                 llvm::Value *EmittedE,
                                                 bool IsDynamic) {
  uint64_t ObjectSize;
  if (!E->tryEvaluateObjectSize(ObjectSize, getContext(), Type))
    return emitBuiltinObjectSize(E, Type, ResType, EmittedE, IsDynamic);
  return ConstantInt::get(ResType, ObjectSize, /*isSigned=*/true);
}

namespace {

/// StructFieldAccess is a simple visitor class to grab the first MemberExpr
/// from an Expr. It records any ArraySubscriptExpr we meet along the way.
class StructFieldAccess
    : public ConstStmtVisitor<StructFieldAccess, const Expr *> {
  bool AddrOfSeen = false;

public:
  const Expr *ArrayIndex = nullptr;
  QualType ArrayElementTy;

  const Expr *VisitMemberExpr(const MemberExpr *E) {
    if (AddrOfSeen && E->getType()->isArrayType())
      // Avoid forms like '&ptr->array'.
      return nullptr;
    return E;
  }

  const Expr *VisitArraySubscriptExpr(const ArraySubscriptExpr *E) {
    if (ArrayIndex)
      // We don't support multiple subscripts.
      return nullptr;

    AddrOfSeen = false; // '&ptr->array[idx]' is okay.
    ArrayIndex = E->getIdx();
    ArrayElementTy = E->getBase()->getType();
    return Visit(E->getBase());
  }
  const Expr *VisitCastExpr(const CastExpr *E) {
    if (E->getCastKind() == CK_LValueToRValue)
      return E;
    return Visit(E->getSubExpr());
  }
  const Expr *VisitParenExpr(const ParenExpr *E) {
    return Visit(E->getSubExpr());
  }
  const Expr *VisitUnaryAddrOf(const clang::UnaryOperator *E) {
    AddrOfSeen = true;
    return Visit(E->getSubExpr());
  }
  const Expr *VisitUnaryDeref(const clang::UnaryOperator *E) {
    AddrOfSeen = false;
    return Visit(E->getSubExpr());
  }
  const Expr *VisitBinaryOperator(const clang::BinaryOperator *Op) {
    return Op->isCommaOp() ? Visit(Op->getRHS()) : nullptr;
  }
};

} // end anonymous namespace

/// Find a struct's flexible array member. It may be embedded inside multiple
/// sub-structs, but must still be the last field.
static const FieldDecl *FindFlexibleArrayMemberField(CodeGenFunction &CGF,
                                                     ASTContext &Ctx,
                                                     const RecordDecl *RD) {
  const LangOptions::StrictFlexArraysLevelKind StrictFlexArraysLevel =
      CGF.getLangOpts().getStrictFlexArraysLevel();

  if (RD->isImplicit())
    return nullptr;

  for (const FieldDecl *FD : RD->fields()) {
    if (Decl::isFlexibleArrayMemberLike(
            Ctx, FD, FD->getType(), StrictFlexArraysLevel,
            /*IgnoreTemplateOrMacroSubstitution=*/true))
      return FD;

    if (const auto *RD = FD->getType()->getAsRecordDecl())
      if (const FieldDecl *FD = FindFlexibleArrayMemberField(CGF, Ctx, RD))
        return FD;
  }

  return nullptr;
}

/// Calculate the offset of a struct field. It may be embedded inside multiple
/// sub-structs.
static bool GetFieldOffset(ASTContext &Ctx, const RecordDecl *RD,
                           const FieldDecl *FD, int64_t &Offset) {
  if (RD->isImplicit())
    return false;

  // Keep track of the field number ourselves, because the other methods
  // (CGRecordLayout::getLLVMFieldNo) aren't always equivalent to how the AST
  // is laid out.
  uint32_t FieldNo = 0;
  const ASTRecordLayout &Layout = Ctx.getASTRecordLayout(RD);

  for (const FieldDecl *Field : RD->fields()) {
    if (Field == FD) {
      Offset += Layout.getFieldOffset(FieldNo);
      return true;
    }

    if (const auto *RD = Field->getType()->getAsRecordDecl()) {
      if (GetFieldOffset(Ctx, RD, FD, Offset)) {
        Offset += Layout.getFieldOffset(FieldNo);
        return true;
      }
    }

    if (!RD->isUnion())
      ++FieldNo;
  }

  return false;
}

static std::optional<int64_t>
GetFieldOffset(ASTContext &Ctx, const RecordDecl *RD, const FieldDecl *FD) {
  int64_t Offset = 0;

  if (GetFieldOffset(Ctx, RD, FD, Offset))
    return std::optional<int64_t>(Offset);

  return std::nullopt;
}

llvm::Value *CodeGenFunction::emitCountedBySize(const Expr *E,
                                                llvm::Value *EmittedE,
                                                unsigned Type,
                                                llvm::IntegerType *ResType) {
  // Note: If the whole struct is specificed in the __bdos (i.e. Visitor
  // returns a DeclRefExpr). The calculation of the whole size of the structure
  // with a flexible array member can be done in two ways:
  //
  //     1) sizeof(struct S) + count * sizeof(typeof(fam))
  //     2) offsetof(struct S, fam) + count * sizeof(typeof(fam))
  //
  // The first will add additional padding after the end of the array
  // allocation while the second method is more precise, but not quite expected
  // from programmers. See
  // https://lore.kernel.org/lkml/ZvV6X5FPBBW7CO1f@archlinux/ for a discussion
  // of the topic.
  //
  // GCC isn't (currently) able to calculate __bdos on a pointer to the whole
  // structure. Therefore, because of the above issue, we choose to match what
  // GCC does for consistency's sake.

  StructFieldAccess Visitor;
  E = Visitor.Visit(E);
  if (!E)
    return nullptr;

  const Expr *Idx = Visitor.ArrayIndex;
  if (Idx) {
    if (Idx->HasSideEffects(getContext()))
      // We can't have side-effects.
      return getDefaultBuiltinObjectSizeResult(Type, ResType);

    if (const auto *IL = dyn_cast<IntegerLiteral>(Idx)) {
      int64_t Val = IL->getValue().getSExtValue();
      if (Val < 0)
        return getDefaultBuiltinObjectSizeResult(Type, ResType);

      // The index is 0, so we don't need to take it into account.
      if (Val == 0)
        Idx = nullptr;
    }
  }

  // __counted_by on either a flexible array member or a pointer into a struct
  // with a flexible array member.
  if (const auto *ME = dyn_cast<MemberExpr>(E))
    return emitCountedByMemberSize(ME, Idx, EmittedE, Visitor.ArrayElementTy,
                                   Type, ResType);

  // __counted_by on a pointer in a struct.
  if (const auto *ICE = dyn_cast<ImplicitCastExpr>(E);
      ICE && ICE->getCastKind() == CK_LValueToRValue)
    return emitCountedByPointerSize(ICE, Idx, EmittedE, Visitor.ArrayElementTy,
                                    Type, ResType);

  return nullptr;
}

static llvm::Value *EmitPositiveResultOrZero(CodeGenFunction &CGF,
                                             llvm::Value *Res,
                                             llvm::Value *Index,
                                             llvm::IntegerType *ResType,
                                             bool IsSigned) {
  //  cmp = (array_size >= 0)
  Value *Cmp = CGF.Builder.CreateIsNotNeg(Res);
  if (Index)
    //  cmp = (cmp && index >= 0)
    Cmp = CGF.Builder.CreateAnd(CGF.Builder.CreateIsNotNeg(Index), Cmp);

  //  return cmp ? result : 0
  return CGF.Builder.CreateSelect(Cmp, Res,
                                  ConstantInt::get(ResType, 0, IsSigned));
}

static std::pair<llvm::Value *, llvm::Value *>
GetCountFieldAndIndex(CodeGenFunction &CGF, const MemberExpr *ME,
                      const FieldDecl *ArrayFD, const FieldDecl *CountFD,
                      const Expr *Idx, llvm::IntegerType *ResType,
                      bool IsSigned) {
  //  count = ptr->count;
  Value *Count = CGF.EmitLoadOfCountedByField(ME, ArrayFD, CountFD);
  if (!Count)
    return std::make_pair<Value *>(nullptr, nullptr);
  Count = CGF.Builder.CreateIntCast(Count, ResType, IsSigned, "count");

  //  index = ptr->index;
  Value *Index = nullptr;
  if (Idx) {
    bool IdxSigned = Idx->getType()->isSignedIntegerType();
    Index = CGF.EmitScalarExpr(Idx);
    Index = CGF.Builder.CreateIntCast(Index, ResType, IdxSigned, "index");
  }

  return std::make_pair(Count, Index);
}

llvm::Value *CodeGenFunction::emitCountedByPointerSize(
    const ImplicitCastExpr *E, const Expr *Idx, llvm::Value *EmittedE,
    QualType CastedArrayElementTy, unsigned Type, llvm::IntegerType *ResType) {
  assert(E->getCastKind() == CK_LValueToRValue &&
         "must be an LValue to RValue cast");

  const MemberExpr *ME =
      dyn_cast<MemberExpr>(E->getSubExpr()->IgnoreParenNoopCasts(getContext()));
  if (!ME)
    return nullptr;

  const auto *ArrayBaseFD = dyn_cast<FieldDecl>(ME->getMemberDecl());
  if (!ArrayBaseFD || !ArrayBaseFD->getType()->isPointerType() ||
      !ArrayBaseFD->getType()->isCountAttributedType())
    return nullptr;

  // Get the 'count' FieldDecl.
  const FieldDecl *CountFD = ArrayBaseFD->findCountedByField();
  if (!CountFD)
    // Can't find the field referenced by the "counted_by" attribute.
    return nullptr;

  // Calculate the array's object size using these formulae. (Note: if the
  // calculation is negative, we return 0.):
  //
  //      struct p;
  //      struct s {
  //          /* ... */
  //          struct p **array __attribute__((counted_by(count)));
  //          int count;
  //      };
  //
  // 1) 'ptr->array':
  //
  //    count = ptr->count;
  //
  //    array_element_size = sizeof (*ptr->array);
  //    array_size = count * array_element_size;
  //
  //    result = array_size;
  //
  //    cmp = (result >= 0)
  //    return cmp ? result : 0;
  //
  // 2) '&((cast) ptr->array)[idx]':
  //
  //    count = ptr->count;
  //    index = idx;
  //
  //    array_element_size = sizeof (*ptr->array);
  //    array_size = count * array_element_size;
  //
  //    casted_array_element_size = sizeof (*((cast) ptr->array));
  //
  //    index_size = index * casted_array_element_size;
  //    result = array_size - index_size;
  //
  //    cmp = (result >= 0)
  //    if (index)
  //        cmp  = (cmp && index > 0)
  //    return cmp ? result : 0;

  auto GetElementBaseSize = [&](QualType ElementTy) {
    CharUnits ElementSize =
        getContext().getTypeSizeInChars(ElementTy->getPointeeType());

    if (ElementSize.isZero()) {
      // This might be a __sized_by on a 'void *', which counts bytes, not
      // elements.
      auto *CAT = ElementTy->getAs<CountAttributedType>();
      if (!CAT || (CAT->getKind() != CountAttributedType::SizedBy &&
                   CAT->getKind() != CountAttributedType::SizedByOrNull))
        // Okay, not sure what it is now.
        // FIXME: Should this be an assert?
        return std::optional<CharUnits>();

      ElementSize = CharUnits::One();
    }

    return std::optional<CharUnits>(ElementSize);
  };

  // Get the sizes of the original array element and the casted array element,
  // if different.
  std::optional<CharUnits> ArrayElementBaseSize =
      GetElementBaseSize(ArrayBaseFD->getType());
  if (!ArrayElementBaseSize)
    return nullptr;

  std::optional<CharUnits> CastedArrayElementBaseSize = ArrayElementBaseSize;
  if (!CastedArrayElementTy.isNull() && CastedArrayElementTy->isPointerType()) {
    CastedArrayElementBaseSize = GetElementBaseSize(CastedArrayElementTy);
    if (!CastedArrayElementBaseSize)
      return nullptr;
  }

  bool IsSigned = CountFD->getType()->isSignedIntegerType();

  //  count = ptr->count;
  //  index = ptr->index;
  Value *Count, *Index;
  std::tie(Count, Index) = GetCountFieldAndIndex(
      *this, ME, ArrayBaseFD, CountFD, Idx, ResType, IsSigned);
  if (!Count)
    return nullptr;

  //  array_element_size = sizeof (*ptr->array)
  auto *ArrayElementSize = llvm::ConstantInt::get(
      ResType, ArrayElementBaseSize->getQuantity(), IsSigned);

  //  casted_array_element_size = sizeof (*((cast) ptr->array));
  auto *CastedArrayElementSize = llvm::ConstantInt::get(
      ResType, CastedArrayElementBaseSize->getQuantity(), IsSigned);

  //  array_size = count * array_element_size;
  Value *ArraySize = Builder.CreateMul(Count, ArrayElementSize, "array_size",
                                       !IsSigned, IsSigned);

  // Option (1) 'ptr->array'
  //  result = array_size
  Value *Result = ArraySize;

  if (Idx) { // Option (2) '&((cast) ptr->array)[idx]'
    //  index_size = index * casted_array_element_size;
    Value *IndexSize = Builder.CreateMul(Index, CastedArrayElementSize,
                                         "index_size", !IsSigned, IsSigned);

    //  result = result - index_size;
    Result =
        Builder.CreateSub(Result, IndexSize, "result", !IsSigned, IsSigned);
  }

  return EmitPositiveResultOrZero(*this, Result, Index, ResType, IsSigned);
}

llvm::Value *CodeGenFunction::emitCountedByMemberSize(
    const MemberExpr *ME, const Expr *Idx, llvm::Value *EmittedE,
    QualType CastedArrayElementTy, unsigned Type, llvm::IntegerType *ResType) {
  const auto *FD = dyn_cast<FieldDecl>(ME->getMemberDecl());
  if (!FD)
    return nullptr;

  // Find the flexible array member and check that it has the __counted_by
  // attribute.
  ASTContext &Ctx = getContext();
  const RecordDecl *RD = FD->getDeclContext()->getOuterLexicalRecordContext();
  const FieldDecl *FlexibleArrayMemberFD = nullptr;

  if (Decl::isFlexibleArrayMemberLike(
          Ctx, FD, FD->getType(), getLangOpts().getStrictFlexArraysLevel(),
          /*IgnoreTemplateOrMacroSubstitution=*/true))
    FlexibleArrayMemberFD = FD;
  else
    FlexibleArrayMemberFD = FindFlexibleArrayMemberField(*this, Ctx, RD);

  if (!FlexibleArrayMemberFD ||
      !FlexibleArrayMemberFD->getType()->isCountAttributedType())
    return nullptr;

  // Get the 'count' FieldDecl.
  const FieldDecl *CountFD = FlexibleArrayMemberFD->findCountedByField();
  if (!CountFD)
    // Can't find the field referenced by the "counted_by" attribute.
    return nullptr;

  // Calculate the flexible array member's object size using these formulae.
  // (Note: if the calculation is negative, we return 0.):
  //
  //      struct p;
  //      struct s {
  //          /* ... */
  //          int count;
  //          struct p *array[] __attribute__((counted_by(count)));
  //      };
  //
  // 1) 'ptr->array':
  //
  //    count = ptr->count;
  //
  //    flexible_array_member_element_size = sizeof (*ptr->array);
  //    flexible_array_member_size =
  //        count * flexible_array_member_element_size;
  //
  //    result = flexible_array_member_size;
  //
  //    cmp = (result >= 0)
  //    return cmp ? result : 0;
  //
  // 2) '&((cast) ptr->array)[idx]':
  //
  //    count = ptr->count;
  //    index = idx;
  //
  //    flexible_array_member_element_size = sizeof (*ptr->array);
  //    flexible_array_member_size =
  //        count * flexible_array_member_element_size;
  //
  //    casted_flexible_array_member_element_size =
  //        sizeof (*((cast) ptr->array));
  //    index_size = index * casted_flexible_array_member_element_size;
  //
  //    result = flexible_array_member_size - index_size;
  //
  //    cmp = (result >= 0)
  //    if (index != 0)
  //        cmp = (cmp && index >= 0)
  //    return cmp ? result : 0;
  //
  // 3) '&ptr->field':
  //
  //    count = ptr->count;
  //    sizeof_struct = sizeof (struct s);
  //
  //    flexible_array_member_element_size = sizeof (*ptr->array);
  //    flexible_array_member_size =
  //        count * flexible_array_member_element_size;
  //
  //    field_offset = offsetof (struct s, field);
  //    offset_diff = sizeof_struct - field_offset;
  //
  //    result = offset_diff + flexible_array_member_size;
  //
  //    cmp = (result >= 0)
  //    return cmp ? result : 0;
  //
  // 4) '&((cast) ptr->field_array)[idx]':
  //
  //    count = ptr->count;
  //    index = idx;
  //    sizeof_struct = sizeof (struct s);
  //
  //    flexible_array_member_element_size = sizeof (*ptr->array);
  //    flexible_array_member_size =
  //        count * flexible_array_member_element_size;
  //
  //    casted_field_element_size = sizeof (*((cast) ptr->field_array));
  //    field_offset = offsetof (struct s, field)
  //    field_offset += index * casted_field_element_size;
  //
  //    offset_diff = sizeof_struct - field_offset;
  //
  //    result = offset_diff + flexible_array_member_size;
  //
  //    cmp = (result >= 0)
  //    if (index != 0)
  //        cmp = (cmp && index >= 0)
  //    return cmp ? result : 0;

  bool IsSigned = CountFD->getType()->isSignedIntegerType();

  QualType FlexibleArrayMemberTy = FlexibleArrayMemberFD->getType();

  // Explicit cast because otherwise the CharWidth will promote an i32's into
  // u64's leading to overflows.
  int64_t CharWidth = static_cast<int64_t>(CGM.getContext().getCharWidth());

  //  field_offset = offsetof (struct s, field);
  Value *FieldOffset = nullptr;
  if (FlexibleArrayMemberFD != FD) {
    std::optional<int64_t> Offset = GetFieldOffset(Ctx, RD, FD);
    if (!Offset)
      return nullptr;
    FieldOffset =
        llvm::ConstantInt::get(ResType, *Offset / CharWidth, IsSigned);
  }

  //  count = ptr->count;
  //  index = ptr->index;
  Value *Count, *Index;
  std::tie(Count, Index) = GetCountFieldAndIndex(
      *this, ME, FlexibleArrayMemberFD, CountFD, Idx, ResType, IsSigned);
  if (!Count)
    return nullptr;

  //  flexible_array_member_element_size = sizeof (*ptr->array);
  const ArrayType *ArrayTy = Ctx.getAsArrayType(FlexibleArrayMemberTy);
  CharUnits BaseSize = Ctx.getTypeSizeInChars(ArrayTy->getElementType());
  auto *FlexibleArrayMemberElementSize =
      llvm::ConstantInt::get(ResType, BaseSize.getQuantity(), IsSigned);

  //  flexible_array_member_size = count * flexible_array_member_element_size;
  Value *FlexibleArrayMemberSize =
      Builder.CreateMul(Count, FlexibleArrayMemberElementSize,
                        "flexible_array_member_size", !IsSigned, IsSigned);

  Value *Result = nullptr;
  if (FlexibleArrayMemberFD == FD) {
    if (Idx) { // Option (2) '&((cast) ptr->array)[idx]'
      //  casted_flexible_array_member_element_size =
      //      sizeof (*((cast) ptr->array));
      llvm::ConstantInt *CastedFlexibleArrayMemberElementSize =
          FlexibleArrayMemberElementSize;
      if (!CastedArrayElementTy.isNull() &&
          CastedArrayElementTy->isPointerType()) {
        CharUnits BaseSize =
            Ctx.getTypeSizeInChars(CastedArrayElementTy->getPointeeType());
        CastedFlexibleArrayMemberElementSize =
            llvm::ConstantInt::get(ResType, BaseSize.getQuantity(), IsSigned);
      }

      //  index_size = index * casted_flexible_array_member_element_size;
      Value *IndexSize =
          Builder.CreateMul(Index, CastedFlexibleArrayMemberElementSize,
                            "index_size", !IsSigned, IsSigned);

      //  result = flexible_array_member_size - index_size;
      Result = Builder.CreateSub(FlexibleArrayMemberSize, IndexSize, "result",
                                 !IsSigned, IsSigned);
    } else { // Option (1) 'ptr->array'
      //  result = flexible_array_member_size;
      Result = FlexibleArrayMemberSize;
    }
  } else {
    //  sizeof_struct = sizeof (struct s);
    llvm::StructType *StructTy = getTypes().getCGRecordLayout(RD).getLLVMType();
    const llvm::DataLayout &Layout = CGM.getDataLayout();
    TypeSize Size = Layout.getTypeSizeInBits(StructTy);
    Value *SizeofStruct =
        llvm::ConstantInt::get(ResType, Size.getKnownMinValue() / CharWidth);

    if (Idx) { // Option (4) '&((cast) ptr->field_array)[idx]'
      //  casted_field_element_size = sizeof (*((cast) ptr->field_array));
      CharUnits BaseSize;
      if (!CastedArrayElementTy.isNull() &&
          CastedArrayElementTy->isPointerType()) {
        BaseSize =
            Ctx.getTypeSizeInChars(CastedArrayElementTy->getPointeeType());
      } else {
        const ArrayType *ArrayTy = Ctx.getAsArrayType(FD->getType());
        BaseSize = Ctx.getTypeSizeInChars(ArrayTy->getElementType());
      }

      llvm::ConstantInt *CastedFieldElementSize =
          llvm::ConstantInt::get(ResType, BaseSize.getQuantity(), IsSigned);

      //  field_offset += index * casted_field_element_size;
      Value *Mul = Builder.CreateMul(Index, CastedFieldElementSize,
                                     "field_offset", !IsSigned, IsSigned);
      FieldOffset = Builder.CreateAdd(FieldOffset, Mul);
    }
    // Option (3) '&ptr->field', and Option (4) continuation.
    //  offset_diff = flexible_array_member_offset - field_offset;
    Value *OffsetDiff = Builder.CreateSub(SizeofStruct, FieldOffset,
                                          "offset_diff", !IsSigned, IsSigned);

    //  result = offset_diff + flexible_array_member_size;
    Result = Builder.CreateAdd(FlexibleArrayMemberSize, OffsetDiff, "result");
  }

  return EmitPositiveResultOrZero(*this, Result, Index, ResType, IsSigned);
}

/// Returns a Value corresponding to the size of the given expression.
/// This Value may be either of the following:
///   - A llvm::Argument (if E is a param with the pass_object_size attribute on
///     it)
///   - A call to the @llvm.objectsize intrinsic
///
/// EmittedE is the result of emitting `E` as a scalar expr. If it's non-null
/// and we wouldn't otherwise try to reference a pass_object_size parameter,
/// we'll call @llvm.objectsize on EmittedE, rather than emitting E.
llvm::Value *
CodeGenFunction::emitBuiltinObjectSize(const Expr *E, unsigned Type,
                                       llvm::IntegerType *ResType,
                                       llvm::Value *EmittedE, bool IsDynamic) {
  // We need to reference an argument if the pointer is a parameter with the
  // pass_object_size attribute.
  if (auto *D = dyn_cast<DeclRefExpr>(E->IgnoreParenImpCasts())) {
    auto *Param = dyn_cast<ParmVarDecl>(D->getDecl());
    auto *PS = D->getDecl()->getAttr<PassObjectSizeAttr>();
    if (Param != nullptr && PS != nullptr &&
        areBOSTypesCompatible(PS->getType(), Type)) {
      auto Iter = SizeArguments.find(Param);
      assert(Iter != SizeArguments.end());

      const ImplicitParamDecl *D = Iter->second;
      auto DIter = LocalDeclMap.find(D);
      assert(DIter != LocalDeclMap.end());

      return EmitLoadOfScalar(DIter->second, /*Volatile=*/false,
                              getContext().getSizeType(), E->getBeginLoc());
    }
  }

  // LLVM can't handle Type=3 appropriately, and __builtin_object_size shouldn't
  // evaluate E for side-effects. In either case, we shouldn't lower to
  // @llvm.objectsize.
  if (Type == 3 || (!EmittedE && E->HasSideEffects(getContext())))
    return getDefaultBuiltinObjectSizeResult(Type, ResType);

  Value *Ptr = EmittedE ? EmittedE : EmitScalarExpr(E);
  assert(Ptr->getType()->isPointerTy() &&
         "Non-pointer passed to __builtin_object_size?");

  if (IsDynamic)
    // Emit special code for a flexible array member with the "counted_by"
    // attribute.
    if (Value *V = emitCountedBySize(E, Ptr, Type, ResType))
      return V;

  Function *F =
      CGM.getIntrinsic(Intrinsic::objectsize, {ResType, Ptr->getType()});

  // LLVM only supports 0 and 2, make sure that we pass along that as a boolean.
  Value *Min = Builder.getInt1((Type & 2) != 0);
  // For GCC compatibility, __builtin_object_size treat NULL as unknown size.
  Value *NullIsUnknown = Builder.getTrue();
  Value *Dynamic = Builder.getInt1(IsDynamic);
  return Builder.CreateCall(F, {Ptr, Min, NullIsUnknown, Dynamic});
}

namespace {
/// A struct to generically describe a bit test intrinsic.
struct BitTest {
  enum ActionKind : uint8_t { TestOnly, Complement, Reset, Set };
  enum InterlockingKind : uint8_t {
    Unlocked,
    Sequential,
    Acquire,
    Release,
    NoFence
  };

  ActionKind Action;
  InterlockingKind Interlocking;
  bool Is64Bit;

  static BitTest decodeBitTestBuiltin(unsigned BuiltinID);
};

} // namespace

BitTest BitTest::decodeBitTestBuiltin(unsigned BuiltinID) {
  switch (BuiltinID) {
    // Main portable variants.
  case Builtin::BI_bittest:
    return {TestOnly, Unlocked, false};
  case Builtin::BI_bittestandcomplement:
    return {Complement, Unlocked, false};
  case Builtin::BI_bittestandreset:
    return {Reset, Unlocked, false};
  case Builtin::BI_bittestandset:
    return {Set, Unlocked, false};
  case Builtin::BI_interlockedbittestandreset:
    return {Reset, Sequential, false};
  case Builtin::BI_interlockedbittestandset:
    return {Set, Sequential, false};

    // 64-bit variants.
  case Builtin::BI_bittest64:
    return {TestOnly, Unlocked, true};
  case Builtin::BI_bittestandcomplement64:
    return {Complement, Unlocked, true};
  case Builtin::BI_bittestandreset64:
    return {Reset, Unlocked, true};
  case Builtin::BI_bittestandset64:
    return {Set, Unlocked, true};
  case Builtin::BI_interlockedbittestandreset64:
    return {Reset, Sequential, true};
  case Builtin::BI_interlockedbittestandset64:
    return {Set, Sequential, true};

    // ARM/AArch64-specific ordering variants.
  case Builtin::BI_interlockedbittestandset_acq:
    return {Set, Acquire, false};
  case Builtin::BI_interlockedbittestandset_rel:
    return {Set, Release, false};
  case Builtin::BI_interlockedbittestandset_nf:
    return {Set, NoFence, false};
  case Builtin::BI_interlockedbittestandreset_acq:
    return {Reset, Acquire, false};
  case Builtin::BI_interlockedbittestandreset_rel:
    return {Reset, Release, false};
  case Builtin::BI_interlockedbittestandreset_nf:
    return {Reset, NoFence, false};
  case Builtin::BI_interlockedbittestandreset64_acq:
    return {Reset, Acquire, false};
  case Builtin::BI_interlockedbittestandreset64_rel:
    return {Reset, Release, false};
  case Builtin::BI_interlockedbittestandreset64_nf:
    return {Reset, NoFence, false};
  case Builtin::BI_interlockedbittestandset64_acq:
    return {Set, Acquire, false};
  case Builtin::BI_interlockedbittestandset64_rel:
    return {Set, Release, false};
  case Builtin::BI_interlockedbittestandset64_nf:
    return {Set, NoFence, false};
  }
  llvm_unreachable("expected only bittest intrinsics");
}

static char bitActionToX86BTCode(BitTest::ActionKind A) {
  switch (A) {
  case BitTest::TestOnly:   return '\0';
  case BitTest::Complement: return 'c';
  case BitTest::Reset:      return 'r';
  case BitTest::Set:        return 's';
  }
  llvm_unreachable("invalid action");
}

static llvm::Value *EmitX86BitTestIntrinsic(CodeGenFunction &CGF,
                                            BitTest BT,
                                            const CallExpr *E, Value *BitBase,
                                            Value *BitPos) {
  char Action = bitActionToX86BTCode(BT.Action);
  char SizeSuffix = BT.Is64Bit ? 'q' : 'l';

  // Build the assembly.
  SmallString<64> Asm;
  raw_svector_ostream AsmOS(Asm);
  if (BT.Interlocking != BitTest::Unlocked)
    AsmOS << "lock ";
  AsmOS << "bt";
  if (Action)
    AsmOS << Action;
  AsmOS << SizeSuffix << " $2, ($1)";

  // Build the constraints. FIXME: We should support immediates when possible.
  std::string Constraints = "={@ccc},r,r,~{cc},~{memory}";
  std::string_view MachineClobbers = CGF.getTarget().getClobbers();
  if (!MachineClobbers.empty()) {
    Constraints += ',';
    Constraints += MachineClobbers;
  }
  llvm::IntegerType *IntType = llvm::IntegerType::get(
      CGF.getLLVMContext(),
      CGF.getContext().getTypeSize(E->getArg(1)->getType()));
  llvm::FunctionType *FTy =
      llvm::FunctionType::get(CGF.Int8Ty, {CGF.UnqualPtrTy, IntType}, false);

  llvm::InlineAsm *IA =
      llvm::InlineAsm::get(FTy, Asm, Constraints, /*hasSideEffects=*/true);
  return CGF.Builder.CreateCall(IA, {BitBase, BitPos});
}

static llvm::AtomicOrdering
getBitTestAtomicOrdering(BitTest::InterlockingKind I) {
  switch (I) {
  case BitTest::Unlocked:   return llvm::AtomicOrdering::NotAtomic;
  case BitTest::Sequential: return llvm::AtomicOrdering::SequentiallyConsistent;
  case BitTest::Acquire:    return llvm::AtomicOrdering::Acquire;
  case BitTest::Release:    return llvm::AtomicOrdering::Release;
  case BitTest::NoFence:    return llvm::AtomicOrdering::Monotonic;
  }
  llvm_unreachable("invalid interlocking");
}

/// Emit a _bittest* intrinsic. These intrinsics take a pointer to an array of
/// bits and a bit position and read and optionally modify the bit at that
/// position. The position index can be arbitrarily large, i.e. it can be larger
/// than 31 or 63, so we need an indexed load in the general case.
static llvm::Value *EmitBitTestIntrinsic(CodeGenFunction &CGF,
                                         unsigned BuiltinID,
                                         const CallExpr *E) {
  Value *BitBase = CGF.EmitScalarExpr(E->getArg(0));
  Value *BitPos = CGF.EmitScalarExpr(E->getArg(1));

  BitTest BT = BitTest::decodeBitTestBuiltin(BuiltinID);

  // X86 has special BT, BTC, BTR, and BTS instructions that handle the array
  // indexing operation internally. Use them if possible.
  if (CGF.getTarget().getTriple().isX86())
    return EmitX86BitTestIntrinsic(CGF, BT, E, BitBase, BitPos);

  // Otherwise, use generic code to load one byte and test the bit. Use all but
  // the bottom three bits as the array index, and the bottom three bits to form
  // a mask.
  // Bit = BitBaseI8[BitPos >> 3] & (1 << (BitPos & 0x7)) != 0;
  Value *ByteIndex = CGF.Builder.CreateAShr(
      BitPos, llvm::ConstantInt::get(BitPos->getType(), 3), "bittest.byteidx");
  Address ByteAddr(CGF.Builder.CreateInBoundsGEP(CGF.Int8Ty, BitBase, ByteIndex,
                                                 "bittest.byteaddr"),
                   CGF.Int8Ty, CharUnits::One());
  Value *PosLow =
      CGF.Builder.CreateAnd(CGF.Builder.CreateTrunc(BitPos, CGF.Int8Ty),
                            llvm::ConstantInt::get(CGF.Int8Ty, 0x7));

  // The updating instructions will need a mask.
  Value *Mask = nullptr;
  if (BT.Action != BitTest::TestOnly) {
    Mask = CGF.Builder.CreateShl(llvm::ConstantInt::get(CGF.Int8Ty, 1), PosLow,
                                 "bittest.mask");
  }

  // Check the action and ordering of the interlocked intrinsics.
  llvm::AtomicOrdering Ordering = getBitTestAtomicOrdering(BT.Interlocking);

  Value *OldByte = nullptr;
  if (Ordering != llvm::AtomicOrdering::NotAtomic) {
    // Emit a combined atomicrmw load/store operation for the interlocked
    // intrinsics.
    llvm::AtomicRMWInst::BinOp RMWOp = llvm::AtomicRMWInst::Or;
    if (BT.Action == BitTest::Reset) {
      Mask = CGF.Builder.CreateNot(Mask);
      RMWOp = llvm::AtomicRMWInst::And;
    }
    OldByte = CGF.Builder.CreateAtomicRMW(RMWOp, ByteAddr, Mask, Ordering);
  } else {
    // Emit a plain load for the non-interlocked intrinsics.
    OldByte = CGF.Builder.CreateLoad(ByteAddr, "bittest.byte");
    Value *NewByte = nullptr;
    switch (BT.Action) {
    case BitTest::TestOnly:
      // Don't store anything.
      break;
    case BitTest::Complement:
      NewByte = CGF.Builder.CreateXor(OldByte, Mask);
      break;
    case BitTest::Reset:
      NewByte = CGF.Builder.CreateAnd(OldByte, CGF.Builder.CreateNot(Mask));
      break;
    case BitTest::Set:
      NewByte = CGF.Builder.CreateOr(OldByte, Mask);
      break;
    }
    if (NewByte)
      CGF.Builder.CreateStore(NewByte, ByteAddr);
  }

  // However we loaded the old byte, either by plain load or atomicrmw, shift
  // the bit into the low position and mask it to 0 or 1.
  Value *ShiftedByte = CGF.Builder.CreateLShr(OldByte, PosLow, "bittest.shr");
  return CGF.Builder.CreateAnd(
      ShiftedByte, llvm::ConstantInt::get(CGF.Int8Ty, 1), "bittest.res");
}

namespace {
enum class MSVCSetJmpKind {
  _setjmpex,
  _setjmp3,
  _setjmp
};
}

/// MSVC handles setjmp a bit differently on different platforms. On every
/// architecture except 32-bit x86, the frame address is passed. On x86, extra
/// parameters can be passed as variadic arguments, but we always pass none.
static RValue EmitMSVCRTSetJmp(CodeGenFunction &CGF, MSVCSetJmpKind SJKind,
                               const CallExpr *E) {
  llvm::Value *Arg1 = nullptr;
  llvm::Type *Arg1Ty = nullptr;
  StringRef Name;
  bool IsVarArg = false;
  if (SJKind == MSVCSetJmpKind::_setjmp3) {
    Name = "_setjmp3";
    Arg1Ty = CGF.Int32Ty;
    Arg1 = llvm::ConstantInt::get(CGF.IntTy, 0);
    IsVarArg = true;
  } else {
    Name = SJKind == MSVCSetJmpKind::_setjmp ? "_setjmp" : "_setjmpex";
    Arg1Ty = CGF.Int8PtrTy;
    if (CGF.getTarget().getTriple().getArch() == llvm::Triple::aarch64) {
      Arg1 = CGF.Builder.CreateCall(
          CGF.CGM.getIntrinsic(Intrinsic::sponentry, CGF.AllocaInt8PtrTy));
    } else
      Arg1 = CGF.Builder.CreateCall(
          CGF.CGM.getIntrinsic(Intrinsic::frameaddress, CGF.AllocaInt8PtrTy),
          llvm::ConstantInt::get(CGF.Int32Ty, 0));
  }

  // Mark the call site and declaration with ReturnsTwice.
  llvm::Type *ArgTypes[2] = {CGF.Int8PtrTy, Arg1Ty};
  llvm::AttributeList ReturnsTwiceAttr = llvm::AttributeList::get(
      CGF.getLLVMContext(), llvm::AttributeList::FunctionIndex,
      llvm::Attribute::ReturnsTwice);
  llvm::FunctionCallee SetJmpFn = CGF.CGM.CreateRuntimeFunction(
      llvm::FunctionType::get(CGF.IntTy, ArgTypes, IsVarArg), Name,
      ReturnsTwiceAttr, /*Local=*/true);

  llvm::Value *Buf = CGF.Builder.CreateBitOrPointerCast(
      CGF.EmitScalarExpr(E->getArg(0)), CGF.Int8PtrTy);
  llvm::Value *Args[] = {Buf, Arg1};
  llvm::CallBase *CB = CGF.EmitRuntimeCallOrInvoke(SetJmpFn, Args);
  CB->setAttributes(ReturnsTwiceAttr);
  return RValue::get(CB);
}

// Emit an MSVC intrinsic. Assumes that arguments have *not* been evaluated.
Value *CodeGenFunction::EmitMSVCBuiltinExpr(MSVCIntrin BuiltinID,
                                            const CallExpr *E) {
  switch (BuiltinID) {
  case MSVCIntrin::_BitScanForward:
  case MSVCIntrin::_BitScanReverse: {
    Address IndexAddress(EmitPointerWithAlignment(E->getArg(0)));
    Value *ArgValue = EmitScalarExpr(E->getArg(1));

    llvm::Type *ArgType = ArgValue->getType();
    llvm::Type *IndexType = IndexAddress.getElementType();
    llvm::Type *ResultType = ConvertType(E->getType());

    Value *ArgZero = llvm::Constant::getNullValue(ArgType);
    Value *ResZero = llvm::Constant::getNullValue(ResultType);
    Value *ResOne = llvm::ConstantInt::get(ResultType, 1);

    BasicBlock *Begin = Builder.GetInsertBlock();
    BasicBlock *End = createBasicBlock("bitscan_end", this->CurFn);
    Builder.SetInsertPoint(End);
    PHINode *Result = Builder.CreatePHI(ResultType, 2, "bitscan_result");

    Builder.SetInsertPoint(Begin);
    Value *IsZero = Builder.CreateICmpEQ(ArgValue, ArgZero);
    BasicBlock *NotZero = createBasicBlock("bitscan_not_zero", this->CurFn);
    Builder.CreateCondBr(IsZero, End, NotZero);
    Result->addIncoming(ResZero, Begin);

    Builder.SetInsertPoint(NotZero);

    if (BuiltinID == MSVCIntrin::_BitScanForward) {
      Function *F = CGM.getIntrinsic(Intrinsic::cttz, ArgType);
      Value *ZeroCount = Builder.CreateCall(F, {ArgValue, Builder.getTrue()});
      ZeroCount = Builder.CreateIntCast(ZeroCount, IndexType, false);
      Builder.CreateStore(ZeroCount, IndexAddress, false);
    } else {
      unsigned ArgWidth = cast<llvm::IntegerType>(ArgType)->getBitWidth();
      Value *ArgTypeLastIndex = llvm::ConstantInt::get(IndexType, ArgWidth - 1);

      Function *F = CGM.getIntrinsic(Intrinsic::ctlz, ArgType);
      Value *ZeroCount = Builder.CreateCall(F, {ArgValue, Builder.getTrue()});
      ZeroCount = Builder.CreateIntCast(ZeroCount, IndexType, false);
      Value *Index = Builder.CreateNSWSub(ArgTypeLastIndex, ZeroCount);
      Builder.CreateStore(Index, IndexAddress, false);
    }
    Builder.CreateBr(End);
    Result->addIncoming(ResOne, NotZero);

    Builder.SetInsertPoint(End);
    return Result;
  }
  case MSVCIntrin::_InterlockedAnd:
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::And, E);
  case MSVCIntrin::_InterlockedExchange:
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::Xchg, E);
  case MSVCIntrin::_InterlockedExchangeAdd:
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::Add, E);
  case MSVCIntrin::_InterlockedExchangeSub:
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::Sub, E);
  case MSVCIntrin::_InterlockedOr:
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::Or, E);
  case MSVCIntrin::_InterlockedXor:
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::Xor, E);
  case MSVCIntrin::_InterlockedExchangeAdd_acq:
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::Add, E,
                                 AtomicOrdering::Acquire);
  case MSVCIntrin::_InterlockedExchangeAdd_rel:
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::Add, E,
                                 AtomicOrdering::Release);
  case MSVCIntrin::_InterlockedExchangeAdd_nf:
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::Add, E,
                                 AtomicOrdering::Monotonic);
  case MSVCIntrin::_InterlockedExchange_acq:
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::Xchg, E,
                                 AtomicOrdering::Acquire);
  case MSVCIntrin::_InterlockedExchange_rel:
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::Xchg, E,
                                 AtomicOrdering::Release);
  case MSVCIntrin::_InterlockedExchange_nf:
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::Xchg, E,
                                 AtomicOrdering::Monotonic);
  case MSVCIntrin::_InterlockedCompareExchange:
    return EmitAtomicCmpXchgForMSIntrin(*this, E);
  case MSVCIntrin::_InterlockedCompareExchange_acq:
    return EmitAtomicCmpXchgForMSIntrin(*this, E, AtomicOrdering::Acquire);
  case MSVCIntrin::_InterlockedCompareExchange_rel:
    return EmitAtomicCmpXchgForMSIntrin(*this, E, AtomicOrdering::Release);
  case MSVCIntrin::_InterlockedCompareExchange_nf:
    return EmitAtomicCmpXchgForMSIntrin(*this, E, AtomicOrdering::Monotonic);
  case MSVCIntrin::_InterlockedCompareExchange128:
    return EmitAtomicCmpXchg128ForMSIntrin(
        *this, E, AtomicOrdering::SequentiallyConsistent);
  case MSVCIntrin::_InterlockedCompareExchange128_acq:
    return EmitAtomicCmpXchg128ForMSIntrin(*this, E, AtomicOrdering::Acquire);
  case MSVCIntrin::_InterlockedCompareExchange128_rel:
    return EmitAtomicCmpXchg128ForMSIntrin(*this, E, AtomicOrdering::Release);
  case MSVCIntrin::_InterlockedCompareExchange128_nf:
    return EmitAtomicCmpXchg128ForMSIntrin(*this, E, AtomicOrdering::Monotonic);
  case MSVCIntrin::_InterlockedOr_acq:
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::Or, E,
                                 AtomicOrdering::Acquire);
  case MSVCIntrin::_InterlockedOr_rel:
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::Or, E,
                                 AtomicOrdering::Release);
  case MSVCIntrin::_InterlockedOr_nf:
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::Or, E,
                                 AtomicOrdering::Monotonic);
  case MSVCIntrin::_InterlockedXor_acq:
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::Xor, E,
                                 AtomicOrdering::Acquire);
  case MSVCIntrin::_InterlockedXor_rel:
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::Xor, E,
                                 AtomicOrdering::Release);
  case MSVCIntrin::_InterlockedXor_nf:
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::Xor, E,
                                 AtomicOrdering::Monotonic);
  case MSVCIntrin::_InterlockedAnd_acq:
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::And, E,
                                 AtomicOrdering::Acquire);
  case MSVCIntrin::_InterlockedAnd_rel:
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::And, E,
                                 AtomicOrdering::Release);
  case MSVCIntrin::_InterlockedAnd_nf:
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::And, E,
                                 AtomicOrdering::Monotonic);
  case MSVCIntrin::_InterlockedIncrement_acq:
    return EmitAtomicIncrementValue(*this, E, AtomicOrdering::Acquire);
  case MSVCIntrin::_InterlockedIncrement_rel:
    return EmitAtomicIncrementValue(*this, E, AtomicOrdering::Release);
  case MSVCIntrin::_InterlockedIncrement_nf:
    return EmitAtomicIncrementValue(*this, E, AtomicOrdering::Monotonic);
  case MSVCIntrin::_InterlockedDecrement_acq:
    return EmitAtomicDecrementValue(*this, E, AtomicOrdering::Acquire);
  case MSVCIntrin::_InterlockedDecrement_rel:
    return EmitAtomicDecrementValue(*this, E, AtomicOrdering::Release);
  case MSVCIntrin::_InterlockedDecrement_nf:
    return EmitAtomicDecrementValue(*this, E, AtomicOrdering::Monotonic);

  case MSVCIntrin::_InterlockedDecrement:
    return EmitAtomicDecrementValue(*this, E);
  case MSVCIntrin::_InterlockedIncrement:
    return EmitAtomicIncrementValue(*this, E);

  case MSVCIntrin::__fastfail: {
    // Request immediate process termination from the kernel. The instruction
    // sequences to do this are documented on MSDN:
    // https://msdn.microsoft.com/en-us/library/dn774154.aspx
    llvm::Triple::ArchType ISA = getTarget().getTriple().getArch();
    StringRef Asm, Constraints;
    switch (ISA) {
    default:
      ErrorUnsupported(E, "__fastfail call for this architecture");
      break;
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
      Asm = "int $$0x29";
      Constraints = "{cx}";
      break;
    case llvm::Triple::thumb:
      Asm = "udf #251";
      Constraints = "{r0}";
      break;
    case llvm::Triple::aarch64:
      Asm = "brk #0xF003";
      Constraints = "{w0}";
    }
    llvm::FunctionType *FTy = llvm::FunctionType::get(VoidTy, {Int32Ty}, false);
    llvm::InlineAsm *IA =
        llvm::InlineAsm::get(FTy, Asm, Constraints, /*hasSideEffects=*/true);
    llvm::AttributeList NoReturnAttr = llvm::AttributeList::get(
        getLLVMContext(), llvm::AttributeList::FunctionIndex,
        llvm::Attribute::NoReturn);
    llvm::CallInst *CI = Builder.CreateCall(IA, EmitScalarExpr(E->getArg(0)));
    CI->setAttributes(NoReturnAttr);
    return CI;
  }
  }
  llvm_unreachable("Incorrect MSVC intrinsic!");
}

namespace {
// ARC cleanup for __builtin_os_log_format
struct CallObjCArcUse final : EHScopeStack::Cleanup {
  CallObjCArcUse(llvm::Value *object) : object(object) {}
  llvm::Value *object;

  void Emit(CodeGenFunction &CGF, Flags flags) override {
    CGF.EmitARCIntrinsicUse(object);
  }
};
}

Value *CodeGenFunction::EmitCheckedArgForBuiltin(const Expr *E,
                                                 BuiltinCheckKind Kind) {
  assert((Kind == BCK_CLZPassedZero || Kind == BCK_CTZPassedZero) &&
         "Unsupported builtin check kind");

  Value *ArgValue = EmitScalarExpr(E);
  if (!SanOpts.has(SanitizerKind::Builtin))
    return ArgValue;

  auto CheckOrdinal = SanitizerKind::SO_Builtin;
  auto CheckHandler = SanitizerHandler::InvalidBuiltin;
  SanitizerDebugLocation SanScope(this, {CheckOrdinal}, CheckHandler);
  Value *Cond = Builder.CreateICmpNE(
      ArgValue, llvm::Constant::getNullValue(ArgValue->getType()));
  EmitCheck(std::make_pair(Cond, CheckOrdinal), CheckHandler,
            {EmitCheckSourceLocation(E->getExprLoc()),
             llvm::ConstantInt::get(Builder.getInt8Ty(), Kind)},
            {});
  return ArgValue;
}

Value *CodeGenFunction::EmitCheckedArgForAssume(const Expr *E) {
  Value *ArgValue = EvaluateExprAsBool(E);
  if (!SanOpts.has(SanitizerKind::Builtin))
    return ArgValue;

  auto CheckOrdinal = SanitizerKind::SO_Builtin;
  auto CheckHandler = SanitizerHandler::InvalidBuiltin;
  SanitizerDebugLocation SanScope(this, {CheckOrdinal}, CheckHandler);
  EmitCheck(
      std::make_pair(ArgValue, CheckOrdinal), CheckHandler,
      {EmitCheckSourceLocation(E->getExprLoc()),
       llvm::ConstantInt::get(Builder.getInt8Ty(), BCK_AssumePassedFalse)},
      {});
  return ArgValue;
}

static Value *EmitAbs(CodeGenFunction &CGF, Value *ArgValue, bool HasNSW) {
  return CGF.Builder.CreateBinaryIntrinsic(
      Intrinsic::abs, ArgValue,
      ConstantInt::get(CGF.Builder.getInt1Ty(), HasNSW));
}

static Value *EmitOverflowCheckedAbs(CodeGenFunction &CGF, const CallExpr *E,
                                     bool SanitizeOverflow) {
  Value *ArgValue = CGF.EmitScalarExpr(E->getArg(0));

  // Try to eliminate overflow check.
  if (const auto *VCI = dyn_cast<llvm::ConstantInt>(ArgValue)) {
    if (!VCI->isMinSignedValue())
      return EmitAbs(CGF, ArgValue, true);
  }

  SmallVector<SanitizerKind::SanitizerOrdinal, 1> Ordinals;
  SanitizerHandler CheckHandler;
  if (SanitizeOverflow) {
    Ordinals.push_back(SanitizerKind::SO_SignedIntegerOverflow);
    CheckHandler = SanitizerHandler::NegateOverflow;
  } else
    CheckHandler = SanitizerHandler::SubOverflow;

  SanitizerDebugLocation SanScope(&CGF, Ordinals, CheckHandler);

  Constant *Zero = Constant::getNullValue(ArgValue->getType());
  Value *ResultAndOverflow = CGF.Builder.CreateBinaryIntrinsic(
      Intrinsic::ssub_with_overflow, Zero, ArgValue);
  Value *Result = CGF.Builder.CreateExtractValue(ResultAndOverflow, 0);
  Value *NotOverflow = CGF.Builder.CreateNot(
      CGF.Builder.CreateExtractValue(ResultAndOverflow, 1));

  // TODO: support -ftrapv-handler.
  if (SanitizeOverflow) {
    CGF.EmitCheck({{NotOverflow, SanitizerKind::SO_SignedIntegerOverflow}},
                  CheckHandler,
                  {CGF.EmitCheckSourceLocation(E->getArg(0)->getExprLoc()),
                   CGF.EmitCheckTypeDescriptor(E->getType())},
                  {ArgValue});
  } else
    CGF.EmitTrapCheck(NotOverflow, CheckHandler);

  Value *CmpResult = CGF.Builder.CreateICmpSLT(ArgValue, Zero, "abscond");
  return CGF.Builder.CreateSelect(CmpResult, Result, ArgValue, "abs");
}

/// Get the argument type for arguments to os_log_helper.
static CanQualType getOSLogArgType(ASTContext &C, int Size) {
  QualType UnsignedTy = C.getIntTypeForBitwidth(Size * 8, /*Signed=*/false);
  return C.getCanonicalType(UnsignedTy);
}

llvm::Function *CodeGenFunction::generateBuiltinOSLogHelperFunction(
    const analyze_os_log::OSLogBufferLayout &Layout,
    CharUnits BufferAlignment) {
  ASTContext &Ctx = getContext();

  llvm::SmallString<64> Name;
  {
    raw_svector_ostream OS(Name);
    OS << "__os_log_helper";
    OS << "_" << BufferAlignment.getQuantity();
    OS << "_" << int(Layout.getSummaryByte());
    OS << "_" << int(Layout.getNumArgsByte());
    for (const auto &Item : Layout.Items)
      OS << "_" << int(Item.getSizeByte()) << "_"
         << int(Item.getDescriptorByte());
  }

  if (llvm::Function *F = CGM.getModule().getFunction(Name))
    return F;

  llvm::SmallVector<QualType, 4> ArgTys;
  FunctionArgList Args;
  Args.push_back(ImplicitParamDecl::Create(
      Ctx, nullptr, SourceLocation(), &Ctx.Idents.get("buffer"), Ctx.VoidPtrTy,
      ImplicitParamKind::Other));
  ArgTys.emplace_back(Ctx.VoidPtrTy);

  for (unsigned int I = 0, E = Layout.Items.size(); I < E; ++I) {
    char Size = Layout.Items[I].getSizeByte();
    if (!Size)
      continue;

    QualType ArgTy = getOSLogArgType(Ctx, Size);
    Args.push_back(ImplicitParamDecl::Create(
        Ctx, nullptr, SourceLocation(),
        &Ctx.Idents.get(std::string("arg") + llvm::to_string(I)), ArgTy,
        ImplicitParamKind::Other));
    ArgTys.emplace_back(ArgTy);
  }

  QualType ReturnTy = Ctx.VoidTy;

  // The helper function has linkonce_odr linkage to enable the linker to merge
  // identical functions. To ensure the merging always happens, 'noinline' is
  // attached to the function when compiling with -Oz.
  const CGFunctionInfo &FI =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(ReturnTy, Args);
  llvm::FunctionType *FuncTy = CGM.getTypes().GetFunctionType(FI);
  llvm::Function *Fn = llvm::Function::Create(
      FuncTy, llvm::GlobalValue::LinkOnceODRLinkage, Name, &CGM.getModule());
  Fn->setVisibility(llvm::GlobalValue::HiddenVisibility);
  CGM.SetLLVMFunctionAttributes(GlobalDecl(), FI, Fn, /*IsThunk=*/false);
  CGM.SetLLVMFunctionAttributesForDefinition(nullptr, Fn);
  Fn->setDoesNotThrow();

  // Attach 'noinline' at -Oz.
  if (CGM.getCodeGenOpts().OptimizeSize == 2)
    Fn->addFnAttr(llvm::Attribute::NoInline);

  auto NL = ApplyDebugLocation::CreateEmpty(*this);
  StartFunction(GlobalDecl(), ReturnTy, Fn, FI, Args);

  // Create a scope with an artificial location for the body of this function.
  auto AL = ApplyDebugLocation::CreateArtificial(*this);

  CharUnits Offset;
  Address BufAddr = makeNaturalAddressForPointer(
      Builder.CreateLoad(GetAddrOfLocalVar(Args[0]), "buf"), Ctx.VoidTy,
      BufferAlignment);
  Builder.CreateStore(Builder.getInt8(Layout.getSummaryByte()),
                      Builder.CreateConstByteGEP(BufAddr, Offset++, "summary"));
  Builder.CreateStore(Builder.getInt8(Layout.getNumArgsByte()),
                      Builder.CreateConstByteGEP(BufAddr, Offset++, "numArgs"));

  unsigned I = 1;
  for (const auto &Item : Layout.Items) {
    Builder.CreateStore(
        Builder.getInt8(Item.getDescriptorByte()),
        Builder.CreateConstByteGEP(BufAddr, Offset++, "argDescriptor"));
    Builder.CreateStore(
        Builder.getInt8(Item.getSizeByte()),
        Builder.CreateConstByteGEP(BufAddr, Offset++, "argSize"));

    CharUnits Size = Item.size();
    if (!Size.getQuantity())
      continue;

    Address Arg = GetAddrOfLocalVar(Args[I]);
    Address Addr = Builder.CreateConstByteGEP(BufAddr, Offset, "argData");
    Addr = Addr.withElementType(Arg.getElementType());
    Builder.CreateStore(Builder.CreateLoad(Arg), Addr);
    Offset += Size;
    ++I;
  }

  FinishFunction();

  return Fn;
}

RValue CodeGenFunction::emitBuiltinOSLogFormat(const CallExpr &E) {
  assert(E.getNumArgs() >= 2 &&
         "__builtin_os_log_format takes at least 2 arguments");
  ASTContext &Ctx = getContext();
  analyze_os_log::OSLogBufferLayout Layout;
  analyze_os_log::computeOSLogBufferLayout(Ctx, &E, Layout);
  Address BufAddr = EmitPointerWithAlignment(E.getArg(0));

  // Ignore argument 1, the format string. It is not currently used.
  CallArgList Args;
  Args.add(RValue::get(BufAddr.emitRawPointer(*this)), Ctx.VoidPtrTy);

  for (const auto &Item : Layout.Items) {
    int Size = Item.getSizeByte();
    if (!Size)
      continue;

    llvm::Value *ArgVal;

    if (Item.getKind() == analyze_os_log::OSLogBufferItem::MaskKind) {
      uint64_t Val = 0;
      for (unsigned I = 0, E = Item.getMaskType().size(); I < E; ++I)
        Val |= ((uint64_t)Item.getMaskType()[I]) << I * 8;
      ArgVal = llvm::Constant::getIntegerValue(Int64Ty, llvm::APInt(64, Val));
    } else if (const Expr *TheExpr = Item.getExpr()) {
      ArgVal = EmitScalarExpr(TheExpr, /*Ignore*/ false);

      // If a temporary object that requires destruction after the full
      // expression is passed, push a lifetime-extended cleanup to extend its
      // lifetime to the end of the enclosing block scope.
      auto LifetimeExtendObject = [&](const Expr *E) {
        E = E->IgnoreParenCasts();
        // Extend lifetimes of objects returned by function calls and message
        // sends.

        // FIXME: We should do this in other cases in which temporaries are
        //        created including arguments of non-ARC types (e.g., C++
        //        temporaries).
        if (isa<CallExpr>(E) || isa<ObjCMessageExpr>(E))
          return true;
        return false;
      };

      if (TheExpr->getType()->isObjCRetainableType() &&
          getLangOpts().ObjCAutoRefCount && LifetimeExtendObject(TheExpr)) {
        assert(getEvaluationKind(TheExpr->getType()) == TEK_Scalar &&
               "Only scalar can be a ObjC retainable type");
        if (!isa<Constant>(ArgVal)) {
          CleanupKind Cleanup = getARCCleanupKind();
          QualType Ty = TheExpr->getType();
          RawAddress Alloca = RawAddress::invalid();
          RawAddress Addr = CreateMemTemp(Ty, "os.log.arg", &Alloca);
          ArgVal = EmitARCRetain(Ty, ArgVal);
          Builder.CreateStore(ArgVal, Addr);
          pushLifetimeExtendedDestroy(Cleanup, Alloca, Ty,
                                      CodeGenFunction::destroyARCStrongPrecise,
                                      Cleanup & EHCleanup);

          // Push a clang.arc.use call to ensure ARC optimizer knows that the
          // argument has to be alive.
          if (CGM.getCodeGenOpts().OptimizationLevel != 0)
            pushCleanupAfterFullExpr<CallObjCArcUse>(Cleanup, ArgVal);
        }
      }
    } else {
      ArgVal = Builder.getInt32(Item.getConstValue().getQuantity());
    }

    unsigned ArgValSize =
        CGM.getDataLayout().getTypeSizeInBits(ArgVal->getType());
    llvm::IntegerType *IntTy = llvm::Type::getIntNTy(getLLVMContext(),
                                                     ArgValSize);
    ArgVal = Builder.CreateBitOrPointerCast(ArgVal, IntTy);
    CanQualType ArgTy = getOSLogArgType(Ctx, Size);
    // If ArgVal has type x86_fp80, zero-extend ArgVal.
    ArgVal = Builder.CreateZExtOrBitCast(ArgVal, ConvertType(ArgTy));
    Args.add(RValue::get(ArgVal), ArgTy);
  }

  const CGFunctionInfo &FI =
      CGM.getTypes().arrangeBuiltinFunctionCall(Ctx.VoidTy, Args);
  llvm::Function *F = CodeGenFunction(CGM).generateBuiltinOSLogHelperFunction(
      Layout, BufAddr.getAlignment());
  EmitCall(FI, CGCallee::forDirect(F), ReturnValueSlot(), Args);
  return RValue::get(BufAddr, *this);
}

static bool isSpecialUnsignedMultiplySignedResult(
    unsigned BuiltinID, WidthAndSignedness Op1Info, WidthAndSignedness Op2Info,
    WidthAndSignedness ResultInfo) {
  return BuiltinID == Builtin::BI__builtin_mul_overflow &&
         Op1Info.Width == Op2Info.Width && Op2Info.Width == ResultInfo.Width &&
         !Op1Info.Signed && !Op2Info.Signed && ResultInfo.Signed;
}

static RValue EmitCheckedUnsignedMultiplySignedResult(
    CodeGenFunction &CGF, const clang::Expr *Op1, WidthAndSignedness Op1Info,
    const clang::Expr *Op2, WidthAndSignedness Op2Info,
    const clang::Expr *ResultArg, QualType ResultQTy,
    WidthAndSignedness ResultInfo) {
  assert(isSpecialUnsignedMultiplySignedResult(
             Builtin::BI__builtin_mul_overflow, Op1Info, Op2Info, ResultInfo) &&
         "Cannot specialize this multiply");

  llvm::Value *V1 = CGF.EmitScalarExpr(Op1);
  llvm::Value *V2 = CGF.EmitScalarExpr(Op2);

  llvm::Value *HasOverflow;
  llvm::Value *Result = EmitOverflowIntrinsic(
      CGF, Intrinsic::umul_with_overflow, V1, V2, HasOverflow);

  // The intrinsic call will detect overflow when the value is > UINT_MAX,
  // however, since the original builtin had a signed result, we need to report
  // an overflow when the result is greater than INT_MAX.
  auto IntMax = llvm::APInt::getSignedMaxValue(ResultInfo.Width);
  llvm::Value *IntMaxValue = llvm::ConstantInt::get(Result->getType(), IntMax);

  llvm::Value *IntMaxOverflow = CGF.Builder.CreateICmpUGT(Result, IntMaxValue);
  HasOverflow = CGF.Builder.CreateOr(HasOverflow, IntMaxOverflow);

  bool isVolatile =
      ResultArg->getType()->getPointeeType().isVolatileQualified();
  Address ResultPtr = CGF.EmitPointerWithAlignment(ResultArg);
  CGF.Builder.CreateStore(CGF.EmitToMemory(Result, ResultQTy), ResultPtr,
                          isVolatile);
  return RValue::get(HasOverflow);
}

/// Determine if a binop is a checked mixed-sign multiply we can specialize.
static bool isSpecialMixedSignMultiply(unsigned BuiltinID,
                                       WidthAndSignedness Op1Info,
                                       WidthAndSignedness Op2Info,
                                       WidthAndSignedness ResultInfo) {
  return BuiltinID == Builtin::BI__builtin_mul_overflow &&
         std::max(Op1Info.Width, Op2Info.Width) >= ResultInfo.Width &&
         Op1Info.Signed != Op2Info.Signed;
}

/// Emit a checked mixed-sign multiply. This is a cheaper specialization of
/// the generic checked-binop irgen.
static RValue
EmitCheckedMixedSignMultiply(CodeGenFunction &CGF, const clang::Expr *Op1,
                             WidthAndSignedness Op1Info, const clang::Expr *Op2,
                             WidthAndSignedness Op2Info,
                             const clang::Expr *ResultArg, QualType ResultQTy,
                             WidthAndSignedness ResultInfo) {
  assert(isSpecialMixedSignMultiply(Builtin::BI__builtin_mul_overflow, Op1Info,
                                    Op2Info, ResultInfo) &&
         "Not a mixed-sign multipliction we can specialize");

  // Emit the signed and unsigned operands.
  const clang::Expr *SignedOp = Op1Info.Signed ? Op1 : Op2;
  const clang::Expr *UnsignedOp = Op1Info.Signed ? Op2 : Op1;
  llvm::Value *Signed = CGF.EmitScalarExpr(SignedOp);
  llvm::Value *Unsigned = CGF.EmitScalarExpr(UnsignedOp);
  unsigned SignedOpWidth = Op1Info.Signed ? Op1Info.Width : Op2Info.Width;
  unsigned UnsignedOpWidth = Op1Info.Signed ? Op2Info.Width : Op1Info.Width;

  // One of the operands may be smaller than the other. If so, [s|z]ext it.
  if (SignedOpWidth < UnsignedOpWidth)
    Signed = CGF.Builder.CreateSExt(Signed, Unsigned->getType(), "op.sext");
  if (UnsignedOpWidth < SignedOpWidth)
    Unsigned = CGF.Builder.CreateZExt(Unsigned, Signed->getType(), "op.zext");

  llvm::Type *OpTy = Signed->getType();
  llvm::Value *Zero = llvm::Constant::getNullValue(OpTy);
  Address ResultPtr = CGF.EmitPointerWithAlignment(ResultArg);
  llvm::Type *ResTy = CGF.getTypes().ConvertType(ResultQTy);
  unsigned OpWidth = std::max(Op1Info.Width, Op2Info.Width);

  // Take the absolute value of the signed operand.
  llvm::Value *IsNegative = CGF.Builder.CreateICmpSLT(Signed, Zero);
  llvm::Value *AbsOfNegative = CGF.Builder.CreateSub(Zero, Signed);
  llvm::Value *AbsSigned =
      CGF.Builder.CreateSelect(IsNegative, AbsOfNegative, Signed);

  // Perform a checked unsigned multiplication.
  llvm::Value *UnsignedOverflow;
  llvm::Value *UnsignedResult =
      EmitOverflowIntrinsic(CGF, Intrinsic::umul_with_overflow, AbsSigned,
                            Unsigned, UnsignedOverflow);

  llvm::Value *Overflow, *Result;
  if (ResultInfo.Signed) {
    // Signed overflow occurs if the result is greater than INT_MAX or lesser
    // than INT_MIN, i.e when |Result| > (INT_MAX + IsNegative).
    auto IntMax =
        llvm::APInt::getSignedMaxValue(ResultInfo.Width).zext(OpWidth);
    llvm::Value *MaxResult =
        CGF.Builder.CreateAdd(llvm::ConstantInt::get(OpTy, IntMax),
                              CGF.Builder.CreateZExt(IsNegative, OpTy));
    llvm::Value *SignedOverflow =
        CGF.Builder.CreateICmpUGT(UnsignedResult, MaxResult);
    Overflow = CGF.Builder.CreateOr(UnsignedOverflow, SignedOverflow);

    // Prepare the signed result (possibly by negating it).
    llvm::Value *NegativeResult = CGF.Builder.CreateNeg(UnsignedResult);
    llvm::Value *SignedResult =
        CGF.Builder.CreateSelect(IsNegative, NegativeResult, UnsignedResult);
    Result = CGF.Builder.CreateTrunc(SignedResult, ResTy);
  } else {
    // Unsigned overflow occurs if the result is < 0 or greater than UINT_MAX.
    llvm::Value *Underflow = CGF.Builder.CreateAnd(
        IsNegative, CGF.Builder.CreateIsNotNull(UnsignedResult));
    Overflow = CGF.Builder.CreateOr(UnsignedOverflow, Underflow);
    if (ResultInfo.Width < OpWidth) {
      auto IntMax =
          llvm::APInt::getMaxValue(ResultInfo.Width).zext(OpWidth);
      llvm::Value *TruncOverflow = CGF.Builder.CreateICmpUGT(
          UnsignedResult, llvm::ConstantInt::get(OpTy, IntMax));
      Overflow = CGF.Builder.CreateOr(Overflow, TruncOverflow);
    }

    // Negate the product if it would be negative in infinite precision.
    Result = CGF.Builder.CreateSelect(
        IsNegative, CGF.Builder.CreateNeg(UnsignedResult), UnsignedResult);

    Result = CGF.Builder.CreateTrunc(Result, ResTy);
  }
  assert(Overflow && Result && "Missing overflow or result");

  bool isVolatile =
      ResultArg->getType()->getPointeeType().isVolatileQualified();
  CGF.Builder.CreateStore(CGF.EmitToMemory(Result, ResultQTy), ResultPtr,
                          isVolatile);
  return RValue::get(Overflow);
}

static bool
TypeRequiresBuiltinLaunderImp(const ASTContext &Ctx, QualType Ty,
                              llvm::SmallPtrSetImpl<const Decl *> &Seen) {
  if (const auto *Arr = Ctx.getAsArrayType(Ty))
    Ty = Ctx.getBaseElementType(Arr);

  const auto *Record = Ty->getAsCXXRecordDecl();
  if (!Record)
    return false;

  // We've already checked this type, or are in the process of checking it.
  if (!Seen.insert(Record).second)
    return false;

  assert(Record->hasDefinition() &&
         "Incomplete types should already be diagnosed");

  if (Record->isDynamicClass())
    return true;

  for (FieldDecl *F : Record->fields()) {
    if (TypeRequiresBuiltinLaunderImp(Ctx, F->getType(), Seen))
      return true;
  }
  return false;
}

/// Determine if the specified type requires laundering by checking if it is a
/// dynamic class type or contains a subobject which is a dynamic class type.
static bool TypeRequiresBuiltinLaunder(CodeGenModule &CGM, QualType Ty) {
  if (!CGM.getCodeGenOpts().StrictVTablePointers)
    return false;
  llvm::SmallPtrSet<const Decl *, 16> Seen;
  return TypeRequiresBuiltinLaunderImp(CGM.getContext(), Ty, Seen);
}

RValue CodeGenFunction::emitRotate(const CallExpr *E, bool IsRotateRight) {
  llvm::Value *Src = EmitScalarExpr(E->getArg(0));
  llvm::Value *ShiftAmt = EmitScalarExpr(E->getArg(1));

  // The builtin's shift arg may have a different type than the source arg and
  // result, but the LLVM intrinsic uses the same type for all values.
  llvm::Type *Ty = Src->getType();
  ShiftAmt = Builder.CreateIntCast(ShiftAmt, Ty, false);

  // Rotate is a special case of LLVM funnel shift - 1st 2 args are the same.
  unsigned IID = IsRotateRight ? Intrinsic::fshr : Intrinsic::fshl;
  Function *F = CGM.getIntrinsic(IID, Ty);
  return RValue::get(Builder.CreateCall(F, { Src, Src, ShiftAmt }));
}

// Map math builtins for long-double to f128 version.
static unsigned mutateLongDoubleBuiltin(unsigned BuiltinID) {
  switch (BuiltinID) {
#define MUTATE_LDBL(func) \
  case Builtin::BI__builtin_##func##l: \
    return Builtin::BI__builtin_##func##f128;
  MUTATE_LDBL(sqrt)
  MUTATE_LDBL(cbrt)
  MUTATE_LDBL(fabs)
  MUTATE_LDBL(log)
  MUTATE_LDBL(log2)
  MUTATE_LDBL(log10)
  MUTATE_LDBL(log1p)
  MUTATE_LDBL(logb)
  MUTATE_LDBL(exp)
  MUTATE_LDBL(exp2)
  MUTATE_LDBL(expm1)
  MUTATE_LDBL(fdim)
  MUTATE_LDBL(hypot)
  MUTATE_LDBL(ilogb)
  MUTATE_LDBL(pow)
  MUTATE_LDBL(fmin)
  MUTATE_LDBL(fmax)
  MUTATE_LDBL(ceil)
  MUTATE_LDBL(trunc)
  MUTATE_LDBL(rint)
  MUTATE_LDBL(nearbyint)
  MUTATE_LDBL(round)
  MUTATE_LDBL(floor)
  MUTATE_LDBL(lround)
  MUTATE_LDBL(llround)
  MUTATE_LDBL(lrint)
  MUTATE_LDBL(llrint)
  MUTATE_LDBL(fmod)
  MUTATE_LDBL(modf)
  MUTATE_LDBL(nan)
  MUTATE_LDBL(nans)
  MUTATE_LDBL(inf)
  MUTATE_LDBL(fma)
  MUTATE_LDBL(sin)
  MUTATE_LDBL(cos)
  MUTATE_LDBL(tan)
  MUTATE_LDBL(sinh)
  MUTATE_LDBL(cosh)
  MUTATE_LDBL(tanh)
  MUTATE_LDBL(asin)
  MUTATE_LDBL(acos)
  MUTATE_LDBL(atan)
  MUTATE_LDBL(asinh)
  MUTATE_LDBL(acosh)
  MUTATE_LDBL(atanh)
  MUTATE_LDBL(atan2)
  MUTATE_LDBL(erf)
  MUTATE_LDBL(erfc)
  MUTATE_LDBL(ldexp)
  MUTATE_LDBL(frexp)
  MUTATE_LDBL(huge_val)
  MUTATE_LDBL(copysign)
  MUTATE_LDBL(nextafter)
  MUTATE_LDBL(nexttoward)
  MUTATE_LDBL(remainder)
  MUTATE_LDBL(remquo)
  MUTATE_LDBL(scalbln)
  MUTATE_LDBL(scalbn)
  MUTATE_LDBL(tgamma)
  MUTATE_LDBL(lgamma)
#undef MUTATE_LDBL
  default:
    return BuiltinID;
  }
}

static Value *tryUseTestFPKind(CodeGenFunction &CGF, unsigned BuiltinID,
                               Value *V) {
  if (CGF.Builder.getIsFPConstrained() &&
      CGF.Builder.getDefaultConstrainedExcept() != fp::ebIgnore) {
    if (Value *Result =
            CGF.getTargetHooks().testFPKind(V, BuiltinID, CGF.Builder, CGF.CGM))
      return Result;
  }
  return nullptr;
}

static RValue EmitHipStdParUnsupportedBuiltin(CodeGenFunction *CGF,
                                              const FunctionDecl *FD) {
  auto Name = FD->getNameAsString() + "__hipstdpar_unsupported";
  auto FnTy = CGF->CGM.getTypes().GetFunctionType(FD);
  auto UBF = CGF->CGM.getModule().getOrInsertFunction(Name, FnTy);

  SmallVector<Value *, 16> Args;
  for (auto &&FormalTy : FnTy->params())
    Args.push_back(llvm::PoisonValue::get(FormalTy));

  return RValue::get(CGF->Builder.CreateCall(UBF, Args));
}

RValue CodeGenFunction::EmitBuiltinExpr(const GlobalDecl GD, unsigned BuiltinID,
                                        const CallExpr *E,
                                        ReturnValueSlot ReturnValue) {
  assert(!getContext().BuiltinInfo.isImmediate(BuiltinID) &&
         "Should not codegen for consteval builtins");

  const FunctionDecl *FD = GD.getDecl()->getAsFunction();
  // See if we can constant fold this builtin.  If so, don't emit it at all.
  // TODO: Extend this handling to all builtin calls that we can constant-fold.
  Expr::EvalResult Result;
  if (E->isPRValue() && E->EvaluateAsRValue(Result, CGM.getContext()) &&
      !Result.hasSideEffects()) {
    if (Result.Val.isInt())
      return RValue::get(llvm::ConstantInt::get(getLLVMContext(),
                                                Result.Val.getInt()));
    if (Result.Val.isFloat())
      return RValue::get(llvm::ConstantFP::get(getLLVMContext(),
                                               Result.Val.getFloat()));
  }

  // If current long-double semantics is IEEE 128-bit, replace math builtins
  // of long-double with f128 equivalent.
  // TODO: This mutation should also be applied to other targets other than PPC,
  // after backend supports IEEE 128-bit style libcalls.
  if (getTarget().getTriple().isPPC64() &&
      &getTarget().getLongDoubleFormat() == &llvm::APFloat::IEEEquad())
    BuiltinID = mutateLongDoubleBuiltin(BuiltinID);

  // If the builtin has been declared explicitly with an assembler label,
  // disable the specialized emitting below. Ideally we should communicate the
  // rename in IR, or at least avoid generating the intrinsic calls that are
  // likely to get lowered to the renamed library functions.
  const unsigned BuiltinIDIfNoAsmLabel =
      FD->hasAttr<AsmLabelAttr>() ? 0 : BuiltinID;

  std::optional<bool> ErrnoOverriden;
  // ErrnoOverriden is true if math-errno is overriden via the
  // '#pragma float_control(precise, on)'. This pragma disables fast-math,
  // which implies math-errno.
  if (E->hasStoredFPFeatures()) {
    FPOptionsOverride OP = E->getFPFeatures();
    if (OP.hasMathErrnoOverride())
      ErrnoOverriden = OP.getMathErrnoOverride();
  }
  // True if 'attribute__((optnone))' is used. This attribute overrides
  // fast-math which implies math-errno.
  bool OptNone = CurFuncDecl && CurFuncDecl->hasAttr<OptimizeNoneAttr>();

  // True if we are compiling at -O2 and errno has been disabled
  // using the '#pragma float_control(precise, off)', and
  // attribute opt-none hasn't been seen.
  bool ErrnoOverridenToFalseWithOpt =
       ErrnoOverriden.has_value() && !ErrnoOverriden.value() && !OptNone &&
       CGM.getCodeGenOpts().OptimizationLevel != 0;

  // There are LLVM math intrinsics/instructions corresponding to math library
  // functions except the LLVM op will never set errno while the math library
  // might. Also, math builtins have the same semantics as their math library
  // twins. Thus, we can transform math library and builtin calls to their
  // LLVM counterparts if the call is marked 'const' (known to never set errno).
  // In case FP exceptions are enabled, the experimental versions of the
  // intrinsics model those.
  bool ConstAlways =
      getContext().BuiltinInfo.isConst(BuiltinID);

  // There's a special case with the fma builtins where they are always const
  // if the target environment is GNU or the target is OS is Windows and we're
  // targeting the MSVCRT.dll environment.
  // FIXME: This list can be become outdated. Need to find a way to get it some
  // other way.
  switch (BuiltinID) {
  case Builtin::BI__builtin_fma:
  case Builtin::BI__builtin_fmaf:
  case Builtin::BI__builtin_fmal:
  case Builtin::BI__builtin_fmaf16:
  case Builtin::BIfma:
  case Builtin::BIfmaf:
  case Builtin::BIfmal: {
    auto &Trip = CGM.getTriple();
    if (Trip.isGNUEnvironment() || Trip.isOSMSVCRT())
      ConstAlways = true;
    break;
  }
  default:
    break;
  }

  bool ConstWithoutErrnoAndExceptions =
      getContext().BuiltinInfo.isConstWithoutErrnoAndExceptions(BuiltinID);
  bool ConstWithoutExceptions =
      getContext().BuiltinInfo.isConstWithoutExceptions(BuiltinID);

  // ConstAttr is enabled in fast-math mode. In fast-math mode, math-errno is
  // disabled.
  // Math intrinsics are generated only when math-errno is disabled. Any pragmas
  // or attributes that affect math-errno should prevent or allow math
  // intrinsics to be generated. Intrinsics are generated:
  //   1- In fast math mode, unless math-errno is overriden
  //      via '#pragma float_control(precise, on)', or via an
  //      'attribute__((optnone))'.
  //   2- If math-errno was enabled on command line but overriden
  //      to false via '#pragma float_control(precise, off))' and
  //      'attribute__((optnone))' hasn't been used.
  //   3- If we are compiling with optimization and errno has been disabled
  //      via '#pragma float_control(precise, off)', and
  //      'attribute__((optnone))' hasn't been used.

  bool ConstWithoutErrnoOrExceptions =
      ConstWithoutErrnoAndExceptions || ConstWithoutExceptions;
  bool GenerateIntrinsics =
      (ConstAlways && !OptNone) ||
      (!getLangOpts().MathErrno &&
       !(ErrnoOverriden.has_value() && ErrnoOverriden.value()) && !OptNone);
  if (!GenerateIntrinsics) {
    GenerateIntrinsics =
        ConstWithoutErrnoOrExceptions && !ConstWithoutErrnoAndExceptions;
    if (!GenerateIntrinsics)
      GenerateIntrinsics =
          ConstWithoutErrnoOrExceptions &&
          (!getLangOpts().MathErrno &&
           !(ErrnoOverriden.has_value() && ErrnoOverriden.value()) && !OptNone);
    if (!GenerateIntrinsics)
      GenerateIntrinsics =
          ConstWithoutErrnoOrExceptions && ErrnoOverridenToFalseWithOpt;
  }
  if (GenerateIntrinsics) {
    switch (BuiltinIDIfNoAsmLabel) {
    case Builtin::BIacos:
    case Builtin::BIacosf:
    case Builtin::BIacosl:
    case Builtin::BI__builtin_acos:
    case Builtin::BI__builtin_acosf:
    case Builtin::BI__builtin_acosf16:
    case Builtin::BI__builtin_acosl:
    case Builtin::BI__builtin_acosf128:
      return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(
          *this, E, Intrinsic::acos, Intrinsic::experimental_constrained_acos));

    case Builtin::BIasin:
    case Builtin::BIasinf:
    case Builtin::BIasinl:
    case Builtin::BI__builtin_asin:
    case Builtin::BI__builtin_asinf:
    case Builtin::BI__builtin_asinf16:
    case Builtin::BI__builtin_asinl:
    case Builtin::BI__builtin_asinf128:
      return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(
          *this, E, Intrinsic::asin, Intrinsic::experimental_constrained_asin));

    case Builtin::BIatan:
    case Builtin::BIatanf:
    case Builtin::BIatanl:
    case Builtin::BI__builtin_atan:
    case Builtin::BI__builtin_atanf:
    case Builtin::BI__builtin_atanf16:
    case Builtin::BI__builtin_atanl:
    case Builtin::BI__builtin_atanf128:
      return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(
          *this, E, Intrinsic::atan, Intrinsic::experimental_constrained_atan));

    case Builtin::BIatan2:
    case Builtin::BIatan2f:
    case Builtin::BIatan2l:
    case Builtin::BI__builtin_atan2:
    case Builtin::BI__builtin_atan2f:
    case Builtin::BI__builtin_atan2f16:
    case Builtin::BI__builtin_atan2l:
    case Builtin::BI__builtin_atan2f128:
      return RValue::get(emitBinaryMaybeConstrainedFPBuiltin(
          *this, E, Intrinsic::atan2,
          Intrinsic::experimental_constrained_atan2));

    case Builtin::BIceil:
    case Builtin::BIceilf:
    case Builtin::BIceill:
    case Builtin::BI__builtin_ceil:
    case Builtin::BI__builtin_ceilf:
    case Builtin::BI__builtin_ceilf16:
    case Builtin::BI__builtin_ceill:
    case Builtin::BI__builtin_ceilf128:
      return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(*this, E,
                                   Intrinsic::ceil,
                                   Intrinsic::experimental_constrained_ceil));

    case Builtin::BIcopysign:
    case Builtin::BIcopysignf:
    case Builtin::BIcopysignl:
    case Builtin::BI__builtin_copysign:
    case Builtin::BI__builtin_copysignf:
    case Builtin::BI__builtin_copysignf16:
    case Builtin::BI__builtin_copysignl:
    case Builtin::BI__builtin_copysignf128:
      return RValue::get(
          emitBuiltinWithOneOverloadedType<2>(*this, E, Intrinsic::copysign));

    case Builtin::BIcos:
    case Builtin::BIcosf:
    case Builtin::BIcosl:
    case Builtin::BI__builtin_cos:
    case Builtin::BI__builtin_cosf:
    case Builtin::BI__builtin_cosf16:
    case Builtin::BI__builtin_cosl:
    case Builtin::BI__builtin_cosf128:
      return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(*this, E,
                                   Intrinsic::cos,
                                   Intrinsic::experimental_constrained_cos));

    case Builtin::BIcosh:
    case Builtin::BIcoshf:
    case Builtin::BIcoshl:
    case Builtin::BI__builtin_cosh:
    case Builtin::BI__builtin_coshf:
    case Builtin::BI__builtin_coshf16:
    case Builtin::BI__builtin_coshl:
    case Builtin::BI__builtin_coshf128:
      return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(
          *this, E, Intrinsic::cosh, Intrinsic::experimental_constrained_cosh));

    case Builtin::BIexp:
    case Builtin::BIexpf:
    case Builtin::BIexpl:
    case Builtin::BI__builtin_exp:
    case Builtin::BI__builtin_expf:
    case Builtin::BI__builtin_expf16:
    case Builtin::BI__builtin_expl:
    case Builtin::BI__builtin_expf128:
      return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(*this, E,
                                   Intrinsic::exp,
                                   Intrinsic::experimental_constrained_exp));

    case Builtin::BIexp2:
    case Builtin::BIexp2f:
    case Builtin::BIexp2l:
    case Builtin::BI__builtin_exp2:
    case Builtin::BI__builtin_exp2f:
    case Builtin::BI__builtin_exp2f16:
    case Builtin::BI__builtin_exp2l:
    case Builtin::BI__builtin_exp2f128:
      return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(*this, E,
                                   Intrinsic::exp2,
                                   Intrinsic::experimental_constrained_exp2));
    case Builtin::BI__builtin_exp10:
    case Builtin::BI__builtin_exp10f:
    case Builtin::BI__builtin_exp10f16:
    case Builtin::BI__builtin_exp10l:
    case Builtin::BI__builtin_exp10f128: {
      // TODO: strictfp support
      if (Builder.getIsFPConstrained())
        break;
      return RValue::get(
          emitBuiltinWithOneOverloadedType<1>(*this, E, Intrinsic::exp10));
    }
    case Builtin::BIfabs:
    case Builtin::BIfabsf:
    case Builtin::BIfabsl:
    case Builtin::BI__builtin_fabs:
    case Builtin::BI__builtin_fabsf:
    case Builtin::BI__builtin_fabsf16:
    case Builtin::BI__builtin_fabsl:
    case Builtin::BI__builtin_fabsf128:
      return RValue::get(
          emitBuiltinWithOneOverloadedType<1>(*this, E, Intrinsic::fabs));

    case Builtin::BIfloor:
    case Builtin::BIfloorf:
    case Builtin::BIfloorl:
    case Builtin::BI__builtin_floor:
    case Builtin::BI__builtin_floorf:
    case Builtin::BI__builtin_floorf16:
    case Builtin::BI__builtin_floorl:
    case Builtin::BI__builtin_floorf128:
      return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(*this, E,
                                   Intrinsic::floor,
                                   Intrinsic::experimental_constrained_floor));

    case Builtin::BIfma:
    case Builtin::BIfmaf:
    case Builtin::BIfmal:
    case Builtin::BI__builtin_fma:
    case Builtin::BI__builtin_fmaf:
    case Builtin::BI__builtin_fmaf16:
    case Builtin::BI__builtin_fmal:
    case Builtin::BI__builtin_fmaf128:
      return RValue::get(emitTernaryMaybeConstrainedFPBuiltin(*this, E,
                                   Intrinsic::fma,
                                   Intrinsic::experimental_constrained_fma));

    case Builtin::BIfmax:
    case Builtin::BIfmaxf:
    case Builtin::BIfmaxl:
    case Builtin::BI__builtin_fmax:
    case Builtin::BI__builtin_fmaxf:
    case Builtin::BI__builtin_fmaxf16:
    case Builtin::BI__builtin_fmaxl:
    case Builtin::BI__builtin_fmaxf128:
      return RValue::get(emitBinaryMaybeConstrainedFPBuiltin(*this, E,
                                   Intrinsic::maxnum,
                                   Intrinsic::experimental_constrained_maxnum));

    case Builtin::BIfmin:
    case Builtin::BIfminf:
    case Builtin::BIfminl:
    case Builtin::BI__builtin_fmin:
    case Builtin::BI__builtin_fminf:
    case Builtin::BI__builtin_fminf16:
    case Builtin::BI__builtin_fminl:
    case Builtin::BI__builtin_fminf128:
      return RValue::get(emitBinaryMaybeConstrainedFPBuiltin(*this, E,
                                   Intrinsic::minnum,
                                   Intrinsic::experimental_constrained_minnum));

    case Builtin::BIfmaximum_num:
    case Builtin::BIfmaximum_numf:
    case Builtin::BIfmaximum_numl:
    case Builtin::BI__builtin_fmaximum_num:
    case Builtin::BI__builtin_fmaximum_numf:
    case Builtin::BI__builtin_fmaximum_numf16:
    case Builtin::BI__builtin_fmaximum_numl:
    case Builtin::BI__builtin_fmaximum_numf128:
      return RValue::get(
          emitBuiltinWithOneOverloadedType<2>(*this, E, Intrinsic::maximumnum));

    case Builtin::BIfminimum_num:
    case Builtin::BIfminimum_numf:
    case Builtin::BIfminimum_numl:
    case Builtin::BI__builtin_fminimum_num:
    case Builtin::BI__builtin_fminimum_numf:
    case Builtin::BI__builtin_fminimum_numf16:
    case Builtin::BI__builtin_fminimum_numl:
    case Builtin::BI__builtin_fminimum_numf128:
      return RValue::get(
          emitBuiltinWithOneOverloadedType<2>(*this, E, Intrinsic::minimumnum));

    // fmod() is a special-case. It maps to the frem instruction rather than an
    // LLVM intrinsic.
    case Builtin::BIfmod:
    case Builtin::BIfmodf:
    case Builtin::BIfmodl:
    case Builtin::BI__builtin_fmod:
    case Builtin::BI__builtin_fmodf:
    case Builtin::BI__builtin_fmodf16:
    case Builtin::BI__builtin_fmodl:
    case Builtin::BI__builtin_fmodf128:
    case Builtin::BI__builtin_elementwise_fmod: {
      CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);
      Value *Arg1 = EmitScalarExpr(E->getArg(0));
      Value *Arg2 = EmitScalarExpr(E->getArg(1));
      return RValue::get(Builder.CreateFRem(Arg1, Arg2, "fmod"));
    }

    case Builtin::BIlog:
    case Builtin::BIlogf:
    case Builtin::BIlogl:
    case Builtin::BI__builtin_log:
    case Builtin::BI__builtin_logf:
    case Builtin::BI__builtin_logf16:
    case Builtin::BI__builtin_logl:
    case Builtin::BI__builtin_logf128:
      return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(*this, E,
                                   Intrinsic::log,
                                   Intrinsic::experimental_constrained_log));

    case Builtin::BIlog10:
    case Builtin::BIlog10f:
    case Builtin::BIlog10l:
    case Builtin::BI__builtin_log10:
    case Builtin::BI__builtin_log10f:
    case Builtin::BI__builtin_log10f16:
    case Builtin::BI__builtin_log10l:
    case Builtin::BI__builtin_log10f128:
      return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(*this, E,
                                   Intrinsic::log10,
                                   Intrinsic::experimental_constrained_log10));

    case Builtin::BIlog2:
    case Builtin::BIlog2f:
    case Builtin::BIlog2l:
    case Builtin::BI__builtin_log2:
    case Builtin::BI__builtin_log2f:
    case Builtin::BI__builtin_log2f16:
    case Builtin::BI__builtin_log2l:
    case Builtin::BI__builtin_log2f128:
      return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(*this, E,
                                   Intrinsic::log2,
                                   Intrinsic::experimental_constrained_log2));

    case Builtin::BInearbyint:
    case Builtin::BInearbyintf:
    case Builtin::BInearbyintl:
    case Builtin::BI__builtin_nearbyint:
    case Builtin::BI__builtin_nearbyintf:
    case Builtin::BI__builtin_nearbyintl:
    case Builtin::BI__builtin_nearbyintf128:
      return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(*this, E,
                                Intrinsic::nearbyint,
                                Intrinsic::experimental_constrained_nearbyint));

    case Builtin::BIpow:
    case Builtin::BIpowf:
    case Builtin::BIpowl:
    case Builtin::BI__builtin_pow:
    case Builtin::BI__builtin_powf:
    case Builtin::BI__builtin_powf16:
    case Builtin::BI__builtin_powl:
    case Builtin::BI__builtin_powf128:
      return RValue::get(emitBinaryMaybeConstrainedFPBuiltin(*this, E,
                                   Intrinsic::pow,
                                   Intrinsic::experimental_constrained_pow));

    case Builtin::BIrint:
    case Builtin::BIrintf:
    case Builtin::BIrintl:
    case Builtin::BI__builtin_rint:
    case Builtin::BI__builtin_rintf:
    case Builtin::BI__builtin_rintf16:
    case Builtin::BI__builtin_rintl:
    case Builtin::BI__builtin_rintf128:
      return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(*this, E,
                                   Intrinsic::rint,
                                   Intrinsic::experimental_constrained_rint));

    case Builtin::BIround:
    case Builtin::BIroundf:
    case Builtin::BIroundl:
    case Builtin::BI__builtin_round:
    case Builtin::BI__builtin_roundf:
    case Builtin::BI__builtin_roundf16:
    case Builtin::BI__builtin_roundl:
    case Builtin::BI__builtin_roundf128:
      return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(*this, E,
                                   Intrinsic::round,
                                   Intrinsic::experimental_constrained_round));

    case Builtin::BIroundeven:
    case Builtin::BIroundevenf:
    case Builtin::BIroundevenl:
    case Builtin::BI__builtin_roundeven:
    case Builtin::BI__builtin_roundevenf:
    case Builtin::BI__builtin_roundevenf16:
    case Builtin::BI__builtin_roundevenl:
    case Builtin::BI__builtin_roundevenf128:
      return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(*this, E,
                                   Intrinsic::roundeven,
                                   Intrinsic::experimental_constrained_roundeven));

    case Builtin::BIsin:
    case Builtin::BIsinf:
    case Builtin::BIsinl:
    case Builtin::BI__builtin_sin:
    case Builtin::BI__builtin_sinf:
    case Builtin::BI__builtin_sinf16:
    case Builtin::BI__builtin_sinl:
    case Builtin::BI__builtin_sinf128:
      return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(*this, E,
                                   Intrinsic::sin,
                                   Intrinsic::experimental_constrained_sin));

    case Builtin::BIsinh:
    case Builtin::BIsinhf:
    case Builtin::BIsinhl:
    case Builtin::BI__builtin_sinh:
    case Builtin::BI__builtin_sinhf:
    case Builtin::BI__builtin_sinhf16:
    case Builtin::BI__builtin_sinhl:
    case Builtin::BI__builtin_sinhf128:
      return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(
          *this, E, Intrinsic::sinh, Intrinsic::experimental_constrained_sinh));

    case Builtin::BI__builtin_sincospi:
    case Builtin::BI__builtin_sincospif:
    case Builtin::BI__builtin_sincospil:
      if (Builder.getIsFPConstrained())
        break; // TODO: Emit constrained sincospi intrinsic once one exists.
      emitSincosBuiltin(*this, E, Intrinsic::sincospi);
      return RValue::get(nullptr);

    case Builtin::BIsincos:
    case Builtin::BIsincosf:
    case Builtin::BIsincosl:
    case Builtin::BI__builtin_sincos:
    case Builtin::BI__builtin_sincosf:
    case Builtin::BI__builtin_sincosf16:
    case Builtin::BI__builtin_sincosl:
    case Builtin::BI__builtin_sincosf128:
      if (Builder.getIsFPConstrained())
        break; // TODO: Emit constrained sincos intrinsic once one exists.
      emitSincosBuiltin(*this, E, Intrinsic::sincos);
      return RValue::get(nullptr);

    case Builtin::BIsqrt:
    case Builtin::BIsqrtf:
    case Builtin::BIsqrtl:
    case Builtin::BI__builtin_sqrt:
    case Builtin::BI__builtin_sqrtf:
    case Builtin::BI__builtin_sqrtf16:
    case Builtin::BI__builtin_sqrtl:
    case Builtin::BI__builtin_sqrtf128:
    case Builtin::BI__builtin_elementwise_sqrt: {
      llvm::Value *Call = emitUnaryMaybeConstrainedFPBuiltin(
          *this, E, Intrinsic::sqrt, Intrinsic::experimental_constrained_sqrt);
      SetSqrtFPAccuracy(Call);
      return RValue::get(Call);
    }

    case Builtin::BItan:
    case Builtin::BItanf:
    case Builtin::BItanl:
    case Builtin::BI__builtin_tan:
    case Builtin::BI__builtin_tanf:
    case Builtin::BI__builtin_tanf16:
    case Builtin::BI__builtin_tanl:
    case Builtin::BI__builtin_tanf128:
      return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(
          *this, E, Intrinsic::tan, Intrinsic::experimental_constrained_tan));

    case Builtin::BItanh:
    case Builtin::BItanhf:
    case Builtin::BItanhl:
    case Builtin::BI__builtin_tanh:
    case Builtin::BI__builtin_tanhf:
    case Builtin::BI__builtin_tanhf16:
    case Builtin::BI__builtin_tanhl:
    case Builtin::BI__builtin_tanhf128:
      return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(
          *this, E, Intrinsic::tanh, Intrinsic::experimental_constrained_tanh));

    case Builtin::BItrunc:
    case Builtin::BItruncf:
    case Builtin::BItruncl:
    case Builtin::BI__builtin_trunc:
    case Builtin::BI__builtin_truncf:
    case Builtin::BI__builtin_truncf16:
    case Builtin::BI__builtin_truncl:
    case Builtin::BI__builtin_truncf128:
      return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(*this, E,
                                   Intrinsic::trunc,
                                   Intrinsic::experimental_constrained_trunc));

    case Builtin::BIlround:
    case Builtin::BIlroundf:
    case Builtin::BIlroundl:
    case Builtin::BI__builtin_lround:
    case Builtin::BI__builtin_lroundf:
    case Builtin::BI__builtin_lroundl:
    case Builtin::BI__builtin_lroundf128:
      return RValue::get(emitMaybeConstrainedFPToIntRoundBuiltin(
          *this, E, Intrinsic::lround,
          Intrinsic::experimental_constrained_lround));

    case Builtin::BIllround:
    case Builtin::BIllroundf:
    case Builtin::BIllroundl:
    case Builtin::BI__builtin_llround:
    case Builtin::BI__builtin_llroundf:
    case Builtin::BI__builtin_llroundl:
    case Builtin::BI__builtin_llroundf128:
      return RValue::get(emitMaybeConstrainedFPToIntRoundBuiltin(
          *this, E, Intrinsic::llround,
          Intrinsic::experimental_constrained_llround));

    case Builtin::BIlrint:
    case Builtin::BIlrintf:
    case Builtin::BIlrintl:
    case Builtin::BI__builtin_lrint:
    case Builtin::BI__builtin_lrintf:
    case Builtin::BI__builtin_lrintl:
    case Builtin::BI__builtin_lrintf128:
      return RValue::get(emitMaybeConstrainedFPToIntRoundBuiltin(
          *this, E, Intrinsic::lrint,
          Intrinsic::experimental_constrained_lrint));

    case Builtin::BIllrint:
    case Builtin::BIllrintf:
    case Builtin::BIllrintl:
    case Builtin::BI__builtin_llrint:
    case Builtin::BI__builtin_llrintf:
    case Builtin::BI__builtin_llrintl:
    case Builtin::BI__builtin_llrintf128:
      return RValue::get(emitMaybeConstrainedFPToIntRoundBuiltin(
          *this, E, Intrinsic::llrint,
          Intrinsic::experimental_constrained_llrint));
    case Builtin::BI__builtin_ldexp:
    case Builtin::BI__builtin_ldexpf:
    case Builtin::BI__builtin_ldexpl:
    case Builtin::BI__builtin_ldexpf16:
    case Builtin::BI__builtin_ldexpf128: {
      return RValue::get(emitBinaryExpMaybeConstrainedFPBuiltin(
          *this, E, Intrinsic::ldexp,
          Intrinsic::experimental_constrained_ldexp));
    }
    default:
      break;
    }
  }

  // Check NonnullAttribute/NullabilityArg and Alignment.
  auto EmitArgCheck = [&](TypeCheckKind Kind, Address A, const Expr *Arg,
                          unsigned ParmNum) {
    Value *Val = A.emitRawPointer(*this);
    EmitNonNullArgCheck(RValue::get(Val), Arg->getType(), Arg->getExprLoc(), FD,
                        ParmNum);

    if (SanOpts.has(SanitizerKind::Alignment)) {
      SanitizerSet SkippedChecks;
      SkippedChecks.set(SanitizerKind::All);
      SkippedChecks.clear(SanitizerKind::Alignment);
      SourceLocation Loc = Arg->getExprLoc();
      // Strip an implicit cast.
      if (auto *CE = dyn_cast<ImplicitCastExpr>(Arg))
        if (CE->getCastKind() == CK_BitCast)
          Arg = CE->getSubExpr();
      EmitTypeCheck(Kind, Loc, Val, Arg->getType(), A.getAlignment(),
                    SkippedChecks);
    }
  };

  switch (BuiltinIDIfNoAsmLabel) {
  default: break;
  case Builtin::BI__builtin___CFStringMakeConstantString:
  case Builtin::BI__builtin___NSStringMakeConstantString:
    return RValue::get(ConstantEmitter(*this).emitAbstract(E, E->getType()));
  case Builtin::BI__builtin_stdarg_start:
  case Builtin::BI__builtin_va_start:
  case Builtin::BI__va_start:
  case Builtin::BI__builtin_c23_va_start:
  case Builtin::BI__builtin_va_end:
    EmitVAStartEnd(BuiltinID == Builtin::BI__va_start
                       ? EmitScalarExpr(E->getArg(0))
                       : EmitVAListRef(E->getArg(0)).emitRawPointer(*this),
                   BuiltinID != Builtin::BI__builtin_va_end);
    return RValue::get(nullptr);
  case Builtin::BI__builtin_va_copy: {
    Value *DstPtr = EmitVAListRef(E->getArg(0)).emitRawPointer(*this);
    Value *SrcPtr = EmitVAListRef(E->getArg(1)).emitRawPointer(*this);
    Builder.CreateCall(CGM.getIntrinsic(Intrinsic::vacopy, {DstPtr->getType()}),
                       {DstPtr, SrcPtr});
    return RValue::get(nullptr);
  }
  case Builtin::BIabs:
  case Builtin::BIlabs:
  case Builtin::BIllabs:
  case Builtin::BI__builtin_abs:
  case Builtin::BI__builtin_labs:
  case Builtin::BI__builtin_llabs: {
    bool SanitizeOverflow = SanOpts.has(SanitizerKind::SignedIntegerOverflow);

    Value *Result;
    switch (getLangOpts().getSignedOverflowBehavior()) {
    case LangOptions::SOB_Defined:
      Result = EmitAbs(*this, EmitScalarExpr(E->getArg(0)), false);
      break;
    case LangOptions::SOB_Undefined:
      if (!SanitizeOverflow) {
        Result = EmitAbs(*this, EmitScalarExpr(E->getArg(0)), true);
        break;
      }
      [[fallthrough]];
    case LangOptions::SOB_Trapping:
      // TODO: Somehow handle the corner case when the address of abs is taken.
      Result = EmitOverflowCheckedAbs(*this, E, SanitizeOverflow);
      break;
    }
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_complex: {
    Value *Real = EmitScalarExpr(E->getArg(0));
    Value *Imag = EmitScalarExpr(E->getArg(1));
    return RValue::getComplex({Real, Imag});
  }
  case Builtin::BI__builtin_conj:
  case Builtin::BI__builtin_conjf:
  case Builtin::BI__builtin_conjl:
  case Builtin::BIconj:
  case Builtin::BIconjf:
  case Builtin::BIconjl: {
    ComplexPairTy ComplexVal = EmitComplexExpr(E->getArg(0));
    Value *Real = ComplexVal.first;
    Value *Imag = ComplexVal.second;
    Imag = Builder.CreateFNeg(Imag, "neg");
    return RValue::getComplex(std::make_pair(Real, Imag));
  }
  case Builtin::BI__builtin_creal:
  case Builtin::BI__builtin_crealf:
  case Builtin::BI__builtin_creall:
  case Builtin::BIcreal:
  case Builtin::BIcrealf:
  case Builtin::BIcreall: {
    ComplexPairTy ComplexVal = EmitComplexExpr(E->getArg(0));
    return RValue::get(ComplexVal.first);
  }

  case Builtin::BI__builtin_preserve_access_index: {
    // Only enabled preserved access index region when debuginfo
    // is available as debuginfo is needed to preserve user-level
    // access pattern.
    if (!getDebugInfo()) {
      CGM.Error(E->getExprLoc(), "using builtin_preserve_access_index() without -g");
      return RValue::get(EmitScalarExpr(E->getArg(0)));
    }

    // Nested builtin_preserve_access_index() not supported
    if (IsInPreservedAIRegion) {
      CGM.Error(E->getExprLoc(), "nested builtin_preserve_access_index() not supported");
      return RValue::get(EmitScalarExpr(E->getArg(0)));
    }

    IsInPreservedAIRegion = true;
    Value *Res = EmitScalarExpr(E->getArg(0));
    IsInPreservedAIRegion = false;
    return RValue::get(Res);
  }

  case Builtin::BI__builtin_cimag:
  case Builtin::BI__builtin_cimagf:
  case Builtin::BI__builtin_cimagl:
  case Builtin::BIcimag:
  case Builtin::BIcimagf:
  case Builtin::BIcimagl: {
    ComplexPairTy ComplexVal = EmitComplexExpr(E->getArg(0));
    return RValue::get(ComplexVal.second);
  }

  case Builtin::BI__builtin_clrsb:
  case Builtin::BI__builtin_clrsbl:
  case Builtin::BI__builtin_clrsbll: {
    // clrsb(x) -> clz(x < 0 ? ~x : x) - 1 or
    Value *ArgValue = EmitScalarExpr(E->getArg(0));

    llvm::Type *ArgType = ArgValue->getType();
    Function *F = CGM.getIntrinsic(Intrinsic::ctlz, ArgType);

    llvm::Type *ResultType = ConvertType(E->getType());
    Value *Zero = llvm::Constant::getNullValue(ArgType);
    Value *IsNeg = Builder.CreateICmpSLT(ArgValue, Zero, "isneg");
    Value *Inverse = Builder.CreateNot(ArgValue, "not");
    Value *Tmp = Builder.CreateSelect(IsNeg, Inverse, ArgValue);
    Value *Ctlz = Builder.CreateCall(F, {Tmp, Builder.getFalse()});
    Value *Result = Builder.CreateSub(Ctlz, llvm::ConstantInt::get(ArgType, 1));
    Result = Builder.CreateIntCast(Result, ResultType, /*isSigned*/true,
                                   "cast");
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_ctzs:
  case Builtin::BI__builtin_ctz:
  case Builtin::BI__builtin_ctzl:
  case Builtin::BI__builtin_ctzll:
  case Builtin::BI__builtin_ctzg:
  case Builtin::BI__builtin_elementwise_cttz: {
    bool HasFallback =
        (BuiltinIDIfNoAsmLabel == Builtin::BI__builtin_ctzg ||
         BuiltinIDIfNoAsmLabel == Builtin::BI__builtin_elementwise_cttz) &&
        E->getNumArgs() > 1;

    Value *ArgValue =
        HasFallback ? EmitScalarExpr(E->getArg(0))
                    : EmitCheckedArgForBuiltin(E->getArg(0), BCK_CTZPassedZero);

    llvm::Type *ArgType = ArgValue->getType();
    Function *F = CGM.getIntrinsic(Intrinsic::cttz, ArgType);

    llvm::Type *ResultType = ConvertType(E->getType());
    // The elementwise builtins always exhibit zero-is-undef behaviour
    Value *ZeroUndef = Builder.getInt1(
        HasFallback || getTarget().isCLZForZeroUndef() ||
        BuiltinIDIfNoAsmLabel == Builtin::BI__builtin_elementwise_cttz);
    Value *Result = Builder.CreateCall(F, {ArgValue, ZeroUndef});
    if (Result->getType() != ResultType)
      Result =
          Builder.CreateIntCast(Result, ResultType, /*isSigned*/ false, "cast");
    if (!HasFallback)
      return RValue::get(Result);

    Value *Zero = Constant::getNullValue(ArgType);
    Value *IsZero = Builder.CreateICmpEQ(ArgValue, Zero, "iszero");
    Value *FallbackValue = EmitScalarExpr(E->getArg(1));
    Value *ResultOrFallback =
        Builder.CreateSelect(IsZero, FallbackValue, Result, "ctzg");
    return RValue::get(ResultOrFallback);
  }
  case Builtin::BI__builtin_clzs:
  case Builtin::BI__builtin_clz:
  case Builtin::BI__builtin_clzl:
  case Builtin::BI__builtin_clzll:
  case Builtin::BI__builtin_clzg:
  case Builtin::BI__builtin_elementwise_ctlz: {
    bool HasFallback =
        (BuiltinIDIfNoAsmLabel == Builtin::BI__builtin_clzg ||
         BuiltinIDIfNoAsmLabel == Builtin::BI__builtin_elementwise_ctlz) &&
        E->getNumArgs() > 1;

    Value *ArgValue =
        HasFallback ? EmitScalarExpr(E->getArg(0))
                    : EmitCheckedArgForBuiltin(E->getArg(0), BCK_CLZPassedZero);

    llvm::Type *ArgType = ArgValue->getType();
    Function *F = CGM.getIntrinsic(Intrinsic::ctlz, ArgType);

    llvm::Type *ResultType = ConvertType(E->getType());
    // The elementwise builtins always exhibit zero-is-undef behaviour
    Value *ZeroUndef = Builder.getInt1(
        HasFallback || getTarget().isCLZForZeroUndef() ||
        BuiltinIDIfNoAsmLabel == Builtin::BI__builtin_elementwise_ctlz);
    Value *Result = Builder.CreateCall(F, {ArgValue, ZeroUndef});
    if (Result->getType() != ResultType)
      Result =
          Builder.CreateIntCast(Result, ResultType, /*isSigned*/ false, "cast");
    if (!HasFallback)
      return RValue::get(Result);

    Value *Zero = Constant::getNullValue(ArgType);
    Value *IsZero = Builder.CreateICmpEQ(ArgValue, Zero, "iszero");
    Value *FallbackValue = EmitScalarExpr(E->getArg(1));
    Value *ResultOrFallback =
        Builder.CreateSelect(IsZero, FallbackValue, Result, "clzg");
    return RValue::get(ResultOrFallback);
  }
  case Builtin::BI__builtin_ffs:
  case Builtin::BI__builtin_ffsl:
  case Builtin::BI__builtin_ffsll: {
    // ffs(x) -> x ? cttz(x) + 1 : 0
    Value *ArgValue = EmitScalarExpr(E->getArg(0));

    llvm::Type *ArgType = ArgValue->getType();
    Function *F = CGM.getIntrinsic(Intrinsic::cttz, ArgType);

    llvm::Type *ResultType = ConvertType(E->getType());
    Value *Tmp =
        Builder.CreateAdd(Builder.CreateCall(F, {ArgValue, Builder.getTrue()}),
                          llvm::ConstantInt::get(ArgType, 1));
    Value *Zero = llvm::Constant::getNullValue(ArgType);
    Value *IsZero = Builder.CreateICmpEQ(ArgValue, Zero, "iszero");
    Value *Result = Builder.CreateSelect(IsZero, Zero, Tmp, "ffs");
    if (Result->getType() != ResultType)
      Result = Builder.CreateIntCast(Result, ResultType, /*isSigned*/true,
                                     "cast");
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_parity:
  case Builtin::BI__builtin_parityl:
  case Builtin::BI__builtin_parityll: {
    // parity(x) -> ctpop(x) & 1
    Value *ArgValue = EmitScalarExpr(E->getArg(0));

    llvm::Type *ArgType = ArgValue->getType();
    Function *F = CGM.getIntrinsic(Intrinsic::ctpop, ArgType);

    llvm::Type *ResultType = ConvertType(E->getType());
    Value *Tmp = Builder.CreateCall(F, ArgValue);
    Value *Result = Builder.CreateAnd(Tmp, llvm::ConstantInt::get(ArgType, 1));
    if (Result->getType() != ResultType)
      Result = Builder.CreateIntCast(Result, ResultType, /*isSigned*/true,
                                     "cast");
    return RValue::get(Result);
  }
  case Builtin::BI__lzcnt16:
  case Builtin::BI__lzcnt:
  case Builtin::BI__lzcnt64: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));

    llvm::Type *ArgType = ArgValue->getType();
    Function *F = CGM.getIntrinsic(Intrinsic::ctlz, ArgType);

    llvm::Type *ResultType = ConvertType(E->getType());
    Value *Result = Builder.CreateCall(F, {ArgValue, Builder.getFalse()});
    if (Result->getType() != ResultType)
      Result = Builder.CreateIntCast(Result, ResultType, /*isSigned*/true,
                                     "cast");
    return RValue::get(Result);
  }
  case Builtin::BI__popcnt16:
  case Builtin::BI__popcnt:
  case Builtin::BI__popcnt64:
  case Builtin::BI__builtin_popcount:
  case Builtin::BI__builtin_popcountl:
  case Builtin::BI__builtin_popcountll:
  case Builtin::BI__builtin_popcountg: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));

    llvm::Type *ArgType = ArgValue->getType();
    Function *F = CGM.getIntrinsic(Intrinsic::ctpop, ArgType);

    llvm::Type *ResultType = ConvertType(E->getType());
    Value *Result = Builder.CreateCall(F, ArgValue);
    if (Result->getType() != ResultType)
      Result =
          Builder.CreateIntCast(Result, ResultType, /*isSigned*/ false, "cast");
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_unpredictable: {
    // Always return the argument of __builtin_unpredictable. LLVM does not
    // handle this builtin. Metadata for this builtin should be added directly
    // to instructions such as branches or switches that use it.
    return RValue::get(EmitScalarExpr(E->getArg(0)));
  }
  case Builtin::BI__builtin_expect: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));
    llvm::Type *ArgType = ArgValue->getType();

    Value *ExpectedValue = EmitScalarExpr(E->getArg(1));
    // Don't generate llvm.expect on -O0 as the backend won't use it for
    // anything.
    // Note, we still IRGen ExpectedValue because it could have side-effects.
    if (CGM.getCodeGenOpts().OptimizationLevel == 0)
      return RValue::get(ArgValue);

    Function *FnExpect = CGM.getIntrinsic(Intrinsic::expect, ArgType);
    Value *Result =
        Builder.CreateCall(FnExpect, {ArgValue, ExpectedValue}, "expval");
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_expect_with_probability: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));
    llvm::Type *ArgType = ArgValue->getType();

    Value *ExpectedValue = EmitScalarExpr(E->getArg(1));
    llvm::APFloat Probability(0.0);
    const Expr *ProbArg = E->getArg(2);
    bool EvalSucceed = ProbArg->EvaluateAsFloat(Probability, CGM.getContext());
    assert(EvalSucceed && "probability should be able to evaluate as float");
    (void)EvalSucceed;
    bool LoseInfo = false;
    Probability.convert(llvm::APFloat::IEEEdouble(),
                        llvm::RoundingMode::Dynamic, &LoseInfo);
    llvm::Type *Ty = ConvertType(ProbArg->getType());
    Constant *Confidence = ConstantFP::get(Ty, Probability);
    // Don't generate llvm.expect.with.probability on -O0 as the backend
    // won't use it for anything.
    // Note, we still IRGen ExpectedValue because it could have side-effects.
    if (CGM.getCodeGenOpts().OptimizationLevel == 0)
      return RValue::get(ArgValue);

    Function *FnExpect =
        CGM.getIntrinsic(Intrinsic::expect_with_probability, ArgType);
    Value *Result = Builder.CreateCall(
        FnExpect, {ArgValue, ExpectedValue, Confidence}, "expval");
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_assume_aligned: {
    const Expr *Ptr = E->getArg(0);
    Value *PtrValue = EmitScalarExpr(Ptr);
    Value *OffsetValue =
      (E->getNumArgs() > 2) ? EmitScalarExpr(E->getArg(2)) : nullptr;

    Value *AlignmentValue = EmitScalarExpr(E->getArg(1));
    ConstantInt *AlignmentCI = cast<ConstantInt>(AlignmentValue);
    if (AlignmentCI->getValue().ugt(llvm::Value::MaximumAlignment))
      AlignmentCI = ConstantInt::get(AlignmentCI->getIntegerType(),
                                     llvm::Value::MaximumAlignment);

    emitAlignmentAssumption(PtrValue, Ptr,
                            /*The expr loc is sufficient.*/ SourceLocation(),
                            AlignmentCI, OffsetValue);
    return RValue::get(PtrValue);
  }
  case Builtin::BI__builtin_assume_dereferenceable: {
    const Expr *Ptr = E->getArg(0);
    const Expr *Size = E->getArg(1);
    Value *PtrValue = EmitScalarExpr(Ptr);
    Value *SizeValue = EmitScalarExpr(Size);
    if (SizeValue->getType() != IntPtrTy)
      SizeValue =
          Builder.CreateIntCast(SizeValue, IntPtrTy, false, "casted.size");
    Builder.CreateDereferenceableAssumption(PtrValue, SizeValue);
    return RValue::get(nullptr);
  }
  case Builtin::BI__assume:
  case Builtin::BI__builtin_assume: {
    if (E->getArg(0)->HasSideEffects(getContext()))
      return RValue::get(nullptr);

    Value *ArgValue = EmitCheckedArgForAssume(E->getArg(0));
    Function *FnAssume = CGM.getIntrinsic(Intrinsic::assume);
    Builder.CreateCall(FnAssume, ArgValue);
    return RValue::get(nullptr);
  }
  case Builtin::BI__builtin_assume_separate_storage: {
    const Expr *Arg0 = E->getArg(0);
    const Expr *Arg1 = E->getArg(1);

    Value *Value0 = EmitScalarExpr(Arg0);
    Value *Value1 = EmitScalarExpr(Arg1);

    Value *Values[] = {Value0, Value1};
    OperandBundleDefT<Value *> OBD("separate_storage", Values);
    Builder.CreateAssumption(ConstantInt::getTrue(getLLVMContext()), {OBD});
    return RValue::get(nullptr);
  }
  case Builtin::BI__builtin_allow_runtime_check: {
    StringRef Kind =
        cast<StringLiteral>(E->getArg(0)->IgnoreParenCasts())->getString();
    LLVMContext &Ctx = CGM.getLLVMContext();
    llvm::Value *Allow = Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::allow_runtime_check),
        llvm::MetadataAsValue::get(Ctx, llvm::MDString::get(Ctx, Kind)));
    return RValue::get(Allow);
  }
  case Builtin::BI__arithmetic_fence: {
    // Create the builtin call if FastMath is selected, and the target
    // supports the builtin, otherwise just return the argument.
    CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);
    llvm::FastMathFlags FMF = Builder.getFastMathFlags();
    bool isArithmeticFenceEnabled =
        FMF.allowReassoc() &&
        getContext().getTargetInfo().checkArithmeticFenceSupported();
    QualType ArgType = E->getArg(0)->getType();
    if (ArgType->isComplexType()) {
      if (isArithmeticFenceEnabled) {
        QualType ElementType = ArgType->castAs<ComplexType>()->getElementType();
        ComplexPairTy ComplexVal = EmitComplexExpr(E->getArg(0));
        Value *Real = Builder.CreateArithmeticFence(ComplexVal.first,
                                                    ConvertType(ElementType));
        Value *Imag = Builder.CreateArithmeticFence(ComplexVal.second,
                                                    ConvertType(ElementType));
        return RValue::getComplex(std::make_pair(Real, Imag));
      }
      ComplexPairTy ComplexVal = EmitComplexExpr(E->getArg(0));
      Value *Real = ComplexVal.first;
      Value *Imag = ComplexVal.second;
      return RValue::getComplex(std::make_pair(Real, Imag));
    }
    Value *ArgValue = EmitScalarExpr(E->getArg(0));
    if (isArithmeticFenceEnabled)
      return RValue::get(
          Builder.CreateArithmeticFence(ArgValue, ConvertType(ArgType)));
    return RValue::get(ArgValue);
  }
  case Builtin::BI__builtin_bswap16:
  case Builtin::BI__builtin_bswap32:
  case Builtin::BI__builtin_bswap64:
  case Builtin::BI_byteswap_ushort:
  case Builtin::BI_byteswap_ulong:
  case Builtin::BI_byteswap_uint64: {
    return RValue::get(
        emitBuiltinWithOneOverloadedType<1>(*this, E, Intrinsic::bswap));
  }
  case Builtin::BI__builtin_bitreverse8:
  case Builtin::BI__builtin_bitreverse16:
  case Builtin::BI__builtin_bitreverse32:
  case Builtin::BI__builtin_bitreverse64: {
    return RValue::get(
        emitBuiltinWithOneOverloadedType<1>(*this, E, Intrinsic::bitreverse));
  }
  case Builtin::BI__builtin_rotateleft8:
  case Builtin::BI__builtin_rotateleft16:
  case Builtin::BI__builtin_rotateleft32:
  case Builtin::BI__builtin_rotateleft64:
  case Builtin::BI_rotl8: // Microsoft variants of rotate left
  case Builtin::BI_rotl16:
  case Builtin::BI_rotl:
  case Builtin::BI_lrotl:
  case Builtin::BI_rotl64:
    return emitRotate(E, false);

  case Builtin::BI__builtin_rotateright8:
  case Builtin::BI__builtin_rotateright16:
  case Builtin::BI__builtin_rotateright32:
  case Builtin::BI__builtin_rotateright64:
  case Builtin::BI_rotr8: // Microsoft variants of rotate right
  case Builtin::BI_rotr16:
  case Builtin::BI_rotr:
  case Builtin::BI_lrotr:
  case Builtin::BI_rotr64:
    return emitRotate(E, true);

  case Builtin::BI__builtin_constant_p: {
    llvm::Type *ResultType = ConvertType(E->getType());

    const Expr *Arg = E->getArg(0);
    QualType ArgType = Arg->getType();
    // FIXME: The allowance for Obj-C pointers and block pointers is historical
    // and likely a mistake.
    if (!ArgType->isIntegralOrEnumerationType() && !ArgType->isFloatingType() &&
        !ArgType->isObjCObjectPointerType() && !ArgType->isBlockPointerType())
      // Per the GCC documentation, only numeric constants are recognized after
      // inlining.
      return RValue::get(ConstantInt::get(ResultType, 0));

    if (Arg->HasSideEffects(getContext()))
      // The argument is unevaluated, so be conservative if it might have
      // side-effects.
      return RValue::get(ConstantInt::get(ResultType, 0));

    Value *ArgValue = EmitScalarExpr(Arg);
    if (ArgType->isObjCObjectPointerType()) {
      // Convert Objective-C objects to id because we cannot distinguish between
      // LLVM types for Obj-C classes as they are opaque.
      ArgType = CGM.getContext().getObjCIdType();
      ArgValue = Builder.CreateBitCast(ArgValue, ConvertType(ArgType));
    }
    Function *F =
        CGM.getIntrinsic(Intrinsic::is_constant, ConvertType(ArgType));
    Value *Result = Builder.CreateCall(F, ArgValue);
    if (Result->getType() != ResultType)
      Result = Builder.CreateIntCast(Result, ResultType, /*isSigned*/false);
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_dynamic_object_size:
  case Builtin::BI__builtin_object_size: {
    unsigned Type =
        E->getArg(1)->EvaluateKnownConstInt(getContext()).getZExtValue();
    auto *ResType = cast<llvm::IntegerType>(ConvertType(E->getType()));

    // We pass this builtin onto the optimizer so that it can figure out the
    // object size in more complex cases.
    bool IsDynamic = BuiltinID == Builtin::BI__builtin_dynamic_object_size;
    return RValue::get(emitBuiltinObjectSize(E->getArg(0), Type, ResType,
                                             /*EmittedE=*/nullptr, IsDynamic));
  }
  case Builtin::BI__builtin_counted_by_ref: {
    // Default to returning '(void *) 0'.
    llvm::Value *Result = llvm::ConstantPointerNull::get(
        llvm::PointerType::getUnqual(getLLVMContext()));

    const Expr *Arg = E->getArg(0)->IgnoreParenImpCasts();

    if (auto *UO = dyn_cast<UnaryOperator>(Arg);
        UO && UO->getOpcode() == UO_AddrOf) {
      Arg = UO->getSubExpr()->IgnoreParenImpCasts();

      if (auto *ASE = dyn_cast<ArraySubscriptExpr>(Arg))
        Arg = ASE->getBase()->IgnoreParenImpCasts();
    }

    if (const MemberExpr *ME = dyn_cast_if_present<MemberExpr>(Arg)) {
      if (auto *CATy =
              ME->getMemberDecl()->getType()->getAs<CountAttributedType>();
          CATy && CATy->getKind() == CountAttributedType::CountedBy) {
        const auto *FAMDecl = cast<FieldDecl>(ME->getMemberDecl());
        if (const FieldDecl *CountFD = FAMDecl->findCountedByField())
          Result = GetCountedByFieldExprGEP(Arg, FAMDecl, CountFD);
        else
          llvm::report_fatal_error("Cannot find the counted_by 'count' field");
      }
    }

    return RValue::get(Result);
  }
  case Builtin::BI__builtin_prefetch: {
    Value *Locality, *RW, *Address = EmitScalarExpr(E->getArg(0));
    // FIXME: Technically these constants should of type 'int', yes?
    RW = (E->getNumArgs() > 1) ? EmitScalarExpr(E->getArg(1)) :
      llvm::ConstantInt::get(Int32Ty, 0);
    Locality = (E->getNumArgs() > 2) ? EmitScalarExpr(E->getArg(2)) :
      llvm::ConstantInt::get(Int32Ty, 3);
    Value *Data = llvm::ConstantInt::get(Int32Ty, 1);
    Function *F = CGM.getIntrinsic(Intrinsic::prefetch, Address->getType());
    Builder.CreateCall(F, {Address, RW, Locality, Data});
    return RValue::get(nullptr);
  }
  case Builtin::BI__builtin_readcyclecounter: {
    Function *F = CGM.getIntrinsic(Intrinsic::readcyclecounter);
    return RValue::get(Builder.CreateCall(F));
  }
  case Builtin::BI__builtin_readsteadycounter: {
    Function *F = CGM.getIntrinsic(Intrinsic::readsteadycounter);
    return RValue::get(Builder.CreateCall(F));
  }
  case Builtin::BI__builtin___clear_cache: {
    Value *Begin = EmitScalarExpr(E->getArg(0));
    Value *End = EmitScalarExpr(E->getArg(1));
    Function *F = CGM.getIntrinsic(Intrinsic::clear_cache);
    return RValue::get(Builder.CreateCall(F, {Begin, End}));
  }
  case Builtin::BI__builtin_trap:
    EmitTrapCall(Intrinsic::trap);
    return RValue::get(nullptr);
  case Builtin::BI__builtin_verbose_trap: {
    llvm::DILocation *TrapLocation = Builder.getCurrentDebugLocation();
    if (getDebugInfo()) {
      TrapLocation = getDebugInfo()->CreateTrapFailureMessageFor(
          TrapLocation, *E->getArg(0)->tryEvaluateString(getContext()),
          *E->getArg(1)->tryEvaluateString(getContext()));
    }
    ApplyDebugLocation ApplyTrapDI(*this, TrapLocation);
    // Currently no attempt is made to prevent traps from being merged.
    EmitTrapCall(Intrinsic::trap);
    return RValue::get(nullptr);
  }
  case Builtin::BI__debugbreak:
    EmitTrapCall(Intrinsic::debugtrap);
    return RValue::get(nullptr);
  case Builtin::BI__builtin_unreachable: {
    EmitUnreachable(E->getExprLoc());

    // We do need to preserve an insertion point.
    EmitBlock(createBasicBlock("unreachable.cont"));

    return RValue::get(nullptr);
  }

  case Builtin::BI__builtin_powi:
  case Builtin::BI__builtin_powif:
  case Builtin::BI__builtin_powil: {
    llvm::Value *Src0 = EmitScalarExpr(E->getArg(0));
    llvm::Value *Src1 = EmitScalarExpr(E->getArg(1));

    if (Builder.getIsFPConstrained()) {
      // FIXME: llvm.powi has 2 mangling types,
      // llvm.experimental.constrained.powi has one.
      CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);
      Function *F = CGM.getIntrinsic(Intrinsic::experimental_constrained_powi,
                                     Src0->getType());
      return RValue::get(Builder.CreateConstrainedFPCall(F, { Src0, Src1 }));
    }

    Function *F = CGM.getIntrinsic(Intrinsic::powi,
                                   { Src0->getType(), Src1->getType() });
    return RValue::get(Builder.CreateCall(F, { Src0, Src1 }));
  }
  case Builtin::BI__builtin_frexpl: {
    // Linux PPC will not be adding additional PPCDoubleDouble support.
    // WIP to switch default to IEEE long double. Will emit libcall for
    // frexpl instead of legalizing this type in the BE.
    if (&getTarget().getLongDoubleFormat() == &llvm::APFloat::PPCDoubleDouble())
      break;
    [[fallthrough]];
  }
  case Builtin::BI__builtin_frexp:
  case Builtin::BI__builtin_frexpf:
  case Builtin::BI__builtin_frexpf128:
  case Builtin::BI__builtin_frexpf16:
    return RValue::get(emitFrexpBuiltin(*this, E, Intrinsic::frexp));
  case Builtin::BImodf:
  case Builtin::BImodff:
  case Builtin::BImodfl:
  case Builtin::BI__builtin_modf:
  case Builtin::BI__builtin_modff:
  case Builtin::BI__builtin_modfl:
    if (Builder.getIsFPConstrained())
      break; // TODO: Emit constrained modf intrinsic once one exists.
    return RValue::get(emitModfBuiltin(*this, E, Intrinsic::modf));
  case Builtin::BI__builtin_isgreater:
  case Builtin::BI__builtin_isgreaterequal:
  case Builtin::BI__builtin_isless:
  case Builtin::BI__builtin_islessequal:
  case Builtin::BI__builtin_islessgreater:
  case Builtin::BI__builtin_isunordered: {
    // Ordered comparisons: we know the arguments to these are matching scalar
    // floating point values.
    CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);
    Value *LHS = EmitScalarExpr(E->getArg(0));
    Value *RHS = EmitScalarExpr(E->getArg(1));

    switch (BuiltinID) {
    default: llvm_unreachable("Unknown ordered comparison");
    case Builtin::BI__builtin_isgreater:
      LHS = Builder.CreateFCmpOGT(LHS, RHS, "cmp");
      break;
    case Builtin::BI__builtin_isgreaterequal:
      LHS = Builder.CreateFCmpOGE(LHS, RHS, "cmp");
      break;
    case Builtin::BI__builtin_isless:
      LHS = Builder.CreateFCmpOLT(LHS, RHS, "cmp");
      break;
    case Builtin::BI__builtin_islessequal:
      LHS = Builder.CreateFCmpOLE(LHS, RHS, "cmp");
      break;
    case Builtin::BI__builtin_islessgreater:
      LHS = Builder.CreateFCmpONE(LHS, RHS, "cmp");
      break;
    case Builtin::BI__builtin_isunordered:
      LHS = Builder.CreateFCmpUNO(LHS, RHS, "cmp");
      break;
    }
    // ZExt bool to int type.
    return RValue::get(Builder.CreateZExt(LHS, ConvertType(E->getType())));
  }

  case Builtin::BI__builtin_isnan: {
    CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);
    Value *V = EmitScalarExpr(E->getArg(0));
    if (Value *Result = tryUseTestFPKind(*this, BuiltinID, V))
      return RValue::get(Result);
    return RValue::get(
        Builder.CreateZExt(Builder.createIsFPClass(V, FPClassTest::fcNan),
                           ConvertType(E->getType())));
  }

  case Builtin::BI__builtin_issignaling: {
    CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);
    Value *V = EmitScalarExpr(E->getArg(0));
    return RValue::get(
        Builder.CreateZExt(Builder.createIsFPClass(V, FPClassTest::fcSNan),
                           ConvertType(E->getType())));
  }

  case Builtin::BI__builtin_isinf: {
    CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);
    Value *V = EmitScalarExpr(E->getArg(0));
    if (Value *Result = tryUseTestFPKind(*this, BuiltinID, V))
      return RValue::get(Result);
    return RValue::get(
        Builder.CreateZExt(Builder.createIsFPClass(V, FPClassTest::fcInf),
                           ConvertType(E->getType())));
  }

  case Builtin::BIfinite:
  case Builtin::BI__finite:
  case Builtin::BIfinitef:
  case Builtin::BI__finitef:
  case Builtin::BIfinitel:
  case Builtin::BI__finitel:
  case Builtin::BI__builtin_isfinite: {
    CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);
    Value *V = EmitScalarExpr(E->getArg(0));
    if (Value *Result = tryUseTestFPKind(*this, BuiltinID, V))
      return RValue::get(Result);
    return RValue::get(
        Builder.CreateZExt(Builder.createIsFPClass(V, FPClassTest::fcFinite),
                           ConvertType(E->getType())));
  }

  case Builtin::BI__builtin_isnormal: {
    CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);
    Value *V = EmitScalarExpr(E->getArg(0));
    return RValue::get(
        Builder.CreateZExt(Builder.createIsFPClass(V, FPClassTest::fcNormal),
                           ConvertType(E->getType())));
  }

  case Builtin::BI__builtin_issubnormal: {
    CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);
    Value *V = EmitScalarExpr(E->getArg(0));
    return RValue::get(
        Builder.CreateZExt(Builder.createIsFPClass(V, FPClassTest::fcSubnormal),
                           ConvertType(E->getType())));
  }

  case Builtin::BI__builtin_iszero: {
    CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);
    Value *V = EmitScalarExpr(E->getArg(0));
    return RValue::get(
        Builder.CreateZExt(Builder.createIsFPClass(V, FPClassTest::fcZero),
                           ConvertType(E->getType())));
  }

  case Builtin::BI__builtin_isfpclass: {
    Expr::EvalResult Result;
    if (!E->getArg(1)->EvaluateAsInt(Result, CGM.getContext()))
      break;
    uint64_t Test = Result.Val.getInt().getLimitedValue();
    CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);
    Value *V = EmitScalarExpr(E->getArg(0));
    return RValue::get(Builder.CreateZExt(Builder.createIsFPClass(V, Test),
                                          ConvertType(E->getType())));
  }

  case Builtin::BI__builtin_nondeterministic_value: {
    llvm::Type *Ty = ConvertType(E->getArg(0)->getType());

    Value *Result = PoisonValue::get(Ty);
    Result = Builder.CreateFreeze(Result);

    return RValue::get(Result);
  }

  case Builtin::BI__builtin_elementwise_abs: {
    Value *Result;
    QualType QT = E->getArg(0)->getType();

    if (auto *VecTy = QT->getAs<VectorType>())
      QT = VecTy->getElementType();
    if (QT->isIntegerType())
      Result = Builder.CreateBinaryIntrinsic(
          Intrinsic::abs, EmitScalarExpr(E->getArg(0)), Builder.getFalse(),
          nullptr, "elt.abs");
    else
      Result = emitBuiltinWithOneOverloadedType<1>(*this, E, Intrinsic::fabs,
                                                   "elt.abs");

    return RValue::get(Result);
  }
  case Builtin::BI__builtin_elementwise_acos:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::acos, "elt.acos"));
  case Builtin::BI__builtin_elementwise_asin:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::asin, "elt.asin"));
  case Builtin::BI__builtin_elementwise_atan:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::atan, "elt.atan"));
  case Builtin::BI__builtin_elementwise_atan2:
    return RValue::get(emitBuiltinWithOneOverloadedType<2>(
        *this, E, Intrinsic::atan2, "elt.atan2"));
  case Builtin::BI__builtin_elementwise_ceil:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::ceil, "elt.ceil"));
  case Builtin::BI__builtin_elementwise_exp:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::exp, "elt.exp"));
  case Builtin::BI__builtin_elementwise_exp2:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::exp2, "elt.exp2"));
  case Builtin::BI__builtin_elementwise_exp10:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::exp10, "elt.exp10"));
  case Builtin::BI__builtin_elementwise_log:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::log, "elt.log"));
  case Builtin::BI__builtin_elementwise_log2:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::log2, "elt.log2"));
  case Builtin::BI__builtin_elementwise_log10:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::log10, "elt.log10"));
  case Builtin::BI__builtin_elementwise_pow: {
    return RValue::get(
        emitBuiltinWithOneOverloadedType<2>(*this, E, Intrinsic::pow));
  }
  case Builtin::BI__builtin_elementwise_bitreverse:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::bitreverse, "elt.bitreverse"));
  case Builtin::BI__builtin_elementwise_cos:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::cos, "elt.cos"));
  case Builtin::BI__builtin_elementwise_cosh:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::cosh, "elt.cosh"));
  case Builtin::BI__builtin_elementwise_floor:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::floor, "elt.floor"));
  case Builtin::BI__builtin_elementwise_popcount:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::ctpop, "elt.ctpop"));
  case Builtin::BI__builtin_elementwise_roundeven:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::roundeven, "elt.roundeven"));
  case Builtin::BI__builtin_elementwise_round:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::round, "elt.round"));
  case Builtin::BI__builtin_elementwise_rint:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::rint, "elt.rint"));
  case Builtin::BI__builtin_elementwise_nearbyint:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::nearbyint, "elt.nearbyint"));
  case Builtin::BI__builtin_elementwise_sin:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::sin, "elt.sin"));
  case Builtin::BI__builtin_elementwise_sinh:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::sinh, "elt.sinh"));
  case Builtin::BI__builtin_elementwise_tan:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::tan, "elt.tan"));
  case Builtin::BI__builtin_elementwise_tanh:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::tanh, "elt.tanh"));
  case Builtin::BI__builtin_elementwise_trunc:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::trunc, "elt.trunc"));
  case Builtin::BI__builtin_elementwise_canonicalize:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::canonicalize, "elt.canonicalize"));
  case Builtin::BI__builtin_elementwise_copysign:
    return RValue::get(
        emitBuiltinWithOneOverloadedType<2>(*this, E, Intrinsic::copysign));
  case Builtin::BI__builtin_elementwise_fma:
    return RValue::get(
        emitBuiltinWithOneOverloadedType<3>(*this, E, Intrinsic::fma));
  case Builtin::BI__builtin_elementwise_fshl:
    return RValue::get(
        emitBuiltinWithOneOverloadedType<3>(*this, E, Intrinsic::fshl));
  case Builtin::BI__builtin_elementwise_fshr:
    return RValue::get(
        emitBuiltinWithOneOverloadedType<3>(*this, E, Intrinsic::fshr));

  case Builtin::BI__builtin_elementwise_add_sat:
  case Builtin::BI__builtin_elementwise_sub_sat: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Result;
    assert(Op0->getType()->isIntOrIntVectorTy() && "integer type expected");
    QualType Ty = E->getArg(0)->getType();
    if (auto *VecTy = Ty->getAs<VectorType>())
      Ty = VecTy->getElementType();
    bool IsSigned = Ty->isSignedIntegerType();
    unsigned Opc;
    if (BuiltinIDIfNoAsmLabel == Builtin::BI__builtin_elementwise_add_sat)
      Opc = IsSigned ? Intrinsic::sadd_sat : Intrinsic::uadd_sat;
    else
      Opc = IsSigned ? Intrinsic::ssub_sat : Intrinsic::usub_sat;
    Result = Builder.CreateBinaryIntrinsic(Opc, Op0, Op1, nullptr, "elt.sat");
    return RValue::get(Result);
  }

  case Builtin::BI__builtin_elementwise_max: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Result;
    if (Op0->getType()->isIntOrIntVectorTy()) {
      QualType Ty = E->getArg(0)->getType();
      if (auto *VecTy = Ty->getAs<VectorType>())
        Ty = VecTy->getElementType();
      Result = Builder.CreateBinaryIntrinsic(
          Ty->isSignedIntegerType() ? Intrinsic::smax : Intrinsic::umax, Op0,
          Op1, nullptr, "elt.max");
    } else
      Result = Builder.CreateMaxNum(Op0, Op1, /*FMFSource=*/nullptr, "elt.max");
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_elementwise_min: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Result;
    if (Op0->getType()->isIntOrIntVectorTy()) {
      QualType Ty = E->getArg(0)->getType();
      if (auto *VecTy = Ty->getAs<VectorType>())
        Ty = VecTy->getElementType();
      Result = Builder.CreateBinaryIntrinsic(
          Ty->isSignedIntegerType() ? Intrinsic::smin : Intrinsic::umin, Op0,
          Op1, nullptr, "elt.min");
    } else
      Result = Builder.CreateMinNum(Op0, Op1, /*FMFSource=*/nullptr, "elt.min");
    return RValue::get(Result);
  }

  case Builtin::BI__builtin_elementwise_maxnum: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Result = Builder.CreateBinaryIntrinsic(llvm::Intrinsic::maxnum, Op0,
                                                  Op1, nullptr, "elt.maxnum");
    return RValue::get(Result);
  }

  case Builtin::BI__builtin_elementwise_minnum: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Result = Builder.CreateBinaryIntrinsic(llvm::Intrinsic::minnum, Op0,
                                                  Op1, nullptr, "elt.minnum");
    return RValue::get(Result);
  }

  case Builtin::BI__builtin_elementwise_maximum: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Result = Builder.CreateBinaryIntrinsic(Intrinsic::maximum, Op0, Op1,
                                                  nullptr, "elt.maximum");
    return RValue::get(Result);
  }

  case Builtin::BI__builtin_elementwise_minimum: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Result = Builder.CreateBinaryIntrinsic(Intrinsic::minimum, Op0, Op1,
                                                  nullptr, "elt.minimum");
    return RValue::get(Result);
  }

  case Builtin::BI__builtin_elementwise_maximumnum: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Result = Builder.CreateBinaryIntrinsic(
        Intrinsic::maximumnum, Op0, Op1, nullptr, "elt.maximumnum");
    return RValue::get(Result);
  }

  case Builtin::BI__builtin_elementwise_minimumnum: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Result = Builder.CreateBinaryIntrinsic(
        Intrinsic::minimumnum, Op0, Op1, nullptr, "elt.minimumnum");
    return RValue::get(Result);
  }

  case Builtin::BI__builtin_reduce_max: {
    auto GetIntrinsicID = [this](QualType QT) {
      if (auto *VecTy = QT->getAs<VectorType>())
        QT = VecTy->getElementType();
      else if (QT->isSizelessVectorType())
        QT = QT->getSizelessVectorEltType(CGM.getContext());

      if (QT->isSignedIntegerType())
        return Intrinsic::vector_reduce_smax;
      if (QT->isUnsignedIntegerType())
        return Intrinsic::vector_reduce_umax;
      assert(QT->isFloatingType() && "must have a float here");
      return Intrinsic::vector_reduce_fmax;
    };
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, GetIntrinsicID(E->getArg(0)->getType()), "rdx.min"));
  }

  case Builtin::BI__builtin_reduce_min: {
    auto GetIntrinsicID = [this](QualType QT) {
      if (auto *VecTy = QT->getAs<VectorType>())
        QT = VecTy->getElementType();
      else if (QT->isSizelessVectorType())
        QT = QT->getSizelessVectorEltType(CGM.getContext());

      if (QT->isSignedIntegerType())
        return Intrinsic::vector_reduce_smin;
      if (QT->isUnsignedIntegerType())
        return Intrinsic::vector_reduce_umin;
      assert(QT->isFloatingType() && "must have a float here");
      return Intrinsic::vector_reduce_fmin;
    };

    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, GetIntrinsicID(E->getArg(0)->getType()), "rdx.min"));
  }

  case Builtin::BI__builtin_reduce_add:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::vector_reduce_add, "rdx.add"));
  case Builtin::BI__builtin_reduce_mul:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::vector_reduce_mul, "rdx.mul"));
  case Builtin::BI__builtin_reduce_xor:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::vector_reduce_xor, "rdx.xor"));
  case Builtin::BI__builtin_reduce_or:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::vector_reduce_or, "rdx.or"));
  case Builtin::BI__builtin_reduce_and:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::vector_reduce_and, "rdx.and"));
  case Builtin::BI__builtin_reduce_maximum:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::vector_reduce_fmaximum, "rdx.maximum"));
  case Builtin::BI__builtin_reduce_minimum:
    return RValue::get(emitBuiltinWithOneOverloadedType<1>(
        *this, E, Intrinsic::vector_reduce_fminimum, "rdx.minimum"));

  case Builtin::BI__builtin_matrix_transpose: {
    auto *MatrixTy = E->getArg(0)->getType()->castAs<ConstantMatrixType>();
    Value *MatValue = EmitScalarExpr(E->getArg(0));
    MatrixBuilder MB(Builder);
    Value *Result = MB.CreateMatrixTranspose(MatValue, MatrixTy->getNumRows(),
                                             MatrixTy->getNumColumns());
    return RValue::get(Result);
  }

  case Builtin::BI__builtin_matrix_column_major_load: {
    MatrixBuilder MB(Builder);
    // Emit everything that isn't dependent on the first parameter type
    Value *Stride = EmitScalarExpr(E->getArg(3));
    const auto *ResultTy = E->getType()->getAs<ConstantMatrixType>();
    auto *PtrTy = E->getArg(0)->getType()->getAs<PointerType>();
    assert(PtrTy && "arg0 must be of pointer type");
    bool IsVolatile = PtrTy->getPointeeType().isVolatileQualified();

    Address Src = EmitPointerWithAlignment(E->getArg(0));
    EmitNonNullArgCheck(RValue::get(Src.emitRawPointer(*this)),
                        E->getArg(0)->getType(), E->getArg(0)->getExprLoc(), FD,
                        0);
    Value *Result = MB.CreateColumnMajorLoad(
        Src.getElementType(), Src.emitRawPointer(*this),
        Align(Src.getAlignment().getQuantity()), Stride, IsVolatile,
        ResultTy->getNumRows(), ResultTy->getNumColumns(), "matrix");
    return RValue::get(Result);
  }

  case Builtin::BI__builtin_matrix_column_major_store: {
    MatrixBuilder MB(Builder);
    Value *Matrix = EmitScalarExpr(E->getArg(0));
    Address Dst = EmitPointerWithAlignment(E->getArg(1));
    Value *Stride = EmitScalarExpr(E->getArg(2));

    const auto *MatrixTy = E->getArg(0)->getType()->getAs<ConstantMatrixType>();
    auto *PtrTy = E->getArg(1)->getType()->getAs<PointerType>();
    assert(PtrTy && "arg1 must be of pointer type");
    bool IsVolatile = PtrTy->getPointeeType().isVolatileQualified();

    EmitNonNullArgCheck(RValue::get(Dst.emitRawPointer(*this)),
                        E->getArg(1)->getType(), E->getArg(1)->getExprLoc(), FD,
                        0);
    Value *Result = MB.CreateColumnMajorStore(
        Matrix, Dst.emitRawPointer(*this),
        Align(Dst.getAlignment().getQuantity()), Stride, IsVolatile,
        MatrixTy->getNumRows(), MatrixTy->getNumColumns());
    addInstToNewSourceAtom(cast<Instruction>(Result), Matrix);
    return RValue::get(Result);
  }

  case Builtin::BI__builtin_masked_load: {
    llvm::Value *Mask = EmitScalarExpr(E->getArg(0));
    llvm::Value *Ptr = EmitScalarExpr(E->getArg(1));

    llvm::Type *RetTy = CGM.getTypes().ConvertType(E->getType());
    CharUnits Align = CGM.getNaturalTypeAlignment(E->getType(), nullptr);
    llvm::Value *AlignVal =
        llvm::ConstantInt::get(Int32Ty, Align.getQuantity());

    llvm::Value *PassThru = llvm::PoisonValue::get(RetTy);

    Function *F =
        CGM.getIntrinsic(Intrinsic::masked_load, {RetTy, UnqualPtrTy});

    llvm::Value *Result =
        Builder.CreateCall(F, {Ptr, AlignVal, Mask, PassThru}, "masked_load");
    return RValue::get(Result);
  };
  case Builtin::BI__builtin_masked_store: {
    llvm::Value *Mask = EmitScalarExpr(E->getArg(0));
    llvm::Value *Val = EmitScalarExpr(E->getArg(1));
    llvm::Value *Ptr = EmitScalarExpr(E->getArg(2));

    QualType ValTy = E->getArg(1)->getType();
    llvm::Type *ValLLTy = CGM.getTypes().ConvertType(ValTy);
    llvm::Type *PtrTy = Ptr->getType();

    CharUnits Align = CGM.getNaturalTypeAlignment(ValTy, nullptr);
    llvm::Value *AlignVal =
        llvm::ConstantInt::get(Int32Ty, Align.getQuantity());

    llvm::Function *F =
        CGM.getIntrinsic(llvm::Intrinsic::masked_store, {ValLLTy, PtrTy});

    Builder.CreateCall(F, {Val, Ptr, AlignVal, Mask});
    return RValue::get(nullptr);
  }

  case Builtin::BI__builtin_isinf_sign: {
    // isinf_sign(x) -> fabs(x) == infinity ? (signbit(x) ? -1 : 1) : 0
    CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);
    // FIXME: for strictfp/IEEE-754 we need to not trap on SNaN here.
    Value *Arg = EmitScalarExpr(E->getArg(0));
    Value *AbsArg = EmitFAbs(*this, Arg);
    Value *IsInf = Builder.CreateFCmpOEQ(
        AbsArg, ConstantFP::getInfinity(Arg->getType()), "isinf");
    Value *IsNeg = EmitSignBit(*this, Arg);

    llvm::Type *IntTy = ConvertType(E->getType());
    Value *Zero = Constant::getNullValue(IntTy);
    Value *One = ConstantInt::get(IntTy, 1);
    Value *NegativeOne = ConstantInt::get(IntTy, -1);
    Value *SignResult = Builder.CreateSelect(IsNeg, NegativeOne, One);
    Value *Result = Builder.CreateSelect(IsInf, SignResult, Zero);
    return RValue::get(Result);
  }

  case Builtin::BI__builtin_flt_rounds: {
    Function *F = CGM.getIntrinsic(Intrinsic::get_rounding);

    llvm::Type *ResultType = ConvertType(E->getType());
    Value *Result = Builder.CreateCall(F);
    if (Result->getType() != ResultType)
      Result = Builder.CreateIntCast(Result, ResultType, /*isSigned*/true,
                                     "cast");
    return RValue::get(Result);
  }

  case Builtin::BI__builtin_set_flt_rounds: {
    Function *F = CGM.getIntrinsic(Intrinsic::set_rounding);

    Value *V = EmitScalarExpr(E->getArg(0));
    Builder.CreateCall(F, V);
    return RValue::get(nullptr);
  }

  case Builtin::BI__builtin_fpclassify: {
    CodeGenFunction::CGFPOptionsRAII FPOptsRAII(*this, E);
    // FIXME: for strictfp/IEEE-754 we need to not trap on SNaN here.
    Value *V = EmitScalarExpr(E->getArg(5));
    llvm::Type *Ty = ConvertType(E->getArg(5)->getType());

    // Create Result
    BasicBlock *Begin = Builder.GetInsertBlock();
    BasicBlock *End = createBasicBlock("fpclassify_end", this->CurFn);
    Builder.SetInsertPoint(End);
    PHINode *Result =
      Builder.CreatePHI(ConvertType(E->getArg(0)->getType()), 4,
                        "fpclassify_result");

    // if (V==0) return FP_ZERO
    Builder.SetInsertPoint(Begin);
    Value *IsZero = Builder.CreateFCmpOEQ(V, Constant::getNullValue(Ty),
                                          "iszero");
    Value *ZeroLiteral = EmitScalarExpr(E->getArg(4));
    BasicBlock *NotZero = createBasicBlock("fpclassify_not_zero", this->CurFn);
    Builder.CreateCondBr(IsZero, End, NotZero);
    Result->addIncoming(ZeroLiteral, Begin);

    // if (V != V) return FP_NAN
    Builder.SetInsertPoint(NotZero);
    Value *IsNan = Builder.CreateFCmpUNO(V, V, "cmp");
    Value *NanLiteral = EmitScalarExpr(E->getArg(0));
    BasicBlock *NotNan = createBasicBlock("fpclassify_not_nan", this->CurFn);
    Builder.CreateCondBr(IsNan, End, NotNan);
    Result->addIncoming(NanLiteral, NotZero);

    // if (fabs(V) == infinity) return FP_INFINITY
    Builder.SetInsertPoint(NotNan);
    Value *VAbs = EmitFAbs(*this, V);
    Value *IsInf =
      Builder.CreateFCmpOEQ(VAbs, ConstantFP::getInfinity(V->getType()),
                            "isinf");
    Value *InfLiteral = EmitScalarExpr(E->getArg(1));
    BasicBlock *NotInf = createBasicBlock("fpclassify_not_inf", this->CurFn);
    Builder.CreateCondBr(IsInf, End, NotInf);
    Result->addIncoming(InfLiteral, NotNan);

    // if (fabs(V) >= MIN_NORMAL) return FP_NORMAL else FP_SUBNORMAL
    Builder.SetInsertPoint(NotInf);
    APFloat Smallest = APFloat::getSmallestNormalized(
        getContext().getFloatTypeSemantics(E->getArg(5)->getType()));
    Value *IsNormal =
      Builder.CreateFCmpUGE(VAbs, ConstantFP::get(V->getContext(), Smallest),
                            "isnormal");
    Value *NormalResult =
      Builder.CreateSelect(IsNormal, EmitScalarExpr(E->getArg(2)),
                           EmitScalarExpr(E->getArg(3)));
    Builder.CreateBr(End);
    Result->addIncoming(NormalResult, NotInf);

    // return Result
    Builder.SetInsertPoint(End);
    return RValue::get(Result);
  }

  // An alloca will always return a pointer to the alloca (stack) address
  // space. This address space need not be the same as the AST / Language
  // default (e.g. in C / C++ auto vars are in the generic address space). At
  // the AST level this is handled within CreateTempAlloca et al., but for the
  // builtin / dynamic alloca we have to handle it here. We use an explicit cast
  // instead of passing an AS to CreateAlloca so as to not inhibit optimisation.
  case Builtin::BIalloca:
  case Builtin::BI_alloca:
  case Builtin::BI__builtin_alloca_uninitialized:
  case Builtin::BI__builtin_alloca: {
    Value *Size = EmitScalarExpr(E->getArg(0));
    const TargetInfo &TI = getContext().getTargetInfo();
    // The alignment of the alloca should correspond to __BIGGEST_ALIGNMENT__.
    const Align SuitableAlignmentInBytes =
        CGM.getContext()
            .toCharUnitsFromBits(TI.getSuitableAlign())
            .getAsAlign();
    AllocaInst *AI = Builder.CreateAlloca(Builder.getInt8Ty(), Size);
    AI->setAlignment(SuitableAlignmentInBytes);
    if (BuiltinID != Builtin::BI__builtin_alloca_uninitialized)
      initializeAlloca(*this, AI, Size, SuitableAlignmentInBytes);
    LangAS AAS = getASTAllocaAddressSpace();
    LangAS EAS = E->getType()->getPointeeType().getAddressSpace();
    if (AAS != EAS) {
      llvm::Type *Ty = CGM.getTypes().ConvertType(E->getType());
      return RValue::get(
          getTargetHooks().performAddrSpaceCast(*this, AI, AAS, Ty));
    }
    return RValue::get(AI);
  }

  case Builtin::BI__builtin_alloca_with_align_uninitialized:
  case Builtin::BI__builtin_alloca_with_align: {
    Value *Size = EmitScalarExpr(E->getArg(0));
    Value *AlignmentInBitsValue = EmitScalarExpr(E->getArg(1));
    auto *AlignmentInBitsCI = cast<ConstantInt>(AlignmentInBitsValue);
    unsigned AlignmentInBits = AlignmentInBitsCI->getZExtValue();
    const Align AlignmentInBytes =
        CGM.getContext().toCharUnitsFromBits(AlignmentInBits).getAsAlign();
    AllocaInst *AI = Builder.CreateAlloca(Builder.getInt8Ty(), Size);
    AI->setAlignment(AlignmentInBytes);
    if (BuiltinID != Builtin::BI__builtin_alloca_with_align_uninitialized)
      initializeAlloca(*this, AI, Size, AlignmentInBytes);
    LangAS AAS = getASTAllocaAddressSpace();
    LangAS EAS = E->getType()->getPointeeType().getAddressSpace();
    if (AAS != EAS) {
      llvm::Type *Ty = CGM.getTypes().ConvertType(E->getType());
      return RValue::get(
          getTargetHooks().performAddrSpaceCast(*this, AI, AAS, Ty));
    }
    return RValue::get(AI);
  }

  case Builtin::BIbzero:
  case Builtin::BI__builtin_bzero: {
    Address Dest = EmitPointerWithAlignment(E->getArg(0));
    Value *SizeVal = EmitScalarExpr(E->getArg(1));
    EmitNonNullArgCheck(Dest, E->getArg(0)->getType(),
                        E->getArg(0)->getExprLoc(), FD, 0);
    auto *I = Builder.CreateMemSet(Dest, Builder.getInt8(0), SizeVal, false);
    addInstToNewSourceAtom(I, nullptr);
    return RValue::get(nullptr);
  }

  case Builtin::BIbcopy:
  case Builtin::BI__builtin_bcopy: {
    Address Src = EmitPointerWithAlignment(E->getArg(0));
    Address Dest = EmitPointerWithAlignment(E->getArg(1));
    Value *SizeVal = EmitScalarExpr(E->getArg(2));
    EmitNonNullArgCheck(RValue::get(Src.emitRawPointer(*this)),
                        E->getArg(0)->getType(), E->getArg(0)->getExprLoc(), FD,
                        0);
    EmitNonNullArgCheck(RValue::get(Dest.emitRawPointer(*this)),
                        E->getArg(1)->getType(), E->getArg(1)->getExprLoc(), FD,
                        0);
    auto *I = Builder.CreateMemMove(Dest, Src, SizeVal, false);
    addInstToNewSourceAtom(I, nullptr);
    return RValue::get(nullptr);
  }

  case Builtin::BImemcpy:
  case Builtin::BI__builtin_memcpy:
  case Builtin::BImempcpy:
  case Builtin::BI__builtin_mempcpy: {
    Address Dest = EmitPointerWithAlignment(E->getArg(0));
    Address Src = EmitPointerWithAlignment(E->getArg(1));
    Value *SizeVal = EmitScalarExpr(E->getArg(2));
    EmitArgCheck(TCK_Store, Dest, E->getArg(0), 0);
    EmitArgCheck(TCK_Load, Src, E->getArg(1), 1);
    auto *I = Builder.CreateMemCpy(Dest, Src, SizeVal, false);
    addInstToNewSourceAtom(I, nullptr);
    if (BuiltinID == Builtin::BImempcpy ||
        BuiltinID == Builtin::BI__builtin_mempcpy)
      return RValue::get(Builder.CreateInBoundsGEP(
          Dest.getElementType(), Dest.emitRawPointer(*this), SizeVal));
    else
      return RValue::get(Dest, *this);
  }

  case Builtin::BI__builtin_memcpy_inline: {
    Address Dest = EmitPointerWithAlignment(E->getArg(0));
    Address Src = EmitPointerWithAlignment(E->getArg(1));
    uint64_t Size =
        E->getArg(2)->EvaluateKnownConstInt(getContext()).getZExtValue();
    EmitArgCheck(TCK_Store, Dest, E->getArg(0), 0);
    EmitArgCheck(TCK_Load, Src, E->getArg(1), 1);
    auto *I = Builder.CreateMemCpyInline(Dest, Src, Size);
    addInstToNewSourceAtom(I, nullptr);
    return RValue::get(nullptr);
  }

  case Builtin::BI__builtin_char_memchr:
    BuiltinID = Builtin::BI__builtin_memchr;
    break;

  case Builtin::BI__builtin___memcpy_chk: {
    // fold __builtin_memcpy_chk(x, y, cst1, cst2) to memcpy iff cst1<=cst2.
    Expr::EvalResult SizeResult, DstSizeResult;
    if (!E->getArg(2)->EvaluateAsInt(SizeResult, CGM.getContext()) ||
        !E->getArg(3)->EvaluateAsInt(DstSizeResult, CGM.getContext()))
      break;
    llvm::APSInt Size = SizeResult.Val.getInt();
    llvm::APSInt DstSize = DstSizeResult.Val.getInt();
    if (Size.ugt(DstSize))
      break;
    Address Dest = EmitPointerWithAlignment(E->getArg(0));
    Address Src = EmitPointerWithAlignment(E->getArg(1));
    Value *SizeVal = llvm::ConstantInt::get(Builder.getContext(), Size);
    auto *I = Builder.CreateMemCpy(Dest, Src, SizeVal, false);
    addInstToNewSourceAtom(I, nullptr);
    return RValue::get(Dest, *this);
  }

  case Builtin::BI__builtin_objc_memmove_collectable: {
    Address DestAddr = EmitPointerWithAlignment(E->getArg(0));
    Address SrcAddr = EmitPointerWithAlignment(E->getArg(1));
    Value *SizeVal = EmitScalarExpr(E->getArg(2));
    CGM.getObjCRuntime().EmitGCMemmoveCollectable(*this,
                                                  DestAddr, SrcAddr, SizeVal);
    return RValue::get(DestAddr, *this);
  }

  case Builtin::BI__builtin___memmove_chk: {
    // fold __builtin_memmove_chk(x, y, cst1, cst2) to memmove iff cst1<=cst2.
    Expr::EvalResult SizeResult, DstSizeResult;
    if (!E->getArg(2)->EvaluateAsInt(SizeResult, CGM.getContext()) ||
        !E->getArg(3)->EvaluateAsInt(DstSizeResult, CGM.getContext()))
      break;
    llvm::APSInt Size = SizeResult.Val.getInt();
    llvm::APSInt DstSize = DstSizeResult.Val.getInt();
    if (Size.ugt(DstSize))
      break;
    Address Dest = EmitPointerWithAlignment(E->getArg(0));
    Address Src = EmitPointerWithAlignment(E->getArg(1));
    Value *SizeVal = llvm::ConstantInt::get(Builder.getContext(), Size);
    auto *I = Builder.CreateMemMove(Dest, Src, SizeVal, false);
    addInstToNewSourceAtom(I, nullptr);
    return RValue::get(Dest, *this);
  }

  case Builtin::BI__builtin_trivially_relocate:
  case Builtin::BImemmove:
  case Builtin::BI__builtin_memmove: {
    Address Dest = EmitPointerWithAlignment(E->getArg(0));
    Address Src = EmitPointerWithAlignment(E->getArg(1));
    Value *SizeVal = EmitScalarExpr(E->getArg(2));
    if (BuiltinIDIfNoAsmLabel == Builtin::BI__builtin_trivially_relocate)
      SizeVal = Builder.CreateMul(
          SizeVal,
          ConstantInt::get(
              SizeVal->getType(),
              getContext()
                  .getTypeSizeInChars(E->getArg(0)->getType()->getPointeeType())
                  .getQuantity()));
    EmitArgCheck(TCK_Store, Dest, E->getArg(0), 0);
    EmitArgCheck(TCK_Load, Src, E->getArg(1), 1);
    auto *I = Builder.CreateMemMove(Dest, Src, SizeVal, false);
    addInstToNewSourceAtom(I, nullptr);
    return RValue::get(Dest, *this);
  }
  case Builtin::BImemset:
  case Builtin::BI__builtin_memset: {
    Address Dest = EmitPointerWithAlignment(E->getArg(0));
    Value *ByteVal = Builder.CreateTrunc(EmitScalarExpr(E->getArg(1)),
                                         Builder.getInt8Ty());
    Value *SizeVal = EmitScalarExpr(E->getArg(2));
    EmitNonNullArgCheck(Dest, E->getArg(0)->getType(),
                        E->getArg(0)->getExprLoc(), FD, 0);
    auto *I = Builder.CreateMemSet(Dest, ByteVal, SizeVal, false);
    addInstToNewSourceAtom(I, ByteVal);
    return RValue::get(Dest, *this);
  }
  case Builtin::BI__builtin_memset_inline: {
    Address Dest = EmitPointerWithAlignment(E->getArg(0));
    Value *ByteVal =
        Builder.CreateTrunc(EmitScalarExpr(E->getArg(1)), Builder.getInt8Ty());
    uint64_t Size =
        E->getArg(2)->EvaluateKnownConstInt(getContext()).getZExtValue();
    EmitNonNullArgCheck(RValue::get(Dest.emitRawPointer(*this)),
                        E->getArg(0)->getType(), E->getArg(0)->getExprLoc(), FD,
                        0);
    auto *I = Builder.CreateMemSetInline(Dest, ByteVal, Size);
    addInstToNewSourceAtom(I, nullptr);
    return RValue::get(nullptr);
  }
  case Builtin::BI__builtin___memset_chk: {
    // fold __builtin_memset_chk(x, y, cst1, cst2) to memset iff cst1<=cst2.
    Expr::EvalResult SizeResult, DstSizeResult;
    if (!E->getArg(2)->EvaluateAsInt(SizeResult, CGM.getContext()) ||
        !E->getArg(3)->EvaluateAsInt(DstSizeResult, CGM.getContext()))
      break;
    llvm::APSInt Size = SizeResult.Val.getInt();
    llvm::APSInt DstSize = DstSizeResult.Val.getInt();
    if (Size.ugt(DstSize))
      break;
    Address Dest = EmitPointerWithAlignment(E->getArg(0));
    Value *ByteVal = Builder.CreateTrunc(EmitScalarExpr(E->getArg(1)),
                                         Builder.getInt8Ty());
    Value *SizeVal = llvm::ConstantInt::get(Builder.getContext(), Size);
    auto *I = Builder.CreateMemSet(Dest, ByteVal, SizeVal, false);
    addInstToNewSourceAtom(I, nullptr);
    return RValue::get(Dest, *this);
  }
  case Builtin::BI__builtin_wmemchr: {
    // The MSVC runtime library does not provide a definition of wmemchr, so we
    // need an inline implementation.
    if (!getTarget().getTriple().isOSMSVCRT())
      break;

    llvm::Type *WCharTy = ConvertType(getContext().WCharTy);
    Value *Str = EmitScalarExpr(E->getArg(0));
    Value *Chr = EmitScalarExpr(E->getArg(1));
    Value *Size = EmitScalarExpr(E->getArg(2));

    BasicBlock *Entry = Builder.GetInsertBlock();
    BasicBlock *CmpEq = createBasicBlock("wmemchr.eq");
    BasicBlock *Next = createBasicBlock("wmemchr.next");
    BasicBlock *Exit = createBasicBlock("wmemchr.exit");
    Value *SizeEq0 = Builder.CreateICmpEQ(Size, ConstantInt::get(SizeTy, 0));
    Builder.CreateCondBr(SizeEq0, Exit, CmpEq);

    EmitBlock(CmpEq);
    PHINode *StrPhi = Builder.CreatePHI(Str->getType(), 2);
    StrPhi->addIncoming(Str, Entry);
    PHINode *SizePhi = Builder.CreatePHI(SizeTy, 2);
    SizePhi->addIncoming(Size, Entry);
    CharUnits WCharAlign =
        getContext().getTypeAlignInChars(getContext().WCharTy);
    Value *StrCh = Builder.CreateAlignedLoad(WCharTy, StrPhi, WCharAlign);
    Value *FoundChr = Builder.CreateConstInBoundsGEP1_32(WCharTy, StrPhi, 0);
    Value *StrEqChr = Builder.CreateICmpEQ(StrCh, Chr);
    Builder.CreateCondBr(StrEqChr, Exit, Next);

    EmitBlock(Next);
    Value *NextStr = Builder.CreateConstInBoundsGEP1_32(WCharTy, StrPhi, 1);
    Value *NextSize = Builder.CreateSub(SizePhi, ConstantInt::get(SizeTy, 1));
    Value *NextSizeEq0 =
        Builder.CreateICmpEQ(NextSize, ConstantInt::get(SizeTy, 0));
    Builder.CreateCondBr(NextSizeEq0, Exit, CmpEq);
    StrPhi->addIncoming(NextStr, Next);
    SizePhi->addIncoming(NextSize, Next);

    EmitBlock(Exit);
    PHINode *Ret = Builder.CreatePHI(Str->getType(), 3);
    Ret->addIncoming(llvm::Constant::getNullValue(Str->getType()), Entry);
    Ret->addIncoming(llvm::Constant::getNullValue(Str->getType()), Next);
    Ret->addIncoming(FoundChr, CmpEq);
    return RValue::get(Ret);
  }
  case Builtin::BI__builtin_wmemcmp: {
    // The MSVC runtime library does not provide a definition of wmemcmp, so we
    // need an inline implementation.
    if (!getTarget().getTriple().isOSMSVCRT())
      break;

    llvm::Type *WCharTy = ConvertType(getContext().WCharTy);

    Value *Dst = EmitScalarExpr(E->getArg(0));
    Value *Src = EmitScalarExpr(E->getArg(1));
    Value *Size = EmitScalarExpr(E->getArg(2));

    BasicBlock *Entry = Builder.GetInsertBlock();
    BasicBlock *CmpGT = createBasicBlock("wmemcmp.gt");
    BasicBlock *CmpLT = createBasicBlock("wmemcmp.lt");
    BasicBlock *Next = createBasicBlock("wmemcmp.next");
    BasicBlock *Exit = createBasicBlock("wmemcmp.exit");
    Value *SizeEq0 = Builder.CreateICmpEQ(Size, ConstantInt::get(SizeTy, 0));
    Builder.CreateCondBr(SizeEq0, Exit, CmpGT);

    EmitBlock(CmpGT);
    PHINode *DstPhi = Builder.CreatePHI(Dst->getType(), 2);
    DstPhi->addIncoming(Dst, Entry);
    PHINode *SrcPhi = Builder.CreatePHI(Src->getType(), 2);
    SrcPhi->addIncoming(Src, Entry);
    PHINode *SizePhi = Builder.CreatePHI(SizeTy, 2);
    SizePhi->addIncoming(Size, Entry);
    CharUnits WCharAlign =
        getContext().getTypeAlignInChars(getContext().WCharTy);
    Value *DstCh = Builder.CreateAlignedLoad(WCharTy, DstPhi, WCharAlign);
    Value *SrcCh = Builder.CreateAlignedLoad(WCharTy, SrcPhi, WCharAlign);
    Value *DstGtSrc = Builder.CreateICmpUGT(DstCh, SrcCh);
    Builder.CreateCondBr(DstGtSrc, Exit, CmpLT);

    EmitBlock(CmpLT);
    Value *DstLtSrc = Builder.CreateICmpULT(DstCh, SrcCh);
    Builder.CreateCondBr(DstLtSrc, Exit, Next);

    EmitBlock(Next);
    Value *NextDst = Builder.CreateConstInBoundsGEP1_32(WCharTy, DstPhi, 1);
    Value *NextSrc = Builder.CreateConstInBoundsGEP1_32(WCharTy, SrcPhi, 1);
    Value *NextSize = Builder.CreateSub(SizePhi, ConstantInt::get(SizeTy, 1));
    Value *NextSizeEq0 =
        Builder.CreateICmpEQ(NextSize, ConstantInt::get(SizeTy, 0));
    Builder.CreateCondBr(NextSizeEq0, Exit, CmpGT);
    DstPhi->addIncoming(NextDst, Next);
    SrcPhi->addIncoming(NextSrc, Next);
    SizePhi->addIncoming(NextSize, Next);

    EmitBlock(Exit);
    PHINode *Ret = Builder.CreatePHI(IntTy, 4);
    Ret->addIncoming(ConstantInt::get(IntTy, 0), Entry);
    Ret->addIncoming(ConstantInt::get(IntTy, 1), CmpGT);
    Ret->addIncoming(ConstantInt::get(IntTy, -1), CmpLT);
    Ret->addIncoming(ConstantInt::get(IntTy, 0), Next);
    return RValue::get(Ret);
  }
  case Builtin::BI__builtin_dwarf_cfa: {
    // The offset in bytes from the first argument to the CFA.
    //
    // Why on earth is this in the frontend?  Is there any reason at
    // all that the backend can't reasonably determine this while
    // lowering llvm.eh.dwarf.cfa()?
    //
    // TODO: If there's a satisfactory reason, add a target hook for
    // this instead of hard-coding 0, which is correct for most targets.
    int32_t Offset = 0;

    Function *F = CGM.getIntrinsic(Intrinsic::eh_dwarf_cfa);
    return RValue::get(Builder.CreateCall(F,
                                      llvm::ConstantInt::get(Int32Ty, Offset)));
  }
  case Builtin::BI__builtin_return_address: {
    Value *Depth = ConstantEmitter(*this).emitAbstract(E->getArg(0),
                                                   getContext().UnsignedIntTy);
    Function *F = CGM.getIntrinsic(Intrinsic::returnaddress);
    return RValue::get(Builder.CreateCall(F, Depth));
  }
  case Builtin::BI_ReturnAddress: {
    Function *F = CGM.getIntrinsic(Intrinsic::returnaddress);
    return RValue::get(Builder.CreateCall(F, Builder.getInt32(0)));
  }
  case Builtin::BI__builtin_frame_address: {
    Value *Depth = ConstantEmitter(*this).emitAbstract(E->getArg(0),
                                                   getContext().UnsignedIntTy);
    Function *F = CGM.getIntrinsic(Intrinsic::frameaddress, AllocaInt8PtrTy);
    return RValue::get(Builder.CreateCall(F, Depth));
  }
  case Builtin::BI__builtin_extract_return_addr: {
    Value *Address = EmitScalarExpr(E->getArg(0));
    Value *Result = getTargetHooks().decodeReturnAddress(*this, Address);
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_frob_return_addr: {
    Value *Address = EmitScalarExpr(E->getArg(0));
    Value *Result = getTargetHooks().encodeReturnAddress(*this, Address);
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_dwarf_sp_column: {
    llvm::IntegerType *Ty
      = cast<llvm::IntegerType>(ConvertType(E->getType()));
    int Column = getTargetHooks().getDwarfEHStackPointer(CGM);
    if (Column == -1) {
      CGM.ErrorUnsupported(E, "__builtin_dwarf_sp_column");
      return RValue::get(llvm::UndefValue::get(Ty));
    }
    return RValue::get(llvm::ConstantInt::get(Ty, Column, true));
  }
  case Builtin::BI__builtin_init_dwarf_reg_size_table: {
    Value *Address = EmitScalarExpr(E->getArg(0));
    if (getTargetHooks().initDwarfEHRegSizeTable(*this, Address))
      CGM.ErrorUnsupported(E, "__builtin_init_dwarf_reg_size_table");
    return RValue::get(llvm::UndefValue::get(ConvertType(E->getType())));
  }
  case Builtin::BI__builtin_eh_return: {
    Value *Int = EmitScalarExpr(E->getArg(0));
    Value *Ptr = EmitScalarExpr(E->getArg(1));

    llvm::IntegerType *IntTy = cast<llvm::IntegerType>(Int->getType());
    assert((IntTy->getBitWidth() == 32 || IntTy->getBitWidth() == 64) &&
           "LLVM's __builtin_eh_return only supports 32- and 64-bit variants");
    Function *F =
        CGM.getIntrinsic(IntTy->getBitWidth() == 32 ? Intrinsic::eh_return_i32
                                                    : Intrinsic::eh_return_i64);
    Builder.CreateCall(F, {Int, Ptr});
    Builder.CreateUnreachable();

    // We do need to preserve an insertion point.
    EmitBlock(createBasicBlock("builtin_eh_return.cont"));

    return RValue::get(nullptr);
  }
  case Builtin::BI__builtin_unwind_init: {
    Function *F = CGM.getIntrinsic(Intrinsic::eh_unwind_init);
    Builder.CreateCall(F);
    return RValue::get(nullptr);
  }
  case Builtin::BI__builtin_extend_pointer: {
    // Extends a pointer to the size of an _Unwind_Word, which is
    // uint64_t on all platforms.  Generally this gets poked into a
    // register and eventually used as an address, so if the
    // addressing registers are wider than pointers and the platform
    // doesn't implicitly ignore high-order bits when doing
    // addressing, we need to make sure we zext / sext based on
    // the platform's expectations.
    //
    // See: http://gcc.gnu.org/ml/gcc-bugs/2002-02/msg00237.html

    // Cast the pointer to intptr_t.
    Value *Ptr = EmitScalarExpr(E->getArg(0));
    Value *Result = Builder.CreatePtrToInt(Ptr, IntPtrTy, "extend.cast");

    // If that's 64 bits, we're done.
    if (IntPtrTy->getBitWidth() == 64)
      return RValue::get(Result);

    // Otherwise, ask the codegen data what to do.
    if (getTargetHooks().extendPointerWithSExt())
      return RValue::get(Builder.CreateSExt(Result, Int64Ty, "extend.sext"));
    else
      return RValue::get(Builder.CreateZExt(Result, Int64Ty, "extend.zext"));
  }
  case Builtin::BI__builtin_setjmp: {
    // Buffer is a void**.
    Address Buf = EmitPointerWithAlignment(E->getArg(0));

    if (getTarget().getTriple().getArch() == llvm::Triple::systemz) {
      // On this target, the back end fills in the context buffer completely.
      // It doesn't really matter if the frontend stores to the buffer before
      // calling setjmp, the back-end is going to overwrite them anyway.
      Function *F = CGM.getIntrinsic(Intrinsic::eh_sjlj_setjmp);
      return RValue::get(Builder.CreateCall(F, Buf.emitRawPointer(*this)));
    }

    // Store the frame pointer to the setjmp buffer.
    Value *FrameAddr = Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::frameaddress, AllocaInt8PtrTy),
        ConstantInt::get(Int32Ty, 0));
    Builder.CreateStore(FrameAddr, Buf);

    // Store the stack pointer to the setjmp buffer.
    Value *StackAddr = Builder.CreateStackSave();
    assert(Buf.emitRawPointer(*this)->getType() == StackAddr->getType());

    Address StackSaveSlot = Builder.CreateConstInBoundsGEP(Buf, 2);
    Builder.CreateStore(StackAddr, StackSaveSlot);

    // Call LLVM's EH setjmp, which is lightweight.
    Function *F = CGM.getIntrinsic(Intrinsic::eh_sjlj_setjmp);
    return RValue::get(Builder.CreateCall(F, Buf.emitRawPointer(*this)));
  }
  case Builtin::BI__builtin_longjmp: {
    Value *Buf = EmitScalarExpr(E->getArg(0));

    // Call LLVM's EH longjmp, which is lightweight.
    Builder.CreateCall(CGM.getIntrinsic(Intrinsic::eh_sjlj_longjmp), Buf);

    // longjmp doesn't return; mark this as unreachable.
    Builder.CreateUnreachable();

    // We do need to preserve an insertion point.
    EmitBlock(createBasicBlock("longjmp.cont"));

    return RValue::get(nullptr);
  }
  case Builtin::BI__builtin_launder: {
    const Expr *Arg = E->getArg(0);
    QualType ArgTy = Arg->getType()->getPointeeType();
    Value *Ptr = EmitScalarExpr(Arg);
    if (TypeRequiresBuiltinLaunder(CGM, ArgTy))
      Ptr = Builder.CreateLaunderInvariantGroup(Ptr);

    return RValue::get(Ptr);
  }
  case Builtin::BI__sync_fetch_and_add:
  case Builtin::BI__sync_fetch_and_sub:
  case Builtin::BI__sync_fetch_and_or:
  case Builtin::BI__sync_fetch_and_and:
  case Builtin::BI__sync_fetch_and_xor:
  case Builtin::BI__sync_fetch_and_nand:
  case Builtin::BI__sync_add_and_fetch:
  case Builtin::BI__sync_sub_and_fetch:
  case Builtin::BI__sync_and_and_fetch:
  case Builtin::BI__sync_or_and_fetch:
  case Builtin::BI__sync_xor_and_fetch:
  case Builtin::BI__sync_nand_and_fetch:
  case Builtin::BI__sync_val_compare_and_swap:
  case Builtin::BI__sync_bool_compare_and_swap:
  case Builtin::BI__sync_lock_test_and_set:
  case Builtin::BI__sync_lock_release:
  case Builtin::BI__sync_swap:
    llvm_unreachable("Shouldn't make it through sema");
  case Builtin::BI__sync_fetch_and_add_1:
  case Builtin::BI__sync_fetch_and_add_2:
  case Builtin::BI__sync_fetch_and_add_4:
  case Builtin::BI__sync_fetch_and_add_8:
  case Builtin::BI__sync_fetch_and_add_16:
    return EmitBinaryAtomic(*this, llvm::AtomicRMWInst::Add, E);
  case Builtin::BI__sync_fetch_and_sub_1:
  case Builtin::BI__sync_fetch_and_sub_2:
  case Builtin::BI__sync_fetch_and_sub_4:
  case Builtin::BI__sync_fetch_and_sub_8:
  case Builtin::BI__sync_fetch_and_sub_16:
    return EmitBinaryAtomic(*this, llvm::AtomicRMWInst::Sub, E);
  case Builtin::BI__sync_fetch_and_or_1:
  case Builtin::BI__sync_fetch_and_or_2:
  case Builtin::BI__sync_fetch_and_or_4:
  case Builtin::BI__sync_fetch_and_or_8:
  case Builtin::BI__sync_fetch_and_or_16:
    return EmitBinaryAtomic(*this, llvm::AtomicRMWInst::Or, E);
  case Builtin::BI__sync_fetch_and_and_1:
  case Builtin::BI__sync_fetch_and_and_2:
  case Builtin::BI__sync_fetch_and_and_4:
  case Builtin::BI__sync_fetch_and_and_8:
  case Builtin::BI__sync_fetch_and_and_16:
    return EmitBinaryAtomic(*this, llvm::AtomicRMWInst::And, E);
  case Builtin::BI__sync_fetch_and_xor_1:
  case Builtin::BI__sync_fetch_and_xor_2:
  case Builtin::BI__sync_fetch_and_xor_4:
  case Builtin::BI__sync_fetch_and_xor_8:
  case Builtin::BI__sync_fetch_and_xor_16:
    return EmitBinaryAtomic(*this, llvm::AtomicRMWInst::Xor, E);
  case Builtin::BI__sync_fetch_and_nand_1:
  case Builtin::BI__sync_fetch_and_nand_2:
  case Builtin::BI__sync_fetch_and_nand_4:
  case Builtin::BI__sync_fetch_and_nand_8:
  case Builtin::BI__sync_fetch_and_nand_16:
    return EmitBinaryAtomic(*this, llvm::AtomicRMWInst::Nand, E);

  // Clang extensions: not overloaded yet.
  case Builtin::BI__sync_fetch_and_min:
    return EmitBinaryAtomic(*this, llvm::AtomicRMWInst::Min, E);
  case Builtin::BI__sync_fetch_and_max:
    return EmitBinaryAtomic(*this, llvm::AtomicRMWInst::Max, E);
  case Builtin::BI__sync_fetch_and_umin:
    return EmitBinaryAtomic(*this, llvm::AtomicRMWInst::UMin, E);
  case Builtin::BI__sync_fetch_and_umax:
    return EmitBinaryAtomic(*this, llvm::AtomicRMWInst::UMax, E);

  case Builtin::BI__sync_add_and_fetch_1:
  case Builtin::BI__sync_add_and_fetch_2:
  case Builtin::BI__sync_add_and_fetch_4:
  case Builtin::BI__sync_add_and_fetch_8:
  case Builtin::BI__sync_add_and_fetch_16:
    return EmitBinaryAtomicPost(*this, llvm::AtomicRMWInst::Add, E,
                                llvm::Instruction::Add);
  case Builtin::BI__sync_sub_and_fetch_1:
  case Builtin::BI__sync_sub_and_fetch_2:
  case Builtin::BI__sync_sub_and_fetch_4:
  case Builtin::BI__sync_sub_and_fetch_8:
  case Builtin::BI__sync_sub_and_fetch_16:
    return EmitBinaryAtomicPost(*this, llvm::AtomicRMWInst::Sub, E,
                                llvm::Instruction::Sub);
  case Builtin::BI__sync_and_and_fetch_1:
  case Builtin::BI__sync_and_and_fetch_2:
  case Builtin::BI__sync_and_and_fetch_4:
  case Builtin::BI__sync_and_and_fetch_8:
  case Builtin::BI__sync_and_and_fetch_16:
    return EmitBinaryAtomicPost(*this, llvm::AtomicRMWInst::And, E,
                                llvm::Instruction::And);
  case Builtin::BI__sync_or_and_fetch_1:
  case Builtin::BI__sync_or_and_fetch_2:
  case Builtin::BI__sync_or_and_fetch_4:
  case Builtin::BI__sync_or_and_fetch_8:
  case Builtin::BI__sync_or_and_fetch_16:
    return EmitBinaryAtomicPost(*this, llvm::AtomicRMWInst::Or, E,
                                llvm::Instruction::Or);
  case Builtin::BI__sync_xor_and_fetch_1:
  case Builtin::BI__sync_xor_and_fetch_2:
  case Builtin::BI__sync_xor_and_fetch_4:
  case Builtin::BI__sync_xor_and_fetch_8:
  case Builtin::BI__sync_xor_and_fetch_16:
    return EmitBinaryAtomicPost(*this, llvm::AtomicRMWInst::Xor, E,
                                llvm::Instruction::Xor);
  case Builtin::BI__sync_nand_and_fetch_1:
  case Builtin::BI__sync_nand_and_fetch_2:
  case Builtin::BI__sync_nand_and_fetch_4:
  case Builtin::BI__sync_nand_and_fetch_8:
  case Builtin::BI__sync_nand_and_fetch_16:
    return EmitBinaryAtomicPost(*this, llvm::AtomicRMWInst::Nand, E,
                                llvm::Instruction::And, true);

  case Builtin::BI__sync_val_compare_and_swap_1:
  case Builtin::BI__sync_val_compare_and_swap_2:
  case Builtin::BI__sync_val_compare_and_swap_4:
  case Builtin::BI__sync_val_compare_and_swap_8:
  case Builtin::BI__sync_val_compare_and_swap_16:
    return RValue::get(MakeAtomicCmpXchgValue(*this, E, false));

  case Builtin::BI__sync_bool_compare_and_swap_1:
  case Builtin::BI__sync_bool_compare_and_swap_2:
  case Builtin::BI__sync_bool_compare_and_swap_4:
  case Builtin::BI__sync_bool_compare_and_swap_8:
  case Builtin::BI__sync_bool_compare_and_swap_16:
    return RValue::get(MakeAtomicCmpXchgValue(*this, E, true));

  case Builtin::BI__sync_swap_1:
  case Builtin::BI__sync_swap_2:
  case Builtin::BI__sync_swap_4:
  case Builtin::BI__sync_swap_8:
  case Builtin::BI__sync_swap_16:
    return EmitBinaryAtomic(*this, llvm::AtomicRMWInst::Xchg, E);

  case Builtin::BI__sync_lock_test_and_set_1:
  case Builtin::BI__sync_lock_test_and_set_2:
  case Builtin::BI__sync_lock_test_and_set_4:
  case Builtin::BI__sync_lock_test_and_set_8:
  case Builtin::BI__sync_lock_test_and_set_16:
    return EmitBinaryAtomic(*this, llvm::AtomicRMWInst::Xchg, E);

  case Builtin::BI__sync_lock_release_1:
  case Builtin::BI__sync_lock_release_2:
  case Builtin::BI__sync_lock_release_4:
  case Builtin::BI__sync_lock_release_8:
  case Builtin::BI__sync_lock_release_16: {
    Address Ptr = CheckAtomicAlignment(*this, E);
    QualType ElTy = E->getArg(0)->getType()->getPointeeType();

    llvm::Type *ITy = llvm::IntegerType::get(getLLVMContext(),
                                             getContext().getTypeSize(ElTy));
    llvm::StoreInst *Store =
        Builder.CreateStore(llvm::Constant::getNullValue(ITy), Ptr);
    Store->setAtomic(llvm::AtomicOrdering::Release);
    return RValue::get(nullptr);
  }

  case Builtin::BI__sync_synchronize: {
    // We assume this is supposed to correspond to a C++0x-style
    // sequentially-consistent fence (i.e. this is only usable for
    // synchronization, not device I/O or anything like that). This intrinsic
    // is really badly designed in the sense that in theory, there isn't
    // any way to safely use it... but in practice, it mostly works
    // to use it with non-atomic loads and stores to get acquire/release
    // semantics.
    Builder.CreateFence(llvm::AtomicOrdering::SequentiallyConsistent);
    return RValue::get(nullptr);
  }

  case Builtin::BI__builtin_nontemporal_load:
    return RValue::get(EmitNontemporalLoad(*this, E));
  case Builtin::BI__builtin_nontemporal_store:
    return RValue::get(EmitNontemporalStore(*this, E));
  case Builtin::BI__c11_atomic_is_lock_free:
  case Builtin::BI__atomic_is_lock_free: {
    // Call "bool __atomic_is_lock_free(size_t size, void *ptr)". For the
    // __c11 builtin, ptr is 0 (indicating a properly-aligned object), since
    // _Atomic(T) is always properly-aligned.
    const char *LibCallName = "__atomic_is_lock_free";
    CallArgList Args;
    Args.add(RValue::get(EmitScalarExpr(E->getArg(0))),
             getContext().getSizeType());
    if (BuiltinID == Builtin::BI__atomic_is_lock_free)
      Args.add(RValue::get(EmitScalarExpr(E->getArg(1))),
               getContext().VoidPtrTy);
    else
      Args.add(RValue::get(llvm::Constant::getNullValue(VoidPtrTy)),
               getContext().VoidPtrTy);
    const CGFunctionInfo &FuncInfo =
        CGM.getTypes().arrangeBuiltinFunctionCall(E->getType(), Args);
    llvm::FunctionType *FTy = CGM.getTypes().GetFunctionType(FuncInfo);
    llvm::FunctionCallee Func = CGM.CreateRuntimeFunction(FTy, LibCallName);
    return EmitCall(FuncInfo, CGCallee::forDirect(Func),
                    ReturnValueSlot(), Args);
  }

  case Builtin::BI__atomic_thread_fence:
  case Builtin::BI__atomic_signal_fence:
  case Builtin::BI__c11_atomic_thread_fence:
  case Builtin::BI__c11_atomic_signal_fence: {
    llvm::SyncScope::ID SSID;
    if (BuiltinID == Builtin::BI__atomic_signal_fence ||
        BuiltinID == Builtin::BI__c11_atomic_signal_fence)
      SSID = llvm::SyncScope::SingleThread;
    else
      SSID = llvm::SyncScope::System;
    Value *Order = EmitScalarExpr(E->getArg(0));
    if (isa<llvm::ConstantInt>(Order)) {
      int ord = cast<llvm::ConstantInt>(Order)->getZExtValue();
      switch (ord) {
      case 0:  // memory_order_relaxed
      default: // invalid order
        break;
      case 1:  // memory_order_consume
      case 2:  // memory_order_acquire
        Builder.CreateFence(llvm::AtomicOrdering::Acquire, SSID);
        break;
      case 3:  // memory_order_release
        Builder.CreateFence(llvm::AtomicOrdering::Release, SSID);
        break;
      case 4:  // memory_order_acq_rel
        Builder.CreateFence(llvm::AtomicOrdering::AcquireRelease, SSID);
        break;
      case 5:  // memory_order_seq_cst
        Builder.CreateFence(llvm::AtomicOrdering::SequentiallyConsistent, SSID);
        break;
      }
      return RValue::get(nullptr);
    }

    llvm::BasicBlock *AcquireBB, *ReleaseBB, *AcqRelBB, *SeqCstBB;
    AcquireBB = createBasicBlock("acquire", CurFn);
    ReleaseBB = createBasicBlock("release", CurFn);
    AcqRelBB = createBasicBlock("acqrel", CurFn);
    SeqCstBB = createBasicBlock("seqcst", CurFn);
    llvm::BasicBlock *ContBB = createBasicBlock("atomic.continue", CurFn);

    Order = Builder.CreateIntCast(Order, Builder.getInt32Ty(), false);
    llvm::SwitchInst *SI = Builder.CreateSwitch(Order, ContBB);

    Builder.SetInsertPoint(AcquireBB);
    Builder.CreateFence(llvm::AtomicOrdering::Acquire, SSID);
    Builder.CreateBr(ContBB);
    SI->addCase(Builder.getInt32(1), AcquireBB);
    SI->addCase(Builder.getInt32(2), AcquireBB);

    Builder.SetInsertPoint(ReleaseBB);
    Builder.CreateFence(llvm::AtomicOrdering::Release, SSID);
    Builder.CreateBr(ContBB);
    SI->addCase(Builder.getInt32(3), ReleaseBB);

    Builder.SetInsertPoint(AcqRelBB);
    Builder.CreateFence(llvm::AtomicOrdering::AcquireRelease, SSID);
    Builder.CreateBr(ContBB);
    SI->addCase(Builder.getInt32(4), AcqRelBB);

    Builder.SetInsertPoint(SeqCstBB);
    Builder.CreateFence(llvm::AtomicOrdering::SequentiallyConsistent, SSID);
    Builder.CreateBr(ContBB);
    SI->addCase(Builder.getInt32(5), SeqCstBB);

    Builder.SetInsertPoint(ContBB);
    return RValue::get(nullptr);
  }
  case Builtin::BI__scoped_atomic_thread_fence: {
    auto ScopeModel = AtomicScopeModel::create(AtomicScopeModelKind::Generic);

    Value *Order = EmitScalarExpr(E->getArg(0));
    Value *Scope = EmitScalarExpr(E->getArg(1));
    auto Ord = dyn_cast<llvm::ConstantInt>(Order);
    auto Scp = dyn_cast<llvm::ConstantInt>(Scope);
    if (Ord && Scp) {
      SyncScope SS = ScopeModel->isValid(Scp->getZExtValue())
                         ? ScopeModel->map(Scp->getZExtValue())
                         : ScopeModel->map(ScopeModel->getFallBackValue());
      switch (Ord->getZExtValue()) {
      case 0:  // memory_order_relaxed
      default: // invalid order
        break;
      case 1: // memory_order_consume
      case 2: // memory_order_acquire
        Builder.CreateFence(
            llvm::AtomicOrdering::Acquire,
            getTargetHooks().getLLVMSyncScopeID(getLangOpts(), SS,
                                                llvm::AtomicOrdering::Acquire,
                                                getLLVMContext()));
        break;
      case 3: // memory_order_release
        Builder.CreateFence(
            llvm::AtomicOrdering::Release,
            getTargetHooks().getLLVMSyncScopeID(getLangOpts(), SS,
                                                llvm::AtomicOrdering::Release,
                                                getLLVMContext()));
        break;
      case 4: // memory_order_acq_rel
        Builder.CreateFence(llvm::AtomicOrdering::AcquireRelease,
                            getTargetHooks().getLLVMSyncScopeID(
                                getLangOpts(), SS,
                                llvm::AtomicOrdering::AcquireRelease,
                                getLLVMContext()));
        break;
      case 5: // memory_order_seq_cst
        Builder.CreateFence(llvm::AtomicOrdering::SequentiallyConsistent,
                            getTargetHooks().getLLVMSyncScopeID(
                                getLangOpts(), SS,
                                llvm::AtomicOrdering::SequentiallyConsistent,
                                getLLVMContext()));
        break;
      }
      return RValue::get(nullptr);
    }

    llvm::BasicBlock *ContBB = createBasicBlock("atomic.scope.continue", CurFn);

    llvm::SmallVector<std::pair<llvm::BasicBlock *, llvm::AtomicOrdering>>
        OrderBBs;
    if (Ord) {
      switch (Ord->getZExtValue()) {
      case 0:  // memory_order_relaxed
      default: // invalid order
        ContBB->eraseFromParent();
        return RValue::get(nullptr);
      case 1: // memory_order_consume
      case 2: // memory_order_acquire
        OrderBBs.emplace_back(Builder.GetInsertBlock(),
                              llvm::AtomicOrdering::Acquire);
        break;
      case 3: // memory_order_release
        OrderBBs.emplace_back(Builder.GetInsertBlock(),
                              llvm::AtomicOrdering::Release);
        break;
      case 4: // memory_order_acq_rel
        OrderBBs.emplace_back(Builder.GetInsertBlock(),
                              llvm::AtomicOrdering::AcquireRelease);
        break;
      case 5: // memory_order_seq_cst
        OrderBBs.emplace_back(Builder.GetInsertBlock(),
                              llvm::AtomicOrdering::SequentiallyConsistent);
        break;
      }
    } else {
      llvm::BasicBlock *AcquireBB = createBasicBlock("acquire", CurFn);
      llvm::BasicBlock *ReleaseBB = createBasicBlock("release", CurFn);
      llvm::BasicBlock *AcqRelBB = createBasicBlock("acqrel", CurFn);
      llvm::BasicBlock *SeqCstBB = createBasicBlock("seqcst", CurFn);

      Order = Builder.CreateIntCast(Order, Builder.getInt32Ty(), false);
      llvm::SwitchInst *SI = Builder.CreateSwitch(Order, ContBB);
      SI->addCase(Builder.getInt32(1), AcquireBB);
      SI->addCase(Builder.getInt32(2), AcquireBB);
      SI->addCase(Builder.getInt32(3), ReleaseBB);
      SI->addCase(Builder.getInt32(4), AcqRelBB);
      SI->addCase(Builder.getInt32(5), SeqCstBB);

      OrderBBs.emplace_back(AcquireBB, llvm::AtomicOrdering::Acquire);
      OrderBBs.emplace_back(ReleaseBB, llvm::AtomicOrdering::Release);
      OrderBBs.emplace_back(AcqRelBB, llvm::AtomicOrdering::AcquireRelease);
      OrderBBs.emplace_back(SeqCstBB,
                            llvm::AtomicOrdering::SequentiallyConsistent);
    }

    for (auto &[OrderBB, Ordering] : OrderBBs) {
      Builder.SetInsertPoint(OrderBB);
      if (Scp) {
        SyncScope SS = ScopeModel->isValid(Scp->getZExtValue())
                           ? ScopeModel->map(Scp->getZExtValue())
                           : ScopeModel->map(ScopeModel->getFallBackValue());
        Builder.CreateFence(Ordering,
                            getTargetHooks().getLLVMSyncScopeID(
                                getLangOpts(), SS, Ordering, getLLVMContext()));
        Builder.CreateBr(ContBB);
      } else {
        llvm::DenseMap<unsigned, llvm::BasicBlock *> BBs;
        for (unsigned Scp : ScopeModel->getRuntimeValues())
          BBs[Scp] = createBasicBlock(getAsString(ScopeModel->map(Scp)), CurFn);

        auto *SC = Builder.CreateIntCast(Scope, Builder.getInt32Ty(), false);
        llvm::SwitchInst *SI = Builder.CreateSwitch(SC, ContBB);
        for (unsigned Scp : ScopeModel->getRuntimeValues()) {
          auto *B = BBs[Scp];
          SI->addCase(Builder.getInt32(Scp), B);

          Builder.SetInsertPoint(B);
          Builder.CreateFence(Ordering, getTargetHooks().getLLVMSyncScopeID(
                                            getLangOpts(), ScopeModel->map(Scp),
                                            Ordering, getLLVMContext()));
          Builder.CreateBr(ContBB);
        }
      }
    }

    Builder.SetInsertPoint(ContBB);
    return RValue::get(nullptr);
  }

  case Builtin::BI__builtin_signbit:
  case Builtin::BI__builtin_signbitf:
  case Builtin::BI__builtin_signbitl: {
    return RValue::get(
        Builder.CreateZExt(EmitSignBit(*this, EmitScalarExpr(E->getArg(0))),
                           ConvertType(E->getType())));
  }
  case Builtin::BI__warn_memset_zero_len:
    return RValue::getIgnored();
  case Builtin::BI__annotation: {
    // Re-encode each wide string to UTF8 and make an MDString.
    SmallVector<Metadata *, 1> Strings;
    for (const Expr *Arg : E->arguments()) {
      const auto *Str = cast<StringLiteral>(Arg->IgnoreParenCasts());
      assert(Str->getCharByteWidth() == 2);
      StringRef WideBytes = Str->getBytes();
      std::string StrUtf8;
      if (!convertUTF16ToUTF8String(
              ArrayRef(WideBytes.data(), WideBytes.size()), StrUtf8)) {
        CGM.ErrorUnsupported(E, "non-UTF16 __annotation argument");
        continue;
      }
      Strings.push_back(llvm::MDString::get(getLLVMContext(), StrUtf8));
    }

    // Build and MDTuple of MDStrings and emit the intrinsic call.
    llvm::Function *F = CGM.getIntrinsic(Intrinsic::codeview_annotation, {});
    MDTuple *StrTuple = MDTuple::get(getLLVMContext(), Strings);
    Builder.CreateCall(F, MetadataAsValue::get(getLLVMContext(), StrTuple));
    return RValue::getIgnored();
  }
  case Builtin::BI__builtin_annotation: {
    llvm::Value *AnnVal = EmitScalarExpr(E->getArg(0));
    llvm::Function *F = CGM.getIntrinsic(
        Intrinsic::annotation, {AnnVal->getType(), CGM.ConstGlobalsPtrTy});

    // Get the annotation string, go through casts. Sema requires this to be a
    // non-wide string literal, potentially casted, so the cast<> is safe.
    const Expr *AnnotationStrExpr = E->getArg(1)->IgnoreParenCasts();
    StringRef Str = cast<StringLiteral>(AnnotationStrExpr)->getString();
    return RValue::get(
        EmitAnnotationCall(F, AnnVal, Str, E->getExprLoc(), nullptr));
  }
  case Builtin::BI__builtin_addcb:
  case Builtin::BI__builtin_addcs:
  case Builtin::BI__builtin_addc:
  case Builtin::BI__builtin_addcl:
  case Builtin::BI__builtin_addcll:
  case Builtin::BI__builtin_subcb:
  case Builtin::BI__builtin_subcs:
  case Builtin::BI__builtin_subc:
  case Builtin::BI__builtin_subcl:
  case Builtin::BI__builtin_subcll: {

    // We translate all of these builtins from expressions of the form:
    //   int x = ..., y = ..., carryin = ..., carryout, result;
    //   result = __builtin_addc(x, y, carryin, &carryout);
    //
    // to LLVM IR of the form:
    //
    //   %tmp1 = call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %x, i32 %y)
    //   %tmpsum1 = extractvalue {i32, i1} %tmp1, 0
    //   %carry1 = extractvalue {i32, i1} %tmp1, 1
    //   %tmp2 = call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %tmpsum1,
    //                                                       i32 %carryin)
    //   %result = extractvalue {i32, i1} %tmp2, 0
    //   %carry2 = extractvalue {i32, i1} %tmp2, 1
    //   %tmp3 = or i1 %carry1, %carry2
    //   %tmp4 = zext i1 %tmp3 to i32
    //   store i32 %tmp4, i32* %carryout

    // Scalarize our inputs.
    llvm::Value *X = EmitScalarExpr(E->getArg(0));
    llvm::Value *Y = EmitScalarExpr(E->getArg(1));
    llvm::Value *Carryin = EmitScalarExpr(E->getArg(2));
    Address CarryOutPtr = EmitPointerWithAlignment(E->getArg(3));

    // Decide if we are lowering to a uadd.with.overflow or usub.with.overflow.
    Intrinsic::ID IntrinsicId;
    switch (BuiltinID) {
    default: llvm_unreachable("Unknown multiprecision builtin id.");
    case Builtin::BI__builtin_addcb:
    case Builtin::BI__builtin_addcs:
    case Builtin::BI__builtin_addc:
    case Builtin::BI__builtin_addcl:
    case Builtin::BI__builtin_addcll:
      IntrinsicId = Intrinsic::uadd_with_overflow;
      break;
    case Builtin::BI__builtin_subcb:
    case Builtin::BI__builtin_subcs:
    case Builtin::BI__builtin_subc:
    case Builtin::BI__builtin_subcl:
    case Builtin::BI__builtin_subcll:
      IntrinsicId = Intrinsic::usub_with_overflow;
      break;
    }

    // Construct our resulting LLVM IR expression.
    llvm::Value *Carry1;
    llvm::Value *Sum1 = EmitOverflowIntrinsic(*this, IntrinsicId,
                                              X, Y, Carry1);
    llvm::Value *Carry2;
    llvm::Value *Sum2 = EmitOverflowIntrinsic(*this, IntrinsicId,
                                              Sum1, Carryin, Carry2);
    llvm::Value *CarryOut = Builder.CreateZExt(Builder.CreateOr(Carry1, Carry2),
                                               X->getType());
    Builder.CreateStore(CarryOut, CarryOutPtr);
    return RValue::get(Sum2);
  }

  case Builtin::BI__builtin_add_overflow:
  case Builtin::BI__builtin_sub_overflow:
  case Builtin::BI__builtin_mul_overflow: {
    const clang::Expr *LeftArg = E->getArg(0);
    const clang::Expr *RightArg = E->getArg(1);
    const clang::Expr *ResultArg = E->getArg(2);

    clang::QualType ResultQTy =
        ResultArg->getType()->castAs<PointerType>()->getPointeeType();

    WidthAndSignedness LeftInfo =
        getIntegerWidthAndSignedness(CGM.getContext(), LeftArg->getType());
    WidthAndSignedness RightInfo =
        getIntegerWidthAndSignedness(CGM.getContext(), RightArg->getType());
    WidthAndSignedness ResultInfo =
        getIntegerWidthAndSignedness(CGM.getContext(), ResultQTy);

    // Handle mixed-sign multiplication as a special case, because adding
    // runtime or backend support for our generic irgen would be too expensive.
    if (isSpecialMixedSignMultiply(BuiltinID, LeftInfo, RightInfo, ResultInfo))
      return EmitCheckedMixedSignMultiply(*this, LeftArg, LeftInfo, RightArg,
                                          RightInfo, ResultArg, ResultQTy,
                                          ResultInfo);

    if (isSpecialUnsignedMultiplySignedResult(BuiltinID, LeftInfo, RightInfo,
                                              ResultInfo))
      return EmitCheckedUnsignedMultiplySignedResult(
          *this, LeftArg, LeftInfo, RightArg, RightInfo, ResultArg, ResultQTy,
          ResultInfo);

    WidthAndSignedness EncompassingInfo =
        EncompassingIntegerType({LeftInfo, RightInfo, ResultInfo});

    llvm::Type *EncompassingLLVMTy =
        llvm::IntegerType::get(CGM.getLLVMContext(), EncompassingInfo.Width);

    llvm::Type *ResultLLVMTy = CGM.getTypes().ConvertType(ResultQTy);

    Intrinsic::ID IntrinsicId;
    switch (BuiltinID) {
    default:
      llvm_unreachable("Unknown overflow builtin id.");
    case Builtin::BI__builtin_add_overflow:
      IntrinsicId = EncompassingInfo.Signed ? Intrinsic::sadd_with_overflow
                                            : Intrinsic::uadd_with_overflow;
      break;
    case Builtin::BI__builtin_sub_overflow:
      IntrinsicId = EncompassingInfo.Signed ? Intrinsic::ssub_with_overflow
                                            : Intrinsic::usub_with_overflow;
      break;
    case Builtin::BI__builtin_mul_overflow:
      IntrinsicId = EncompassingInfo.Signed ? Intrinsic::smul_with_overflow
                                            : Intrinsic::umul_with_overflow;
      break;
    }

    llvm::Value *Left = EmitScalarExpr(LeftArg);
    llvm::Value *Right = EmitScalarExpr(RightArg);
    Address ResultPtr = EmitPointerWithAlignment(ResultArg);

    // Extend each operand to the encompassing type.
    Left = Builder.CreateIntCast(Left, EncompassingLLVMTy, LeftInfo.Signed);
    Right = Builder.CreateIntCast(Right, EncompassingLLVMTy, RightInfo.Signed);

    // Perform the operation on the extended values.
    llvm::Value *Overflow, *Result;
    Result = EmitOverflowIntrinsic(*this, IntrinsicId, Left, Right, Overflow);

    if (EncompassingInfo.Width > ResultInfo.Width) {
      // The encompassing type is wider than the result type, so we need to
      // truncate it.
      llvm::Value *ResultTrunc = Builder.CreateTrunc(Result, ResultLLVMTy);

      // To see if the truncation caused an overflow, we will extend
      // the result and then compare it to the original result.
      llvm::Value *ResultTruncExt = Builder.CreateIntCast(
          ResultTrunc, EncompassingLLVMTy, ResultInfo.Signed);
      llvm::Value *TruncationOverflow =
          Builder.CreateICmpNE(Result, ResultTruncExt);

      Overflow = Builder.CreateOr(Overflow, TruncationOverflow);
      Result = ResultTrunc;
    }

    // Finally, store the result using the pointer.
    bool isVolatile =
      ResultArg->getType()->getPointeeType().isVolatileQualified();
    Builder.CreateStore(EmitToMemory(Result, ResultQTy), ResultPtr, isVolatile);

    return RValue::get(Overflow);
  }

  case Builtin::BI__builtin_uadd_overflow:
  case Builtin::BI__builtin_uaddl_overflow:
  case Builtin::BI__builtin_uaddll_overflow:
  case Builtin::BI__builtin_usub_overflow:
  case Builtin::BI__builtin_usubl_overflow:
  case Builtin::BI__builtin_usubll_overflow:
  case Builtin::BI__builtin_umul_overflow:
  case Builtin::BI__builtin_umull_overflow:
  case Builtin::BI__builtin_umulll_overflow:
  case Builtin::BI__builtin_sadd_overflow:
  case Builtin::BI__builtin_saddl_overflow:
  case Builtin::BI__builtin_saddll_overflow:
  case Builtin::BI__builtin_ssub_overflow:
  case Builtin::BI__builtin_ssubl_overflow:
  case Builtin::BI__builtin_ssubll_overflow:
  case Builtin::BI__builtin_smul_overflow:
  case Builtin::BI__builtin_smull_overflow:
  case Builtin::BI__builtin_smulll_overflow: {

    // We translate all of these builtins directly to the relevant llvm IR node.

    // Scalarize our inputs.
    llvm::Value *X = EmitScalarExpr(E->getArg(0));
    llvm::Value *Y = EmitScalarExpr(E->getArg(1));
    Address SumOutPtr = EmitPointerWithAlignment(E->getArg(2));

    // Decide which of the overflow intrinsics we are lowering to:
    Intrinsic::ID IntrinsicId;
    switch (BuiltinID) {
    default: llvm_unreachable("Unknown overflow builtin id.");
    case Builtin::BI__builtin_uadd_overflow:
    case Builtin::BI__builtin_uaddl_overflow:
    case Builtin::BI__builtin_uaddll_overflow:
      IntrinsicId = Intrinsic::uadd_with_overflow;
      break;
    case Builtin::BI__builtin_usub_overflow:
    case Builtin::BI__builtin_usubl_overflow:
    case Builtin::BI__builtin_usubll_overflow:
      IntrinsicId = Intrinsic::usub_with_overflow;
      break;
    case Builtin::BI__builtin_umul_overflow:
    case Builtin::BI__builtin_umull_overflow:
    case Builtin::BI__builtin_umulll_overflow:
      IntrinsicId = Intrinsic::umul_with_overflow;
      break;
    case Builtin::BI__builtin_sadd_overflow:
    case Builtin::BI__builtin_saddl_overflow:
    case Builtin::BI__builtin_saddll_overflow:
      IntrinsicId = Intrinsic::sadd_with_overflow;
      break;
    case Builtin::BI__builtin_ssub_overflow:
    case Builtin::BI__builtin_ssubl_overflow:
    case Builtin::BI__builtin_ssubll_overflow:
      IntrinsicId = Intrinsic::ssub_with_overflow;
      break;
    case Builtin::BI__builtin_smul_overflow:
    case Builtin::BI__builtin_smull_overflow:
    case Builtin::BI__builtin_smulll_overflow:
      IntrinsicId = Intrinsic::smul_with_overflow;
      break;
    }


    llvm::Value *Carry;
    llvm::Value *Sum = EmitOverflowIntrinsic(*this, IntrinsicId, X, Y, Carry);
    Builder.CreateStore(Sum, SumOutPtr);

    return RValue::get(Carry);
  }
  case Builtin::BIaddressof:
  case Builtin::BI__addressof:
  case Builtin::BI__builtin_addressof:
    return RValue::get(EmitLValue(E->getArg(0)).getPointer(*this));
  case Builtin::BI__builtin_function_start:
    return RValue::get(CGM.GetFunctionStart(
        E->getArg(0)->getAsBuiltinConstantDeclRef(CGM.getContext())));
  case Builtin::BI__builtin_operator_new:
    return EmitBuiltinNewDeleteCall(
        E->getCallee()->getType()->castAs<FunctionProtoType>(), E, false);
  case Builtin::BI__builtin_operator_delete:
    EmitBuiltinNewDeleteCall(
        E->getCallee()->getType()->castAs<FunctionProtoType>(), E, true);
    return RValue::get(nullptr);

  case Builtin::BI__builtin_is_aligned:
    return EmitBuiltinIsAligned(E);
  case Builtin::BI__builtin_align_up:
    return EmitBuiltinAlignTo(E, true);
  case Builtin::BI__builtin_align_down:
    return EmitBuiltinAlignTo(E, false);

  case Builtin::BI__noop:
    // __noop always evaluates to an integer literal zero.
    return RValue::get(ConstantInt::get(IntTy, 0));
  case Builtin::BI__builtin_call_with_static_chain: {
    const CallExpr *Call = cast<CallExpr>(E->getArg(0));
    const Expr *Chain = E->getArg(1);
    return EmitCall(Call->getCallee()->getType(),
                    EmitCallee(Call->getCallee()), Call, ReturnValue,
                    EmitScalarExpr(Chain));
  }
  case Builtin::BI_InterlockedExchange8:
  case Builtin::BI_InterlockedExchange16:
  case Builtin::BI_InterlockedExchange:
  case Builtin::BI_InterlockedExchangePointer:
    return RValue::get(
        EmitMSVCBuiltinExpr(MSVCIntrin::_InterlockedExchange, E));
  case Builtin::BI_InterlockedCompareExchangePointer:
    return RValue::get(
        EmitMSVCBuiltinExpr(MSVCIntrin::_InterlockedCompareExchange, E));
  case Builtin::BI_InterlockedCompareExchangePointer_nf:
    return RValue::get(
        EmitMSVCBuiltinExpr(MSVCIntrin::_InterlockedCompareExchange_nf, E));
  case Builtin::BI_InterlockedCompareExchange8:
  case Builtin::BI_InterlockedCompareExchange16:
  case Builtin::BI_InterlockedCompareExchange:
  case Builtin::BI_InterlockedCompareExchange64:
    return RValue::get(EmitAtomicCmpXchgForMSIntrin(*this, E));
  case Builtin::BI_InterlockedIncrement16:
  case Builtin::BI_InterlockedIncrement:
    return RValue::get(
        EmitMSVCBuiltinExpr(MSVCIntrin::_InterlockedIncrement, E));
  case Builtin::BI_InterlockedDecrement16:
  case Builtin::BI_InterlockedDecrement:
    return RValue::get(
        EmitMSVCBuiltinExpr(MSVCIntrin::_InterlockedDecrement, E));
  case Builtin::BI_InterlockedAnd8:
  case Builtin::BI_InterlockedAnd16:
  case Builtin::BI_InterlockedAnd:
    return RValue::get(EmitMSVCBuiltinExpr(MSVCIntrin::_InterlockedAnd, E));
  case Builtin::BI_InterlockedExchangeAdd8:
  case Builtin::BI_InterlockedExchangeAdd16:
  case Builtin::BI_InterlockedExchangeAdd:
    return RValue::get(
        EmitMSVCBuiltinExpr(MSVCIntrin::_InterlockedExchangeAdd, E));
  case Builtin::BI_InterlockedExchangeSub8:
  case Builtin::BI_InterlockedExchangeSub16:
  case Builtin::BI_InterlockedExchangeSub:
    return RValue::get(
        EmitMSVCBuiltinExpr(MSVCIntrin::_InterlockedExchangeSub, E));
  case Builtin::BI_InterlockedOr8:
  case Builtin::BI_InterlockedOr16:
  case Builtin::BI_InterlockedOr:
    return RValue::get(EmitMSVCBuiltinExpr(MSVCIntrin::_InterlockedOr, E));
  case Builtin::BI_InterlockedXor8:
  case Builtin::BI_InterlockedXor16:
  case Builtin::BI_InterlockedXor:
    return RValue::get(EmitMSVCBuiltinExpr(MSVCIntrin::_InterlockedXor, E));

  case Builtin::BI_bittest64:
  case Builtin::BI_bittest:
  case Builtin::BI_bittestandcomplement64:
  case Builtin::BI_bittestandcomplement:
  case Builtin::BI_bittestandreset64:
  case Builtin::BI_bittestandreset:
  case Builtin::BI_bittestandset64:
  case Builtin::BI_bittestandset:
  case Builtin::BI_interlockedbittestandreset:
  case Builtin::BI_interlockedbittestandreset64:
  case Builtin::BI_interlockedbittestandreset64_acq:
  case Builtin::BI_interlockedbittestandreset64_rel:
  case Builtin::BI_interlockedbittestandreset64_nf:
  case Builtin::BI_interlockedbittestandset64:
  case Builtin::BI_interlockedbittestandset64_acq:
  case Builtin::BI_interlockedbittestandset64_rel:
  case Builtin::BI_interlockedbittestandset64_nf:
  case Builtin::BI_interlockedbittestandset:
  case Builtin::BI_interlockedbittestandset_acq:
  case Builtin::BI_interlockedbittestandset_rel:
  case Builtin::BI_interlockedbittestandset_nf:
  case Builtin::BI_interlockedbittestandreset_acq:
  case Builtin::BI_interlockedbittestandreset_rel:
  case Builtin::BI_interlockedbittestandreset_nf:
    return RValue::get(EmitBitTestIntrinsic(*this, BuiltinID, E));

    // These builtins exist to emit regular volatile loads and stores not
    // affected by the -fms-volatile setting.
  case Builtin::BI__iso_volatile_load8:
  case Builtin::BI__iso_volatile_load16:
  case Builtin::BI__iso_volatile_load32:
  case Builtin::BI__iso_volatile_load64:
    return RValue::get(EmitISOVolatileLoad(*this, E));
  case Builtin::BI__iso_volatile_store8:
  case Builtin::BI__iso_volatile_store16:
  case Builtin::BI__iso_volatile_store32:
  case Builtin::BI__iso_volatile_store64:
    return RValue::get(EmitISOVolatileStore(*this, E));

  case Builtin::BI__builtin_ptrauth_sign_constant:
    return RValue::get(ConstantEmitter(*this).emitAbstract(E, E->getType()));

  case Builtin::BI__builtin_ptrauth_auth:
  case Builtin::BI__builtin_ptrauth_auth_and_resign:
  case Builtin::BI__builtin_ptrauth_blend_discriminator:
  case Builtin::BI__builtin_ptrauth_sign_generic_data:
  case Builtin::BI__builtin_ptrauth_sign_unauthenticated:
  case Builtin::BI__builtin_ptrauth_strip: {
    // Emit the arguments.
    SmallVector<llvm::Value *, 5> Args;
    for (auto argExpr : E->arguments())
      Args.push_back(EmitScalarExpr(argExpr));

    // Cast the value to intptr_t, saving its original type.
    llvm::Type *OrigValueType = Args[0]->getType();
    if (OrigValueType->isPointerTy())
      Args[0] = Builder.CreatePtrToInt(Args[0], IntPtrTy);

    switch (BuiltinID) {
    case Builtin::BI__builtin_ptrauth_auth_and_resign:
      if (Args[4]->getType()->isPointerTy())
        Args[4] = Builder.CreatePtrToInt(Args[4], IntPtrTy);
      [[fallthrough]];

    case Builtin::BI__builtin_ptrauth_auth:
    case Builtin::BI__builtin_ptrauth_sign_unauthenticated:
      if (Args[2]->getType()->isPointerTy())
        Args[2] = Builder.CreatePtrToInt(Args[2], IntPtrTy);
      break;

    case Builtin::BI__builtin_ptrauth_sign_generic_data:
      if (Args[1]->getType()->isPointerTy())
        Args[1] = Builder.CreatePtrToInt(Args[1], IntPtrTy);
      break;

    case Builtin::BI__builtin_ptrauth_blend_discriminator:
    case Builtin::BI__builtin_ptrauth_strip:
      break;
    }

    // Call the intrinsic.
    auto IntrinsicID = [&]() -> unsigned {
      switch (BuiltinID) {
      case Builtin::BI__builtin_ptrauth_auth:
        return Intrinsic::ptrauth_auth;
      case Builtin::BI__builtin_ptrauth_auth_and_resign:
        return Intrinsic::ptrauth_resign;
      case Builtin::BI__builtin_ptrauth_blend_discriminator:
        return Intrinsic::ptrauth_blend;
      case Builtin::BI__builtin_ptrauth_sign_generic_data:
        return Intrinsic::ptrauth_sign_generic;
      case Builtin::BI__builtin_ptrauth_sign_unauthenticated:
        return Intrinsic::ptrauth_sign;
      case Builtin::BI__builtin_ptrauth_strip:
        return Intrinsic::ptrauth_strip;
      }
      llvm_unreachable("bad ptrauth intrinsic");
    }();
    auto Intrinsic = CGM.getIntrinsic(IntrinsicID);
    llvm::Value *Result = EmitRuntimeCall(Intrinsic, Args);

    if (BuiltinID != Builtin::BI__builtin_ptrauth_sign_generic_data &&
        BuiltinID != Builtin::BI__builtin_ptrauth_blend_discriminator &&
        OrigValueType->isPointerTy()) {
      Result = Builder.CreateIntToPtr(Result, OrigValueType);
    }
    return RValue::get(Result);
  }

  case Builtin::BI__builtin_get_vtable_pointer: {
    const Expr *Target = E->getArg(0);
    QualType TargetType = Target->getType();
    const CXXRecordDecl *Decl = TargetType->getPointeeCXXRecordDecl();
    assert(Decl);
    auto ThisAddress = EmitPointerWithAlignment(Target);
    assert(ThisAddress.isValid());
    llvm::Value *VTablePointer =
        GetVTablePtr(ThisAddress, Int8PtrTy, Decl, VTableAuthMode::MustTrap);
    return RValue::get(VTablePointer);
  }

  case Builtin::BI__exception_code:
  case Builtin::BI_exception_code:
    return RValue::get(EmitSEHExceptionCode());
  case Builtin::BI__exception_info:
  case Builtin::BI_exception_info:
    return RValue::get(EmitSEHExceptionInfo());
  case Builtin::BI__abnormal_termination:
  case Builtin::BI_abnormal_termination:
    return RValue::get(EmitSEHAbnormalTermination());
  case Builtin::BI_setjmpex:
    if (getTarget().getTriple().isOSMSVCRT() && E->getNumArgs() == 1 &&
        E->getArg(0)->getType()->isPointerType())
      return EmitMSVCRTSetJmp(*this, MSVCSetJmpKind::_setjmpex, E);
    break;
  case Builtin::BI_setjmp:
    if (getTarget().getTriple().isOSMSVCRT() && E->getNumArgs() == 1 &&
        E->getArg(0)->getType()->isPointerType()) {
      if (getTarget().getTriple().getArch() == llvm::Triple::x86)
        return EmitMSVCRTSetJmp(*this, MSVCSetJmpKind::_setjmp3, E);
      else if (getTarget().getTriple().getArch() == llvm::Triple::aarch64)
        return EmitMSVCRTSetJmp(*this, MSVCSetJmpKind::_setjmpex, E);
      return EmitMSVCRTSetJmp(*this, MSVCSetJmpKind::_setjmp, E);
    }
    break;

  // C++ std:: builtins.
  case Builtin::BImove:
  case Builtin::BImove_if_noexcept:
  case Builtin::BIforward:
  case Builtin::BIforward_like:
  case Builtin::BIas_const:
    return RValue::get(EmitLValue(E->getArg(0)).getPointer(*this));
  case Builtin::BI__GetExceptionInfo: {
    if (llvm::GlobalVariable *GV =
            CGM.getCXXABI().getThrowInfo(FD->getParamDecl(0)->getType()))
      return RValue::get(GV);
    break;
  }

  case Builtin::BI__fastfail:
    return RValue::get(EmitMSVCBuiltinExpr(MSVCIntrin::__fastfail, E));

  case Builtin::BI__builtin_coro_id:
    return EmitCoroutineIntrinsic(E, Intrinsic::coro_id);
  case Builtin::BI__builtin_coro_promise:
    return EmitCoroutineIntrinsic(E, Intrinsic::coro_promise);
  case Builtin::BI__builtin_coro_resume:
    EmitCoroutineIntrinsic(E, Intrinsic::coro_resume);
    return RValue::get(nullptr);
  case Builtin::BI__builtin_coro_frame:
    return EmitCoroutineIntrinsic(E, Intrinsic::coro_frame);
  case Builtin::BI__builtin_coro_noop:
    return EmitCoroutineIntrinsic(E, Intrinsic::coro_noop);
  case Builtin::BI__builtin_coro_free:
    return EmitCoroutineIntrinsic(E, Intrinsic::coro_free);
  case Builtin::BI__builtin_coro_destroy:
    EmitCoroutineIntrinsic(E, Intrinsic::coro_destroy);
    return RValue::get(nullptr);
  case Builtin::BI__builtin_coro_done:
    return EmitCoroutineIntrinsic(E, Intrinsic::coro_done);
  case Builtin::BI__builtin_coro_alloc:
    return EmitCoroutineIntrinsic(E, Intrinsic::coro_alloc);
  case Builtin::BI__builtin_coro_begin:
    return EmitCoroutineIntrinsic(E, Intrinsic::coro_begin);
  case Builtin::BI__builtin_coro_end:
    return EmitCoroutineIntrinsic(E, Intrinsic::coro_end);
  case Builtin::BI__builtin_coro_suspend:
    return EmitCoroutineIntrinsic(E, Intrinsic::coro_suspend);
  case Builtin::BI__builtin_coro_size:
    return EmitCoroutineIntrinsic(E, Intrinsic::coro_size);
  case Builtin::BI__builtin_coro_align:
    return EmitCoroutineIntrinsic(E, Intrinsic::coro_align);

  // OpenCL v2.0 s6.13.16.2, Built-in pipe read and write functions
  case Builtin::BIread_pipe:
  case Builtin::BIwrite_pipe: {
    Value *Arg0 = EmitScalarExpr(E->getArg(0)),
          *Arg1 = EmitScalarExpr(E->getArg(1));
    CGOpenCLRuntime OpenCLRT(CGM);
    Value *PacketSize = OpenCLRT.getPipeElemSize(E->getArg(0));
    Value *PacketAlign = OpenCLRT.getPipeElemAlign(E->getArg(0));

    // Type of the generic packet parameter.
    unsigned GenericAS =
        getContext().getTargetAddressSpace(LangAS::opencl_generic);
    llvm::Type *I8PTy = llvm::PointerType::get(getLLVMContext(), GenericAS);

    // Testing which overloaded version we should generate the call for.
    if (2U == E->getNumArgs()) {
      const char *Name = (BuiltinID == Builtin::BIread_pipe) ? "__read_pipe_2"
                                                             : "__write_pipe_2";
      // Creating a generic function type to be able to call with any builtin or
      // user defined type.
      llvm::Type *ArgTys[] = {Arg0->getType(), I8PTy, Int32Ty, Int32Ty};
      llvm::FunctionType *FTy = llvm::FunctionType::get(Int32Ty, ArgTys, false);
      Value *ACast = Builder.CreateAddrSpaceCast(Arg1, I8PTy);
      return RValue::get(
          EmitRuntimeCall(CGM.CreateRuntimeFunction(FTy, Name),
                          {Arg0, ACast, PacketSize, PacketAlign}));
    } else {
      assert(4 == E->getNumArgs() &&
             "Illegal number of parameters to pipe function");
      const char *Name = (BuiltinID == Builtin::BIread_pipe) ? "__read_pipe_4"
                                                             : "__write_pipe_4";

      llvm::Type *ArgTys[] = {Arg0->getType(), Arg1->getType(), Int32Ty, I8PTy,
                              Int32Ty, Int32Ty};
      Value *Arg2 = EmitScalarExpr(E->getArg(2)),
            *Arg3 = EmitScalarExpr(E->getArg(3));
      llvm::FunctionType *FTy = llvm::FunctionType::get(Int32Ty, ArgTys, false);
      Value *ACast = Builder.CreateAddrSpaceCast(Arg3, I8PTy);
      // We know the third argument is an integer type, but we may need to cast
      // it to i32.
      if (Arg2->getType() != Int32Ty)
        Arg2 = Builder.CreateZExtOrTrunc(Arg2, Int32Ty);
      return RValue::get(
          EmitRuntimeCall(CGM.CreateRuntimeFunction(FTy, Name),
                          {Arg0, Arg1, Arg2, ACast, PacketSize, PacketAlign}));
    }
  }
  // OpenCL v2.0 s6.13.16 ,s9.17.3.5 - Built-in pipe reserve read and write
  // functions
  case Builtin::BIreserve_read_pipe:
  case Builtin::BIreserve_write_pipe:
  case Builtin::BIwork_group_reserve_read_pipe:
  case Builtin::BIwork_group_reserve_write_pipe:
  case Builtin::BIsub_group_reserve_read_pipe:
  case Builtin::BIsub_group_reserve_write_pipe: {
    // Composing the mangled name for the function.
    const char *Name;
    if (BuiltinID == Builtin::BIreserve_read_pipe)
      Name = "__reserve_read_pipe";
    else if (BuiltinID == Builtin::BIreserve_write_pipe)
      Name = "__reserve_write_pipe";
    else if (BuiltinID == Builtin::BIwork_group_reserve_read_pipe)
      Name = "__work_group_reserve_read_pipe";
    else if (BuiltinID == Builtin::BIwork_group_reserve_write_pipe)
      Name = "__work_group_reserve_write_pipe";
    else if (BuiltinID == Builtin::BIsub_group_reserve_read_pipe)
      Name = "__sub_group_reserve_read_pipe";
    else
      Name = "__sub_group_reserve_write_pipe";

    Value *Arg0 = EmitScalarExpr(E->getArg(0)),
          *Arg1 = EmitScalarExpr(E->getArg(1));
    llvm::Type *ReservedIDTy = ConvertType(getContext().OCLReserveIDTy);
    CGOpenCLRuntime OpenCLRT(CGM);
    Value *PacketSize = OpenCLRT.getPipeElemSize(E->getArg(0));
    Value *PacketAlign = OpenCLRT.getPipeElemAlign(E->getArg(0));

    // Building the generic function prototype.
    llvm::Type *ArgTys[] = {Arg0->getType(), Int32Ty, Int32Ty, Int32Ty};
    llvm::FunctionType *FTy =
        llvm::FunctionType::get(ReservedIDTy, ArgTys, false);
    // We know the second argument is an integer type, but we may need to cast
    // it to i32.
    if (Arg1->getType() != Int32Ty)
      Arg1 = Builder.CreateZExtOrTrunc(Arg1, Int32Ty);
    return RValue::get(EmitRuntimeCall(CGM.CreateRuntimeFunction(FTy, Name),
                                       {Arg0, Arg1, PacketSize, PacketAlign}));
  }
  // OpenCL v2.0 s6.13.16, s9.17.3.5 - Built-in pipe commit read and write
  // functions
  case Builtin::BIcommit_read_pipe:
  case Builtin::BIcommit_write_pipe:
  case Builtin::BIwork_group_commit_read_pipe:
  case Builtin::BIwork_group_commit_write_pipe:
  case Builtin::BIsub_group_commit_read_pipe:
  case Builtin::BIsub_group_commit_write_pipe: {
    const char *Name;
    if (BuiltinID == Builtin::BIcommit_read_pipe)
      Name = "__commit_read_pipe";
    else if (BuiltinID == Builtin::BIcommit_write_pipe)
      Name = "__commit_write_pipe";
    else if (BuiltinID == Builtin::BIwork_group_commit_read_pipe)
      Name = "__work_group_commit_read_pipe";
    else if (BuiltinID == Builtin::BIwork_group_commit_write_pipe)
      Name = "__work_group_commit_write_pipe";
    else if (BuiltinID == Builtin::BIsub_group_commit_read_pipe)
      Name = "__sub_group_commit_read_pipe";
    else
      Name = "__sub_group_commit_write_pipe";

    Value *Arg0 = EmitScalarExpr(E->getArg(0)),
          *Arg1 = EmitScalarExpr(E->getArg(1));
    CGOpenCLRuntime OpenCLRT(CGM);
    Value *PacketSize = OpenCLRT.getPipeElemSize(E->getArg(0));
    Value *PacketAlign = OpenCLRT.getPipeElemAlign(E->getArg(0));

    // Building the generic function prototype.
    llvm::Type *ArgTys[] = {Arg0->getType(), Arg1->getType(), Int32Ty, Int32Ty};
    llvm::FunctionType *FTy = llvm::FunctionType::get(
        llvm::Type::getVoidTy(getLLVMContext()), ArgTys, false);

    return RValue::get(EmitRuntimeCall(CGM.CreateRuntimeFunction(FTy, Name),
                                       {Arg0, Arg1, PacketSize, PacketAlign}));
  }
  // OpenCL v2.0 s6.13.16.4 Built-in pipe query functions
  case Builtin::BIget_pipe_num_packets:
  case Builtin::BIget_pipe_max_packets: {
    const char *BaseName;
    const auto *PipeTy = E->getArg(0)->getType()->castAs<PipeType>();
    if (BuiltinID == Builtin::BIget_pipe_num_packets)
      BaseName = "__get_pipe_num_packets";
    else
      BaseName = "__get_pipe_max_packets";
    std::string Name = std::string(BaseName) +
                       std::string(PipeTy->isReadOnly() ? "_ro" : "_wo");

    // Building the generic function prototype.
    Value *Arg0 = EmitScalarExpr(E->getArg(0));
    CGOpenCLRuntime OpenCLRT(CGM);
    Value *PacketSize = OpenCLRT.getPipeElemSize(E->getArg(0));
    Value *PacketAlign = OpenCLRT.getPipeElemAlign(E->getArg(0));
    llvm::Type *ArgTys[] = {Arg0->getType(), Int32Ty, Int32Ty};
    llvm::FunctionType *FTy = llvm::FunctionType::get(Int32Ty, ArgTys, false);

    return RValue::get(EmitRuntimeCall(CGM.CreateRuntimeFunction(FTy, Name),
                                       {Arg0, PacketSize, PacketAlign}));
  }

  // OpenCL v2.0 s6.13.9 - Address space qualifier functions.
  case Builtin::BIto_global:
  case Builtin::BIto_local:
  case Builtin::BIto_private: {
    auto Arg0 = EmitScalarExpr(E->getArg(0));
    auto NewArgT = llvm::PointerType::get(
        getLLVMContext(),
        CGM.getContext().getTargetAddressSpace(LangAS::opencl_generic));
    auto NewRetT = llvm::PointerType::get(
        getLLVMContext(),
        CGM.getContext().getTargetAddressSpace(
            E->getType()->getPointeeType().getAddressSpace()));
    auto FTy = llvm::FunctionType::get(NewRetT, {NewArgT}, false);
    llvm::Value *NewArg;
    if (Arg0->getType()->getPointerAddressSpace() !=
        NewArgT->getPointerAddressSpace())
      NewArg = Builder.CreateAddrSpaceCast(Arg0, NewArgT);
    else
      NewArg = Builder.CreateBitOrPointerCast(Arg0, NewArgT);
    auto NewName = std::string("__") + E->getDirectCallee()->getName().str();
    auto NewCall =
        EmitRuntimeCall(CGM.CreateRuntimeFunction(FTy, NewName), {NewArg});
    return RValue::get(Builder.CreateBitOrPointerCast(NewCall,
      ConvertType(E->getType())));
  }

  // OpenCL v2.0, s6.13.17 - Enqueue kernel function.
  // Table 6.13.17.1 specifies four overload forms of enqueue_kernel.
  // The code below expands the builtin call to a call to one of the following
  // functions that an OpenCL runtime library will have to provide:
  //   __enqueue_kernel_basic
  //   __enqueue_kernel_varargs
  //   __enqueue_kernel_basic_events
  //   __enqueue_kernel_events_varargs
  case Builtin::BIenqueue_kernel: {
    StringRef Name; // Generated function call name
    unsigned NumArgs = E->getNumArgs();

    llvm::Type *QueueTy = ConvertType(getContext().OCLQueueTy);
    llvm::Type *GenericVoidPtrTy = Builder.getPtrTy(
        getContext().getTargetAddressSpace(LangAS::opencl_generic));

    llvm::Value *Queue = EmitScalarExpr(E->getArg(0));
    llvm::Value *Flags = EmitScalarExpr(E->getArg(1));
    LValue NDRangeL = EmitAggExprToLValue(E->getArg(2));
    llvm::Value *Range = NDRangeL.getAddress().emitRawPointer(*this);

    // FIXME: Look through the addrspacecast which may exist to the stack
    // temporary as a hack.
    //
    // This is hardcoding the assumed ABI of the target function. This assumes
    // direct passing for every argument except NDRange, which is assumed to be
    // byval or byref indirect passed.
    //
    // This should be fixed to query a signature from CGOpenCLRuntime, and go
    // through EmitCallArgs to get the correct target ABI.
    Range = Range->stripPointerCasts();

    llvm::Type *RangePtrTy = Range->getType();

    if (NumArgs == 4) {
      // The most basic form of the call with parameters:
      // queue_t, kernel_enqueue_flags_t, ndrange_t, block(void)
      Name = "__enqueue_kernel_basic";
      llvm::Type *ArgTys[] = {QueueTy, Int32Ty, RangePtrTy, GenericVoidPtrTy,
                              GenericVoidPtrTy};
      llvm::FunctionType *FTy = llvm::FunctionType::get(Int32Ty, ArgTys, false);

      auto Info =
          CGM.getOpenCLRuntime().emitOpenCLEnqueuedBlock(*this, E->getArg(3));
      llvm::Value *Kernel =
          Builder.CreatePointerCast(Info.KernelHandle, GenericVoidPtrTy);
      llvm::Value *Block =
          Builder.CreatePointerCast(Info.BlockArg, GenericVoidPtrTy);

      auto RTCall = EmitRuntimeCall(CGM.CreateRuntimeFunction(FTy, Name),
                                    {Queue, Flags, Range, Kernel, Block});
      return RValue::get(RTCall);
    }
    assert(NumArgs >= 5 && "Invalid enqueue_kernel signature");

    // Create a temporary array to hold the sizes of local pointer arguments
    // for the block. \p First is the position of the first size argument.
    auto CreateArrayForSizeVar =
        [=](unsigned First) -> std::pair<llvm::Value *, llvm::Value *> {
      llvm::APInt ArraySize(32, NumArgs - First);
      QualType SizeArrayTy = getContext().getConstantArrayType(
          getContext().getSizeType(), ArraySize, nullptr,
          ArraySizeModifier::Normal,
          /*IndexTypeQuals=*/0);
      auto Tmp = CreateMemTemp(SizeArrayTy, "block_sizes");
      llvm::Value *TmpPtr = Tmp.getPointer();
      // The EmitLifetime* pair expect a naked Alloca as their last argument,
      // however for cases where the default AS is not the Alloca AS, Tmp is
      // actually the Alloca ascasted to the default AS, hence the
      // stripPointerCasts()
      llvm::Value *Alloca = TmpPtr->stripPointerCasts();
      llvm::Value *ElemPtr;
      EmitLifetimeStart(Alloca);
      // Each of the following arguments specifies the size of the corresponding
      // argument passed to the enqueued block.
      auto *Zero = llvm::ConstantInt::get(IntTy, 0);
      for (unsigned I = First; I < NumArgs; ++I) {
        auto *Index = llvm::ConstantInt::get(IntTy, I - First);
        auto *GEP =
            Builder.CreateGEP(Tmp.getElementType(), Alloca, {Zero, Index});
        if (I == First)
          ElemPtr = GEP;
        auto *V =
            Builder.CreateZExtOrTrunc(EmitScalarExpr(E->getArg(I)), SizeTy);
        Builder.CreateAlignedStore(
            V, GEP, CGM.getDataLayout().getPrefTypeAlign(SizeTy));
      }
      // Return the Alloca itself rather than a potential ascast as this is only
      // used by the paired EmitLifetimeEnd.
      return {ElemPtr, Alloca};
    };

    // Could have events and/or varargs.
    if (E->getArg(3)->getType()->isBlockPointerType()) {
      // No events passed, but has variadic arguments.
      Name = "__enqueue_kernel_varargs";
      auto Info =
          CGM.getOpenCLRuntime().emitOpenCLEnqueuedBlock(*this, E->getArg(3));
      llvm::Value *Kernel =
          Builder.CreatePointerCast(Info.KernelHandle, GenericVoidPtrTy);
      auto *Block = Builder.CreatePointerCast(Info.BlockArg, GenericVoidPtrTy);
      auto [ElemPtr, TmpPtr] = CreateArrayForSizeVar(4);

      // Create a vector of the arguments, as well as a constant value to
      // express to the runtime the number of variadic arguments.
      llvm::Value *const Args[] = {Queue,  Flags,
                                   Range,  Kernel,
                                   Block,  ConstantInt::get(IntTy, NumArgs - 4),
                                   ElemPtr};
      llvm::Type *const ArgTys[] = {
          QueueTy,          IntTy, RangePtrTy,        GenericVoidPtrTy,
          GenericVoidPtrTy, IntTy, ElemPtr->getType()};

      llvm::FunctionType *FTy = llvm::FunctionType::get(Int32Ty, ArgTys, false);
      auto Call = RValue::get(
          EmitRuntimeCall(CGM.CreateRuntimeFunction(FTy, Name), Args));
      EmitLifetimeEnd(TmpPtr);
      return Call;
    }
    // Any calls now have event arguments passed.
    if (NumArgs >= 7) {
      llvm::PointerType *PtrTy = llvm::PointerType::get(
          CGM.getLLVMContext(),
          CGM.getContext().getTargetAddressSpace(LangAS::opencl_generic));

      llvm::Value *NumEvents =
          Builder.CreateZExtOrTrunc(EmitScalarExpr(E->getArg(3)), Int32Ty);

      // Since SemaOpenCLBuiltinEnqueueKernel allows fifth and sixth arguments
      // to be a null pointer constant (including `0` literal), we can take it
      // into account and emit null pointer directly.
      llvm::Value *EventWaitList = nullptr;
      if (E->getArg(4)->isNullPointerConstant(
              getContext(), Expr::NPC_ValueDependentIsNotNull)) {
        EventWaitList = llvm::ConstantPointerNull::get(PtrTy);
      } else {
        EventWaitList =
            E->getArg(4)->getType()->isArrayType()
                ? EmitArrayToPointerDecay(E->getArg(4)).emitRawPointer(*this)
                : EmitScalarExpr(E->getArg(4));
        // Convert to generic address space.
        EventWaitList = Builder.CreatePointerCast(EventWaitList, PtrTy);
      }
      llvm::Value *EventRet = nullptr;
      if (E->getArg(5)->isNullPointerConstant(
              getContext(), Expr::NPC_ValueDependentIsNotNull)) {
        EventRet = llvm::ConstantPointerNull::get(PtrTy);
      } else {
        EventRet =
            Builder.CreatePointerCast(EmitScalarExpr(E->getArg(5)), PtrTy);
      }

      auto Info =
          CGM.getOpenCLRuntime().emitOpenCLEnqueuedBlock(*this, E->getArg(6));
      llvm::Value *Kernel =
          Builder.CreatePointerCast(Info.KernelHandle, GenericVoidPtrTy);
      llvm::Value *Block =
          Builder.CreatePointerCast(Info.BlockArg, GenericVoidPtrTy);

      std::vector<llvm::Type *> ArgTys = {
          QueueTy, Int32Ty, RangePtrTy,       Int32Ty,
          PtrTy,   PtrTy,   GenericVoidPtrTy, GenericVoidPtrTy};

      std::vector<llvm::Value *> Args = {Queue,     Flags,         Range,
                                         NumEvents, EventWaitList, EventRet,
                                         Kernel,    Block};

      if (NumArgs == 7) {
        // Has events but no variadics.
        Name = "__enqueue_kernel_basic_events";
        llvm::FunctionType *FTy =
            llvm::FunctionType::get(Int32Ty, ArgTys, false);
        return RValue::get(
            EmitRuntimeCall(CGM.CreateRuntimeFunction(FTy, Name), Args));
      }
      // Has event info and variadics
      // Pass the number of variadics to the runtime function too.
      Args.push_back(ConstantInt::get(Int32Ty, NumArgs - 7));
      ArgTys.push_back(Int32Ty);
      Name = "__enqueue_kernel_events_varargs";

      auto [ElemPtr, TmpPtr] = CreateArrayForSizeVar(7);
      Args.push_back(ElemPtr);
      ArgTys.push_back(ElemPtr->getType());

      llvm::FunctionType *FTy = llvm::FunctionType::get(Int32Ty, ArgTys, false);
      auto Call = RValue::get(
          EmitRuntimeCall(CGM.CreateRuntimeFunction(FTy, Name), Args));
      EmitLifetimeEnd(TmpPtr);
      return Call;
    }
    llvm_unreachable("Unexpected enqueue_kernel signature");
  }
  // OpenCL v2.0 s6.13.17.6 - Kernel query functions need bitcast of block
  // parameter.
  case Builtin::BIget_kernel_work_group_size: {
    llvm::Type *GenericVoidPtrTy = Builder.getPtrTy(
        getContext().getTargetAddressSpace(LangAS::opencl_generic));
    auto Info =
        CGM.getOpenCLRuntime().emitOpenCLEnqueuedBlock(*this, E->getArg(0));
    Value *Kernel =
        Builder.CreatePointerCast(Info.KernelHandle, GenericVoidPtrTy);
    Value *Arg = Builder.CreatePointerCast(Info.BlockArg, GenericVoidPtrTy);
    return RValue::get(EmitRuntimeCall(
        CGM.CreateRuntimeFunction(
            llvm::FunctionType::get(IntTy, {GenericVoidPtrTy, GenericVoidPtrTy},
                                    false),
            "__get_kernel_work_group_size_impl"),
        {Kernel, Arg}));
  }
  case Builtin::BIget_kernel_preferred_work_group_size_multiple: {
    llvm::Type *GenericVoidPtrTy = Builder.getPtrTy(
        getContext().getTargetAddressSpace(LangAS::opencl_generic));
    auto Info =
        CGM.getOpenCLRuntime().emitOpenCLEnqueuedBlock(*this, E->getArg(0));
    Value *Kernel =
        Builder.CreatePointerCast(Info.KernelHandle, GenericVoidPtrTy);
    Value *Arg = Builder.CreatePointerCast(Info.BlockArg, GenericVoidPtrTy);
    return RValue::get(EmitRuntimeCall(
        CGM.CreateRuntimeFunction(
            llvm::FunctionType::get(IntTy, {GenericVoidPtrTy, GenericVoidPtrTy},
                                    false),
            "__get_kernel_preferred_work_group_size_multiple_impl"),
        {Kernel, Arg}));
  }
  case Builtin::BIget_kernel_max_sub_group_size_for_ndrange:
  case Builtin::BIget_kernel_sub_group_count_for_ndrange: {
    llvm::Type *GenericVoidPtrTy = Builder.getPtrTy(
        getContext().getTargetAddressSpace(LangAS::opencl_generic));
    LValue NDRangeL = EmitAggExprToLValue(E->getArg(0));
    llvm::Value *NDRange = NDRangeL.getAddress().emitRawPointer(*this);
    auto Info =
        CGM.getOpenCLRuntime().emitOpenCLEnqueuedBlock(*this, E->getArg(1));
    Value *Kernel =
        Builder.CreatePointerCast(Info.KernelHandle, GenericVoidPtrTy);
    Value *Block = Builder.CreatePointerCast(Info.BlockArg, GenericVoidPtrTy);
    const char *Name =
        BuiltinID == Builtin::BIget_kernel_max_sub_group_size_for_ndrange
            ? "__get_kernel_max_sub_group_size_for_ndrange_impl"
            : "__get_kernel_sub_group_count_for_ndrange_impl";
    return RValue::get(EmitRuntimeCall(
        CGM.CreateRuntimeFunction(
            llvm::FunctionType::get(
                IntTy, {NDRange->getType(), GenericVoidPtrTy, GenericVoidPtrTy},
                false),
            Name),
        {NDRange, Kernel, Block}));
  }
  case Builtin::BI__builtin_store_half:
  case Builtin::BI__builtin_store_halff: {
    Value *Val = EmitScalarExpr(E->getArg(0));
    Address Address = EmitPointerWithAlignment(E->getArg(1));
    Value *HalfVal = Builder.CreateFPTrunc(Val, Builder.getHalfTy());
    Builder.CreateStore(HalfVal, Address);
    return RValue::get(nullptr);
  }
  case Builtin::BI__builtin_load_half: {
    Address Address = EmitPointerWithAlignment(E->getArg(0));
    Value *HalfVal = Builder.CreateLoad(Address);
    return RValue::get(Builder.CreateFPExt(HalfVal, Builder.getDoubleTy()));
  }
  case Builtin::BI__builtin_load_halff: {
    Address Address = EmitPointerWithAlignment(E->getArg(0));
    Value *HalfVal = Builder.CreateLoad(Address);
    return RValue::get(Builder.CreateFPExt(HalfVal, Builder.getFloatTy()));
  }
  case Builtin::BI__builtin_printf:
  case Builtin::BIprintf:
    if (getTarget().getTriple().isNVPTX() ||
        getTarget().getTriple().isAMDGCN() ||
        (getTarget().getTriple().isSPIRV() &&
         getTarget().getTriple().getVendor() == Triple::VendorType::AMD)) {
      if (getTarget().getTriple().isNVPTX())
        return EmitNVPTXDevicePrintfCallExpr(E);
      if ((getTarget().getTriple().isAMDGCN() ||
           getTarget().getTriple().isSPIRV()) &&
          getLangOpts().HIP)
        return EmitAMDGPUDevicePrintfCallExpr(E);
    }

    break;
  case Builtin::BI__builtin_canonicalize:
  case Builtin::BI__builtin_canonicalizef:
  case Builtin::BI__builtin_canonicalizef16:
  case Builtin::BI__builtin_canonicalizel:
    return RValue::get(
        emitBuiltinWithOneOverloadedType<1>(*this, E, Intrinsic::canonicalize));

  case Builtin::BI__builtin_thread_pointer: {
    if (!getContext().getTargetInfo().isTLSSupported())
      CGM.ErrorUnsupported(E, "__builtin_thread_pointer");

    return RValue::get(Builder.CreateIntrinsic(llvm::Intrinsic::thread_pointer,
                                               {GlobalsInt8PtrTy}, {}));
  }
  case Builtin::BI__builtin_os_log_format:
    return emitBuiltinOSLogFormat(*E);

  case Builtin::BI__xray_customevent: {
    if (!ShouldXRayInstrumentFunction())
      return RValue::getIgnored();

    if (!CGM.getCodeGenOpts().XRayInstrumentationBundle.has(
            XRayInstrKind::Custom))
      return RValue::getIgnored();

    if (const auto *XRayAttr = CurFuncDecl->getAttr<XRayInstrumentAttr>())
      if (XRayAttr->neverXRayInstrument() && !AlwaysEmitXRayCustomEvents())
        return RValue::getIgnored();

    Function *F = CGM.getIntrinsic(Intrinsic::xray_customevent);
    auto FTy = F->getFunctionType();
    auto Arg0 = E->getArg(0);
    auto Arg0Val = EmitScalarExpr(Arg0);
    auto Arg0Ty = Arg0->getType();
    auto PTy0 = FTy->getParamType(0);
    if (PTy0 != Arg0Val->getType()) {
      if (Arg0Ty->isArrayType())
        Arg0Val = EmitArrayToPointerDecay(Arg0).emitRawPointer(*this);
      else
        Arg0Val = Builder.CreatePointerCast(Arg0Val, PTy0);
    }
    auto Arg1 = EmitScalarExpr(E->getArg(1));
    auto PTy1 = FTy->getParamType(1);
    if (PTy1 != Arg1->getType())
      Arg1 = Builder.CreateTruncOrBitCast(Arg1, PTy1);
    return RValue::get(Builder.CreateCall(F, {Arg0Val, Arg1}));
  }

  case Builtin::BI__xray_typedevent: {
    // TODO: There should be a way to always emit events even if the current
    // function is not instrumented. Losing events in a stream can cripple
    // a trace.
    if (!ShouldXRayInstrumentFunction())
      return RValue::getIgnored();

    if (!CGM.getCodeGenOpts().XRayInstrumentationBundle.has(
            XRayInstrKind::Typed))
      return RValue::getIgnored();

    if (const auto *XRayAttr = CurFuncDecl->getAttr<XRayInstrumentAttr>())
      if (XRayAttr->neverXRayInstrument() && !AlwaysEmitXRayTypedEvents())
        return RValue::getIgnored();

    Function *F = CGM.getIntrinsic(Intrinsic::xray_typedevent);
    auto FTy = F->getFunctionType();
    auto Arg0 = EmitScalarExpr(E->getArg(0));
    auto PTy0 = FTy->getParamType(0);
    if (PTy0 != Arg0->getType())
      Arg0 = Builder.CreateTruncOrBitCast(Arg0, PTy0);
    auto Arg1 = E->getArg(1);
    auto Arg1Val = EmitScalarExpr(Arg1);
    auto Arg1Ty = Arg1->getType();
    auto PTy1 = FTy->getParamType(1);
    if (PTy1 != Arg1Val->getType()) {
      if (Arg1Ty->isArrayType())
        Arg1Val = EmitArrayToPointerDecay(Arg1).emitRawPointer(*this);
      else
        Arg1Val = Builder.CreatePointerCast(Arg1Val, PTy1);
    }
    auto Arg2 = EmitScalarExpr(E->getArg(2));
    auto PTy2 = FTy->getParamType(2);
    if (PTy2 != Arg2->getType())
      Arg2 = Builder.CreateTruncOrBitCast(Arg2, PTy2);
    return RValue::get(Builder.CreateCall(F, {Arg0, Arg1Val, Arg2}));
  }

  case Builtin::BI__builtin_ms_va_start:
  case Builtin::BI__builtin_ms_va_end:
    return RValue::get(
        EmitVAStartEnd(EmitMSVAListRef(E->getArg(0)).emitRawPointer(*this),
                       BuiltinID == Builtin::BI__builtin_ms_va_start));

  case Builtin::BI__builtin_ms_va_copy: {
    // Lower this manually. We can't reliably determine whether or not any
    // given va_copy() is for a Win64 va_list from the calling convention
    // alone, because it's legal to do this from a System V ABI function.
    // With opaque pointer types, we won't have enough information in LLVM
    // IR to determine this from the argument types, either. Best to do it
    // now, while we have enough information.
    Address DestAddr = EmitMSVAListRef(E->getArg(0));
    Address SrcAddr = EmitMSVAListRef(E->getArg(1));

    DestAddr = DestAddr.withElementType(Int8PtrTy);
    SrcAddr = SrcAddr.withElementType(Int8PtrTy);

    Value *ArgPtr = Builder.CreateLoad(SrcAddr, "ap.val");
    return RValue::get(Builder.CreateStore(ArgPtr, DestAddr));
  }

  case Builtin::BI__builtin_get_device_side_mangled_name: {
    auto Name = CGM.getCUDARuntime().getDeviceSideName(
        cast<DeclRefExpr>(E->getArg(0)->IgnoreImpCasts())->getDecl());
    auto Str = CGM.GetAddrOfConstantCString(Name, "");
    return RValue::get(Str.getPointer());
  }
  }

  // If this is an alias for a lib function (e.g. __builtin_sin), emit
  // the call using the normal call path, but using the unmangled
  // version of the function name.
  const auto &BI = getContext().BuiltinInfo;
  if (!shouldEmitBuiltinAsIR(BuiltinID, BI, *this) &&
      BI.isLibFunction(BuiltinID))
    return emitLibraryCall(*this, FD, E,
                           CGM.getBuiltinLibFunction(FD, BuiltinID));

  // If this is a predefined lib function (e.g. malloc), emit the call
  // using exactly the normal call path.
  if (BI.isPredefinedLibFunction(BuiltinID))
    return emitLibraryCall(*this, FD, E, CGM.getRawFunctionPointer(FD));

  // Check that a call to a target specific builtin has the correct target
  // features.
  // This is down here to avoid non-target specific builtins, however, if
  // generic builtins start to require generic target features then we
  // can move this up to the beginning of the function.
  checkTargetFeatures(E, FD);

  if (unsigned VectorWidth = getContext().BuiltinInfo.getRequiredVectorWidth(BuiltinID))
    LargestVectorWidth = std::max(LargestVectorWidth, VectorWidth);

  // See if we have a target specific intrinsic.
  std::string Name = getContext().BuiltinInfo.getName(BuiltinID);
  Intrinsic::ID IntrinsicID = Intrinsic::not_intrinsic;
  StringRef Prefix =
      llvm::Triple::getArchTypePrefix(getTarget().getTriple().getArch());
  if (!Prefix.empty()) {
    IntrinsicID = Intrinsic::getIntrinsicForClangBuiltin(Prefix.data(), Name);
    if (IntrinsicID == Intrinsic::not_intrinsic && Prefix == "spv" &&
        getTarget().getTriple().getOS() == llvm::Triple::OSType::AMDHSA)
      IntrinsicID = Intrinsic::getIntrinsicForClangBuiltin("amdgcn", Name);
    // NOTE we don't need to perform a compatibility flag check here since the
    // intrinsics are declared in Builtins*.def via LANGBUILTIN which filter the
    // MS builtins via ALL_MS_LANGUAGES and are filtered earlier.
    if (IntrinsicID == Intrinsic::not_intrinsic)
      IntrinsicID = Intrinsic::getIntrinsicForMSBuiltin(Prefix.data(), Name);
  }

  if (IntrinsicID != Intrinsic::not_intrinsic) {
    SmallVector<Value*, 16> Args;

    // Find out if any arguments are required to be integer constant
    // expressions.
    unsigned ICEArguments = 0;
    ASTContext::GetBuiltinTypeError Error;
    getContext().GetBuiltinType(BuiltinID, Error, &ICEArguments);
    assert(Error == ASTContext::GE_None && "Should not codegen an error");

    Function *F = CGM.getIntrinsic(IntrinsicID);
    llvm::FunctionType *FTy = F->getFunctionType();

    for (unsigned i = 0, e = E->getNumArgs(); i != e; ++i) {
      Value *ArgValue = EmitScalarOrConstFoldImmArg(ICEArguments, i, E);
      // If the intrinsic arg type is different from the builtin arg type
      // we need to do a bit cast.
      llvm::Type *PTy = FTy->getParamType(i);
      if (PTy != ArgValue->getType()) {
        // XXX - vector of pointers?
        if (auto *PtrTy = dyn_cast<llvm::PointerType>(PTy)) {
          if (PtrTy->getAddressSpace() !=
              ArgValue->getType()->getPointerAddressSpace()) {
            ArgValue = Builder.CreateAddrSpaceCast(
                ArgValue, llvm::PointerType::get(getLLVMContext(),
                                                 PtrTy->getAddressSpace()));
          }
        }

        // Cast vector type (e.g., v256i32) to x86_amx, this only happen
        // in amx intrinsics.
        if (PTy->isX86_AMXTy())
          ArgValue = Builder.CreateIntrinsic(Intrinsic::x86_cast_vector_to_tile,
                                             {ArgValue->getType()}, {ArgValue});
        else
          ArgValue = Builder.CreateBitCast(ArgValue, PTy);
      }

      Args.push_back(ArgValue);
    }

    Value *V = Builder.CreateCall(F, Args);
    QualType BuiltinRetType = E->getType();

    llvm::Type *RetTy = VoidTy;
    if (!BuiltinRetType->isVoidType())
      RetTy = ConvertType(BuiltinRetType);

    if (RetTy != V->getType()) {
      // XXX - vector of pointers?
      if (auto *PtrTy = dyn_cast<llvm::PointerType>(RetTy)) {
        if (PtrTy->getAddressSpace() != V->getType()->getPointerAddressSpace()) {
          V = Builder.CreateAddrSpaceCast(
              V, llvm::PointerType::get(getLLVMContext(),
                                        PtrTy->getAddressSpace()));
        }
      }

      // Cast x86_amx to vector type (e.g., v256i32), this only happen
      // in amx intrinsics.
      if (V->getType()->isX86_AMXTy())
        V = Builder.CreateIntrinsic(Intrinsic::x86_cast_tile_to_vector, {RetTy},
                                    {V});
      else
        V = Builder.CreateBitCast(V, RetTy);
    }

    if (RetTy->isVoidTy())
      return RValue::get(nullptr);

    return RValue::get(V);
  }

  // Some target-specific builtins can have aggregate return values, e.g.
  // __builtin_arm_mve_vld2q_u32. So if the result is an aggregate, force
  // ReturnValue to be non-null, so that the target-specific emission code can
  // always just emit into it.
  TypeEvaluationKind EvalKind = getEvaluationKind(E->getType());
  if (EvalKind == TEK_Aggregate && ReturnValue.isNull()) {
    Address DestPtr = CreateMemTemp(E->getType(), "agg.tmp");
    ReturnValue = ReturnValueSlot(DestPtr, false);
  }

  // Now see if we can emit a target-specific builtin.
  if (Value *V = EmitTargetBuiltinExpr(BuiltinID, E, ReturnValue)) {
    switch (EvalKind) {
    case TEK_Scalar:
      if (V->getType()->isVoidTy())
        return RValue::get(nullptr);
      return RValue::get(V);
    case TEK_Aggregate:
      return RValue::getAggregate(ReturnValue.getAddress(),
                                  ReturnValue.isVolatile());
    case TEK_Complex:
      llvm_unreachable("No current target builtin returns complex");
    }
    llvm_unreachable("Bad evaluation kind in EmitBuiltinExpr");
  }

  // EmitHLSLBuiltinExpr will check getLangOpts().HLSL
  if (Value *V = EmitHLSLBuiltinExpr(BuiltinID, E, ReturnValue)) {
    switch (EvalKind) {
    case TEK_Scalar:
      if (V->getType()->isVoidTy())
        return RValue::get(nullptr);
      return RValue::get(V);
    case TEK_Aggregate:
      return RValue::getAggregate(ReturnValue.getAddress(),
                                  ReturnValue.isVolatile());
    case TEK_Complex:
      llvm_unreachable("No current hlsl builtin returns complex");
    }
    llvm_unreachable("Bad evaluation kind in EmitBuiltinExpr");
  }

  if (getLangOpts().HIPStdPar && getLangOpts().CUDAIsDevice)
    return EmitHipStdParUnsupportedBuiltin(this, FD);

  ErrorUnsupported(E, "builtin function");

  // Unknown builtin, for now just dump it out and return undef.
  return GetUndefRValue(E->getType());
}

namespace {
struct BuiltinAlignArgs {
  llvm::Value *Src = nullptr;
  llvm::Type *SrcType = nullptr;
  llvm::Value *Alignment = nullptr;
  llvm::Value *Mask = nullptr;
  llvm::IntegerType *IntType = nullptr;

  BuiltinAlignArgs(const CallExpr *E, CodeGenFunction &CGF) {
    QualType AstType = E->getArg(0)->getType();
    if (AstType->isArrayType())
      Src = CGF.EmitArrayToPointerDecay(E->getArg(0)).emitRawPointer(CGF);
    else
      Src = CGF.EmitScalarExpr(E->getArg(0));
    SrcType = Src->getType();
    if (SrcType->isPointerTy()) {
      IntType = IntegerType::get(
          CGF.getLLVMContext(),
          CGF.CGM.getDataLayout().getIndexTypeSizeInBits(SrcType));
    } else {
      assert(SrcType->isIntegerTy());
      IntType = cast<llvm::IntegerType>(SrcType);
    }
    Alignment = CGF.EmitScalarExpr(E->getArg(1));
    Alignment = CGF.Builder.CreateZExtOrTrunc(Alignment, IntType, "alignment");
    auto *One = llvm::ConstantInt::get(IntType, 1);
    Mask = CGF.Builder.CreateSub(Alignment, One, "mask");
  }
};
} // namespace

/// Generate (x & (y-1)) == 0.
RValue CodeGenFunction::EmitBuiltinIsAligned(const CallExpr *E) {
  BuiltinAlignArgs Args(E, *this);
  llvm::Value *SrcAddress = Args.Src;
  if (Args.SrcType->isPointerTy())
    SrcAddress =
        Builder.CreateBitOrPointerCast(Args.Src, Args.IntType, "src_addr");
  return RValue::get(Builder.CreateICmpEQ(
      Builder.CreateAnd(SrcAddress, Args.Mask, "set_bits"),
      llvm::Constant::getNullValue(Args.IntType), "is_aligned"));
}

/// Generate (x & ~(y-1)) to align down or ((x+(y-1)) & ~(y-1)) to align up.
/// Note: For pointer types we can avoid ptrtoint/inttoptr pairs by using the
/// llvm.ptrmask intrinsic (with a GEP before in the align_up case).
RValue CodeGenFunction::EmitBuiltinAlignTo(const CallExpr *E, bool AlignUp) {
  BuiltinAlignArgs Args(E, *this);
  llvm::Value *SrcForMask = Args.Src;
  if (AlignUp) {
    // When aligning up we have to first add the mask to ensure we go over the
    // next alignment value and then align down to the next valid multiple.
    // By adding the mask, we ensure that align_up on an already aligned
    // value will not change the value.
    if (Args.Src->getType()->isPointerTy()) {
      if (getLangOpts().PointerOverflowDefined)
        SrcForMask =
            Builder.CreateGEP(Int8Ty, SrcForMask, Args.Mask, "over_boundary");
      else
        SrcForMask = EmitCheckedInBoundsGEP(Int8Ty, SrcForMask, Args.Mask,
                                            /*SignedIndices=*/true,
                                            /*isSubtraction=*/false,
                                            E->getExprLoc(), "over_boundary");
    } else {
      SrcForMask = Builder.CreateAdd(SrcForMask, Args.Mask, "over_boundary");
    }
  }
  // Invert the mask to only clear the lower bits.
  llvm::Value *InvertedMask = Builder.CreateNot(Args.Mask, "inverted_mask");
  llvm::Value *Result = nullptr;
  if (Args.Src->getType()->isPointerTy()) {
    Result = Builder.CreateIntrinsic(
        Intrinsic::ptrmask, {Args.SrcType, Args.IntType},
        {SrcForMask, InvertedMask}, nullptr, "aligned_result");
  } else {
    Result = Builder.CreateAnd(SrcForMask, InvertedMask, "aligned_result");
  }
  assert(Result->getType() == Args.SrcType);
  return RValue::get(Result);
}
