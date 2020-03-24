//===------ CGGPUBuiltin.cpp - Codegen for GPU builtins -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generates code for built-in GPU calls which are not runtime-specific.
// (Runtime-specific codegen lives in programming model specific files.)
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "clang/Basic/Builtins.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/Utils/AMDGPUEmitPrintf.h"

using namespace clang;
using namespace CodeGen;

static llvm::Function *GetVprintfDeclaration(llvm::Module &M) {
  llvm::Type *ArgTypes[] = {llvm::Type::getInt8PtrTy(M.getContext()),
                            llvm::Type::getInt8PtrTy(M.getContext())};
  llvm::FunctionType *VprintfFuncType = llvm::FunctionType::get(
      llvm::Type::getInt32Ty(M.getContext()), ArgTypes, false);

  if (auto* F = M.getFunction("vprintf")) {
    // Our CUDA system header declares vprintf with the right signature, so
    // nobody else should have been able to declare vprintf with a bogus
    // signature.
    assert(F->getFunctionType() == VprintfFuncType);
    return F;
  }

  // vprintf doesn't already exist; create a declaration and insert it into the
  // module.
  return llvm::Function::Create(
      VprintfFuncType, llvm::GlobalVariable::ExternalLinkage, "vprintf", &M);
}

// Transforms a call to printf into a call to the NVPTX vprintf syscall (which
// isn't particularly special; it's invoked just like a regular function).
// vprintf takes two args: A format string, and a pointer to a buffer containing
// the varargs.
//
// For example, the call
//
//   printf("format string", arg1, arg2, arg3);
//
// is converted into something resembling
//
//   struct Tmp {
//     Arg1 a1;
//     Arg2 a2;
//     Arg3 a3;
//   };
//   char* buf = alloca(sizeof(Tmp));
//   *(Tmp*)buf = {a1, a2, a3};
//   vprintf("format string", buf);
//
// buf is aligned to the max of {alignof(Arg1), ...}.  Furthermore, each of the
// args is itself aligned to its preferred alignment.
//
// Note that by the time this function runs, E's args have already undergone the
// standard C vararg promotion (short -> int, float -> double, etc.).
RValue
CodeGenFunction::EmitNVPTXDevicePrintfCallExpr(const CallExpr *E,
                                               ReturnValueSlot ReturnValue) {
  assert(getTarget().getTriple().isNVPTX());
  assert(E->getBuiltinCallee() == Builtin::BIprintf);
  assert(E->getNumArgs() >= 1); // printf always has at least one arg.

  const llvm::DataLayout &DL = CGM.getDataLayout();
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();

  CallArgList Args;
  EmitCallArgs(Args,
               E->getDirectCallee()->getType()->getAs<FunctionProtoType>(),
               E->arguments(), E->getDirectCallee(),
               /* ParamsToSkip = */ 0);

  // We don't know how to emit non-scalar varargs.
  if (std::any_of(Args.begin() + 1, Args.end(), [&](const CallArg &A) {
        return !A.getRValue(*this).isScalar();
      })) {
    CGM.ErrorUnsupported(E, "non-scalar arg to printf");
    return RValue::get(llvm::ConstantInt::get(IntTy, 0));
  }

  // Construct and fill the args buffer that we'll pass to vprintf.
  llvm::Value *BufferPtr;
  if (Args.size() <= 1) {
    // If there are no args, pass a null pointer to vprintf.
    BufferPtr = llvm::ConstantPointerNull::get(llvm::Type::getInt8PtrTy(Ctx));
  } else {
    llvm::SmallVector<llvm::Type *, 8> ArgTypes;
    for (unsigned I = 1, NumArgs = Args.size(); I < NumArgs; ++I)
      ArgTypes.push_back(Args[I].getRValue(*this).getScalarVal()->getType());

    // Using llvm::StructType is correct only because printf doesn't accept
    // aggregates.  If we had to handle aggregates here, we'd have to manually
    // compute the offsets within the alloca -- we wouldn't be able to assume
    // that the alignment of the llvm type was the same as the alignment of the
    // clang type.
    llvm::Type *AllocaTy = llvm::StructType::create(ArgTypes, "printf_args");
    llvm::Value *Alloca = CreateTempAlloca(AllocaTy);

    for (unsigned I = 1, NumArgs = Args.size(); I < NumArgs; ++I) {
      llvm::Value *P = Builder.CreateStructGEP(AllocaTy, Alloca, I - 1);
      llvm::Value *Arg = Args[I].getRValue(*this).getScalarVal();
      Builder.CreateAlignedStore(Arg, P, DL.getPrefTypeAlign(Arg->getType()));
    }
    BufferPtr = Builder.CreatePointerCast(Alloca, llvm::Type::getInt8PtrTy(Ctx));
  }

  // Invoke vprintf and return.
  llvm::Function* VprintfFunc = GetVprintfDeclaration(CGM.getModule());
  return RValue::get(Builder.CreateCall(
      VprintfFunc, {Args[0].getRValue(*this).getScalarVal(), BufferPtr}));
}

RValue
CodeGenFunction::EmitAMDGPUDevicePrintfCallExpr(const CallExpr *E,
                                                ReturnValueSlot ReturnValue) {
  assert(getTarget().getTriple().isAMDGCN());
  assert(E->getBuiltinCallee() == Builtin::BIprintf ||
         E->getBuiltinCallee() == Builtin::BI__builtin_printf);
  assert(E->getNumArgs() >= 1); // printf always has at least one arg.

  CallArgList CallArgs;
  EmitCallArgs(CallArgs,
               E->getDirectCallee()->getType()->getAs<FunctionProtoType>(),
               E->arguments(), E->getDirectCallee(),
               /* ParamsToSkip = */ 0);

  SmallVector<llvm::Value *, 8> Args;
  for (auto A : CallArgs) {
    // We don't know how to emit non-scalar varargs.
    if (!A.getRValue(*this).isScalar()) {
      CGM.ErrorUnsupported(E, "non-scalar arg to printf");
      return RValue::get(llvm::ConstantInt::get(IntTy, -1));
    }

    llvm::Value *Arg = A.getRValue(*this).getScalarVal();
    Args.push_back(Arg);
  }

  llvm::IRBuilder<> IRB(Builder.GetInsertBlock(), Builder.GetInsertPoint());
  IRB.SetCurrentDebugLocation(Builder.getCurrentDebugLocation());
  auto Printf = llvm::emitAMDGPUPrintfCall(IRB, Args);
  Builder.SetInsertPoint(IRB.GetInsertBlock(), IRB.GetInsertPoint());
  return RValue::get(Printf);
}

// For printf in OpenMP on amdgcn, we build a struct of numerics where string
// pointers are converted to their lengths and then all the strings are
// written after the struct. We write the length of the numerics struct
// in the first eelement of the struct. For example
//
// printf("format string %d %s %ld\n", arg1, "string2", arg3);
// is converted into
//
//  { struct Tmp {24; 25; int arg1; 7; long arg3} ,
//    "format string %d %s %ld\n",
//    "string2"
//  }
// Return value from printf_alloc is a thread-specific pointer to global
// memory that will contain all the data values and all strings, INCLUDING
// the format string. The first data value is the length of the data values.
// This is followed by the data values themselves.  The data value for a
// string is the LENGTH of the string, NOT the string itself.  All the strings
// including the format string are stored consecutively at the end of all the
// data values. Since the first value is the combined size of all the data
// values, the second data value is the length of the format string because
// the format string is always the first argument of a printf.
// The host routine to parse this buffer, must find the format string by
// using the length of the data values in the 1st field and jumping that many
// bytes to the end of the data values. If the format string has any
// args, their values will start in the 3rd field. If the host parser
// encounters a string (%s) in the format string, it must get this string
// value by going to the next string value after the format string.
// It does this by using the length of the format string. Subsequent strings
// will be found by using previous string lengths.

static llvm::Function *GetOmpPrintfAllocDeclaration(CodeGenModule &CGM) {
  auto &M = CGM.getModule();
  llvm::Type *ArgTypes[] = {CGM.Int32Ty};
  llvm::FunctionType *OmpPrintfAllocFuncType = llvm::FunctionType::get(
      llvm::PointerType::getUnqual(CGM.Int8Ty), ArgTypes, false);
  if (auto *F = M.getFunction("printf_alloc")) {
    assert(F->getFunctionType() == OmpPrintfAllocFuncType);
    return F;
  }
  llvm::Function *FN = llvm::Function::Create(
      OmpPrintfAllocFuncType, llvm::GlobalVariable::ExternalLinkage,
      "printf_alloc", &M);
  return FN;
}
static llvm::Function *GetOmpPrintfExecuteDeclaration(CodeGenModule &CGM) {
  auto &M = CGM.getModule();
  llvm::Type *ArgTypes[] = {llvm::PointerType::getUnqual(CGM.Int8Ty),
                            CGM.Int32Ty};
  llvm::FunctionType *OmpPrintfExecuteFuncType =
      llvm::FunctionType::get(CGM.Int32Ty, ArgTypes, false);
  if (auto *F = M.getFunction("printf_execute")) {
    assert(F->getFunctionType() == OmpPrintfExecuteFuncType);
    return F;
  }
  llvm::Function *FN = llvm::Function::Create(
      OmpPrintfExecuteFuncType, llvm::GlobalVariable::ExternalLinkage,
      "printf_execute", &M);
  return FN;
}

static llvm::Function *GetOmpStrlenDeclaration(CodeGenModule &CGM) {
  auto &M = CGM.getModule();
  // Args are pointer to char and maxstringlen
  llvm::Type *ArgTypes[] = {CGM.Int8PtrTy, CGM.Int32Ty};
  llvm::FunctionType *OmpStrlenFTy =
      llvm::FunctionType::get(CGM.Int32Ty, ArgTypes, false);
  if (auto *F = M.getFunction("__strlen_max")) {
    assert(F->getFunctionType() == OmpStrlenFTy);
    return F;
  }
  llvm::Function *FN = llvm::Function::Create(
      OmpStrlenFTy, llvm::GlobalVariable::ExternalLinkage, "__strlen_max", &M);
  return FN;
}

static bool isVarString(const clang::Expr *argX, const clang::Type *argXTy,
                        const llvm::Value *Arg) {
  if ((argXTy->isPointerType() || argXTy->isConstantArrayType()) &&
      argXTy->getPointeeOrArrayElementType()->isCharType() && !argX->isLValue())
    return true;
  // Ensure the VarDecl has an inititalizer
  if (const auto *DRE = dyn_cast<DeclRefExpr>(argX))
    if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()))
      if (!VD->getInit())
        return true;
  return false;
}

static bool isString(const clang::Type *argXTy) {
  if ((argXTy->isPointerType() || argXTy->isConstantArrayType()) &&
      argXTy->getPointeeOrArrayElementType()->isCharType())
    return true;
  else
    return false;
}

static const StringLiteral *getSL(const clang::Expr *argX,
                                  const clang::Type *argXTy) {
  // String in argX has known constant length
  if (!argXTy->isConstantArrayType()) {
    // Allow constant string to be a declared variable,
    // But it must be constant and initialized.
    const DeclRefExpr *DRE = cast<DeclRefExpr>(argX);
    const VarDecl *VarD = cast<VarDecl>(DRE->getDecl());
    argX = VarD->getInit()->IgnoreImplicit();
  }
  const StringLiteral *SL = cast<StringLiteral>(argX);
  return SL;
}

RValue CodeGenFunction::EmitAMDGPUDevicePrintfCallExprOMP(
    const CallExpr *E, ReturnValueSlot ReturnValue) {
  assert(getTarget().getTriple().isAMDGCN());
  assert(E->getBuiltinCallee() == Builtin::BIprintf);
  assert(E->getNumArgs() >= 1); // printf always has at least one arg.

  const llvm::DataLayout &DL = CGM.getDataLayout();

  CallArgList Args;
  EmitCallArgs(Args,
               E->getDirectCallee()->getType()->getAs<FunctionProtoType>(),
               E->arguments(), E->getDirectCallee(),
               /* ParamsToSkip = */ 0);

  // We don't know how to emit non-scalar varargs.
  if (std::any_of(Args.begin() + 1, Args.end(), [&](const CallArg &A) {
        return !A.getRValue(*this).isScalar();
      })) {
    CGM.ErrorUnsupported(E, "non-scalar arg to printf");
    return RValue::get(llvm::ConstantInt::get(IntTy, 0));
  }

  // ---  1st Pass over Args to create ArgTypes and count size ---

  llvm::SmallVector<llvm::Type *, 16> ArgTypes;
  llvm::SmallVector<llvm::Value *, 16> VarStrLengths;
  llvm::Value *TotalVarStrsLength = llvm::ConstantInt::get(Int32Ty, 0);
  int AllStringsLen_CT = 0;
  int DataLen_CT = (int)DL.getTypeAllocSize(Int32Ty);
  bool hasVarStrings = false;
  ArgTypes.push_back(Int32Ty); // First field in struct will be total DataLen
  for (unsigned I = 0, NumArgs = Args.size(); I < NumArgs; ++I) {
    llvm::Value *Arg = Args[I].getRValue(*this).getScalarVal();
    llvm::Type *ArgType = Arg->getType();
    const Expr *argX = E->getArg(I)->IgnoreParenCasts();
    auto *argXTy = argX->getType().getTypePtr();
    if (isString(argXTy)) {
      if (isVarString(argX, argXTy, Arg)) {
        hasVarStrings = true;
        if (auto *PtrTy = dyn_cast<llvm::PointerType>(ArgType))
          if (PtrTy->getPointerAddressSpace()) {
            Arg = Builder.CreateAddrSpaceCast(Arg, CGM.Int8PtrTy);
            ArgType = Arg->getType();
          }
        llvm::Value *VarStrLen =
            Builder.CreateCall(GetOmpStrlenDeclaration(CGM),
                               {Arg, llvm::ConstantInt::get(Int32Ty, 1024)});
        VarStrLengths.push_back(VarStrLen);
        TotalVarStrsLength = Builder.CreateAdd(TotalVarStrsLength, VarStrLen,
                                               "sum_of_var_strings_length");
        ArgType = Int32Ty;
      } else {
        const StringLiteral *SL = getSL(argX, argXTy);
        StringRef ArgString = SL->getString();
        AllStringsLen_CT += ((int)ArgString.size() + 1);
        // change ArgType from char ptr to int to contain string length
        ArgType = Int32Ty;
      }
    }
    DataLen_CT += (int)DL.getTypeAllocSize(ArgType);
    ArgTypes.push_back(ArgType);
  }

  // ---  Generate call to printf_alloc to get pointer to data structure  ---

  if (hasVarStrings)
    TotalVarStrsLength = Builder.CreateAdd(
        TotalVarStrsLength,
        llvm::ConstantInt::get(Int32Ty, AllStringsLen_CT + DataLen_CT,
                               "const_length_adder"),
        "total_buffer_size");

  llvm::Value *BufferLen =
      hasVarStrings
          ? TotalVarStrsLength
          : llvm::ConstantInt::get(Int32Ty, AllStringsLen_CT + DataLen_CT);

  llvm::Value *DataStructPtr =
      Builder.CreateCall(GetOmpPrintfAllocDeclaration(CGM), {BufferLen});

  // cast the generic return pointer to be a struct in device global memory
  llvm::StructType *DataStructTy =
      llvm::StructType::create(ArgTypes, "printf_args");
  unsigned AS = getContext().getTargetAddressSpace(LangAS::cuda_device);
  llvm::Value *BufferPtr = Builder.CreatePointerCast(
      DataStructPtr, llvm::PointerType::get(DataStructTy, AS),
      "printf_args_casted");

  // ---  2nd Pass: Store thread-specfic data values to global memory buffer ---

  // Start with length of data structure which is not a user arg
  llvm::Value *DataLen = llvm::ConstantInt::get(Int32Ty, DataLen_CT);
  llvm::Value *P = Builder.CreateStructGEP(DataStructTy, BufferPtr, 0);
  Builder.CreateAlignedStore(DataLen, P,
                             DL.getPrefTypeAlignment(DataLen->getType()));
  unsigned varstring_index = 0;
  for (unsigned I = 0, NumArgs = Args.size(); I < NumArgs; ++I) {
    llvm::Value *Arg;
    const Expr *argX = E->getArg(I)->IgnoreParenCasts();
    auto *argXTy = argX->getType().getTypePtr();
    if (isString(argXTy)) {
      if (isVarString(argX, argXTy, Arg)) {
        Arg = VarStrLengths[varstring_index];
        varstring_index++;
      } else {
        const StringLiteral *SL = getSL(argX, argXTy);
        StringRef ArgString = SL->getString();
        int ArgStrLen = (int)ArgString.size() + 1;
        // Change Arg from a char pointer to the integer string length
        Arg = llvm::ConstantInt::get(Int32Ty, ArgStrLen);
      }
    } else
      Arg = Args[I].getKnownRValue().getScalarVal();
    P = Builder.CreateStructGEP(DataStructTy, BufferPtr, I + 1);
    Builder.CreateAlignedStore(Arg, P, DL.getPrefTypeAlignment(Arg->getType()));
  }

  // ---  3rd Pass: memcpy all strings after the data values ---

  // bitcast the struct in device global memory as a char buffer
  Address BufferPtrByteAddr = Address(
      Builder.CreatePointerCast(BufferPtr, llvm::PointerType::get(Int8Ty, AS)),
      CharUnits::fromQuantity(1));
  // BufferPtrByteAddr is a pointer to where we want to write the next string
  BufferPtrByteAddr = Builder.CreateConstInBoundsByteGEP(
      BufferPtrByteAddr, CharUnits::fromQuantity(DataLen_CT));
  varstring_index = 0;
  for (unsigned I = 0, NumArgs = Args.size(); I < NumArgs; ++I) {
    llvm::Value *Arg = Args[I].getKnownRValue().getScalarVal();
    const Expr *argX = E->getArg(I)->IgnoreParenCasts();
    auto *argXTy = argX->getType().getTypePtr();
    if (isString(argXTy)) {
      if (isVarString(argX, argXTy, Arg)) {
        llvm::Value *varStrLength = VarStrLengths[varstring_index];
        varstring_index++;
        Address SrcAddr = Address(Arg, CharUnits::fromQuantity(1));
        Builder.CreateMemCpy(BufferPtrByteAddr, SrcAddr, varStrLength);
        // update BufferPtrByteAddr for next string memcpy
        llvm::Type *OrigTy = BufferPtrByteAddr.getType();
        llvm::Value *PtrAsInt = BufferPtrByteAddr.getPointer();
        PtrAsInt = Builder.CreatePtrToInt(PtrAsInt, IntPtrTy);
        auto *bigint =
            Builder.CreateZExt(varStrLength, PtrAsInt->getType(), "PtyAdder");
        PtrAsInt = Builder.CreateAdd(PtrAsInt, bigint);
        // Instead of recreating a new Address here, can we just set
        // it's pointer to Builder.CreateIntToPtr(PtrAsInt, OrigTy) ???
        BufferPtrByteAddr = Address(
            Builder.CreatePointerCast(Builder.CreateIntToPtr(PtrAsInt, OrigTy),
                                      llvm::PointerType::get(Int8Ty, AS)),
            CharUnits::fromQuantity(1));
      } else {
        const StringLiteral *SL = getSL(argX, argXTy);
        StringRef ArgString = SL->getString();
        int ArgStrLen = (int)ArgString.size() + 1;
        Address SrcAddr = CGM.GetAddrOfConstantStringFromLiteral(SL);
        Builder.CreateMemCpy(BufferPtrByteAddr, SrcAddr, ArgStrLen);
        // update BufferPtrByteAddr for next memcpy
        BufferPtrByteAddr = Builder.CreateConstInBoundsByteGEP(
            BufferPtrByteAddr, CharUnits::fromQuantity(ArgStrLen));
      }
    }
  }
  return RValue::get(Builder.CreateCall(GetOmpPrintfExecuteDeclaration(CGM),
                                        {DataStructPtr, BufferLen}));
}
