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

namespace {
llvm::Value *packArgsIntoNVPTXFormatBuffer(CodeGenFunction *CGF,
                                           const CallArgList &Args) {
  const llvm::DataLayout &DL = CGF->CGM.getDataLayout();
  llvm::LLVMContext &Ctx = CGF->CGM.getLLVMContext();
  CGBuilderTy &Builder = CGF->Builder;

  // Construct and fill the args buffer that we'll pass to vprintf.
  if (Args.size() <= 1) {
    // If there are no args, pass a null pointer to vprintf.
    return llvm::ConstantPointerNull::get(llvm::Type::getInt8PtrTy(Ctx));
  } else {
    llvm::SmallVector<llvm::Type *, 8> ArgTypes;
    for (unsigned I = 1, NumArgs = Args.size(); I < NumArgs; ++I)
      ArgTypes.push_back(Args[I].getRValue(*CGF).getScalarVal()->getType());

    // Using llvm::StructType is correct only because printf doesn't accept
    // aggregates.  If we had to handle aggregates here, we'd have to manually
    // compute the offsets within the alloca -- we wouldn't be able to assume
    // that the alignment of the llvm type was the same as the alignment of the
    // clang type.
    llvm::Type *AllocaTy = llvm::StructType::create(ArgTypes, "printf_args");
    llvm::Value *Alloca = CGF->CreateTempAlloca(AllocaTy);

    for (unsigned I = 1, NumArgs = Args.size(); I < NumArgs; ++I) {
      llvm::Value *P = Builder.CreateStructGEP(AllocaTy, Alloca, I - 1);
      llvm::Value *Arg = Args[I].getRValue(*CGF).getScalarVal();
      Builder.CreateAlignedStore(Arg, P, DL.getPrefTypeAlign(Arg->getType()));
    }
    return Builder.CreatePointerCast(Alloca, llvm::Type::getInt8PtrTy(Ctx));
  }
}
} // namespace

RValue
CodeGenFunction::EmitNVPTXDevicePrintfCallExpr(const CallExpr *E,
                                               ReturnValueSlot ReturnValue) {
  assert(getTarget().getTriple().isNVPTX());
  assert(E->getBuiltinCallee() == Builtin::BIprintf);
  assert(E->getNumArgs() >= 1); // printf always has at least one arg.

  CallArgList Args;
  EmitCallArgs(Args,
               E->getDirectCallee()->getType()->getAs<FunctionProtoType>(),
               E->arguments(), E->getDirectCallee(),
               /* ParamsToSkip = */ 0);

  // We don't know how to emit non-scalar varargs.
  if (llvm::any_of(llvm::drop_begin(Args), [&](const CallArg &A) {
        return !A.getRValue(*this).isScalar();
      })) {
    CGM.ErrorUnsupported(E, "non-scalar arg to printf");
    return RValue::get(llvm::ConstantInt::get(IntTy, 0));
  }

  llvm::Value *BufferPtr = packArgsIntoNVPTXFormatBuffer(this, Args);

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

// EmitHostrpcVargsFn:
//
// For printf in an OpenMP Target region on amdgn and for variable argument
// functions that have a supporting host service function (hostrpc) a struct
// is created to represent the vargs for each call site.
// The struct contains the length, number of args, an array of 4-byte keys
// that represent the type of of each arg, an array of aligned "data" values
// for each arg, and finally the runtime string values. If an arg is a string
// the data value is the runtime length of the string.  Each 4-byte key
// contains the llvm type ID and the number of bits for the type.
// encoded by the macro PACK_TY_BITLEN(x,y) ((uint32_t)x << 16) | ((uint32_t)y)
// The llvm type ID of a string is pointer. To distinguish string pointers
// from non-string pointers, the number of bitlen is set to 1.
//
// For example, here is a 4 arg printf function
//
// printf("format string %d %s %f \n", (int) 1, "string2", (double) 1.234);
//
// is represented by a struct with these 13 elements.
//
//  {81, 4, 983041, 720928, 983041, 196672, 25, int 1, 7, 0, double 1.234,
//     "format string %d %s %ld\n", "string2" }
//
// 81 is the total length of the buffer that must be allocated.
// 4 is the number of arguments.
// The next 4 key values represent the data types of the 4 args.
// The format string length is 25.
// The integer field is next.
// The string argument "string2" has length 7
// The 4-byte dummy arg 0 is inserted so the next double arg is aligned.
// The string arguments follows the header, keys, and data args.
//
// Before the struct is written, a hostrpc call is is emitted  to allocate
// memory for the transfer. Then the struct is emitted.  Then a call
// to the execute the GPU stub function that initiates the service
// on the host.  The host runtime passes the buffer to the service routine
// for processing.

// These static helper functions support EmitHostrpcVargsFn.

// For strings that vary in length at runtime this strlen_max
// will stop at a provided maximum.
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

// Deterimines if an expression is a string with variable lenth
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

// Deterimines if an argument is a string
static bool isString(const clang::Type *argXTy) {
  if ((argXTy->isPointerType() || argXTy->isConstantArrayType()) &&
      argXTy->getPointeeOrArrayElementType()->isCharType())
    return true;
  else
    return false;
}

// Gets a string literal to write into the transfer buffer
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

// Returns a function pointer to the memory allocation routine
static llvm::Function *GetVargsFnAllocDeclaration(CodeGenModule &CGM,
                                                  const char *GPUAllocateName) {
  auto &M = CGM.getModule();
  llvm::Type *ArgTypes[] = {CGM.Int32Ty};
  llvm::Function *FN;
  llvm::FunctionType *VargsFnAllocFuncType = llvm::FunctionType::get(
      llvm::PointerType::getUnqual(CGM.Int8Ty), ArgTypes, false);

  if (!(FN = M.getFunction(GPUAllocateName)))
    FN = llvm::Function::Create(VargsFnAllocFuncType,
                                llvm::GlobalVariable::ExternalLinkage,
                                GPUAllocateName, &M);
  assert(FN->getFunctionType() == VargsFnAllocFuncType);
  return FN;
}

// Returns a function pointer to the GPU stub function
static llvm::Function *
hostrpcVargsReturnsFnDeclaration(CodeGenModule &CGM, QualType Ty,
                                 const char *GPUStubFunctionName) {
  auto &M = CGM.getModule();
  llvm::Type *ArgTypes[] = {llvm::PointerType::getUnqual(CGM.Int8Ty),
                            CGM.Int32Ty};
  llvm::Function *FN;
  llvm::FunctionType *VarfnFuncType =
      llvm::FunctionType::get(CGM.getTypes().ConvertType(Ty), ArgTypes, false);
  if (!(FN = M.getFunction(GPUStubFunctionName)))
    FN = llvm::Function::Create(VarfnFuncType,
                                llvm::GlobalVariable::ExternalLinkage,
                                GPUStubFunctionName, &M);
  assert(FN->getFunctionType() == VarfnFuncType);
  return FN;
}

// The macro to pack the llvm type ID and numbits into 4-byte key
#define PACK_TY_BITLEN(x, y) ((uint32_t)x << 16) | ((uint32_t)y)

// Emit the code to support a host vargs function such as printf.
RValue CodeGenFunction::EmitHostrpcVargsFn(const CallExpr *E,
                                           const char *GPUAllocateName,
                                           const char *GPUStubFunctionName,
                                           ReturnValueSlot ReturnValue) {
  assert(getTarget().getTriple().isAMDGCN());
  // assert(E->getBuiltinCallee() == Builtin::BIprintf);
  assert(E->getNumArgs() >= 1); // rpc varfn always has at least one arg.

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
    CGM.ErrorUnsupported(E, "non-scalar arg in GPU vargs function");
    return RValue::get(llvm::ConstantInt::get(IntTy, 0));
  }

  unsigned NumArgs = (unsigned)Args.size();
  llvm::SmallVector<llvm::Type *, 32> ArgTypes;
  llvm::SmallVector<llvm::Value *, 32> VarStrLengths;
  llvm::Value *TotalVarStrsLength = llvm::ConstantInt::get(Int32Ty, 0);
  bool hasVarStrings = false;
  ArgTypes.push_back(Int32Ty); // First field in struct will be total DataLen
  ArgTypes.push_back(Int32Ty); // 2nd field in struct will be num args
  // An array of 4-byte keys that describe the arg type
  for (unsigned I = 0; I < NumArgs; ++I)
    ArgTypes.push_back(Int32Ty);

  // Track the size of the numeric data length and string length
  unsigned DataLen_CT =
      (unsigned)(DL.getTypeAllocSize(Int32Ty)) * (NumArgs + 2);
  unsigned AllStringsLen_CT = 0;

  // ---  1st Pass over Args to create ArgTypes and count size ---

  size_t structOffset = 4 * (NumArgs + 2);
  for (unsigned I = 0; I < NumArgs; I++) {
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
    } // end of processing string argument
    // if ArgTypeSize is >4 bytes we need to insert dummy align
    // values in the struct so all stores can be aligned .
    // These dummy fields must be inserted before the arg.
    //
    // In the pass below where the stores are generated careful
    // tracking of the index into the struct is necessary.
    size_t needsPadding = (structOffset % (size_t)DL.getTypeAllocSize(ArgType));
    if (needsPadding) {
      DataLen_CT += (unsigned)needsPadding;
      structOffset += needsPadding;
      ArgTypes.push_back(Int32Ty); // should assert that needsPadding == 4 here
    }

    ArgTypes.push_back(ArgType);
    DataLen_CT += ((int)DL.getTypeAllocSize(ArgType));
    structOffset += (size_t)DL.getTypeAllocSize(ArgType);
  }

  // ---  Generate call to printf_alloc to get pointer to data structure  ---
  if (hasVarStrings)
    TotalVarStrsLength = Builder.CreateAdd(
        TotalVarStrsLength,
        llvm::ConstantInt::get(Int32Ty, AllStringsLen_CT + DataLen_CT),
        "total_buffer_size");
  llvm::Value *BufferLen =
      hasVarStrings
          ? TotalVarStrsLength
          : llvm::ConstantInt::get(Int32Ty, AllStringsLen_CT + DataLen_CT);

  llvm::Value *DataStructPtr = Builder.CreateCall(
      GetVargsFnAllocDeclaration(CGM, GPUAllocateName), {BufferLen});

  // cast the generic return pointer to be a struct in device global memory
  llvm::StructType *DataStructTy =
      llvm::StructType::create(ArgTypes, "varfn_args_store");
  unsigned AS = getContext().getTargetAddressSpace(LangAS::cuda_device);
  llvm::Value *BufferPtr = Builder.CreatePointerCast(
      DataStructPtr, llvm::PointerType::get(DataStructTy, AS),
      "varfn_args_store_casted");

  // ---  Header of struct contains length and NumArgs ---
  llvm::Value *DataLenField = llvm::ConstantInt::get(Int32Ty, DataLen_CT);
  llvm::Value *P = Builder.CreateStructGEP(DataStructTy, BufferPtr, 0);
  Builder.CreateAlignedStore(
      DataLenField, P, DL.getPrefTypeAlign(DataLenField->getType()));
  llvm::Value *NumArgsField = llvm::ConstantInt::get(Int32Ty, NumArgs);
  P = Builder.CreateStructGEP(DataStructTy, BufferPtr, 1);
  Builder.CreateAlignedStore(
      NumArgsField, P, DL.getPrefTypeAlign(NumArgsField->getType()));

  // ---  2nd Pass: create array of 4-byte keys to describe each arg

  for (unsigned I = 0; I < NumArgs; I++) {
    llvm::Type *ty = Args[I].getRValue(*this).getScalarVal()->getType();
    llvm::Type::TypeID argtypeid =
        Args[I].getRValue(*this).getScalarVal()->getType()->getTypeID();

    // Get type size in bits. Usually 64 or 32.
    uint32_t numbits = 0;
    if (isString(E->getArg(I)->IgnoreParenCasts()->getType().getTypePtr()))
      // The llvm typeID for string is pointer.  Since pointer numbits is 0,
      // we set numbits to 1 to distinguish pointer type ID as string pointer.
      numbits = 1;
    else
      numbits = ty->getScalarSizeInBits();
    // Create a key that combines llvm typeID and size
    llvm::Value *Key =
        llvm::ConstantInt::get(Int32Ty, PACK_TY_BITLEN(argtypeid, numbits));
    P = Builder.CreateStructGEP(DataStructTy, BufferPtr, I + 2);
    Builder.CreateAlignedStore(Key, P, DL.getPrefTypeAlign(Key->getType()));
  }

  // ---  3rd Pass: Store thread-specfic data values for each arg ---

  unsigned varstring_index = 0;
  unsigned structIndex = 2 + NumArgs;
  structOffset = 4 * structIndex;
  for (unsigned I = 0; I < NumArgs; I++) {
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
    } else {
      Arg = Args[I].getKnownRValue().getScalarVal();
    }
    size_t structElementSize = (size_t)DL.getTypeAllocSize(Arg->getType());
    size_t needsPadding = (structOffset % structElementSize);
    if (needsPadding) {
      // Skip over dummy fields in struct to align
      structOffset += needsPadding; // should assert needsPadding == 4
      structIndex++;
    }
    P = Builder.CreateStructGEP(DataStructTy, BufferPtr, structIndex);
    Builder.CreateAlignedStore(Arg, P, DL.getPrefTypeAlign(Arg->getType()));
    structOffset += structElementSize;
    structIndex++;
  }

  // ---  4th Pass: memcpy all strings after the data values ---

  // bitcast the struct in device global memory as a char buffer
  Address BufferPtrByteAddr = Address(
      Builder.CreatePointerCast(BufferPtr, llvm::PointerType::get(Int8Ty, AS)),
      CharUnits::fromQuantity(1));
  // BufferPtrByteAddr is a pointer to where we want to write the next string
  BufferPtrByteAddr = Builder.CreateConstInBoundsByteGEP(
      BufferPtrByteAddr, CharUnits::fromQuantity(DataLen_CT));
  varstring_index = 0;
  for (unsigned I = 0; I < NumArgs; ++I) {
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
        llvm::Value *PtrAsInt = BufferPtrByteAddr.getPointer();
        BufferPtrByteAddr = Address(
            Builder.CreateGEP(PtrAsInt->getType()->getScalarType()->getPointerElementType(),
		    PtrAsInt, ArrayRef<llvm::Value*>(varStrLength)),
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
  return RValue::get(Builder.CreateCall(
      hostrpcVargsReturnsFnDeclaration(CGM, E->getType(), GPUStubFunctionName),
      {DataStructPtr, BufferLen}));
}
