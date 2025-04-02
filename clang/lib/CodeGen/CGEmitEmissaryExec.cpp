//===------- CGEmitEmissaryExec.cpp - Codegen for _emissary_exec --==------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emits device code for an encountered call to vargs functions _emissary_exec
// The emitted code has three parts:
// 1  call __llvm_omp_emissary_prealloc for memory buffer to contain all args
// 2. Store each arg into the buffer.
// 3. call to __llvm_omp_emissary_rpc function.
//===----------------------------------------------------------------------===//

#include "../../offload/DeviceRTL/include/EmissaryIds.h"
#include "CodeGenFunction.h"
#include "clang/Basic/Builtins.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/Utils/AMDGPUEmitPrintf.h"

using namespace clang;
using namespace CodeGen;

// EmitEmissaryExec:
//
// When a device call to the varadic function _emissary_exec is encountered
// (in CGExpr.cpp) EmitEmissaryExec does these steps:
//
// 1. If string lens are runtime dependent, Emit code to determine runtime len.
// 2. Emits call to allocate memory __llvm_omp_emissary_premalloc,
// 3. Emit stores of each arg into arg buffer,
// 4. Emits call to function __llvm_omp_emissary_rpc.
//
// The arg buffer is a struct that contains the length, number of args, an
// array of 4-byte keys that represent the type of of each arg, an array of
// aligned "data" values for each arg, and finally the runtime string values.
// If an arg is a string the data value is the runtime length of the string.
// Each 4-byte key contains the llvm type ID and the number of bits for the
// type. encoded by the macro _PACK_TY_BITLEN(x,y) ((uint32_t)x << 16) |
// ((uint32_t)y)
//
// TODO: Add example of call to _emissary_exec() and the corresponding struct

// These static helper functions support EmitEmissaryExec.
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
      if (!VD->getInit() ||
          !llvm::isa<StringLiteral>(VD->getInit()->IgnoreImplicit()))
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

// Returns a function pointer to __llvm_omp_emissary_premalloc
static llvm::Function *GetEmissaryAllocDeclaration(CodeGenModule &CGM) {
  auto &M = CGM.getModule();
  const char *_executeName = "__llvm_omp_emissary_premalloc";
  llvm::Type *ArgTypes[] = {CGM.Int32Ty};
  llvm::Function *FN;
  llvm::FunctionType *VargsFnAllocFuncType = llvm::FunctionType::get(
      llvm::PointerType::getUnqual(CGM.Int8Ty), ArgTypes, false);

  if (!(FN = M.getFunction(_executeName)))
    FN = llvm::Function::Create(VargsFnAllocFuncType,
                                llvm::GlobalVariable::ExternalLinkage,
                                _executeName, &M);
  assert(FN->getFunctionType() == VargsFnAllocFuncType);
  return FN;
}

// Returns a function pointer to __llvm_omp_emissary_rpc
static llvm::Function *GetEmissaryExecDeclaration(CodeGenModule &CGM) {
  const char *_executeName = "__llvm_omp_emissary_rpc";
  auto &M = CGM.getModule();
  llvm::Type *ArgTypes[] = {CGM.Int64Ty,
	  llvm::PointerType::getUnqual(CGM.Int8Ty)};
  llvm::Function *FN;
  llvm::FunctionType *VarfnFuncType =
      llvm::FunctionType::get(CGM.Int64Ty, ArgTypes, false);
  if (!(FN = M.getFunction(_executeName)))
    FN = llvm::Function::Create(
        VarfnFuncType, llvm::GlobalVariable::ExternalLinkage, _executeName, &M);
  assert(FN->getFunctionType() == VarfnFuncType);
  return FN;
}

// A macro to pack the llvm type ID and numbits into 4-byte key
#define _PACK_TY_BITLEN(x, y) ((uint32_t)x << 16) | ((uint32_t)y)

//  ----- External function EmitEmissaryExec called from CGExpr.cpp -----
RValue CodeGenFunction::EmitEmissaryExec(const CallExpr *E) {
  assert(getTarget().getTriple().isAMDGCN() ||
         getTarget().getTriple().isNVPTX());
  assert(E->getNumArgs() >= 1); // _emissary_exec always has at least one arg.

  const llvm::DataLayout &DL = CGM.getDataLayout();

  CallArgList Args;

  // --- Insert 1st emisid arg if emiting fprintf or printf.
  unsigned int AOE = 0;
  if (E->getDirectCallee()->getNameAsString() == "fprintf") {
    constexpr unsigned long long emisid =
        ((unsigned long long)EMIS_ID_PRINT << 32) |
        (unsigned long long)_fprintf_idx;
    Args.add(
        RValue::get(llvm::ConstantInt::get(Int64Ty, emisid)),
        getContext().getIntTypeForBitwidth(/*DestWidth=*/64, /*Signed=*/false));
    AOE = 1; // Arg# offset to E->arguments to use with E->getArg(I-AOE)
  }
  if (E->getDirectCallee()->getNameAsString() == "printf") {
    constexpr unsigned long long emisid =
        ((unsigned long long)EMIS_ID_PRINT << 32) |
        (unsigned long long)_printf_idx;
    Args.add(
        RValue::get(llvm::ConstantInt::get(Int64Ty, emisid)),
        getContext().getIntTypeForBitwidth(/*DestWidth=*/64, /*Signed=*/false));
    AOE = 1; // Arg# offset to E->arguments to use with E->getArg(I-AOE)
  }

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

  // NumArgs always includes emisid, but E->getNumArgs() could be 1 less if
  // inserted it above.
  unsigned NumArgs = (unsigned)Args.size();
  llvm::SmallVector<llvm::Type *, 32> ArgTypes;
  llvm::SmallVector<llvm::Value *, 32> VarStrLengths;
  llvm::Value *TotalVarStrsLength = llvm::ConstantInt::get(Int32Ty, 0);
  bool hasVarStrings = false;
  ArgTypes.push_back(
      Int32Ty); // First field in struct will be total DataLen FIXME
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
    // Skip string processing on arg0 which may not be in E->getArg(0)
    if (I != 0) {
      const Expr *argX = E->getArg(I - AOE)->IgnoreParenCasts();
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
    } // End of skip 1st arg
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
      ArgTypes.push_back(Int32Ty); // could assert that needsPadding == 4 here
    }

    ArgTypes.push_back(ArgType);
    DataLen_CT += ((int)DL.getTypeAllocSize(ArgType));
    structOffset += (size_t)DL.getTypeAllocSize(ArgType);
  }

  // ---  Generate call to __llvm_omp_emissary_premalloc to get data pointer
  if (hasVarStrings)
    TotalVarStrsLength = Builder.CreateAdd(
        TotalVarStrsLength,
        llvm::ConstantInt::get(Int32Ty, AllStringsLen_CT + DataLen_CT),
        "total_buffer_size");
  llvm::Value *BufferLen =
      hasVarStrings
          ? TotalVarStrsLength
          : llvm::ConstantInt::get(Int32Ty, AllStringsLen_CT + DataLen_CT);
  llvm::Value *DataStructPtr =
      Builder.CreateCall(GetEmissaryAllocDeclaration(CGM), {BufferLen});

  // --- Cast the generic return pointer to be a struct in device global memory
  llvm::StructType *DataStructTy =
      llvm::StructType::create(ArgTypes, "varfn_args_store");
  unsigned AS = getContext().getTargetAddressSpace(LangAS::cuda_device);
  llvm::Value *BufferPtr = Builder.CreatePointerCast(
      DataStructPtr, llvm::PointerType::get(DataStructTy, AS),
      "varfn_args_store_casted");

  // ---  Header of struct contains length and NumArgs ---
  llvm::Value *DataLenField = llvm::ConstantInt::get(Int32Ty, DataLen_CT);
  llvm::Value *P = Builder.CreateStructGEP(DataStructTy, BufferPtr, 0);
  Builder.CreateAlignedStore(DataLenField, P,
                             DL.getPrefTypeAlign(DataLenField->getType()));
  llvm::Value *NumArgsField = llvm::ConstantInt::get(Int32Ty, NumArgs);
  P = Builder.CreateStructGEP(DataStructTy, BufferPtr, 1);
  Builder.CreateAlignedStore(NumArgsField, P,
                             DL.getPrefTypeAlign(NumArgsField->getType()));

  // ---  2nd Pass: create array of 4-byte keys to describe each arg
  for (unsigned I = 0; I < NumArgs; I++) {
    llvm::Type *ty = Args[I].getRValue(*this).getScalarVal()->getType();
    llvm::Type::TypeID argtypeid =
        Args[I].getRValue(*this).getScalarVal()->getType()->getTypeID();

    // Get type size in bits. Usually 64 or 32.
    uint32_t numbits = 0;
    if (I > 0 &&
        isString(
            E->getArg(I - AOE)->IgnoreParenCasts()->getType().getTypePtr()))
      // The llvm typeID for string is pointer.  Since pointer numbits is 0,
      // we set numbits to 1 to distinguish pointer type ID as string pointer.
      numbits = 1;
    else
      numbits = ty->getScalarSizeInBits();
    // Create a key that combines llvm typeID and size
    llvm::Value *Key =
        llvm::ConstantInt::get(Int32Ty, _PACK_TY_BITLEN(argtypeid, numbits));
    P = Builder.CreateStructGEP(DataStructTy, BufferPtr, I + 2);
    Builder.CreateAlignedStore(Key, P, DL.getPrefTypeAlign(Key->getType()));
  }

  // ---  3rd Pass: Store data values for each arg ---
  unsigned varstring_index = 0;
  unsigned structIndex = 2 + NumArgs;
  structOffset = 4 * structIndex;
  for (unsigned I = 0; I < NumArgs; I++) {
    llvm::Value *Arg;
    if (I == 0) {
      Arg = Args[I].getKnownRValue().getScalarVal();
    } else {
      const Expr *argX = E->getArg(I - AOE)->IgnoreParenCasts();
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
      Int8Ty, CharUnits::fromQuantity(1));
  // BufferPtrByteAddr is a pointer to where we want to write the next string
  BufferPtrByteAddr = Builder.CreateConstInBoundsByteGEP(
      BufferPtrByteAddr, CharUnits::fromQuantity(DataLen_CT));
  varstring_index = 0;
  // Skip string processing on arg0 which may not be in E->getArg(0)
  for (unsigned I = 1; I < NumArgs; ++I) {
    llvm::Value *Arg = Args[I].getKnownRValue().getScalarVal();
    const Expr *argX = E->getArg(I - AOE)->IgnoreParenCasts();
    auto *argXTy = argX->getType().getTypePtr();
    if (isString(argXTy)) {
      if (isVarString(argX, argXTy, Arg)) {
        llvm::Value *varStrLength = VarStrLengths[varstring_index];
        varstring_index++;
        Address SrcAddr = Address(Arg, Int8Ty, CharUnits::fromQuantity(1));
        Builder.CreateMemCpy(BufferPtrByteAddr, SrcAddr, varStrLength);
        // update BufferPtrByteAddr for next string memcpy
        llvm::Value *PtrAsInt = BufferPtrByteAddr.emitRawPointer(*this);
        BufferPtrByteAddr =
            Address(Builder.CreateGEP(Int8Ty, PtrAsInt,
                                      ArrayRef<llvm::Value *>(varStrLength)),
                    Int8Ty, CharUnits::fromQuantity(1));
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
  // --- Generate call to __llvm_omp_emissary_rpc and return RValue
  llvm::Value *EmisIds = Args[0].getRValue(*this).getScalarVal();
  return RValue::get(Builder.CreateCall(
      GetEmissaryExecDeclaration(CGM), {EmisIds, DataStructPtr}));
}
