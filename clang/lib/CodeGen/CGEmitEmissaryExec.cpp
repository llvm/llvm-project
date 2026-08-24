//===- CGEmitEmissaryExec.cpp - Codegen for _emissary_exec ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// EmitEmissaryExec:
//
// When a device call to the variadic function _emissary_exec is encountered
// (in CGExpr.cpp) EmitEmissaryExec does these steps:
//
// 1. If string lens are runtime dependent, Emit code to determine runtime len.
// 2. Emits call to allocate memory __llvm_emissary_premalloc,
// 3. Emit stores of each arg into arg buffer,
// 4. Emits call to function __llvm_emissary_rpc or __llvm_emissary_rpc_dm
//
// The arg buffer is a struct that contains the length, number of args, an
// array of 4-byte keys that represent the type of each arg, an array of
// aligned "data" values for each arg, and finally the runtime string values.
// If an arg is a string the data value is the runtime length of the string.
// Each 4-byte key contains the llvm type ID and the number of bits for the
// type. encoded by the macro PACK_TY_BITLEN(x,y) ((uint32_t)x << 16) |
// ((uint32_t)y)
//
//===----------------------------------------------------------------------===//

#include "../../../clang/lib/Headers/EmissaryIds.h"
#include "CodeGenFunction.h"
#include "clang/Basic/Builtins.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Transforms/Utils/AMDGPUEmitPrintf.h"

using namespace clang;
using namespace CodeGen;

// These static helper functions support EmitEmissaryExec.
static llvm::Function *getOmpStrlenDeclaration(CodeGenModule &CGM) {
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

// Determines if an expression is a string with variable length
static bool isVarString(const clang::Expr *ArgX, const clang::Type *ArgXTy,
                        const llvm::Value *Arg) {
  if ((ArgXTy->isPointerType() || ArgXTy->isConstantArrayType()) &&
      ArgXTy->getPointeeOrArrayElementType()->isCharType() && !ArgX->isLValue())
    return true;
  // Ensure the VarDecl has an initializer
  if (const auto *DRE = dyn_cast<DeclRefExpr>(ArgX))
    if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()))
      if (!VD->getInit() ||
          !llvm::isa<StringLiteral>(VD->getInit()->IgnoreImplicit()))
        return true;
  return false;
}

// Determines if an argument is a string
static bool isString(const clang::Type *ArgXTy) {
  if ((ArgXTy->isPointerType() || ArgXTy->isConstantArrayType()) &&
      ArgXTy->getPointeeOrArrayElementType()->isCharType())
    return true;
  else
    return false;
}

// Gets a string literal to write into the transfer buffer
static const StringLiteral *getSL(const clang::Expr *ArgX,
                                  const clang::Type *ArgXTy) {
  // String in ArgX has known constant length
  if (!ArgXTy->isConstantArrayType()) {
    // Allow constant string to be a declared variable,
    // But it must be constant and initialized.
    const DeclRefExpr *DRE = cast<DeclRefExpr>(ArgX);
    const VarDecl *VarD = cast<VarDecl>(DRE->getDecl());
    ArgX = VarD->getInit()->IgnoreImplicit();
  }
  const StringLiteral *SL = cast<StringLiteral>(ArgX);
  return SL;
}

// Returns a function pointer to __llvm_emissary_premalloc
static llvm::Function *getEmissaryAllocDeclaration(CodeGenModule &CGM) {
  auto &M = CGM.getModule();
  const char *ExecuteName = "__llvm_emissary_premalloc";
  llvm::Type *ArgTypes[] = {CGM.Int32Ty};
  llvm::Function *FN;
  // Maybe this should be pointer to char instead of pointer to void
  llvm::FunctionType *VargsFnAllocFuncType = llvm::FunctionType::get(
      CGM.getTypes().ConvertType(
          CGM.getContext().getPointerType(CGM.getContext().VoidTy)),
      ArgTypes, false);
  if (!(FN = M.getFunction(ExecuteName)))
    FN = llvm::Function::Create(VargsFnAllocFuncType,
                                llvm::GlobalVariable::ExternalLinkage,
                                ExecuteName, &M);
  assert(FN->getFunctionType() == VargsFnAllocFuncType);
  return FN;
}

// Returns a function pointer to __llvm_emissary_rpc
static llvm::Function *getEmissaryExecDeclaration(CodeGenModule &CGM,
                                                  bool HasXfers) {
  const char *ExecuteName =
      HasXfers ? "__llvm_emissary_rpc_dm" : "__llvm_emissary_rpc";
  auto &M = CGM.getModule();
  llvm::Type *ArgTypes[] = {
      CGM.Int32Ty, CGM.getTypes().ConvertType(CGM.getContext().getPointerType(
                       CGM.getContext().VoidTy))};
  llvm::Function *FN;
  llvm::FunctionType *VarfnFuncType =
      llvm::FunctionType::get(CGM.Int64Ty, ArgTypes, false);
  if (!(FN = M.getFunction(ExecuteName)))
    FN = llvm::Function::Create(
        VarfnFuncType, llvm::GlobalVariable::ExternalLinkage, ExecuteName, &M);
  assert(FN->getFunctionType() == VarfnFuncType);
  return FN;
}

// A macro to pack the llvm type ID and NumBits into 4-byte key
#define PACK_TY_BITLEN(x, y) ((uint32_t)x << 16) | ((uint32_t)y)

static EmisTyID getEmisTyID(llvm::Type::TypeID TyId) {
  switch (TyId) {
  case llvm::Type::HalfTyID:     ///< 16-bit floating point type
  case llvm::Type::X86_FP80TyID: ///< 80-bit floating point type (X87)
  case llvm::Type::BFloatTyID:   ///< 16-bit floating point type (7-bit
                                 ///< significand)
    return EmisInvalidTy;
  case llvm::Type::FloatTyID:  ///< 32-bit floating point type
  case llvm::Type::DoubleTyID: ///< 64-bit floating point type
  case llvm::Type::FP128TyID:  ///< 128-bit floating point type (112-bit
                               ///< significand)
    return EmisFloatTy;
  case llvm::Type::PPC_FP128TyID: ///< 128-bit floating point type (two 64-bits,
                                  ///< PowerPC)
  case llvm::Type::VoidTyID:      ///< type with no size
  case llvm::Type::LabelTyID:     ///< Labels
  case llvm::Type::MetadataTyID:  ///< Metadata
  case llvm::Type::X86_AMXTyID:   ///< AMX vectors (8192 bits, X86 specific)
  case llvm::Type::TokenTyID:     ///< Tokens
    return EmisInvalidTy;
  // Derived types... see DerivedTypes.h file.
  case llvm::Type::IntegerTyID: ///< Arbitrary bit width integers
    return EmisIntegerTy;
  case llvm::Type::ByteTyID:     ///< Arbitrary bit width bytes
  case llvm::Type::FunctionTyID: ///< Functions
    return EmisInvalidTy;
  case llvm::Type::PointerTyID: ///< Pointers
    return EmisPointerTy;
  case llvm::Type::StructTyID:         ///< Structures
  case llvm::Type::ArrayTyID:          ///< Arrays
  case llvm::Type::FixedVectorTyID:    ///< Fixed width SIMD vector type
  case llvm::Type::ScalableVectorTyID: ///< Scalable SIMD vector type
  case llvm::Type::TypedPointerTyID: ///< Typed pointer used by some GPU targets
  case llvm::Type::TargetExtTyID:    ///< Target extension type
    return EmisInvalidTy;
  break;
  }
  return EmisInvalidTy;
}

//  ----- External function EmitEmissaryExec called from CGExpr.cpp -----
RValue CodeGenFunction::EmitEmissaryExec(const CallExpr *E) {
  assert(getTarget().getTriple().isAMDGCN() ||
         getTarget().getTriple().isNVPTX());
  assert(E->getNumArgs() >= 1); // _emissary_exec always has at least one arg.
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
  // Arg 0 is the packed emisid supplied by the caller, so Args maps 1:1 onto
  // E->arguments(). It has to be a compile-time constant because the buffer
  // layout below depends on the transfer counts encoded in it. _PACK_EMIS_IDS()
  // folds to a constant, so this only fires on a malformed hand-written call --
  // diagnose it instead of crashing on the cast.
  RValue EmisIdRV = Args[0].getKnownRValue();
  if (!EmisIdRV.isScalar() ||
      !llvm::isa<llvm::ConstantInt>(EmisIdRV.getScalarVal())) {
    CGM.ErrorUnsupported(E, "non-constant emissary id in _emissary_exec call");
    return RValue::get(llvm::ConstantInt::get(IntTy, 0));
  }

  unsigned NumArgs = (unsigned)Args.size();
  llvm::SmallVector<llvm::Type *, 32> ArgTypes;
  llvm::SmallVector<llvm::Value *, 32> VarStrLengths;
  llvm::Value *TotalVarStrsLength = llvm::ConstantInt::get(Int32Ty, 0);
  bool HasVarStrings = false;
  ArgTypes.push_back(Int32Ty); // 1st field in struct is total DataLen
  ArgTypes.push_back(Int32Ty); // 2nd field in struct will be num args
  // An array of 4-byte keys that describe the arg type
  for (unsigned I = 0; I < NumArgs; ++I)
    ArgTypes.push_back(Int32Ty);

  // Track the size of the numeric data length and string length
  unsigned DataLenCT = (unsigned)(DL.getTypeAllocSize(Int32Ty)) * (NumArgs + 2);
  unsigned AllStringsLenCT = 0;

  // ---  1st Pass over Args to create ArgTypes and count size ---
  size_t StructOffset = 4 * (NumArgs + 2);
  for (unsigned I = 0; I < NumArgs; I++) {
    llvm::Value *Arg = Args[I].getRValue(*this).getScalarVal();
    llvm::Type *ArgType = Arg->getType();
    // Skip string processing on arg0 which may not be in E->getArg(0)
    if (I != 0) {
      const Expr *ArgX = E->getArg(I)->IgnoreParenCasts();
      auto *ArgXTy = ArgX->getType().getTypePtr();
      if (isString(ArgXTy)) {
        if (isVarString(ArgX, ArgXTy, Arg)) {
          HasVarStrings = true;
          if (auto *PtrTy = dyn_cast<llvm::PointerType>(ArgType))
            if (PtrTy->getPointerAddressSpace()) {
              Arg = Builder.CreateAddrSpaceCast(Arg, CGM.Int8PtrTy);
              ArgType = Arg->getType();
            }
          llvm::Value *VarStrLen =
              Builder.CreateCall(getOmpStrlenDeclaration(CGM),
                                 {Arg, llvm::ConstantInt::get(Int32Ty, 1024)});
          VarStrLengths.push_back(VarStrLen);
          TotalVarStrsLength = Builder.CreateAdd(TotalVarStrsLength, VarStrLen,
                                                 "sum_of_var_strings_length");
          ArgType = Int32Ty;
        } else {
          const StringLiteral *SL = getSL(ArgX, ArgXTy);
          StringRef ArgString = SL->getString();
          AllStringsLenCT += ((int)ArgString.size() + 1);
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
    size_t NeedsPadding = (StructOffset % (size_t)DL.getTypeAllocSize(ArgType));
    if (NeedsPadding) {
      DataLenCT += (unsigned)NeedsPadding;
      StructOffset += NeedsPadding;
      ArgTypes.push_back(Int32Ty); // could assert that NeedsPadding == 4 here
    }

    ArgTypes.push_back(ArgType);
    DataLenCT += ((int)DL.getTypeAllocSize(ArgType));
    StructOffset += (size_t)DL.getTypeAllocSize(ArgType);
  }

  // ---  Generate call to __llvm_emissary_premalloc to get data pointer
  if (HasVarStrings)
    TotalVarStrsLength = Builder.CreateAdd(
        TotalVarStrsLength,
        llvm::ConstantInt::get(Int32Ty, AllStringsLenCT + DataLenCT),
        "total_buffer_size");
  llvm::Value *BufferLen =
      HasVarStrings
          ? TotalVarStrsLength
          : llvm::ConstantInt::get(Int32Ty, AllStringsLenCT + DataLenCT);
  llvm::Value *DataStructPtr =
      Builder.CreateCall(getEmissaryAllocDeclaration(CGM), {BufferLen});

  // --- Cast the generic return pointer to be a struct in device global memory
  llvm::StructType *DataStructTy =
      llvm::StructType::create(ArgTypes, "varfn_args_store");
  unsigned AS = getContext().getTargetAddressSpace(LangAS::cuda_device);
  llvm::Value *BufferPtr = Builder.CreatePointerCast(
      DataStructPtr, llvm::PointerType::get(CGM.getLLVMContext(), AS),
      "varfn_args_store_casted");
  // ---  Header of struct contains length and NumArgs ---
  llvm::Value *DataLenField = llvm::ConstantInt::get(Int32Ty, DataLenCT);
  llvm::Value *P = Builder.CreateStructGEP(DataStructTy, BufferPtr, 0);
  Builder.CreateAlignedStore(DataLenField, P,
                             DL.getPrefTypeAlign(DataLenField->getType()));
  llvm::Value *NumArgsField = llvm::ConstantInt::get(Int32Ty, NumArgs);
  P = Builder.CreateStructGEP(DataStructTy, BufferPtr, 1);
  Builder.CreateAlignedStore(NumArgsField, P,
                             DL.getPrefTypeAlign(NumArgsField->getType()));

  // ---  2nd Pass: create array of 4-byte keys to describe each arg
  for (unsigned I = 0; I < NumArgs; I++) {
    llvm::Type *Ty = Args[I].getRValue(*this).getScalarVal()->getType();
    llvm::Type::TypeID ArgTypeId =
        Args[I].getRValue(*this).getScalarVal()->getType()->getTypeID();
    EmisTyID EmisTypeId = getEmisTyID(ArgTypeId);

    // Get type size in bits. Usually 64 or 32.
    uint32_t NumBits = 0;
    if (I > 0 &&
        isString(E->getArg(I)->IgnoreParenCasts()->getType().getTypePtr()))
      // The llvm typeID for string is pointer.  Since pointer NumBits is 0,
      // we set NumBits to 1 to distinguish pointer type ID as string pointer.
      NumBits = 1;
    else
      NumBits = Ty->getScalarSizeInBits();
    // Create a key that combines llvm typeID and size
    llvm::Value *Key =
        llvm::ConstantInt::get(Int32Ty, PACK_TY_BITLEN(EmisTypeId, NumBits));
    P = Builder.CreateStructGEP(DataStructTy, BufferPtr, I + 2);
    Builder.CreateAlignedStore(Key, P, DL.getPrefTypeAlign(Key->getType()));
  }

  // ---  3rd Pass: Store data values for each arg ---
  unsigned VarStringIndex = 0;
  unsigned StructIndex = 2 + NumArgs;
  StructOffset = 4 * StructIndex;
  bool HasXfers;
  for (unsigned I = 0; I < NumArgs; I++) {
    llvm::Value *Arg = nullptr;
    if (I == 0) {
      Arg = Args[I].getKnownRValue().getScalarVal();
      uint64_t UInt64Value = llvm::cast<llvm::ConstantInt>(Arg)->getZExtValue();
      uint32_t Lower32 = (uint32_t)(UInt64Value & 0xFFFFFFFF);
      HasXfers = Lower32 ? true : false;
    } else {
      const Expr *ArgX = E->getArg(I)->IgnoreParenCasts();
      auto *ArgXTy = ArgX->getType().getTypePtr();
      if (isString(ArgXTy)) {
        if (isVarString(ArgX, ArgXTy, Arg)) {
          Arg = VarStrLengths[VarStringIndex];
          VarStringIndex++;
        } else {
          const StringLiteral *SL = getSL(ArgX, ArgXTy);
          StringRef ArgString = SL->getString();
          int ArgStrLen = (int)ArgString.size() + 1;
          // Change Arg from a char pointer to the integer string length
          Arg = llvm::ConstantInt::get(Int32Ty, ArgStrLen);
        }
      } else {
        Arg = Args[I].getKnownRValue().getScalarVal();
      }
    }
    size_t StructElementSize = (size_t)DL.getTypeAllocSize(Arg->getType());
    size_t NeedsPadding = (StructOffset % StructElementSize);
    if (NeedsPadding) {
      // Skip over dummy fields in struct to align
      StructOffset += NeedsPadding; // should assert NeedsPadding == 4
      StructIndex++;
    }
    P = Builder.CreateStructGEP(DataStructTy, BufferPtr, StructIndex);
    Builder.CreateAlignedStore(Arg, P, DL.getPrefTypeAlign(Arg->getType()));
    StructOffset += StructElementSize;
    StructIndex++;
  }

  // ---  4th Pass: memcpy all strings after the data values ---
  // bitcast the struct in device global memory as a char buffer
  Address BufferPtrByteAddr =
      Address(Builder.CreatePointerCast(
                  BufferPtr, llvm::PointerType::get(CGM.getLLVMContext(), AS),
                  "_casted"),
              Int8Ty, CharUnits::fromQuantity(1));

  // BufferPtrByteAddr is a pointer to where we want to write the next string
  BufferPtrByteAddr = Builder.CreateConstInBoundsByteGEP(
      BufferPtrByteAddr, CharUnits::fromQuantity(DataLenCT));
  VarStringIndex = 0;
  // Skip string processing on arg0 which may not be in E->getArg(0)
  for (unsigned I = 1; I < NumArgs; ++I) {
    llvm::Value *Arg = Args[I].getKnownRValue().getScalarVal();
    const Expr *ArgX = E->getArg(I)->IgnoreParenCasts();
    auto *ArgXTy = ArgX->getType().getTypePtr();
    if (isString(ArgXTy)) {
      if (isVarString(ArgX, ArgXTy, Arg)) {
        llvm::Value *VarStrLength = VarStrLengths[VarStringIndex];
        VarStringIndex++;
        Address SrcAddr = Address(Arg, Int8Ty, CharUnits::fromQuantity(1));
        Builder.CreateMemCpy(BufferPtrByteAddr, SrcAddr, VarStrLength);
        // update BufferPtrByteAddr for next string memcpy
        llvm::Value *PtrAsInt = BufferPtrByteAddr.emitRawPointer(*this);
        BufferPtrByteAddr =
            Address(Builder.CreateGEP(Int8Ty, PtrAsInt,
                                      ArrayRef<llvm::Value *>(VarStrLength)),
                    Int8Ty, CharUnits::fromQuantity(1));
      } else {
        const StringLiteral *SL = getSL(ArgX, ArgXTy);
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
  // --- Generate call to __llvm_emissary_rpc and return RValue
  llvm::Value *EmisRc = Builder.CreateCall(
      getEmissaryExecDeclaration(CGM, HasXfers), {BufferLen, DataStructPtr});
  // truncate long long int to int for printf return value.
  if ((E->getDirectCallee()->getNameAsString() == "fprintf") ||
      (E->getDirectCallee()->getNameAsString() == "printf"))
    EmisRc = Builder.CreateTrunc(EmisRc, CGM.Int32Ty, "emis_rc");
  return RValue::get(EmisRc);
}
