//===------- CGHLSLBuiltins.cpp - Emit LLVM Code for HLSL builtins --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit HLSL Builtin calls as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CGBuiltin.h"
#include "CGHLSLRuntime.h"
#include "CodeGenFunction.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm;

static Value *handleAsDoubleBuiltin(CodeGenFunction &CGF, const CallExpr *E) {
  assert((E->getArg(0)->getType()->hasUnsignedIntegerRepresentation() &&
          E->getArg(1)->getType()->hasUnsignedIntegerRepresentation()) &&
         "asdouble operands types mismatch");
  Value *OpLowBits = CGF.EmitScalarExpr(E->getArg(0));
  Value *OpHighBits = CGF.EmitScalarExpr(E->getArg(1));

  llvm::Type *ResultType = CGF.DoubleTy;
  int N = 1;
  if (auto *VTy = E->getArg(0)->getType()->getAs<clang::VectorType>()) {
    N = VTy->getNumElements();
    ResultType = llvm::FixedVectorType::get(CGF.DoubleTy, N);
  }

  if (CGF.CGM.getTarget().getTriple().isDXIL())
    return CGF.Builder.CreateIntrinsic(
        /*ReturnType=*/ResultType, Intrinsic::dx_asdouble,
        {OpLowBits, OpHighBits}, nullptr, "hlsl.asdouble");

  if (!E->getArg(0)->getType()->isVectorType()) {
    OpLowBits = CGF.Builder.CreateVectorSplat(1, OpLowBits);
    OpHighBits = CGF.Builder.CreateVectorSplat(1, OpHighBits);
  }

  llvm::SmallVector<int> Mask;
  for (int i = 0; i < N; i++) {
    Mask.push_back(i);
    Mask.push_back(i + N);
  }

  Value *BitVec = CGF.Builder.CreateShuffleVector(OpLowBits, OpHighBits, Mask);

  return CGF.Builder.CreateBitCast(BitVec, ResultType);
}

static Value *handleHlslClip(const CallExpr *E, CodeGenFunction *CGF) {
  Value *Op0 = CGF->EmitScalarExpr(E->getArg(0));

  Constant *FZeroConst = ConstantFP::getZero(CGF->FloatTy);
  Value *CMP;
  Value *LastInstr;

  if (const auto *VecTy = E->getArg(0)->getType()->getAs<clang::VectorType>()) {
    FZeroConst = ConstantVector::getSplat(
        ElementCount::getFixed(VecTy->getNumElements()), FZeroConst);
    auto *FCompInst = CGF->Builder.CreateFCmpOLT(Op0, FZeroConst);
    CMP = CGF->Builder.CreateIntrinsic(
        CGF->Builder.getInt1Ty(), CGF->CGM.getHLSLRuntime().getAnyIntrinsic(),
        {FCompInst});
  } else {
    CMP = CGF->Builder.CreateFCmpOLT(Op0, FZeroConst);
  }

  if (CGF->CGM.getTarget().getTriple().isDXIL()) {
    LastInstr = CGF->Builder.CreateIntrinsic(Intrinsic::dx_discard, {CMP});
  } else if (CGF->CGM.getTarget().getTriple().isSPIRV()) {
    BasicBlock *LT0 = CGF->createBasicBlock("lt0", CGF->CurFn);
    BasicBlock *End = CGF->createBasicBlock("end", CGF->CurFn);

    CGF->Builder.CreateCondBr(CMP, LT0, End);

    CGF->Builder.SetInsertPoint(LT0);

    CGF->Builder.CreateIntrinsic(Intrinsic::spv_discard, {});

    LastInstr = CGF->Builder.CreateBr(End);
    CGF->Builder.SetInsertPoint(End);
  } else {
    llvm_unreachable("Backend Codegen not supported.");
  }

  return LastInstr;
}

static Value *handleHlslSplitdouble(const CallExpr *E, CodeGenFunction *CGF) {
  Value *Op0 = CGF->EmitScalarExpr(E->getArg(0));
  const auto *OutArg1 = dyn_cast<HLSLOutArgExpr>(E->getArg(1));
  const auto *OutArg2 = dyn_cast<HLSLOutArgExpr>(E->getArg(2));

  CallArgList Args;
  LValue Op1TmpLValue =
      CGF->EmitHLSLOutArgExpr(OutArg1, Args, OutArg1->getType());
  LValue Op2TmpLValue =
      CGF->EmitHLSLOutArgExpr(OutArg2, Args, OutArg2->getType());

  if (CGF->getTarget().getCXXABI().areArgsDestroyedLeftToRightInCallee())
    Args.reverseWritebacks();

  Value *LowBits = nullptr;
  Value *HighBits = nullptr;

  if (CGF->CGM.getTarget().getTriple().isDXIL()) {
    llvm::Type *RetElementTy = CGF->Int32Ty;
    if (auto *Op0VecTy = E->getArg(0)->getType()->getAs<clang::VectorType>())
      RetElementTy = llvm::VectorType::get(
          CGF->Int32Ty, ElementCount::getFixed(Op0VecTy->getNumElements()));
    auto *RetTy = llvm::StructType::get(RetElementTy, RetElementTy);

    CallInst *CI = CGF->Builder.CreateIntrinsic(
        RetTy, Intrinsic::dx_splitdouble, {Op0}, nullptr, "hlsl.splitdouble");

    LowBits = CGF->Builder.CreateExtractValue(CI, 0);
    HighBits = CGF->Builder.CreateExtractValue(CI, 1);
  } else {
    // For Non DXIL targets we generate the instructions.

    if (!Op0->getType()->isVectorTy()) {
      FixedVectorType *DestTy = FixedVectorType::get(CGF->Int32Ty, 2);
      Value *Bitcast = CGF->Builder.CreateBitCast(Op0, DestTy);

      LowBits = CGF->Builder.CreateExtractElement(Bitcast, (uint64_t)0);
      HighBits = CGF->Builder.CreateExtractElement(Bitcast, 1);
    } else {
      int NumElements = 1;
      if (const auto *VecTy =
              E->getArg(0)->getType()->getAs<clang::VectorType>())
        NumElements = VecTy->getNumElements();

      FixedVectorType *Uint32VecTy =
          FixedVectorType::get(CGF->Int32Ty, NumElements * 2);
      Value *Uint32Vec = CGF->Builder.CreateBitCast(Op0, Uint32VecTy);
      if (NumElements == 1) {
        LowBits = CGF->Builder.CreateExtractElement(Uint32Vec, (uint64_t)0);
        HighBits = CGF->Builder.CreateExtractElement(Uint32Vec, 1);
      } else {
        SmallVector<int> EvenMask, OddMask;
        for (int I = 0, E = NumElements; I != E; ++I) {
          EvenMask.push_back(I * 2);
          OddMask.push_back(I * 2 + 1);
        }
        LowBits = CGF->Builder.CreateShuffleVector(Uint32Vec, EvenMask);
        HighBits = CGF->Builder.CreateShuffleVector(Uint32Vec, OddMask);
      }
    }
  }
  CGF->Builder.CreateStore(LowBits, Op1TmpLValue.getAddress());
  auto *LastInst =
      CGF->Builder.CreateStore(HighBits, Op2TmpLValue.getAddress());
  CGF->EmitWritebacks(Args);
  return LastInst;
}

// Return dot product intrinsic that corresponds to the QT scalar type
static Intrinsic::ID getDotProductIntrinsic(CGHLSLRuntime &RT, QualType QT) {
  if (QT->isFloatingType())
    return RT.getFDotIntrinsic();
  if (QT->isSignedIntegerType())
    return RT.getSDotIntrinsic();
  assert(QT->isUnsignedIntegerType());
  return RT.getUDotIntrinsic();
}

static Intrinsic::ID getFirstBitHighIntrinsic(CGHLSLRuntime &RT, QualType QT) {
  if (QT->hasSignedIntegerRepresentation()) {
    return RT.getFirstBitSHighIntrinsic();
  }

  assert(QT->hasUnsignedIntegerRepresentation());
  return RT.getFirstBitUHighIntrinsic();
}

// Return wave active sum that corresponds to the QT scalar type
static Intrinsic::ID getWaveActiveSumIntrinsic(llvm::Triple::ArchType Arch,
                                               CGHLSLRuntime &RT, QualType QT) {
  switch (Arch) {
  case llvm::Triple::spirv:
    return Intrinsic::spv_wave_reduce_sum;
  case llvm::Triple::dxil: {
    if (QT->isUnsignedIntegerType())
      return Intrinsic::dx_wave_reduce_usum;
    return Intrinsic::dx_wave_reduce_sum;
  }
  default:
    llvm_unreachable("Intrinsic WaveActiveSum"
                     " not supported by target architecture");
  }
}

// Return wave active sum that corresponds to the QT scalar type
static Intrinsic::ID getWaveActiveMaxIntrinsic(llvm::Triple::ArchType Arch,
                                               CGHLSLRuntime &RT, QualType QT) {
  switch (Arch) {
  case llvm::Triple::spirv:
    if (QT->isUnsignedIntegerType())
      return Intrinsic::spv_wave_reduce_umax;
    return Intrinsic::spv_wave_reduce_max;
  case llvm::Triple::dxil: {
    if (QT->isUnsignedIntegerType())
      return Intrinsic::dx_wave_reduce_umax;
    return Intrinsic::dx_wave_reduce_max;
  }
  default:
    llvm_unreachable("Intrinsic WaveActiveMax"
                     " not supported by target architecture");
  }
}

// Returns the mangled name for a builtin function that the SPIR-V backend
// will expand into a spec Constant.
static std::string getSpecConstantFunctionName(clang::QualType SpecConstantType,
                                               ASTContext &Context) {
  // The parameter types for our conceptual intrinsic function.
  QualType ClangParamTypes[] = {Context.IntTy, SpecConstantType};

  // Create a temporary FunctionDecl for the builtin fuction. It won't be
  // added to the AST.
  FunctionProtoType::ExtProtoInfo EPI;
  QualType FnType =
      Context.getFunctionType(SpecConstantType, ClangParamTypes, EPI);
  DeclarationName FuncName = &Context.Idents.get("__spirv_SpecConstant");
  FunctionDecl *FnDeclForMangling = FunctionDecl::Create(
      Context, Context.getTranslationUnitDecl(), SourceLocation(),
      SourceLocation(), FuncName, FnType, /*TSI=*/nullptr, SC_Extern);

  // Attach the created parameter declarations to the function declaration.
  SmallVector<ParmVarDecl *, 2> ParamDecls;
  for (QualType ParamType : ClangParamTypes) {
    ParmVarDecl *PD = ParmVarDecl::Create(
        Context, FnDeclForMangling, SourceLocation(), SourceLocation(),
        /*IdentifierInfo*/ nullptr, ParamType, /*TSI*/ nullptr, SC_None,
        /*DefaultArg*/ nullptr);
    ParamDecls.push_back(PD);
  }
  FnDeclForMangling->setParams(ParamDecls);

  // Get the mangled name.
  std::string Name;
  llvm::raw_string_ostream MangledNameStream(Name);
  std::unique_ptr<MangleContext> Mangler(Context.createMangleContext());
  Mangler->mangleName(FnDeclForMangling, MangledNameStream);
  MangledNameStream.flush();

  return Name;
}

Value *CodeGenFunction::EmitHLSLBuiltinExpr(unsigned BuiltinID,
                                            const CallExpr *E,
                                            ReturnValueSlot ReturnValue) {
  if (!getLangOpts().HLSL)
    return nullptr;

  switch (BuiltinID) {
  case Builtin::BI__builtin_hlsl_adduint64: {
    Value *OpA = EmitScalarExpr(E->getArg(0));
    Value *OpB = EmitScalarExpr(E->getArg(1));
    QualType Arg0Ty = E->getArg(0)->getType();
    uint64_t NumElements = Arg0Ty->castAs<VectorType>()->getNumElements();
    assert(Arg0Ty == E->getArg(1)->getType() &&
           "AddUint64 operand types must match");
    assert(Arg0Ty->hasIntegerRepresentation() &&
           "AddUint64 operands must have an integer representation");
    assert((NumElements == 2 || NumElements == 4) &&
           "AddUint64 operands must have 2 or 4 elements");

    llvm::Value *LowA;
    llvm::Value *HighA;
    llvm::Value *LowB;
    llvm::Value *HighB;

    // Obtain low and high words of inputs A and B
    if (NumElements == 2) {
      LowA = Builder.CreateExtractElement(OpA, (uint64_t)0, "LowA");
      HighA = Builder.CreateExtractElement(OpA, (uint64_t)1, "HighA");
      LowB = Builder.CreateExtractElement(OpB, (uint64_t)0, "LowB");
      HighB = Builder.CreateExtractElement(OpB, (uint64_t)1, "HighB");
    } else {
      LowA = Builder.CreateShuffleVector(OpA, {0, 2}, "LowA");
      HighA = Builder.CreateShuffleVector(OpA, {1, 3}, "HighA");
      LowB = Builder.CreateShuffleVector(OpB, {0, 2}, "LowB");
      HighB = Builder.CreateShuffleVector(OpB, {1, 3}, "HighB");
    }

    // Use an uadd_with_overflow to compute the sum of low words and obtain a
    // carry value
    llvm::Value *Carry;
    llvm::Value *LowSum = EmitOverflowIntrinsic(
        *this, Intrinsic::uadd_with_overflow, LowA, LowB, Carry);
    llvm::Value *ZExtCarry =
        Builder.CreateZExt(Carry, HighA->getType(), "CarryZExt");

    // Sum the high words and the carry
    llvm::Value *HighSum = Builder.CreateAdd(HighA, HighB, "HighSum");
    llvm::Value *HighSumPlusCarry =
        Builder.CreateAdd(HighSum, ZExtCarry, "HighSumPlusCarry");

    if (NumElements == 4) {
      return Builder.CreateShuffleVector(LowSum, HighSumPlusCarry, {0, 2, 1, 3},
                                         "hlsl.AddUint64");
    }

    llvm::Value *Result = PoisonValue::get(OpA->getType());
    Result = Builder.CreateInsertElement(Result, LowSum, (uint64_t)0,
                                         "hlsl.AddUint64.upto0");
    Result = Builder.CreateInsertElement(Result, HighSumPlusCarry, (uint64_t)1,
                                         "hlsl.AddUint64");
    return Result;
  }
  case Builtin::BI__builtin_hlsl_resource_getpointer: {
    Value *HandleOp = EmitScalarExpr(E->getArg(0));
    Value *IndexOp = EmitScalarExpr(E->getArg(1));

    llvm::Type *RetTy = ConvertType(E->getType());
    return Builder.CreateIntrinsic(
        RetTy, CGM.getHLSLRuntime().getCreateResourceGetPointerIntrinsic(),
        ArrayRef<Value *>{HandleOp, IndexOp});
  }
  case Builtin::BI__builtin_hlsl_resource_uninitializedhandle: {
    llvm::Type *HandleTy = CGM.getTypes().ConvertType(E->getType());
    return llvm::PoisonValue::get(HandleTy);
  }
  case Builtin::BI__builtin_hlsl_resource_handlefrombinding: {
    llvm::Type *HandleTy = CGM.getTypes().ConvertType(E->getType());
    Value *RegisterOp = EmitScalarExpr(E->getArg(1));
    Value *SpaceOp = EmitScalarExpr(E->getArg(2));
    Value *RangeOp = EmitScalarExpr(E->getArg(3));
    Value *IndexOp = EmitScalarExpr(E->getArg(4));
    Value *Name = EmitScalarExpr(E->getArg(5));
    llvm::Intrinsic::ID IntrinsicID =
        CGM.getHLSLRuntime().getCreateHandleFromBindingIntrinsic();
    SmallVector<Value *> Args{SpaceOp, RegisterOp, RangeOp, IndexOp, Name};
    return Builder.CreateIntrinsic(HandleTy, IntrinsicID, Args);
  }
  case Builtin::BI__builtin_hlsl_resource_handlefromimplicitbinding: {
    llvm::Type *HandleTy = CGM.getTypes().ConvertType(E->getType());
    Value *OrderID = EmitScalarExpr(E->getArg(1));
    Value *SpaceOp = EmitScalarExpr(E->getArg(2));
    Value *RangeOp = EmitScalarExpr(E->getArg(3));
    Value *IndexOp = EmitScalarExpr(E->getArg(4));
    Value *Name = EmitScalarExpr(E->getArg(5));
    llvm::Intrinsic::ID IntrinsicID =
        CGM.getHLSLRuntime().getCreateHandleFromImplicitBindingIntrinsic();
    SmallVector<Value *> Args{OrderID, SpaceOp, RangeOp, IndexOp, Name};
    return Builder.CreateIntrinsic(HandleTy, IntrinsicID, Args);
  }
  case Builtin::BI__builtin_hlsl_all: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    return Builder.CreateIntrinsic(
        /*ReturnType=*/llvm::Type::getInt1Ty(getLLVMContext()),
        CGM.getHLSLRuntime().getAllIntrinsic(), ArrayRef<Value *>{Op0}, nullptr,
        "hlsl.all");
  }
  case Builtin::BI__builtin_hlsl_and: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    return Builder.CreateAnd(Op0, Op1, "hlsl.and");
  }
  case Builtin::BI__builtin_hlsl_or: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    return Builder.CreateOr(Op0, Op1, "hlsl.or");
  }
  case Builtin::BI__builtin_hlsl_any: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    return Builder.CreateIntrinsic(
        /*ReturnType=*/llvm::Type::getInt1Ty(getLLVMContext()),
        CGM.getHLSLRuntime().getAnyIntrinsic(), ArrayRef<Value *>{Op0}, nullptr,
        "hlsl.any");
  }
  case Builtin::BI__builtin_hlsl_asdouble:
    return handleAsDoubleBuiltin(*this, E);
  case Builtin::BI__builtin_hlsl_elementwise_clamp: {
    Value *OpX = EmitScalarExpr(E->getArg(0));
    Value *OpMin = EmitScalarExpr(E->getArg(1));
    Value *OpMax = EmitScalarExpr(E->getArg(2));

    QualType Ty = E->getArg(0)->getType();
    if (auto *VecTy = Ty->getAs<VectorType>())
      Ty = VecTy->getElementType();

    Intrinsic::ID Intr;
    if (Ty->isFloatingType()) {
      Intr = CGM.getHLSLRuntime().getNClampIntrinsic();
    } else if (Ty->isUnsignedIntegerType()) {
      Intr = CGM.getHLSLRuntime().getUClampIntrinsic();
    } else {
      assert(Ty->isSignedIntegerType());
      Intr = CGM.getHLSLRuntime().getSClampIntrinsic();
    }
    return Builder.CreateIntrinsic(
        /*ReturnType=*/OpX->getType(), Intr,
        ArrayRef<Value *>{OpX, OpMin, OpMax}, nullptr, "hlsl.clamp");
  }
  case Builtin::BI__builtin_hlsl_crossf16:
  case Builtin::BI__builtin_hlsl_crossf32: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    assert(E->getArg(0)->getType()->hasFloatingRepresentation() &&
           E->getArg(1)->getType()->hasFloatingRepresentation() &&
           "cross operands must have a float representation");
    // make sure each vector has exactly 3 elements
    assert(
        E->getArg(0)->getType()->castAs<VectorType>()->getNumElements() == 3 &&
        E->getArg(1)->getType()->castAs<VectorType>()->getNumElements() == 3 &&
        "input vectors must have 3 elements each");
    return Builder.CreateIntrinsic(
        /*ReturnType=*/Op0->getType(), CGM.getHLSLRuntime().getCrossIntrinsic(),
        ArrayRef<Value *>{Op0, Op1}, nullptr, "hlsl.cross");
  }
  case Builtin::BI__builtin_hlsl_dot: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    llvm::Type *T0 = Op0->getType();
    llvm::Type *T1 = Op1->getType();

    // If the arguments are scalars, just emit a multiply
    if (!T0->isVectorTy() && !T1->isVectorTy()) {
      if (T0->isFloatingPointTy())
        return Builder.CreateFMul(Op0, Op1, "hlsl.dot");

      if (T0->isIntegerTy())
        return Builder.CreateMul(Op0, Op1, "hlsl.dot");

      llvm_unreachable(
          "Scalar dot product is only supported on ints and floats.");
    }
    // For vectors, validate types and emit the appropriate intrinsic
    assert(CGM.getContext().hasSameUnqualifiedType(E->getArg(0)->getType(),
                                                   E->getArg(1)->getType()) &&
           "Dot product operands must have the same type.");

    auto *VecTy0 = E->getArg(0)->getType()->castAs<VectorType>();
    assert(VecTy0 && "Dot product argument must be a vector.");

    return Builder.CreateIntrinsic(
        /*ReturnType=*/T0->getScalarType(),
        getDotProductIntrinsic(CGM.getHLSLRuntime(), VecTy0->getElementType()),
        ArrayRef<Value *>{Op0, Op1}, nullptr, "hlsl.dot");
  }
  case Builtin::BI__builtin_hlsl_dot4add_i8packed: {
    Value *X = EmitScalarExpr(E->getArg(0));
    Value *Y = EmitScalarExpr(E->getArg(1));
    Value *Acc = EmitScalarExpr(E->getArg(2));

    Intrinsic::ID ID = CGM.getHLSLRuntime().getDot4AddI8PackedIntrinsic();
    // Note that the argument order disagrees between the builtin and the
    // intrinsic here.
    return Builder.CreateIntrinsic(
        /*ReturnType=*/Acc->getType(), ID, ArrayRef<Value *>{Acc, X, Y},
        nullptr, "hlsl.dot4add.i8packed");
  }
  case Builtin::BI__builtin_hlsl_dot4add_u8packed: {
    Value *X = EmitScalarExpr(E->getArg(0));
    Value *Y = EmitScalarExpr(E->getArg(1));
    Value *Acc = EmitScalarExpr(E->getArg(2));

    Intrinsic::ID ID = CGM.getHLSLRuntime().getDot4AddU8PackedIntrinsic();
    // Note that the argument order disagrees between the builtin and the
    // intrinsic here.
    return Builder.CreateIntrinsic(
        /*ReturnType=*/Acc->getType(), ID, ArrayRef<Value *>{Acc, X, Y},
        nullptr, "hlsl.dot4add.u8packed");
  }
  case Builtin::BI__builtin_hlsl_elementwise_firstbithigh: {
    Value *X = EmitScalarExpr(E->getArg(0));

    return Builder.CreateIntrinsic(
        /*ReturnType=*/ConvertType(E->getType()),
        getFirstBitHighIntrinsic(CGM.getHLSLRuntime(), E->getArg(0)->getType()),
        ArrayRef<Value *>{X}, nullptr, "hlsl.firstbithigh");
  }
  case Builtin::BI__builtin_hlsl_elementwise_firstbitlow: {
    Value *X = EmitScalarExpr(E->getArg(0));

    return Builder.CreateIntrinsic(
        /*ReturnType=*/ConvertType(E->getType()),
        CGM.getHLSLRuntime().getFirstBitLowIntrinsic(), ArrayRef<Value *>{X},
        nullptr, "hlsl.firstbitlow");
  }
  case Builtin::BI__builtin_hlsl_lerp: {
    Value *X = EmitScalarExpr(E->getArg(0));
    Value *Y = EmitScalarExpr(E->getArg(1));
    Value *S = EmitScalarExpr(E->getArg(2));
    if (!E->getArg(0)->getType()->hasFloatingRepresentation())
      llvm_unreachable("lerp operand must have a float representation");
    return Builder.CreateIntrinsic(
        /*ReturnType=*/X->getType(), CGM.getHLSLRuntime().getLerpIntrinsic(),
        ArrayRef<Value *>{X, Y, S}, nullptr, "hlsl.lerp");
  }
  case Builtin::BI__builtin_hlsl_normalize: {
    Value *X = EmitScalarExpr(E->getArg(0));

    assert(E->getArg(0)->getType()->hasFloatingRepresentation() &&
           "normalize operand must have a float representation");

    return Builder.CreateIntrinsic(
        /*ReturnType=*/X->getType(),
        CGM.getHLSLRuntime().getNormalizeIntrinsic(), ArrayRef<Value *>{X},
        nullptr, "hlsl.normalize");
  }
  case Builtin::BI__builtin_hlsl_elementwise_degrees: {
    Value *X = EmitScalarExpr(E->getArg(0));

    assert(E->getArg(0)->getType()->hasFloatingRepresentation() &&
           "degree operand must have a float representation");

    return Builder.CreateIntrinsic(
        /*ReturnType=*/X->getType(), CGM.getHLSLRuntime().getDegreesIntrinsic(),
        ArrayRef<Value *>{X}, nullptr, "hlsl.degrees");
  }
  case Builtin::BI__builtin_hlsl_elementwise_frac: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    if (!E->getArg(0)->getType()->hasFloatingRepresentation())
      llvm_unreachable("frac operand must have a float representation");
    return Builder.CreateIntrinsic(
        /*ReturnType=*/Op0->getType(), CGM.getHLSLRuntime().getFracIntrinsic(),
        ArrayRef<Value *>{Op0}, nullptr, "hlsl.frac");
  }
  case Builtin::BI__builtin_hlsl_elementwise_isinf: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    llvm::Type *Xty = Op0->getType();
    llvm::Type *retType = llvm::Type::getInt1Ty(this->getLLVMContext());
    if (Xty->isVectorTy()) {
      auto *XVecTy = E->getArg(0)->getType()->castAs<VectorType>();
      retType = llvm::VectorType::get(
          retType, ElementCount::getFixed(XVecTy->getNumElements()));
    }
    if (!E->getArg(0)->getType()->hasFloatingRepresentation())
      llvm_unreachable("isinf operand must have a float representation");
    return Builder.CreateIntrinsic(
        retType, CGM.getHLSLRuntime().getIsInfIntrinsic(),
        ArrayRef<Value *>{Op0}, nullptr, "hlsl.isinf");
  }
  case Builtin::BI__builtin_hlsl_mad: {
    Value *M = EmitScalarExpr(E->getArg(0));
    Value *A = EmitScalarExpr(E->getArg(1));
    Value *B = EmitScalarExpr(E->getArg(2));
    if (E->getArg(0)->getType()->hasFloatingRepresentation())
      return Builder.CreateIntrinsic(
          /*ReturnType*/ M->getType(), Intrinsic::fmuladd,
          ArrayRef<Value *>{M, A, B}, nullptr, "hlsl.fmad");

    if (E->getArg(0)->getType()->hasSignedIntegerRepresentation()) {
      if (CGM.getTarget().getTriple().getArch() == llvm::Triple::dxil)
        return Builder.CreateIntrinsic(
            /*ReturnType*/ M->getType(), Intrinsic::dx_imad,
            ArrayRef<Value *>{M, A, B}, nullptr, "dx.imad");

      Value *Mul = Builder.CreateNSWMul(M, A);
      return Builder.CreateNSWAdd(Mul, B);
    }
    assert(E->getArg(0)->getType()->hasUnsignedIntegerRepresentation());
    if (CGM.getTarget().getTriple().getArch() == llvm::Triple::dxil)
      return Builder.CreateIntrinsic(
          /*ReturnType=*/M->getType(), Intrinsic::dx_umad,
          ArrayRef<Value *>{M, A, B}, nullptr, "dx.umad");

    Value *Mul = Builder.CreateNUWMul(M, A);
    return Builder.CreateNUWAdd(Mul, B);
  }
  case Builtin::BI__builtin_hlsl_elementwise_rcp: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    if (!E->getArg(0)->getType()->hasFloatingRepresentation())
      llvm_unreachable("rcp operand must have a float representation");
    llvm::Type *Ty = Op0->getType();
    llvm::Type *EltTy = Ty->getScalarType();
    Constant *One = Ty->isVectorTy()
                        ? ConstantVector::getSplat(
                              ElementCount::getFixed(
                                  cast<FixedVectorType>(Ty)->getNumElements()),
                              ConstantFP::get(EltTy, 1.0))
                        : ConstantFP::get(EltTy, 1.0);
    return Builder.CreateFDiv(One, Op0, "hlsl.rcp");
  }
  case Builtin::BI__builtin_hlsl_elementwise_rsqrt: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    if (!E->getArg(0)->getType()->hasFloatingRepresentation())
      llvm_unreachable("rsqrt operand must have a float representation");
    return Builder.CreateIntrinsic(
        /*ReturnType=*/Op0->getType(), CGM.getHLSLRuntime().getRsqrtIntrinsic(),
        ArrayRef<Value *>{Op0}, nullptr, "hlsl.rsqrt");
  }
  case Builtin::BI__builtin_hlsl_elementwise_saturate: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    assert(E->getArg(0)->getType()->hasFloatingRepresentation() &&
           "saturate operand must have a float representation");
    return Builder.CreateIntrinsic(
        /*ReturnType=*/Op0->getType(),
        CGM.getHLSLRuntime().getSaturateIntrinsic(), ArrayRef<Value *>{Op0},
        nullptr, "hlsl.saturate");
  }
  case Builtin::BI__builtin_hlsl_select: {
    Value *OpCond = EmitScalarExpr(E->getArg(0));
    RValue RValTrue = EmitAnyExpr(E->getArg(1));
    Value *OpTrue =
        RValTrue.isScalar()
            ? RValTrue.getScalarVal()
            : RValTrue.getAggregatePointer(E->getArg(1)->getType(), *this);
    RValue RValFalse = EmitAnyExpr(E->getArg(2));
    Value *OpFalse =
        RValFalse.isScalar()
            ? RValFalse.getScalarVal()
            : RValFalse.getAggregatePointer(E->getArg(2)->getType(), *this);
    if (auto *VTy = E->getType()->getAs<VectorType>()) {
      if (!OpTrue->getType()->isVectorTy())
        OpTrue =
            Builder.CreateVectorSplat(VTy->getNumElements(), OpTrue, "splat");
      if (!OpFalse->getType()->isVectorTy())
        OpFalse =
            Builder.CreateVectorSplat(VTy->getNumElements(), OpFalse, "splat");
    }

    Value *SelectVal =
        Builder.CreateSelect(OpCond, OpTrue, OpFalse, "hlsl.select");
    if (!RValTrue.isScalar())
      Builder.CreateStore(SelectVal, ReturnValue.getAddress(),
                          ReturnValue.isVolatile());

    return SelectVal;
  }
  case Builtin::BI__builtin_hlsl_step: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    assert(E->getArg(0)->getType()->hasFloatingRepresentation() &&
           E->getArg(1)->getType()->hasFloatingRepresentation() &&
           "step operands must have a float representation");
    return Builder.CreateIntrinsic(
        /*ReturnType=*/Op0->getType(), CGM.getHLSLRuntime().getStepIntrinsic(),
        ArrayRef<Value *>{Op0, Op1}, nullptr, "hlsl.step");
  }
  case Builtin::BI__builtin_hlsl_wave_active_all_true: {
    Value *Op = EmitScalarExpr(E->getArg(0));
    assert(Op->getType()->isIntegerTy(1) &&
           "Intrinsic WaveActiveAllTrue operand must be a bool");

    Intrinsic::ID ID = CGM.getHLSLRuntime().getWaveActiveAllTrueIntrinsic();
    return EmitRuntimeCall(
        Intrinsic::getOrInsertDeclaration(&CGM.getModule(), ID), {Op});
  }
  case Builtin::BI__builtin_hlsl_wave_active_any_true: {
    Value *Op = EmitScalarExpr(E->getArg(0));
    assert(Op->getType()->isIntegerTy(1) &&
           "Intrinsic WaveActiveAnyTrue operand must be a bool");

    Intrinsic::ID ID = CGM.getHLSLRuntime().getWaveActiveAnyTrueIntrinsic();
    return EmitRuntimeCall(
        Intrinsic::getOrInsertDeclaration(&CGM.getModule(), ID), {Op});
  }
  case Builtin::BI__builtin_hlsl_wave_active_count_bits: {
    Value *OpExpr = EmitScalarExpr(E->getArg(0));
    Intrinsic::ID ID = CGM.getHLSLRuntime().getWaveActiveCountBitsIntrinsic();
    return EmitRuntimeCall(
        Intrinsic::getOrInsertDeclaration(&CGM.getModule(), ID),
        ArrayRef{OpExpr});
  }
  case Builtin::BI__builtin_hlsl_wave_active_sum: {
    // Due to the use of variadic arguments, explicitly retreive argument
    Value *OpExpr = EmitScalarExpr(E->getArg(0));
    Intrinsic::ID IID = getWaveActiveSumIntrinsic(
        getTarget().getTriple().getArch(), CGM.getHLSLRuntime(),
        E->getArg(0)->getType());

    return EmitRuntimeCall(Intrinsic::getOrInsertDeclaration(
                               &CGM.getModule(), IID, {OpExpr->getType()}),
                           ArrayRef{OpExpr}, "hlsl.wave.active.sum");
  }
  case Builtin::BI__builtin_hlsl_wave_active_max: {
    // Due to the use of variadic arguments, explicitly retreive argument
    Value *OpExpr = EmitScalarExpr(E->getArg(0));
    Intrinsic::ID IID = getWaveActiveMaxIntrinsic(
        getTarget().getTriple().getArch(), CGM.getHLSLRuntime(),
        E->getArg(0)->getType());

    return EmitRuntimeCall(Intrinsic::getOrInsertDeclaration(
                               &CGM.getModule(), IID, {OpExpr->getType()}),
                           ArrayRef{OpExpr}, "hlsl.wave.active.max");
  }
  case Builtin::BI__builtin_hlsl_wave_get_lane_index: {
    // We don't define a SPIR-V intrinsic, instead it is a SPIR-V built-in
    // defined in SPIRVBuiltins.td. So instead we manually get the matching name
    // for the DirectX intrinsic and the demangled builtin name
    switch (CGM.getTarget().getTriple().getArch()) {
    case llvm::Triple::dxil:
      return EmitRuntimeCall(Intrinsic::getOrInsertDeclaration(
          &CGM.getModule(), Intrinsic::dx_wave_getlaneindex));
    case llvm::Triple::spirv:
      return EmitRuntimeCall(CGM.CreateRuntimeFunction(
          llvm::FunctionType::get(IntTy, {}, false),
          "__hlsl_wave_get_lane_index", {}, false, true));
    default:
      llvm_unreachable(
          "Intrinsic WaveGetLaneIndex not supported by target architecture");
    }
  }
  case Builtin::BI__builtin_hlsl_wave_is_first_lane: {
    Intrinsic::ID ID = CGM.getHLSLRuntime().getWaveIsFirstLaneIntrinsic();
    return EmitRuntimeCall(
        Intrinsic::getOrInsertDeclaration(&CGM.getModule(), ID));
  }
  case Builtin::BI__builtin_hlsl_wave_get_lane_count: {
    Intrinsic::ID ID = CGM.getHLSLRuntime().getWaveGetLaneCountIntrinsic();
    return EmitRuntimeCall(
        Intrinsic::getOrInsertDeclaration(&CGM.getModule(), ID));
  }
  case Builtin::BI__builtin_hlsl_wave_read_lane_at: {
    // Due to the use of variadic arguments we must explicitly retreive them and
    // create our function type.
    Value *OpExpr = EmitScalarExpr(E->getArg(0));
    Value *OpIndex = EmitScalarExpr(E->getArg(1));
    return EmitRuntimeCall(
        Intrinsic::getOrInsertDeclaration(
            &CGM.getModule(), CGM.getHLSLRuntime().getWaveReadLaneAtIntrinsic(),
            {OpExpr->getType()}),
        ArrayRef{OpExpr, OpIndex}, "hlsl.wave.readlane");
  }
  case Builtin::BI__builtin_hlsl_elementwise_sign: {
    auto *Arg0 = E->getArg(0);
    Value *Op0 = EmitScalarExpr(Arg0);
    llvm::Type *Xty = Op0->getType();
    llvm::Type *retType = llvm::Type::getInt32Ty(this->getLLVMContext());
    if (Xty->isVectorTy()) {
      auto *XVecTy = Arg0->getType()->castAs<VectorType>();
      retType = llvm::VectorType::get(
          retType, ElementCount::getFixed(XVecTy->getNumElements()));
    }
    assert((Arg0->getType()->hasFloatingRepresentation() ||
            Arg0->getType()->hasIntegerRepresentation()) &&
           "sign operand must have a float or int representation");

    if (Arg0->getType()->hasUnsignedIntegerRepresentation()) {
      Value *Cmp = Builder.CreateICmpEQ(Op0, ConstantInt::get(Xty, 0));
      return Builder.CreateSelect(Cmp, ConstantInt::get(retType, 0),
                                  ConstantInt::get(retType, 1), "hlsl.sign");
    }

    return Builder.CreateIntrinsic(
        retType, CGM.getHLSLRuntime().getSignIntrinsic(),
        ArrayRef<Value *>{Op0}, nullptr, "hlsl.sign");
  }
  case Builtin::BI__builtin_hlsl_elementwise_radians: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    assert(E->getArg(0)->getType()->hasFloatingRepresentation() &&
           "radians operand must have a float representation");
    return Builder.CreateIntrinsic(
        /*ReturnType=*/Op0->getType(),
        CGM.getHLSLRuntime().getRadiansIntrinsic(), ArrayRef<Value *>{Op0},
        nullptr, "hlsl.radians");
  }
  case Builtin::BI__builtin_hlsl_buffer_update_counter: {
    Value *ResHandle = EmitScalarExpr(E->getArg(0));
    Value *Offset = EmitScalarExpr(E->getArg(1));
    Value *OffsetI8 = Builder.CreateIntCast(Offset, Int8Ty, true);
    return Builder.CreateIntrinsic(
        /*ReturnType=*/Offset->getType(),
        CGM.getHLSLRuntime().getBufferUpdateCounterIntrinsic(),
        ArrayRef<Value *>{ResHandle, OffsetI8}, nullptr);
  }
  case Builtin::BI__builtin_hlsl_elementwise_splitdouble: {

    assert((E->getArg(0)->getType()->hasFloatingRepresentation() &&
            E->getArg(1)->getType()->hasUnsignedIntegerRepresentation() &&
            E->getArg(2)->getType()->hasUnsignedIntegerRepresentation()) &&
           "asuint operands types mismatch");
    return handleHlslSplitdouble(E, this);
  }
  case Builtin::BI__builtin_hlsl_elementwise_clip:
    assert(E->getArg(0)->getType()->hasFloatingRepresentation() &&
           "clip operands types mismatch");
    return handleHlslClip(E, this);
  case Builtin::BI__builtin_hlsl_group_memory_barrier_with_group_sync: {
    Intrinsic::ID ID =
        CGM.getHLSLRuntime().getGroupMemoryBarrierWithGroupSyncIntrinsic();
    return EmitRuntimeCall(
        Intrinsic::getOrInsertDeclaration(&CGM.getModule(), ID));
  }
  case Builtin::BI__builtin_get_spirv_spec_constant_bool:
  case Builtin::BI__builtin_get_spirv_spec_constant_short:
  case Builtin::BI__builtin_get_spirv_spec_constant_ushort:
  case Builtin::BI__builtin_get_spirv_spec_constant_int:
  case Builtin::BI__builtin_get_spirv_spec_constant_uint:
  case Builtin::BI__builtin_get_spirv_spec_constant_longlong:
  case Builtin::BI__builtin_get_spirv_spec_constant_ulonglong:
  case Builtin::BI__builtin_get_spirv_spec_constant_half:
  case Builtin::BI__builtin_get_spirv_spec_constant_float:
  case Builtin::BI__builtin_get_spirv_spec_constant_double: {
    llvm::Function *SpecConstantFn = getSpecConstantFunction(E->getType());
    llvm::Value *SpecId = EmitScalarExpr(E->getArg(0));
    llvm::Value *DefaultVal = EmitScalarExpr(E->getArg(1));
    llvm::Value *Args[] = {SpecId, DefaultVal};
    return Builder.CreateCall(SpecConstantFn, Args);
  }
  }
  return nullptr;
}

llvm::Function *clang::CodeGen::CodeGenFunction::getSpecConstantFunction(
    const clang::QualType &SpecConstantType) {

  // Find or create the declaration for the function.
  llvm::Module *M = &CGM.getModule();
  std::string MangledName =
      getSpecConstantFunctionName(SpecConstantType, getContext());
  llvm::Function *SpecConstantFn = M->getFunction(MangledName);

  if (!SpecConstantFn) {
    llvm::Type *IntType = ConvertType(getContext().IntTy);
    llvm::Type *RetTy = ConvertType(SpecConstantType);
    llvm::Type *ArgTypes[] = {IntType, RetTy};
    llvm::FunctionType *FnTy = llvm::FunctionType::get(RetTy, ArgTypes, false);
    SpecConstantFn = llvm::Function::Create(
        FnTy, llvm::GlobalValue::ExternalLinkage, MangledName, M);
  }
  return SpecConstantFn;
}
