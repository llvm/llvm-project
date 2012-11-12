//===---- CGBuiltin.cpp - Emit LLVM Code for builtins ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Builtin calls as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "TargetInfo.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "CGObjCRuntime.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/Intrinsics.h"
#include "llvm/DataLayout.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm;

/// getBuiltinLibFunction - Given a builtin id for a function like
/// "__builtin_fabsf", return a Function* for "fabsf".
llvm::Value *CodeGenModule::getBuiltinLibFunction(const FunctionDecl *FD,
                                                  unsigned BuiltinID) {
  assert(Context.BuiltinInfo.isLibFunction(BuiltinID));

  // Get the name, skip over the __builtin_ prefix (if necessary).
  StringRef Name;
  GlobalDecl D(FD);

  // If the builtin has been declared explicitly with an assembler label,
  // use the mangled name. This differs from the plain label on platforms
  // that prefix labels.
  if (FD->hasAttr<AsmLabelAttr>())
    Name = getMangledName(D);
  else
    Name = Context.BuiltinInfo.GetName(BuiltinID) + 10;

  llvm::FunctionType *Ty =
    cast<llvm::FunctionType>(getTypes().ConvertType(FD->getType()));

  return GetOrCreateLLVMFunction(Name, Ty, D, /*ForVTable=*/false);
}

/// Emit the conversions required to turn the given value into an
/// integer of the given size.
static Value *EmitToInt(CodeGenFunction &CGF, llvm::Value *V,
                        QualType T, llvm::IntegerType *IntType) {
  V = CGF.EmitToMemory(V, T);

  if (V->getType()->isPointerTy())
    return CGF.Builder.CreatePtrToInt(V, IntType);

  assert(V->getType() == IntType);
  return V;
}

static Value *EmitFromInt(CodeGenFunction &CGF, llvm::Value *V,
                          QualType T, llvm::Type *ResultType) {
  V = CGF.EmitFromMemory(V, T);

  if (ResultType->isPointerTy())
    return CGF.Builder.CreateIntToPtr(V, ResultType);

  assert(V->getType() == ResultType);
  return V;
}

/// Utility to insert an atomic instruction based on Instrinsic::ID
/// and the expression node.
static RValue EmitBinaryAtomic(CodeGenFunction &CGF,
                               llvm::AtomicRMWInst::BinOp Kind,
                               const CallExpr *E) {
  QualType T = E->getType();
  assert(E->getArg(0)->getType()->isPointerType());
  assert(CGF.getContext().hasSameUnqualifiedType(T,
                                  E->getArg(0)->getType()->getPointeeType()));
  assert(CGF.getContext().hasSameUnqualifiedType(T, E->getArg(1)->getType()));

  llvm::Value *DestPtr = CGF.EmitScalarExpr(E->getArg(0));
  unsigned AddrSpace = DestPtr->getType()->getPointerAddressSpace();

  llvm::IntegerType *IntType =
    llvm::IntegerType::get(CGF.getLLVMContext(),
                           CGF.getContext().getTypeSize(T));
  llvm::Type *IntPtrType = IntType->getPointerTo(AddrSpace);

  llvm::Value *Args[2];
  Args[0] = CGF.Builder.CreateBitCast(DestPtr, IntPtrType);
  Args[1] = CGF.EmitScalarExpr(E->getArg(1));
  llvm::Type *ValueType = Args[1]->getType();
  Args[1] = EmitToInt(CGF, Args[1], T, IntType);

  llvm::Value *Result =
      CGF.Builder.CreateAtomicRMW(Kind, Args[0], Args[1],
                                  llvm::SequentiallyConsistent);
  Result = EmitFromInt(CGF, Result, T, ValueType);
  return RValue::get(Result);
}

/// Utility to insert an atomic instruction based Instrinsic::ID and
/// the expression node, where the return value is the result of the
/// operation.
static RValue EmitBinaryAtomicPost(CodeGenFunction &CGF,
                                   llvm::AtomicRMWInst::BinOp Kind,
                                   const CallExpr *E,
                                   Instruction::BinaryOps Op) {
  QualType T = E->getType();
  assert(E->getArg(0)->getType()->isPointerType());
  assert(CGF.getContext().hasSameUnqualifiedType(T,
                                  E->getArg(0)->getType()->getPointeeType()));
  assert(CGF.getContext().hasSameUnqualifiedType(T, E->getArg(1)->getType()));

  llvm::Value *DestPtr = CGF.EmitScalarExpr(E->getArg(0));
  unsigned AddrSpace = DestPtr->getType()->getPointerAddressSpace();

  llvm::IntegerType *IntType =
    llvm::IntegerType::get(CGF.getLLVMContext(),
                           CGF.getContext().getTypeSize(T));
  llvm::Type *IntPtrType = IntType->getPointerTo(AddrSpace);

  llvm::Value *Args[2];
  Args[1] = CGF.EmitScalarExpr(E->getArg(1));
  llvm::Type *ValueType = Args[1]->getType();
  Args[1] = EmitToInt(CGF, Args[1], T, IntType);
  Args[0] = CGF.Builder.CreateBitCast(DestPtr, IntPtrType);

  llvm::Value *Result =
      CGF.Builder.CreateAtomicRMW(Kind, Args[0], Args[1],
                                  llvm::SequentiallyConsistent);
  Result = CGF.Builder.CreateBinOp(Op, Result, Args[1]);
  Result = EmitFromInt(CGF, Result, T, ValueType);
  return RValue::get(Result);
}

/// EmitFAbs - Emit a call to fabs/fabsf/fabsl, depending on the type of ValTy,
/// which must be a scalar floating point type.
static Value *EmitFAbs(CodeGenFunction &CGF, Value *V, QualType ValTy) {
  const BuiltinType *ValTyP = ValTy->getAs<BuiltinType>();
  assert(ValTyP && "isn't scalar fp type!");

  StringRef FnName;
  switch (ValTyP->getKind()) {
  default: llvm_unreachable("Isn't a scalar fp type!");
  case BuiltinType::Float:      FnName = "fabsf"; break;
  case BuiltinType::Double:     FnName = "fabs"; break;
  case BuiltinType::LongDouble: FnName = "fabsl"; break;
  }

  // The prototype is something that takes and returns whatever V's type is.
  llvm::FunctionType *FT = llvm::FunctionType::get(V->getType(), V->getType(),
                                                   false);
  llvm::Value *Fn = CGF.CGM.CreateRuntimeFunction(FT, FnName);

  return CGF.Builder.CreateCall(Fn, V, "abs");
}

static RValue emitLibraryCall(CodeGenFunction &CGF, const FunctionDecl *Fn,
                              const CallExpr *E, llvm::Value *calleeValue) {
  return CGF.EmitCall(E->getCallee()->getType(), calleeValue,
                      ReturnValueSlot(), E->arg_begin(), E->arg_end(), Fn);
}

RValue CodeGenFunction::EmitBuiltinExpr(const FunctionDecl *FD,
                                        unsigned BuiltinID, const CallExpr *E) {
  // See if we can constant fold this builtin.  If so, don't emit it at all.
  Expr::EvalResult Result;
  if (E->EvaluateAsRValue(Result, CGM.getContext()) &&
      !Result.hasSideEffects()) {
    if (Result.Val.isInt())
      return RValue::get(llvm::ConstantInt::get(getLLVMContext(),
                                                Result.Val.getInt()));
    if (Result.Val.isFloat())
      return RValue::get(llvm::ConstantFP::get(getLLVMContext(),
                                               Result.Val.getFloat()));
  }

  switch (BuiltinID) {
  default: break;  // Handle intrinsics and libm functions below.
  case Builtin::BI__builtin___CFStringMakeConstantString:
  case Builtin::BI__builtin___NSStringMakeConstantString:
    return RValue::get(CGM.EmitConstantExpr(E, E->getType(), 0));
  case Builtin::BI__builtin_stdarg_start:
  case Builtin::BI__builtin_va_start:
  case Builtin::BI__builtin_va_end: {
    Value *ArgValue = EmitVAListRef(E->getArg(0));
    llvm::Type *DestType = Int8PtrTy;
    if (ArgValue->getType() != DestType)
      ArgValue = Builder.CreateBitCast(ArgValue, DestType,
                                       ArgValue->getName().data());

    Intrinsic::ID inst = (BuiltinID == Builtin::BI__builtin_va_end) ?
      Intrinsic::vaend : Intrinsic::vastart;
    return RValue::get(Builder.CreateCall(CGM.getIntrinsic(inst), ArgValue));
  }
  case Builtin::BI__builtin_va_copy: {
    Value *DstPtr = EmitVAListRef(E->getArg(0));
    Value *SrcPtr = EmitVAListRef(E->getArg(1));

    llvm::Type *Type = Int8PtrTy;

    DstPtr = Builder.CreateBitCast(DstPtr, Type);
    SrcPtr = Builder.CreateBitCast(SrcPtr, Type);
    return RValue::get(Builder.CreateCall2(CGM.getIntrinsic(Intrinsic::vacopy),
                                           DstPtr, SrcPtr));
  }
  case Builtin::BI__builtin_abs:
  case Builtin::BI__builtin_labs:
  case Builtin::BI__builtin_llabs: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));

    Value *NegOp = Builder.CreateNeg(ArgValue, "neg");
    Value *CmpResult =
    Builder.CreateICmpSGE(ArgValue,
                          llvm::Constant::getNullValue(ArgValue->getType()),
                                                            "abscond");
    Value *Result =
      Builder.CreateSelect(CmpResult, ArgValue, NegOp, "abs");

    return RValue::get(Result);
  }

  case Builtin::BI__builtin_conj:
  case Builtin::BI__builtin_conjf:
  case Builtin::BI__builtin_conjl: {
    ComplexPairTy ComplexVal = EmitComplexExpr(E->getArg(0));
    Value *Real = ComplexVal.first;
    Value *Imag = ComplexVal.second;
    Value *Zero =
      Imag->getType()->isFPOrFPVectorTy()
        ? llvm::ConstantFP::getZeroValueForNegation(Imag->getType())
        : llvm::Constant::getNullValue(Imag->getType());

    Imag = Builder.CreateFSub(Zero, Imag, "sub");
    return RValue::getComplex(std::make_pair(Real, Imag));
  }
  case Builtin::BI__builtin_creal:
  case Builtin::BI__builtin_crealf:
  case Builtin::BI__builtin_creall: {
    ComplexPairTy ComplexVal = EmitComplexExpr(E->getArg(0));
    return RValue::get(ComplexVal.first);
  }

  case Builtin::BI__builtin_cimag:
  case Builtin::BI__builtin_cimagf:
  case Builtin::BI__builtin_cimagl: {
    ComplexPairTy ComplexVal = EmitComplexExpr(E->getArg(0));
    return RValue::get(ComplexVal.second);
  }

  case Builtin::BI__builtin_ctzs:
  case Builtin::BI__builtin_ctz:
  case Builtin::BI__builtin_ctzl:
  case Builtin::BI__builtin_ctzll: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));

    llvm::Type *ArgType = ArgValue->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::cttz, ArgType);

    llvm::Type *ResultType = ConvertType(E->getType());
    Value *ZeroUndef = Builder.getInt1(Target.isCLZForZeroUndef());
    Value *Result = Builder.CreateCall2(F, ArgValue, ZeroUndef);
    if (Result->getType() != ResultType)
      Result = Builder.CreateIntCast(Result, ResultType, /*isSigned*/true,
                                     "cast");
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_clzs:
  case Builtin::BI__builtin_clz:
  case Builtin::BI__builtin_clzl:
  case Builtin::BI__builtin_clzll: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));

    llvm::Type *ArgType = ArgValue->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::ctlz, ArgType);

    llvm::Type *ResultType = ConvertType(E->getType());
    Value *ZeroUndef = Builder.getInt1(Target.isCLZForZeroUndef());
    Value *Result = Builder.CreateCall2(F, ArgValue, ZeroUndef);
    if (Result->getType() != ResultType)
      Result = Builder.CreateIntCast(Result, ResultType, /*isSigned*/true,
                                     "cast");
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_ffs:
  case Builtin::BI__builtin_ffsl:
  case Builtin::BI__builtin_ffsll: {
    // ffs(x) -> x ? cttz(x) + 1 : 0
    Value *ArgValue = EmitScalarExpr(E->getArg(0));

    llvm::Type *ArgType = ArgValue->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::cttz, ArgType);

    llvm::Type *ResultType = ConvertType(E->getType());
    Value *Tmp = Builder.CreateAdd(Builder.CreateCall2(F, ArgValue,
                                                       Builder.getTrue()),
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
    Value *F = CGM.getIntrinsic(Intrinsic::ctpop, ArgType);

    llvm::Type *ResultType = ConvertType(E->getType());
    Value *Tmp = Builder.CreateCall(F, ArgValue);
    Value *Result = Builder.CreateAnd(Tmp, llvm::ConstantInt::get(ArgType, 1));
    if (Result->getType() != ResultType)
      Result = Builder.CreateIntCast(Result, ResultType, /*isSigned*/true,
                                     "cast");
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_popcount:
  case Builtin::BI__builtin_popcountl:
  case Builtin::BI__builtin_popcountll: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));

    llvm::Type *ArgType = ArgValue->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::ctpop, ArgType);

    llvm::Type *ResultType = ConvertType(E->getType());
    Value *Result = Builder.CreateCall(F, ArgValue);
    if (Result->getType() != ResultType)
      Result = Builder.CreateIntCast(Result, ResultType, /*isSigned*/true,
                                     "cast");
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_expect: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));
    llvm::Type *ArgType = ArgValue->getType();

    Value *FnExpect = CGM.getIntrinsic(Intrinsic::expect, ArgType);
    Value *ExpectedValue = EmitScalarExpr(E->getArg(1));

    Value *Result = Builder.CreateCall2(FnExpect, ArgValue, ExpectedValue,
                                        "expval");
    return RValue::get(Result);
  }
  case Builtin::BI__builtin_bswap16:
  case Builtin::BI__builtin_bswap32:
  case Builtin::BI__builtin_bswap64: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));
    llvm::Type *ArgType = ArgValue->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::bswap, ArgType);
    return RValue::get(Builder.CreateCall(F, ArgValue));
  }
  case Builtin::BI__builtin_object_size: {
    // We rely on constant folding to deal with expressions with side effects.
    assert(!E->getArg(0)->HasSideEffects(getContext()) &&
           "should have been constant folded");

    // We pass this builtin onto the optimizer so that it can
    // figure out the object size in more complex cases.
    llvm::Type *ResType = ConvertType(E->getType());

    // LLVM only supports 0 and 2, make sure that we pass along that
    // as a boolean.
    Value *Ty = EmitScalarExpr(E->getArg(1));
    ConstantInt *CI = dyn_cast<ConstantInt>(Ty);
    assert(CI);
    uint64_t val = CI->getZExtValue();
    CI = ConstantInt::get(Builder.getInt1Ty(), (val & 0x2) >> 1);

    Value *F = CGM.getIntrinsic(Intrinsic::objectsize, ResType);
    return RValue::get(Builder.CreateCall2(F, EmitScalarExpr(E->getArg(0)),CI));
  }
  case Builtin::BI__builtin_prefetch: {
    Value *Locality, *RW, *Address = EmitScalarExpr(E->getArg(0));
    // FIXME: Technically these constants should of type 'int', yes?
    RW = (E->getNumArgs() > 1) ? EmitScalarExpr(E->getArg(1)) :
      llvm::ConstantInt::get(Int32Ty, 0);
    Locality = (E->getNumArgs() > 2) ? EmitScalarExpr(E->getArg(2)) :
      llvm::ConstantInt::get(Int32Ty, 3);
    Value *Data = llvm::ConstantInt::get(Int32Ty, 1);
    Value *F = CGM.getIntrinsic(Intrinsic::prefetch);
    return RValue::get(Builder.CreateCall4(F, Address, RW, Locality, Data));
  }
  case Builtin::BI__builtin_readcyclecounter: {
    Value *F = CGM.getIntrinsic(Intrinsic::readcyclecounter);
    return RValue::get(Builder.CreateCall(F));
  }
  case Builtin::BI__builtin_trap: {
    Value *F = CGM.getIntrinsic(Intrinsic::trap);
    return RValue::get(Builder.CreateCall(F));
  }
  case Builtin::BI__debugbreak: {
    Value *F = CGM.getIntrinsic(Intrinsic::debugtrap);
    return RValue::get(Builder.CreateCall(F));
  }
  case Builtin::BI__builtin_unreachable: {
    if (getLangOpts().SanitizeUnreachable)
      EmitCheck(Builder.getFalse(), "builtin_unreachable",
                EmitCheckSourceLocation(E->getExprLoc()),
                llvm::ArrayRef<llvm::Value *>());
    else
      Builder.CreateUnreachable();

    // We do need to preserve an insertion point.
    EmitBlock(createBasicBlock("unreachable.cont"));

    return RValue::get(0);
  }

  case Builtin::BI__builtin_powi:
  case Builtin::BI__builtin_powif:
  case Builtin::BI__builtin_powil: {
    Value *Base = EmitScalarExpr(E->getArg(0));
    Value *Exponent = EmitScalarExpr(E->getArg(1));
    llvm::Type *ArgType = Base->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::powi, ArgType);
    return RValue::get(Builder.CreateCall2(F, Base, Exponent));
  }

  case Builtin::BI__builtin_isgreater:
  case Builtin::BI__builtin_isgreaterequal:
  case Builtin::BI__builtin_isless:
  case Builtin::BI__builtin_islessequal:
  case Builtin::BI__builtin_islessgreater:
  case Builtin::BI__builtin_isunordered: {
    // Ordered comparisons: we know the arguments to these are matching scalar
    // floating point values.
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
    Value *V = EmitScalarExpr(E->getArg(0));
    V = Builder.CreateFCmpUNO(V, V, "cmp");
    return RValue::get(Builder.CreateZExt(V, ConvertType(E->getType())));
  }

  case Builtin::BI__builtin_isinf: {
    // isinf(x) --> fabs(x) == infinity
    Value *V = EmitScalarExpr(E->getArg(0));
    V = EmitFAbs(*this, V, E->getArg(0)->getType());

    V = Builder.CreateFCmpOEQ(V, ConstantFP::getInfinity(V->getType()),"isinf");
    return RValue::get(Builder.CreateZExt(V, ConvertType(E->getType())));
  }

  // TODO: BI__builtin_isinf_sign
  //   isinf_sign(x) -> isinf(x) ? (signbit(x) ? -1 : 1) : 0

  case Builtin::BI__builtin_isnormal: {
    // isnormal(x) --> x == x && fabsf(x) < infinity && fabsf(x) >= float_min
    Value *V = EmitScalarExpr(E->getArg(0));
    Value *Eq = Builder.CreateFCmpOEQ(V, V, "iseq");

    Value *Abs = EmitFAbs(*this, V, E->getArg(0)->getType());
    Value *IsLessThanInf =
      Builder.CreateFCmpULT(Abs, ConstantFP::getInfinity(V->getType()),"isinf");
    APFloat Smallest = APFloat::getSmallestNormalized(
                   getContext().getFloatTypeSemantics(E->getArg(0)->getType()));
    Value *IsNormal =
      Builder.CreateFCmpUGE(Abs, ConstantFP::get(V->getContext(), Smallest),
                            "isnormal");
    V = Builder.CreateAnd(Eq, IsLessThanInf, "and");
    V = Builder.CreateAnd(V, IsNormal, "and");
    return RValue::get(Builder.CreateZExt(V, ConvertType(E->getType())));
  }

  case Builtin::BI__builtin_isfinite: {
    // isfinite(x) --> x == x && fabs(x) != infinity;
    Value *V = EmitScalarExpr(E->getArg(0));
    Value *Eq = Builder.CreateFCmpOEQ(V, V, "iseq");

    Value *Abs = EmitFAbs(*this, V, E->getArg(0)->getType());
    Value *IsNotInf =
      Builder.CreateFCmpUNE(Abs, ConstantFP::getInfinity(V->getType()),"isinf");

    V = Builder.CreateAnd(Eq, IsNotInf, "and");
    return RValue::get(Builder.CreateZExt(V, ConvertType(E->getType())));
  }

  case Builtin::BI__builtin_fpclassify: {
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
    Value *VAbs = EmitFAbs(*this, V, E->getArg(5)->getType());
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

  case Builtin::BIalloca:
  case Builtin::BI__builtin_alloca: {
    Value *Size = EmitScalarExpr(E->getArg(0));
    return RValue::get(Builder.CreateAlloca(Builder.getInt8Ty(), Size));
  }
  case Builtin::BIbzero:
  case Builtin::BI__builtin_bzero: {
    std::pair<llvm::Value*, unsigned> Dest =
        EmitPointerWithAlignment(E->getArg(0));
    Value *SizeVal = EmitScalarExpr(E->getArg(1));
    Builder.CreateMemSet(Dest.first, Builder.getInt8(0), SizeVal,
                         Dest.second, false);
    return RValue::get(Dest.first);
  }
  case Builtin::BImemcpy:
  case Builtin::BI__builtin_memcpy: {
    std::pair<llvm::Value*, unsigned> Dest =
        EmitPointerWithAlignment(E->getArg(0));
    std::pair<llvm::Value*, unsigned> Src =
        EmitPointerWithAlignment(E->getArg(1));
    Value *SizeVal = EmitScalarExpr(E->getArg(2));
    unsigned Align = std::min(Dest.second, Src.second);
    Builder.CreateMemCpy(Dest.first, Src.first, SizeVal, Align, false);
    return RValue::get(Dest.first);
  }

  case Builtin::BI__builtin___memcpy_chk: {
    // fold __builtin_memcpy_chk(x, y, cst1, cst2) to memcpy iff cst1<=cst2.
    llvm::APSInt Size, DstSize;
    if (!E->getArg(2)->EvaluateAsInt(Size, CGM.getContext()) ||
        !E->getArg(3)->EvaluateAsInt(DstSize, CGM.getContext()))
      break;
    if (Size.ugt(DstSize))
      break;
    std::pair<llvm::Value*, unsigned> Dest =
        EmitPointerWithAlignment(E->getArg(0));
    std::pair<llvm::Value*, unsigned> Src =
        EmitPointerWithAlignment(E->getArg(1));
    Value *SizeVal = llvm::ConstantInt::get(Builder.getContext(), Size);
    unsigned Align = std::min(Dest.second, Src.second);
    Builder.CreateMemCpy(Dest.first, Src.first, SizeVal, Align, false);
    return RValue::get(Dest.first);
  }

  case Builtin::BI__builtin_objc_memmove_collectable: {
    Value *Address = EmitScalarExpr(E->getArg(0));
    Value *SrcAddr = EmitScalarExpr(E->getArg(1));
    Value *SizeVal = EmitScalarExpr(E->getArg(2));
    CGM.getObjCRuntime().EmitGCMemmoveCollectable(*this,
                                                  Address, SrcAddr, SizeVal);
    return RValue::get(Address);
  }

  case Builtin::BI__builtin___memmove_chk: {
    // fold __builtin_memmove_chk(x, y, cst1, cst2) to memmove iff cst1<=cst2.
    llvm::APSInt Size, DstSize;
    if (!E->getArg(2)->EvaluateAsInt(Size, CGM.getContext()) ||
        !E->getArg(3)->EvaluateAsInt(DstSize, CGM.getContext()))
      break;
    if (Size.ugt(DstSize))
      break;
    std::pair<llvm::Value*, unsigned> Dest =
        EmitPointerWithAlignment(E->getArg(0));
    std::pair<llvm::Value*, unsigned> Src =
        EmitPointerWithAlignment(E->getArg(1));
    Value *SizeVal = llvm::ConstantInt::get(Builder.getContext(), Size);
    unsigned Align = std::min(Dest.second, Src.second);
    Builder.CreateMemMove(Dest.first, Src.first, SizeVal, Align, false);
    return RValue::get(Dest.first);
  }

  case Builtin::BImemmove:
  case Builtin::BI__builtin_memmove: {
    std::pair<llvm::Value*, unsigned> Dest =
        EmitPointerWithAlignment(E->getArg(0));
    std::pair<llvm::Value*, unsigned> Src =
        EmitPointerWithAlignment(E->getArg(1));
    Value *SizeVal = EmitScalarExpr(E->getArg(2));
    unsigned Align = std::min(Dest.second, Src.second);
    Builder.CreateMemMove(Dest.first, Src.first, SizeVal, Align, false);
    return RValue::get(Dest.first);
  }
  case Builtin::BImemset:
  case Builtin::BI__builtin_memset: {
    std::pair<llvm::Value*, unsigned> Dest =
        EmitPointerWithAlignment(E->getArg(0));
    Value *ByteVal = Builder.CreateTrunc(EmitScalarExpr(E->getArg(1)),
                                         Builder.getInt8Ty());
    Value *SizeVal = EmitScalarExpr(E->getArg(2));
    Builder.CreateMemSet(Dest.first, ByteVal, SizeVal, Dest.second, false);
    return RValue::get(Dest.first);
  }
  case Builtin::BI__builtin___memset_chk: {
    // fold __builtin_memset_chk(x, y, cst1, cst2) to memset iff cst1<=cst2.
    llvm::APSInt Size, DstSize;
    if (!E->getArg(2)->EvaluateAsInt(Size, CGM.getContext()) ||
        !E->getArg(3)->EvaluateAsInt(DstSize, CGM.getContext()))
      break;
    if (Size.ugt(DstSize))
      break;
    std::pair<llvm::Value*, unsigned> Dest =
        EmitPointerWithAlignment(E->getArg(0));
    Value *ByteVal = Builder.CreateTrunc(EmitScalarExpr(E->getArg(1)),
                                         Builder.getInt8Ty());
    Value *SizeVal = llvm::ConstantInt::get(Builder.getContext(), Size);
    Builder.CreateMemSet(Dest.first, ByteVal, SizeVal, Dest.second, false);
    return RValue::get(Dest.first);
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

    Value *F = CGM.getIntrinsic(Intrinsic::eh_dwarf_cfa);
    return RValue::get(Builder.CreateCall(F,
                                      llvm::ConstantInt::get(Int32Ty, Offset)));
  }
  case Builtin::BI__builtin_return_address: {
    Value *Depth = EmitScalarExpr(E->getArg(0));
    Depth = Builder.CreateIntCast(Depth, Int32Ty, false);
    Value *F = CGM.getIntrinsic(Intrinsic::returnaddress);
    return RValue::get(Builder.CreateCall(F, Depth));
  }
  case Builtin::BI__builtin_frame_address: {
    Value *Depth = EmitScalarExpr(E->getArg(0));
    Depth = Builder.CreateIntCast(Depth, Int32Ty, false);
    Value *F = CGM.getIntrinsic(Intrinsic::frameaddress);
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
    Value *F = CGM.getIntrinsic(IntTy->getBitWidth() == 32
                                  ? Intrinsic::eh_return_i32
                                  : Intrinsic::eh_return_i64);
    Builder.CreateCall2(F, Int, Ptr);
    Builder.CreateUnreachable();

    // We do need to preserve an insertion point.
    EmitBlock(createBasicBlock("builtin_eh_return.cont"));

    return RValue::get(0);
  }
  case Builtin::BI__builtin_unwind_init: {
    Value *F = CGM.getIntrinsic(Intrinsic::eh_unwind_init);
    return RValue::get(Builder.CreateCall(F));
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
    Value *Buf = EmitScalarExpr(E->getArg(0));

    // Store the frame pointer to the setjmp buffer.
    Value *FrameAddr =
      Builder.CreateCall(CGM.getIntrinsic(Intrinsic::frameaddress),
                         ConstantInt::get(Int32Ty, 0));
    Builder.CreateStore(FrameAddr, Buf);

    // Store the stack pointer to the setjmp buffer.
    Value *StackAddr =
      Builder.CreateCall(CGM.getIntrinsic(Intrinsic::stacksave));
    Value *StackSaveSlot =
      Builder.CreateGEP(Buf, ConstantInt::get(Int32Ty, 2));
    Builder.CreateStore(StackAddr, StackSaveSlot);

    // Call LLVM's EH setjmp, which is lightweight.
    Value *F = CGM.getIntrinsic(Intrinsic::eh_sjlj_setjmp);
    Buf = Builder.CreateBitCast(Buf, Int8PtrTy);
    return RValue::get(Builder.CreateCall(F, Buf));
  }
  case Builtin::BI__builtin_longjmp: {
    Value *Buf = EmitScalarExpr(E->getArg(0));
    Buf = Builder.CreateBitCast(Buf, Int8PtrTy);

    // Call LLVM's EH longjmp, which is lightweight.
    Builder.CreateCall(CGM.getIntrinsic(Intrinsic::eh_sjlj_longjmp), Buf);

    // longjmp doesn't return; mark this as unreachable.
    Builder.CreateUnreachable();

    // We do need to preserve an insertion point.
    EmitBlock(createBasicBlock("longjmp.cont"));

    return RValue::get(0);
  }
  case Builtin::BI__sync_fetch_and_add:
  case Builtin::BI__sync_fetch_and_sub:
  case Builtin::BI__sync_fetch_and_or:
  case Builtin::BI__sync_fetch_and_and:
  case Builtin::BI__sync_fetch_and_xor:
  case Builtin::BI__sync_add_and_fetch:
  case Builtin::BI__sync_sub_and_fetch:
  case Builtin::BI__sync_and_and_fetch:
  case Builtin::BI__sync_or_and_fetch:
  case Builtin::BI__sync_xor_and_fetch:
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

  case Builtin::BI__sync_val_compare_and_swap_1:
  case Builtin::BI__sync_val_compare_and_swap_2:
  case Builtin::BI__sync_val_compare_and_swap_4:
  case Builtin::BI__sync_val_compare_and_swap_8:
  case Builtin::BI__sync_val_compare_and_swap_16: {
    QualType T = E->getType();
    llvm::Value *DestPtr = EmitScalarExpr(E->getArg(0));
    unsigned AddrSpace = DestPtr->getType()->getPointerAddressSpace();

    llvm::IntegerType *IntType =
      llvm::IntegerType::get(getLLVMContext(),
                             getContext().getTypeSize(T));
    llvm::Type *IntPtrType = IntType->getPointerTo(AddrSpace);

    Value *Args[3];
    Args[0] = Builder.CreateBitCast(DestPtr, IntPtrType);
    Args[1] = EmitScalarExpr(E->getArg(1));
    llvm::Type *ValueType = Args[1]->getType();
    Args[1] = EmitToInt(*this, Args[1], T, IntType);
    Args[2] = EmitToInt(*this, EmitScalarExpr(E->getArg(2)), T, IntType);

    Value *Result = Builder.CreateAtomicCmpXchg(Args[0], Args[1], Args[2],
                                                llvm::SequentiallyConsistent);
    Result = EmitFromInt(*this, Result, T, ValueType);
    return RValue::get(Result);
  }

  case Builtin::BI__sync_bool_compare_and_swap_1:
  case Builtin::BI__sync_bool_compare_and_swap_2:
  case Builtin::BI__sync_bool_compare_and_swap_4:
  case Builtin::BI__sync_bool_compare_and_swap_8:
  case Builtin::BI__sync_bool_compare_and_swap_16: {
    QualType T = E->getArg(1)->getType();
    llvm::Value *DestPtr = EmitScalarExpr(E->getArg(0));
    unsigned AddrSpace = DestPtr->getType()->getPointerAddressSpace();

    llvm::IntegerType *IntType =
      llvm::IntegerType::get(getLLVMContext(),
                             getContext().getTypeSize(T));
    llvm::Type *IntPtrType = IntType->getPointerTo(AddrSpace);

    Value *Args[3];
    Args[0] = Builder.CreateBitCast(DestPtr, IntPtrType);
    Args[1] = EmitToInt(*this, EmitScalarExpr(E->getArg(1)), T, IntType);
    Args[2] = EmitToInt(*this, EmitScalarExpr(E->getArg(2)), T, IntType);

    Value *OldVal = Args[1];
    Value *PrevVal = Builder.CreateAtomicCmpXchg(Args[0], Args[1], Args[2],
                                                 llvm::SequentiallyConsistent);
    Value *Result = Builder.CreateICmpEQ(PrevVal, OldVal);
    // zext bool to int.
    Result = Builder.CreateZExt(Result, ConvertType(E->getType()));
    return RValue::get(Result);
  }

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
    Value *Ptr = EmitScalarExpr(E->getArg(0));
    QualType ElTy = E->getArg(0)->getType()->getPointeeType();
    CharUnits StoreSize = getContext().getTypeSizeInChars(ElTy);
    llvm::Type *ITy = llvm::IntegerType::get(getLLVMContext(),
                                             StoreSize.getQuantity() * 8);
    Ptr = Builder.CreateBitCast(Ptr, ITy->getPointerTo());
    llvm::StoreInst *Store =
      Builder.CreateStore(llvm::Constant::getNullValue(ITy), Ptr);
    Store->setAlignment(StoreSize.getQuantity());
    Store->setAtomic(llvm::Release);
    return RValue::get(0);
  }

  case Builtin::BI__sync_synchronize: {
    // We assume this is supposed to correspond to a C++0x-style
    // sequentially-consistent fence (i.e. this is only usable for
    // synchonization, not device I/O or anything like that). This intrinsic
    // is really badly designed in the sense that in theory, there isn't
    // any way to safely use it... but in practice, it mostly works
    // to use it with non-atomic loads and stores to get acquire/release
    // semantics.
    Builder.CreateFence(llvm::SequentiallyConsistent);
    return RValue::get(0);
  }

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
        CGM.getTypes().arrangeFreeFunctionCall(E->getType(), Args,
                                               FunctionType::ExtInfo(),
                                               RequiredArgs::All);
    llvm::FunctionType *FTy = CGM.getTypes().GetFunctionType(FuncInfo);
    llvm::Constant *Func = CGM.CreateRuntimeFunction(FTy, LibCallName);
    return EmitCall(FuncInfo, Func, ReturnValueSlot(), Args);
  }

  case Builtin::BI__atomic_test_and_set: {
    // Look at the argument type to determine whether this is a volatile
    // operation. The parameter type is always volatile.
    QualType PtrTy = E->getArg(0)->IgnoreImpCasts()->getType();
    bool Volatile =
        PtrTy->castAs<PointerType>()->getPointeeType().isVolatileQualified();

    Value *Ptr = EmitScalarExpr(E->getArg(0));
    unsigned AddrSpace = Ptr->getType()->getPointerAddressSpace();
    Ptr = Builder.CreateBitCast(Ptr, Int8Ty->getPointerTo(AddrSpace));
    Value *NewVal = Builder.getInt8(1);
    Value *Order = EmitScalarExpr(E->getArg(1));
    if (isa<llvm::ConstantInt>(Order)) {
      int ord = cast<llvm::ConstantInt>(Order)->getZExtValue();
      AtomicRMWInst *Result = 0;
      switch (ord) {
      case 0:  // memory_order_relaxed
      default: // invalid order
        Result = Builder.CreateAtomicRMW(llvm::AtomicRMWInst::Xchg,
                                         Ptr, NewVal,
                                         llvm::Monotonic);
        break;
      case 1:  // memory_order_consume
      case 2:  // memory_order_acquire
        Result = Builder.CreateAtomicRMW(llvm::AtomicRMWInst::Xchg,
                                         Ptr, NewVal,
                                         llvm::Acquire);
        break;
      case 3:  // memory_order_release
        Result = Builder.CreateAtomicRMW(llvm::AtomicRMWInst::Xchg,
                                         Ptr, NewVal,
                                         llvm::Release);
        break;
      case 4:  // memory_order_acq_rel
        Result = Builder.CreateAtomicRMW(llvm::AtomicRMWInst::Xchg,
                                         Ptr, NewVal,
                                         llvm::AcquireRelease);
        break;
      case 5:  // memory_order_seq_cst
        Result = Builder.CreateAtomicRMW(llvm::AtomicRMWInst::Xchg,
                                         Ptr, NewVal,
                                         llvm::SequentiallyConsistent);
        break;
      }
      Result->setVolatile(Volatile);
      return RValue::get(Builder.CreateIsNotNull(Result, "tobool"));
    }

    llvm::BasicBlock *ContBB = createBasicBlock("atomic.continue", CurFn);

    llvm::BasicBlock *BBs[5] = {
      createBasicBlock("monotonic", CurFn),
      createBasicBlock("acquire", CurFn),
      createBasicBlock("release", CurFn),
      createBasicBlock("acqrel", CurFn),
      createBasicBlock("seqcst", CurFn)
    };
    llvm::AtomicOrdering Orders[5] = {
      llvm::Monotonic, llvm::Acquire, llvm::Release,
      llvm::AcquireRelease, llvm::SequentiallyConsistent
    };

    Order = Builder.CreateIntCast(Order, Builder.getInt32Ty(), false);
    llvm::SwitchInst *SI = Builder.CreateSwitch(Order, BBs[0]);

    Builder.SetInsertPoint(ContBB);
    PHINode *Result = Builder.CreatePHI(Int8Ty, 5, "was_set");

    for (unsigned i = 0; i < 5; ++i) {
      Builder.SetInsertPoint(BBs[i]);
      AtomicRMWInst *RMW = Builder.CreateAtomicRMW(llvm::AtomicRMWInst::Xchg,
                                                   Ptr, NewVal, Orders[i]);
      RMW->setVolatile(Volatile);
      Result->addIncoming(RMW, BBs[i]);
      Builder.CreateBr(ContBB);
    }

    SI->addCase(Builder.getInt32(0), BBs[0]);
    SI->addCase(Builder.getInt32(1), BBs[1]);
    SI->addCase(Builder.getInt32(2), BBs[1]);
    SI->addCase(Builder.getInt32(3), BBs[2]);
    SI->addCase(Builder.getInt32(4), BBs[3]);
    SI->addCase(Builder.getInt32(5), BBs[4]);

    Builder.SetInsertPoint(ContBB);
    return RValue::get(Builder.CreateIsNotNull(Result, "tobool"));
  }

  case Builtin::BI__atomic_clear: {
    QualType PtrTy = E->getArg(0)->IgnoreImpCasts()->getType();
    bool Volatile =
        PtrTy->castAs<PointerType>()->getPointeeType().isVolatileQualified();

    Value *Ptr = EmitScalarExpr(E->getArg(0));
    unsigned AddrSpace = Ptr->getType()->getPointerAddressSpace();
    Ptr = Builder.CreateBitCast(Ptr, Int8Ty->getPointerTo(AddrSpace));
    Value *NewVal = Builder.getInt8(0);
    Value *Order = EmitScalarExpr(E->getArg(1));
    if (isa<llvm::ConstantInt>(Order)) {
      int ord = cast<llvm::ConstantInt>(Order)->getZExtValue();
      StoreInst *Store = Builder.CreateStore(NewVal, Ptr, Volatile);
      Store->setAlignment(1);
      switch (ord) {
      case 0:  // memory_order_relaxed
      default: // invalid order
        Store->setOrdering(llvm::Monotonic);
        break;
      case 3:  // memory_order_release
        Store->setOrdering(llvm::Release);
        break;
      case 5:  // memory_order_seq_cst
        Store->setOrdering(llvm::SequentiallyConsistent);
        break;
      }
      return RValue::get(0);
    }

    llvm::BasicBlock *ContBB = createBasicBlock("atomic.continue", CurFn);

    llvm::BasicBlock *BBs[3] = {
      createBasicBlock("monotonic", CurFn),
      createBasicBlock("release", CurFn),
      createBasicBlock("seqcst", CurFn)
    };
    llvm::AtomicOrdering Orders[3] = {
      llvm::Monotonic, llvm::Release, llvm::SequentiallyConsistent
    };

    Order = Builder.CreateIntCast(Order, Builder.getInt32Ty(), false);
    llvm::SwitchInst *SI = Builder.CreateSwitch(Order, BBs[0]);

    for (unsigned i = 0; i < 3; ++i) {
      Builder.SetInsertPoint(BBs[i]);
      StoreInst *Store = Builder.CreateStore(NewVal, Ptr, Volatile);
      Store->setAlignment(1);
      Store->setOrdering(Orders[i]);
      Builder.CreateBr(ContBB);
    }

    SI->addCase(Builder.getInt32(0), BBs[0]);
    SI->addCase(Builder.getInt32(3), BBs[1]);
    SI->addCase(Builder.getInt32(5), BBs[2]);

    Builder.SetInsertPoint(ContBB);
    return RValue::get(0);
  }

  case Builtin::BI__atomic_thread_fence:
  case Builtin::BI__atomic_signal_fence:
  case Builtin::BI__c11_atomic_thread_fence:
  case Builtin::BI__c11_atomic_signal_fence: {
    llvm::SynchronizationScope Scope;
    if (BuiltinID == Builtin::BI__atomic_signal_fence ||
        BuiltinID == Builtin::BI__c11_atomic_signal_fence)
      Scope = llvm::SingleThread;
    else
      Scope = llvm::CrossThread;
    Value *Order = EmitScalarExpr(E->getArg(0));
    if (isa<llvm::ConstantInt>(Order)) {
      int ord = cast<llvm::ConstantInt>(Order)->getZExtValue();
      switch (ord) {
      case 0:  // memory_order_relaxed
      default: // invalid order
        break;
      case 1:  // memory_order_consume
      case 2:  // memory_order_acquire
        Builder.CreateFence(llvm::Acquire, Scope);
        break;
      case 3:  // memory_order_release
        Builder.CreateFence(llvm::Release, Scope);
        break;
      case 4:  // memory_order_acq_rel
        Builder.CreateFence(llvm::AcquireRelease, Scope);
        break;
      case 5:  // memory_order_seq_cst
        Builder.CreateFence(llvm::SequentiallyConsistent, Scope);
        break;
      }
      return RValue::get(0);
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
    Builder.CreateFence(llvm::Acquire, Scope);
    Builder.CreateBr(ContBB);
    SI->addCase(Builder.getInt32(1), AcquireBB);
    SI->addCase(Builder.getInt32(2), AcquireBB);

    Builder.SetInsertPoint(ReleaseBB);
    Builder.CreateFence(llvm::Release, Scope);
    Builder.CreateBr(ContBB);
    SI->addCase(Builder.getInt32(3), ReleaseBB);

    Builder.SetInsertPoint(AcqRelBB);
    Builder.CreateFence(llvm::AcquireRelease, Scope);
    Builder.CreateBr(ContBB);
    SI->addCase(Builder.getInt32(4), AcqRelBB);

    Builder.SetInsertPoint(SeqCstBB);
    Builder.CreateFence(llvm::SequentiallyConsistent, Scope);
    Builder.CreateBr(ContBB);
    SI->addCase(Builder.getInt32(5), SeqCstBB);

    Builder.SetInsertPoint(ContBB);
    return RValue::get(0);
  }

    // Library functions with special handling.
  case Builtin::BIsqrt:
  case Builtin::BIsqrtf:
  case Builtin::BIsqrtl: {
    // TODO: there is currently no set of optimizer flags
    // sufficient for us to rewrite sqrt to @llvm.sqrt.
    // -fmath-errno=0 is not good enough; we need finiteness.
    // We could probably precondition the call with an ult
    // against 0, but is that worth the complexity?
    break;
  }

  case Builtin::BIpow:
  case Builtin::BIpowf:
  case Builtin::BIpowl: {
    // Rewrite sqrt to intrinsic if allowed.
    if (!FD->hasAttr<ConstAttr>())
      break;
    Value *Base = EmitScalarExpr(E->getArg(0));
    Value *Exponent = EmitScalarExpr(E->getArg(1));
    llvm::Type *ArgType = Base->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::pow, ArgType);
    return RValue::get(Builder.CreateCall2(F, Base, Exponent));
  }

  case Builtin::BIfma:
  case Builtin::BIfmaf:
  case Builtin::BIfmal:
  case Builtin::BI__builtin_fma:
  case Builtin::BI__builtin_fmaf:
  case Builtin::BI__builtin_fmal: {
    // Rewrite fma to intrinsic.
    Value *FirstArg = EmitScalarExpr(E->getArg(0));
    llvm::Type *ArgType = FirstArg->getType();
    Value *F = CGM.getIntrinsic(Intrinsic::fma, ArgType);
    return RValue::get(Builder.CreateCall3(F, FirstArg,
                                              EmitScalarExpr(E->getArg(1)),
                                              EmitScalarExpr(E->getArg(2))));
  }

  case Builtin::BI__builtin_signbit:
  case Builtin::BI__builtin_signbitf:
  case Builtin::BI__builtin_signbitl: {
    LLVMContext &C = CGM.getLLVMContext();

    Value *Arg = EmitScalarExpr(E->getArg(0));
    llvm::Type *ArgTy = Arg->getType();
    if (ArgTy->isPPC_FP128Ty())
      break; // FIXME: I'm not sure what the right implementation is here.
    int ArgWidth = ArgTy->getPrimitiveSizeInBits();
    llvm::Type *ArgIntTy = llvm::IntegerType::get(C, ArgWidth);
    Value *BCArg = Builder.CreateBitCast(Arg, ArgIntTy);
    Value *ZeroCmp = llvm::Constant::getNullValue(ArgIntTy);
    Value *Result = Builder.CreateICmpSLT(BCArg, ZeroCmp);
    return RValue::get(Builder.CreateZExt(Result, ConvertType(E->getType())));
  }
  case Builtin::BI__builtin_annotation: {
    llvm::Value *AnnVal = EmitScalarExpr(E->getArg(0));
    llvm::Value *F = CGM.getIntrinsic(llvm::Intrinsic::annotation,
                                      AnnVal->getType());

    // Get the annotation string, go through casts. Sema requires this to be a
    // non-wide string literal, potentially casted, so the cast<> is safe.
    const Expr *AnnotationStrExpr = E->getArg(1)->IgnoreParenCasts();
    llvm::StringRef Str = cast<StringLiteral>(AnnotationStrExpr)->getString();
    return RValue::get(EmitAnnotationCall(F, AnnVal, Str, E->getExprLoc()));
  }
  case Builtin::BI__noop:
    return RValue::get(0);
  }

  // If this is an alias for a lib function (e.g. __builtin_sin), emit
  // the call using the normal call path, but using the unmangled
  // version of the function name.
  if (getContext().BuiltinInfo.isLibFunction(BuiltinID))
    return emitLibraryCall(*this, FD, E,
                           CGM.getBuiltinLibFunction(FD, BuiltinID));

  // If this is a predefined lib function (e.g. malloc), emit the call
  // using exactly the normal call path.
  if (getContext().BuiltinInfo.isPredefinedLibFunction(BuiltinID))
    return emitLibraryCall(*this, FD, E, EmitScalarExpr(E->getCallee()));

  // See if we have a target specific intrinsic.
  const char *Name = getContext().BuiltinInfo.GetName(BuiltinID);
  Intrinsic::ID IntrinsicID = Intrinsic::not_intrinsic;
  if (const char *Prefix =
      llvm::Triple::getArchTypePrefix(Target.getTriple().getArch()))
    IntrinsicID = Intrinsic::getIntrinsicForGCCBuiltin(Prefix, Name);

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
      Value *ArgValue;
      // If this is a normal argument, just emit it as a scalar.
      if ((ICEArguments & (1 << i)) == 0) {
        ArgValue = EmitScalarExpr(E->getArg(i));
      } else {
        // If this is required to be a constant, constant fold it so that we
        // know that the generated intrinsic gets a ConstantInt.
        llvm::APSInt Result;
        bool IsConst = E->getArg(i)->isIntegerConstantExpr(Result,getContext());
        assert(IsConst && "Constant arg isn't actually constant?");
        (void)IsConst;
        ArgValue = llvm::ConstantInt::get(getLLVMContext(), Result);
      }

      // If the intrinsic arg type is different from the builtin arg type
      // we need to do a bit cast.
      llvm::Type *PTy = FTy->getParamType(i);
      if (PTy != ArgValue->getType()) {
        assert(PTy->canLosslesslyBitCastTo(FTy->getParamType(i)) &&
               "Must be able to losslessly bit cast to param");
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
      assert(V->getType()->canLosslesslyBitCastTo(RetTy) &&
             "Must be able to losslessly bit cast result type");
      V = Builder.CreateBitCast(V, RetTy);
    }

    return RValue::get(V);
  }

  // See if we have a target specific builtin that needs to be lowered.
  if (Value *V = EmitTargetBuiltinExpr(BuiltinID, E))
    return RValue::get(V);

  ErrorUnsupported(E, "builtin function");

  // Unknown builtin, for now just dump it out and return undef.
  if (hasAggregateLLVMType(E->getType()))
    return RValue::getAggregate(CreateMemTemp(E->getType()));
  return RValue::get(llvm::UndefValue::get(ConvertType(E->getType())));
}

Value *CodeGenFunction::EmitTargetBuiltinExpr(unsigned BuiltinID,
                                              const CallExpr *E) {
  switch (Target.getTriple().getArch()) {
  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    return EmitARMBuiltinExpr(BuiltinID, E);
  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    return EmitX86BuiltinExpr(BuiltinID, E);
  case llvm::Triple::ppc:
  case llvm::Triple::ppc64:
    return EmitPPCBuiltinExpr(BuiltinID, E);
  default:
    return 0;
  }
}

static llvm::VectorType *GetNeonType(CodeGenFunction *CGF,
                                     NeonTypeFlags TypeFlags) {
  int IsQuad = TypeFlags.isQuad();
  switch (TypeFlags.getEltType()) {
  case NeonTypeFlags::Int8:
  case NeonTypeFlags::Poly8:
    return llvm::VectorType::get(CGF->Int8Ty, 8 << IsQuad);
  case NeonTypeFlags::Int16:
  case NeonTypeFlags::Poly16:
  case NeonTypeFlags::Float16:
    return llvm::VectorType::get(CGF->Int16Ty, 4 << IsQuad);
  case NeonTypeFlags::Int32:
    return llvm::VectorType::get(CGF->Int32Ty, 2 << IsQuad);
  case NeonTypeFlags::Int64:
    return llvm::VectorType::get(CGF->Int64Ty, 1 << IsQuad);
  case NeonTypeFlags::Float32:
    return llvm::VectorType::get(CGF->FloatTy, 2 << IsQuad);
  }
  llvm_unreachable("Invalid NeonTypeFlags element type!");
}

Value *CodeGenFunction::EmitNeonSplat(Value *V, Constant *C) {
  unsigned nElts = cast<llvm::VectorType>(V->getType())->getNumElements();
  Value* SV = llvm::ConstantVector::getSplat(nElts, C);
  return Builder.CreateShuffleVector(V, V, SV, "lane");
}

Value *CodeGenFunction::EmitNeonCall(Function *F, SmallVectorImpl<Value*> &Ops,
                                     const char *name,
                                     unsigned shift, bool rightshift) {
  unsigned j = 0;
  for (Function::const_arg_iterator ai = F->arg_begin(), ae = F->arg_end();
       ai != ae; ++ai, ++j)
    if (shift > 0 && shift == j)
      Ops[j] = EmitNeonShiftVector(Ops[j], ai->getType(), rightshift);
    else
      Ops[j] = Builder.CreateBitCast(Ops[j], ai->getType(), name);

  return Builder.CreateCall(F, Ops, name);
}

Value *CodeGenFunction::EmitNeonShiftVector(Value *V, llvm::Type *Ty,
                                            bool neg) {
  int SV = cast<ConstantInt>(V)->getSExtValue();

  llvm::VectorType *VTy = cast<llvm::VectorType>(Ty);
  llvm::Constant *C = ConstantInt::get(VTy->getElementType(), neg ? -SV : SV);
  return llvm::ConstantVector::getSplat(VTy->getNumElements(), C);
}

/// GetPointeeAlignment - Given an expression with a pointer type, find the
/// alignment of the type referenced by the pointer.  Skip over implicit
/// casts.
std::pair<llvm::Value*, unsigned>
CodeGenFunction::EmitPointerWithAlignment(const Expr *Addr) {
  assert(Addr->getType()->isPointerType());
  Addr = Addr->IgnoreParens();
  if (const ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(Addr)) {
    if ((ICE->getCastKind() == CK_BitCast || ICE->getCastKind() == CK_NoOp) &&
        ICE->getSubExpr()->getType()->isPointerType()) {
      std::pair<llvm::Value*, unsigned> Ptr =
          EmitPointerWithAlignment(ICE->getSubExpr());
      Ptr.first = Builder.CreateBitCast(Ptr.first,
                                        ConvertType(Addr->getType()));
      return Ptr;
    } else if (ICE->getCastKind() == CK_ArrayToPointerDecay) {
      LValue LV = EmitLValue(ICE->getSubExpr());
      unsigned Align = LV.getAlignment().getQuantity();
      if (!Align) {
        // FIXME: Once LValues are fixed to always set alignment,
        // zap this code.
        QualType PtTy = ICE->getSubExpr()->getType();
        if (!PtTy->isIncompleteType())
          Align = getContext().getTypeAlignInChars(PtTy).getQuantity();
        else
          Align = 1;
      }
      return std::make_pair(LV.getAddress(), Align);
    }
  }
  if (const UnaryOperator *UO = dyn_cast<UnaryOperator>(Addr)) {
    if (UO->getOpcode() == UO_AddrOf) {
      LValue LV = EmitLValue(UO->getSubExpr());
      unsigned Align = LV.getAlignment().getQuantity();
      if (!Align) {
        // FIXME: Once LValues are fixed to always set alignment,
        // zap this code.
        QualType PtTy = UO->getSubExpr()->getType();
        if (!PtTy->isIncompleteType())
          Align = getContext().getTypeAlignInChars(PtTy).getQuantity();
        else
          Align = 1;
      }
      return std::make_pair(LV.getAddress(), Align);
    }
  }

  unsigned Align = 1;
  QualType PtTy = Addr->getType()->getPointeeType();
  if (!PtTy->isIncompleteType())
    Align = getContext().getTypeAlignInChars(PtTy).getQuantity();

  return std::make_pair(EmitScalarExpr(Addr), Align);
}

Value *CodeGenFunction::EmitARMBuiltinExpr(unsigned BuiltinID,
                                           const CallExpr *E) {
  if (BuiltinID == ARM::BI__clear_cache) {
    const FunctionDecl *FD = E->getDirectCallee();
    // Oddly people write this call without args on occasion and gcc accepts
    // it - it's also marked as varargs in the description file.
    SmallVector<Value*, 2> Ops;
    for (unsigned i = 0; i < E->getNumArgs(); i++)
      Ops.push_back(EmitScalarExpr(E->getArg(i)));
    llvm::Type *Ty = CGM.getTypes().ConvertType(FD->getType());
    llvm::FunctionType *FTy = cast<llvm::FunctionType>(Ty);
    StringRef Name = FD->getName();
    return Builder.CreateCall(CGM.CreateRuntimeFunction(FTy, Name), Ops);
  }

  if (BuiltinID == ARM::BI__builtin_arm_ldrexd) {
    Function *F = CGM.getIntrinsic(Intrinsic::arm_ldrexd);

    Value *LdPtr = EmitScalarExpr(E->getArg(0));
    Value *Val = Builder.CreateCall(F, LdPtr, "ldrexd");

    Value *Val0 = Builder.CreateExtractValue(Val, 1);
    Value *Val1 = Builder.CreateExtractValue(Val, 0);
    Val0 = Builder.CreateZExt(Val0, Int64Ty);
    Val1 = Builder.CreateZExt(Val1, Int64Ty);

    Value *ShiftCst = llvm::ConstantInt::get(Int64Ty, 32);
    Val = Builder.CreateShl(Val0, ShiftCst, "shl", true /* nuw */);
    return Builder.CreateOr(Val, Val1);
  }

  if (BuiltinID == ARM::BI__builtin_arm_strexd) {
    Function *F = CGM.getIntrinsic(Intrinsic::arm_strexd);
    llvm::Type *STy = llvm::StructType::get(Int32Ty, Int32Ty, NULL);

    Value *One = llvm::ConstantInt::get(Int32Ty, 1);
    Value *Tmp = Builder.CreateAlloca(Int64Ty, One);
    Value *Val = EmitScalarExpr(E->getArg(0));
    Builder.CreateStore(Val, Tmp);

    Value *LdPtr = Builder.CreateBitCast(Tmp,llvm::PointerType::getUnqual(STy));
    Val = Builder.CreateLoad(LdPtr);

    Value *Arg0 = Builder.CreateExtractValue(Val, 0);
    Value *Arg1 = Builder.CreateExtractValue(Val, 1);
    Value *StPtr = EmitScalarExpr(E->getArg(1));
    return Builder.CreateCall3(F, Arg0, Arg1, StPtr, "strexd");
  }

  SmallVector<Value*, 4> Ops;
  llvm::Value *Align = 0;
  for (unsigned i = 0, e = E->getNumArgs() - 1; i != e; i++) {
    if (i == 0) {
      switch (BuiltinID) {
      case ARM::BI__builtin_neon_vld1_v:
      case ARM::BI__builtin_neon_vld1q_v:
      case ARM::BI__builtin_neon_vld1q_lane_v:
      case ARM::BI__builtin_neon_vld1_lane_v:
      case ARM::BI__builtin_neon_vld1_dup_v:
      case ARM::BI__builtin_neon_vld1q_dup_v:
      case ARM::BI__builtin_neon_vst1_v:
      case ARM::BI__builtin_neon_vst1q_v:
      case ARM::BI__builtin_neon_vst1q_lane_v:
      case ARM::BI__builtin_neon_vst1_lane_v:
      case ARM::BI__builtin_neon_vst2_v:
      case ARM::BI__builtin_neon_vst2q_v:
      case ARM::BI__builtin_neon_vst2_lane_v:
      case ARM::BI__builtin_neon_vst2q_lane_v:
      case ARM::BI__builtin_neon_vst3_v:
      case ARM::BI__builtin_neon_vst3q_v:
      case ARM::BI__builtin_neon_vst3_lane_v:
      case ARM::BI__builtin_neon_vst3q_lane_v:
      case ARM::BI__builtin_neon_vst4_v:
      case ARM::BI__builtin_neon_vst4q_v:
      case ARM::BI__builtin_neon_vst4_lane_v:
      case ARM::BI__builtin_neon_vst4q_lane_v:
        // Get the alignment for the argument in addition to the value;
        // we'll use it later.
        std::pair<llvm::Value*, unsigned> Src =
            EmitPointerWithAlignment(E->getArg(0));
        Ops.push_back(Src.first);
        Align = Builder.getInt32(Src.second);
        continue;
      }
    }
    if (i == 1) {
      switch (BuiltinID) {
      case ARM::BI__builtin_neon_vld2_v:
      case ARM::BI__builtin_neon_vld2q_v:
      case ARM::BI__builtin_neon_vld3_v:
      case ARM::BI__builtin_neon_vld3q_v:
      case ARM::BI__builtin_neon_vld4_v:
      case ARM::BI__builtin_neon_vld4q_v:
      case ARM::BI__builtin_neon_vld2_lane_v:
      case ARM::BI__builtin_neon_vld2q_lane_v:
      case ARM::BI__builtin_neon_vld3_lane_v:
      case ARM::BI__builtin_neon_vld3q_lane_v:
      case ARM::BI__builtin_neon_vld4_lane_v:
      case ARM::BI__builtin_neon_vld4q_lane_v:
      case ARM::BI__builtin_neon_vld2_dup_v:
      case ARM::BI__builtin_neon_vld3_dup_v:
      case ARM::BI__builtin_neon_vld4_dup_v:
        // Get the alignment for the argument in addition to the value;
        // we'll use it later.
        std::pair<llvm::Value*, unsigned> Src =
            EmitPointerWithAlignment(E->getArg(1));
        Ops.push_back(Src.first);
        Align = Builder.getInt32(Src.second);
        continue;
      }
    }
    Ops.push_back(EmitScalarExpr(E->getArg(i)));
  }

  // vget_lane and vset_lane are not overloaded and do not have an extra
  // argument that specifies the vector type.
  switch (BuiltinID) {
  default: break;
  case ARM::BI__builtin_neon_vget_lane_i8:
  case ARM::BI__builtin_neon_vget_lane_i16:
  case ARM::BI__builtin_neon_vget_lane_i32:
  case ARM::BI__builtin_neon_vget_lane_i64:
  case ARM::BI__builtin_neon_vget_lane_f32:
  case ARM::BI__builtin_neon_vgetq_lane_i8:
  case ARM::BI__builtin_neon_vgetq_lane_i16:
  case ARM::BI__builtin_neon_vgetq_lane_i32:
  case ARM::BI__builtin_neon_vgetq_lane_i64:
  case ARM::BI__builtin_neon_vgetq_lane_f32:
    return Builder.CreateExtractElement(Ops[0], EmitScalarExpr(E->getArg(1)),
                                        "vget_lane");
  case ARM::BI__builtin_neon_vset_lane_i8:
  case ARM::BI__builtin_neon_vset_lane_i16:
  case ARM::BI__builtin_neon_vset_lane_i32:
  case ARM::BI__builtin_neon_vset_lane_i64:
  case ARM::BI__builtin_neon_vset_lane_f32:
  case ARM::BI__builtin_neon_vsetq_lane_i8:
  case ARM::BI__builtin_neon_vsetq_lane_i16:
  case ARM::BI__builtin_neon_vsetq_lane_i32:
  case ARM::BI__builtin_neon_vsetq_lane_i64:
  case ARM::BI__builtin_neon_vsetq_lane_f32:
    Ops.push_back(EmitScalarExpr(E->getArg(2)));
    return Builder.CreateInsertElement(Ops[1], Ops[0], Ops[2], "vset_lane");
  }

  // Get the last argument, which specifies the vector type.
  llvm::APSInt Result;
  const Expr *Arg = E->getArg(E->getNumArgs()-1);
  if (!Arg->isIntegerConstantExpr(Result, getContext()))
    return 0;

  if (BuiltinID == ARM::BI__builtin_arm_vcvtr_f ||
      BuiltinID == ARM::BI__builtin_arm_vcvtr_d) {
    // Determine the overloaded type of this builtin.
    llvm::Type *Ty;
    if (BuiltinID == ARM::BI__builtin_arm_vcvtr_f)
      Ty = FloatTy;
    else
      Ty = DoubleTy;

    // Determine whether this is an unsigned conversion or not.
    bool usgn = Result.getZExtValue() == 1;
    unsigned Int = usgn ? Intrinsic::arm_vcvtru : Intrinsic::arm_vcvtr;

    // Call the appropriate intrinsic.
    Function *F = CGM.getIntrinsic(Int, Ty);
    return Builder.CreateCall(F, Ops, "vcvtr");
  }

  // Determine the type of this overloaded NEON intrinsic.
  NeonTypeFlags Type(Result.getZExtValue());
  bool usgn = Type.isUnsigned();
  bool quad = Type.isQuad();
  bool rightShift = false;

  llvm::VectorType *VTy = GetNeonType(this, Type);
  llvm::Type *Ty = VTy;
  if (!Ty)
    return 0;

  unsigned Int;
  switch (BuiltinID) {
  default: return 0;
  case ARM::BI__builtin_neon_vbsl_v:
  case ARM::BI__builtin_neon_vbslq_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vbsl, Ty),
                        Ops, "vbsl");
  case ARM::BI__builtin_neon_vabd_v:
  case ARM::BI__builtin_neon_vabdq_v:
    Int = usgn ? Intrinsic::arm_neon_vabdu : Intrinsic::arm_neon_vabds;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vabd");
  case ARM::BI__builtin_neon_vabs_v:
  case ARM::BI__builtin_neon_vabsq_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vabs, Ty),
                        Ops, "vabs");
  case ARM::BI__builtin_neon_vaddhn_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vaddhn, Ty),
                        Ops, "vaddhn");
  case ARM::BI__builtin_neon_vcale_v:
    std::swap(Ops[0], Ops[1]);
  case ARM::BI__builtin_neon_vcage_v: {
    Function *F = CGM.getIntrinsic(Intrinsic::arm_neon_vacged);
    return EmitNeonCall(F, Ops, "vcage");
  }
  case ARM::BI__builtin_neon_vcaleq_v:
    std::swap(Ops[0], Ops[1]);
  case ARM::BI__builtin_neon_vcageq_v: {
    Function *F = CGM.getIntrinsic(Intrinsic::arm_neon_vacgeq);
    return EmitNeonCall(F, Ops, "vcage");
  }
  case ARM::BI__builtin_neon_vcalt_v:
    std::swap(Ops[0], Ops[1]);
  case ARM::BI__builtin_neon_vcagt_v: {
    Function *F = CGM.getIntrinsic(Intrinsic::arm_neon_vacgtd);
    return EmitNeonCall(F, Ops, "vcagt");
  }
  case ARM::BI__builtin_neon_vcaltq_v:
    std::swap(Ops[0], Ops[1]);
  case ARM::BI__builtin_neon_vcagtq_v: {
    Function *F = CGM.getIntrinsic(Intrinsic::arm_neon_vacgtq);
    return EmitNeonCall(F, Ops, "vcagt");
  }
  case ARM::BI__builtin_neon_vcls_v:
  case ARM::BI__builtin_neon_vclsq_v: {
    Function *F = CGM.getIntrinsic(Intrinsic::arm_neon_vcls, Ty);
    return EmitNeonCall(F, Ops, "vcls");
  }
  case ARM::BI__builtin_neon_vclz_v:
  case ARM::BI__builtin_neon_vclzq_v: {
    // Generate target-independent intrinsic; also need to add second argument
    // for whether or not clz of zero is undefined; on ARM it isn't.
    Function *F = CGM.getIntrinsic(Intrinsic::ctlz, Ty);
    Ops.push_back(Builder.getInt1(Target.isCLZForZeroUndef()));
    return EmitNeonCall(F, Ops, "vclz");
  }
  case ARM::BI__builtin_neon_vcnt_v:
  case ARM::BI__builtin_neon_vcntq_v: {
    // generate target-independent intrinsic
    Function *F = CGM.getIntrinsic(Intrinsic::ctpop, Ty);
    return EmitNeonCall(F, Ops, "vctpop");
  }
  case ARM::BI__builtin_neon_vcvt_f16_v: {
    assert(Type.getEltType() == NeonTypeFlags::Float16 && !quad &&
           "unexpected vcvt_f16_v builtin");
    Function *F = CGM.getIntrinsic(Intrinsic::arm_neon_vcvtfp2hf);
    return EmitNeonCall(F, Ops, "vcvt");
  }
  case ARM::BI__builtin_neon_vcvt_f32_f16: {
    assert(Type.getEltType() == NeonTypeFlags::Float16 && !quad &&
           "unexpected vcvt_f32_f16 builtin");
    Function *F = CGM.getIntrinsic(Intrinsic::arm_neon_vcvthf2fp);
    return EmitNeonCall(F, Ops, "vcvt");
  }
  case ARM::BI__builtin_neon_vcvt_f32_v:
  case ARM::BI__builtin_neon_vcvtq_f32_v:
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ty = GetNeonType(this, NeonTypeFlags(NeonTypeFlags::Float32, false, quad));
    return usgn ? Builder.CreateUIToFP(Ops[0], Ty, "vcvt")
                : Builder.CreateSIToFP(Ops[0], Ty, "vcvt");
  case ARM::BI__builtin_neon_vcvt_s32_v:
  case ARM::BI__builtin_neon_vcvt_u32_v:
  case ARM::BI__builtin_neon_vcvtq_s32_v:
  case ARM::BI__builtin_neon_vcvtq_u32_v: {
    llvm::Type *FloatTy =
      GetNeonType(this, NeonTypeFlags(NeonTypeFlags::Float32, false, quad));
    Ops[0] = Builder.CreateBitCast(Ops[0], FloatTy);
    return usgn ? Builder.CreateFPToUI(Ops[0], Ty, "vcvt")
                : Builder.CreateFPToSI(Ops[0], Ty, "vcvt");
  }
  case ARM::BI__builtin_neon_vcvt_n_f32_v:
  case ARM::BI__builtin_neon_vcvtq_n_f32_v: {
    llvm::Type *FloatTy =
      GetNeonType(this, NeonTypeFlags(NeonTypeFlags::Float32, false, quad));
    llvm::Type *Tys[2] = { FloatTy, Ty };
    Int = usgn ? Intrinsic::arm_neon_vcvtfxu2fp
               : Intrinsic::arm_neon_vcvtfxs2fp;
    Function *F = CGM.getIntrinsic(Int, Tys);
    return EmitNeonCall(F, Ops, "vcvt_n");
  }
  case ARM::BI__builtin_neon_vcvt_n_s32_v:
  case ARM::BI__builtin_neon_vcvt_n_u32_v:
  case ARM::BI__builtin_neon_vcvtq_n_s32_v:
  case ARM::BI__builtin_neon_vcvtq_n_u32_v: {
    llvm::Type *FloatTy =
      GetNeonType(this, NeonTypeFlags(NeonTypeFlags::Float32, false, quad));
    llvm::Type *Tys[2] = { Ty, FloatTy };
    Int = usgn ? Intrinsic::arm_neon_vcvtfp2fxu
               : Intrinsic::arm_neon_vcvtfp2fxs;
    Function *F = CGM.getIntrinsic(Int, Tys);
    return EmitNeonCall(F, Ops, "vcvt_n");
  }
  case ARM::BI__builtin_neon_vext_v:
  case ARM::BI__builtin_neon_vextq_v: {
    int CV = cast<ConstantInt>(Ops[2])->getSExtValue();
    SmallVector<Constant*, 16> Indices;
    for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i)
      Indices.push_back(ConstantInt::get(Int32Ty, i+CV));

    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Value *SV = llvm::ConstantVector::get(Indices);
    return Builder.CreateShuffleVector(Ops[0], Ops[1], SV, "vext");
  }
  case ARM::BI__builtin_neon_vhadd_v:
  case ARM::BI__builtin_neon_vhaddq_v:
    Int = usgn ? Intrinsic::arm_neon_vhaddu : Intrinsic::arm_neon_vhadds;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vhadd");
  case ARM::BI__builtin_neon_vhsub_v:
  case ARM::BI__builtin_neon_vhsubq_v:
    Int = usgn ? Intrinsic::arm_neon_vhsubu : Intrinsic::arm_neon_vhsubs;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vhsub");
  case ARM::BI__builtin_neon_vld1_v:
  case ARM::BI__builtin_neon_vld1q_v:
    Ops.push_back(Align);
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vld1, Ty),
                        Ops, "vld1");
  case ARM::BI__builtin_neon_vld1q_lane_v:
    // Handle 64-bit integer elements as a special case.  Use shuffles of
    // one-element vectors to avoid poor code for i64 in the backend.
    if (VTy->getElementType()->isIntegerTy(64)) {
      // Extract the other lane.
      Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
      int Lane = cast<ConstantInt>(Ops[2])->getZExtValue();
      Value *SV = llvm::ConstantVector::get(ConstantInt::get(Int32Ty, 1-Lane));
      Ops[1] = Builder.CreateShuffleVector(Ops[1], Ops[1], SV);
      // Load the value as a one-element vector.
      Ty = llvm::VectorType::get(VTy->getElementType(), 1);
      Function *F = CGM.getIntrinsic(Intrinsic::arm_neon_vld1, Ty);
      Value *Ld = Builder.CreateCall2(F, Ops[0], Align);
      // Combine them.
      SmallVector<Constant*, 2> Indices;
      Indices.push_back(ConstantInt::get(Int32Ty, 1-Lane));
      Indices.push_back(ConstantInt::get(Int32Ty, Lane));
      SV = llvm::ConstantVector::get(Indices);
      return Builder.CreateShuffleVector(Ops[1], Ld, SV, "vld1q_lane");
    }
    // fall through
  case ARM::BI__builtin_neon_vld1_lane_v: {
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ty = llvm::PointerType::getUnqual(VTy->getElementType());
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    LoadInst *Ld = Builder.CreateLoad(Ops[0]);
    Ld->setAlignment(cast<ConstantInt>(Align)->getZExtValue());
    return Builder.CreateInsertElement(Ops[1], Ld, Ops[2], "vld1_lane");
  }
  case ARM::BI__builtin_neon_vld1_dup_v:
  case ARM::BI__builtin_neon_vld1q_dup_v: {
    Value *V = UndefValue::get(Ty);
    Ty = llvm::PointerType::getUnqual(VTy->getElementType());
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    LoadInst *Ld = Builder.CreateLoad(Ops[0]);
    Ld->setAlignment(cast<ConstantInt>(Align)->getZExtValue());
    llvm::Constant *CI = ConstantInt::get(Int32Ty, 0);
    Ops[0] = Builder.CreateInsertElement(V, Ld, CI);
    return EmitNeonSplat(Ops[0], CI);
  }
  case ARM::BI__builtin_neon_vld2_v:
  case ARM::BI__builtin_neon_vld2q_v: {
    Function *F = CGM.getIntrinsic(Intrinsic::arm_neon_vld2, Ty);
    Ops[1] = Builder.CreateCall2(F, Ops[1], Align, "vld2");
    Ty = llvm::PointerType::getUnqual(Ops[1]->getType());
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    return Builder.CreateStore(Ops[1], Ops[0]);
  }
  case ARM::BI__builtin_neon_vld3_v:
  case ARM::BI__builtin_neon_vld3q_v: {
    Function *F = CGM.getIntrinsic(Intrinsic::arm_neon_vld3, Ty);
    Ops[1] = Builder.CreateCall2(F, Ops[1], Align, "vld3");
    Ty = llvm::PointerType::getUnqual(Ops[1]->getType());
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    return Builder.CreateStore(Ops[1], Ops[0]);
  }
  case ARM::BI__builtin_neon_vld4_v:
  case ARM::BI__builtin_neon_vld4q_v: {
    Function *F = CGM.getIntrinsic(Intrinsic::arm_neon_vld4, Ty);
    Ops[1] = Builder.CreateCall2(F, Ops[1], Align, "vld4");
    Ty = llvm::PointerType::getUnqual(Ops[1]->getType());
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    return Builder.CreateStore(Ops[1], Ops[0]);
  }
  case ARM::BI__builtin_neon_vld2_lane_v:
  case ARM::BI__builtin_neon_vld2q_lane_v: {
    Function *F = CGM.getIntrinsic(Intrinsic::arm_neon_vld2lane, Ty);
    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);
    Ops[3] = Builder.CreateBitCast(Ops[3], Ty);
    Ops.push_back(Align);
    Ops[1] = Builder.CreateCall(F, makeArrayRef(Ops).slice(1), "vld2_lane");
    Ty = llvm::PointerType::getUnqual(Ops[1]->getType());
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    return Builder.CreateStore(Ops[1], Ops[0]);
  }
  case ARM::BI__builtin_neon_vld3_lane_v:
  case ARM::BI__builtin_neon_vld3q_lane_v: {
    Function *F = CGM.getIntrinsic(Intrinsic::arm_neon_vld3lane, Ty);
    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);
    Ops[3] = Builder.CreateBitCast(Ops[3], Ty);
    Ops[4] = Builder.CreateBitCast(Ops[4], Ty);
    Ops.push_back(Align);
    Ops[1] = Builder.CreateCall(F, makeArrayRef(Ops).slice(1), "vld3_lane");
    Ty = llvm::PointerType::getUnqual(Ops[1]->getType());
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    return Builder.CreateStore(Ops[1], Ops[0]);
  }
  case ARM::BI__builtin_neon_vld4_lane_v:
  case ARM::BI__builtin_neon_vld4q_lane_v: {
    Function *F = CGM.getIntrinsic(Intrinsic::arm_neon_vld4lane, Ty);
    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);
    Ops[3] = Builder.CreateBitCast(Ops[3], Ty);
    Ops[4] = Builder.CreateBitCast(Ops[4], Ty);
    Ops[5] = Builder.CreateBitCast(Ops[5], Ty);
    Ops.push_back(Align);
    Ops[1] = Builder.CreateCall(F, makeArrayRef(Ops).slice(1), "vld3_lane");
    Ty = llvm::PointerType::getUnqual(Ops[1]->getType());
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    return Builder.CreateStore(Ops[1], Ops[0]);
  }
  case ARM::BI__builtin_neon_vld2_dup_v:
  case ARM::BI__builtin_neon_vld3_dup_v:
  case ARM::BI__builtin_neon_vld4_dup_v: {
    // Handle 64-bit elements as a special-case.  There is no "dup" needed.
    if (VTy->getElementType()->getPrimitiveSizeInBits() == 64) {
      switch (BuiltinID) {
      case ARM::BI__builtin_neon_vld2_dup_v:
        Int = Intrinsic::arm_neon_vld2;
        break;
      case ARM::BI__builtin_neon_vld3_dup_v:
        Int = Intrinsic::arm_neon_vld3;
        break;
      case ARM::BI__builtin_neon_vld4_dup_v:
        Int = Intrinsic::arm_neon_vld4;
        break;
      default: llvm_unreachable("unknown vld_dup intrinsic?");
      }
      Function *F = CGM.getIntrinsic(Int, Ty);
      Ops[1] = Builder.CreateCall2(F, Ops[1], Align, "vld_dup");
      Ty = llvm::PointerType::getUnqual(Ops[1]->getType());
      Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
      return Builder.CreateStore(Ops[1], Ops[0]);
    }
    switch (BuiltinID) {
    case ARM::BI__builtin_neon_vld2_dup_v:
      Int = Intrinsic::arm_neon_vld2lane;
      break;
    case ARM::BI__builtin_neon_vld3_dup_v:
      Int = Intrinsic::arm_neon_vld3lane;
      break;
    case ARM::BI__builtin_neon_vld4_dup_v:
      Int = Intrinsic::arm_neon_vld4lane;
      break;
    default: llvm_unreachable("unknown vld_dup intrinsic?");
    }
    Function *F = CGM.getIntrinsic(Int, Ty);
    llvm::StructType *STy = cast<llvm::StructType>(F->getReturnType());

    SmallVector<Value*, 6> Args;
    Args.push_back(Ops[1]);
    Args.append(STy->getNumElements(), UndefValue::get(Ty));

    llvm::Constant *CI = ConstantInt::get(Int32Ty, 0);
    Args.push_back(CI);
    Args.push_back(Align);

    Ops[1] = Builder.CreateCall(F, Args, "vld_dup");
    // splat lane 0 to all elts in each vector of the result.
    for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
      Value *Val = Builder.CreateExtractValue(Ops[1], i);
      Value *Elt = Builder.CreateBitCast(Val, Ty);
      Elt = EmitNeonSplat(Elt, CI);
      Elt = Builder.CreateBitCast(Elt, Val->getType());
      Ops[1] = Builder.CreateInsertValue(Ops[1], Elt, i);
    }
    Ty = llvm::PointerType::getUnqual(Ops[1]->getType());
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    return Builder.CreateStore(Ops[1], Ops[0]);
  }
  case ARM::BI__builtin_neon_vmax_v:
  case ARM::BI__builtin_neon_vmaxq_v:
    Int = usgn ? Intrinsic::arm_neon_vmaxu : Intrinsic::arm_neon_vmaxs;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vmax");
  case ARM::BI__builtin_neon_vmin_v:
  case ARM::BI__builtin_neon_vminq_v:
    Int = usgn ? Intrinsic::arm_neon_vminu : Intrinsic::arm_neon_vmins;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vmin");
  case ARM::BI__builtin_neon_vmovl_v: {
    llvm::Type *DTy =llvm::VectorType::getTruncatedElementVectorType(VTy);
    Ops[0] = Builder.CreateBitCast(Ops[0], DTy);
    if (usgn)
      return Builder.CreateZExt(Ops[0], Ty, "vmovl");
    return Builder.CreateSExt(Ops[0], Ty, "vmovl");
  }
  case ARM::BI__builtin_neon_vmovn_v: {
    llvm::Type *QTy = llvm::VectorType::getExtendedElementVectorType(VTy);
    Ops[0] = Builder.CreateBitCast(Ops[0], QTy);
    return Builder.CreateTrunc(Ops[0], Ty, "vmovn");
  }
  case ARM::BI__builtin_neon_vmul_v:
  case ARM::BI__builtin_neon_vmulq_v:
    assert(Type.isPoly() && "vmul builtin only supported for polynomial types");
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vmulp, Ty),
                        Ops, "vmul");
  case ARM::BI__builtin_neon_vmull_v:
    Int = usgn ? Intrinsic::arm_neon_vmullu : Intrinsic::arm_neon_vmulls;
    Int = Type.isPoly() ? (unsigned)Intrinsic::arm_neon_vmullp : Int;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vmull");
  case ARM::BI__builtin_neon_vfma_v:
  case ARM::BI__builtin_neon_vfmaq_v: {
    Value *F = CGM.getIntrinsic(Intrinsic::fma, Ty);
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);
    return Builder.CreateCall3(F, Ops[0], Ops[1], Ops[2]);
  }
  case ARM::BI__builtin_neon_vpadal_v:
  case ARM::BI__builtin_neon_vpadalq_v: {
    Int = usgn ? Intrinsic::arm_neon_vpadalu : Intrinsic::arm_neon_vpadals;
    // The source operand type has twice as many elements of half the size.
    unsigned EltBits = VTy->getElementType()->getPrimitiveSizeInBits();
    llvm::Type *EltTy =
      llvm::IntegerType::get(getLLVMContext(), EltBits / 2);
    llvm::Type *NarrowTy =
      llvm::VectorType::get(EltTy, VTy->getNumElements() * 2);
    llvm::Type *Tys[2] = { Ty, NarrowTy };
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vpadal");
  }
  case ARM::BI__builtin_neon_vpadd_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vpadd, Ty),
                        Ops, "vpadd");
  case ARM::BI__builtin_neon_vpaddl_v:
  case ARM::BI__builtin_neon_vpaddlq_v: {
    Int = usgn ? Intrinsic::arm_neon_vpaddlu : Intrinsic::arm_neon_vpaddls;
    // The source operand type has twice as many elements of half the size.
    unsigned EltBits = VTy->getElementType()->getPrimitiveSizeInBits();
    llvm::Type *EltTy = llvm::IntegerType::get(getLLVMContext(), EltBits / 2);
    llvm::Type *NarrowTy =
      llvm::VectorType::get(EltTy, VTy->getNumElements() * 2);
    llvm::Type *Tys[2] = { Ty, NarrowTy };
    return EmitNeonCall(CGM.getIntrinsic(Int, Tys), Ops, "vpaddl");
  }
  case ARM::BI__builtin_neon_vpmax_v:
    Int = usgn ? Intrinsic::arm_neon_vpmaxu : Intrinsic::arm_neon_vpmaxs;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vpmax");
  case ARM::BI__builtin_neon_vpmin_v:
    Int = usgn ? Intrinsic::arm_neon_vpminu : Intrinsic::arm_neon_vpmins;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vpmin");
  case ARM::BI__builtin_neon_vqabs_v:
  case ARM::BI__builtin_neon_vqabsq_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vqabs, Ty),
                        Ops, "vqabs");
  case ARM::BI__builtin_neon_vqadd_v:
  case ARM::BI__builtin_neon_vqaddq_v:
    Int = usgn ? Intrinsic::arm_neon_vqaddu : Intrinsic::arm_neon_vqadds;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vqadd");
  case ARM::BI__builtin_neon_vqdmlal_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vqdmlal, Ty),
                        Ops, "vqdmlal");
  case ARM::BI__builtin_neon_vqdmlsl_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vqdmlsl, Ty),
                        Ops, "vqdmlsl");
  case ARM::BI__builtin_neon_vqdmulh_v:
  case ARM::BI__builtin_neon_vqdmulhq_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vqdmulh, Ty),
                        Ops, "vqdmulh");
  case ARM::BI__builtin_neon_vqdmull_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vqdmull, Ty),
                        Ops, "vqdmull");
  case ARM::BI__builtin_neon_vqmovn_v:
    Int = usgn ? Intrinsic::arm_neon_vqmovnu : Intrinsic::arm_neon_vqmovns;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vqmovn");
  case ARM::BI__builtin_neon_vqmovun_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vqmovnsu, Ty),
                        Ops, "vqdmull");
  case ARM::BI__builtin_neon_vqneg_v:
  case ARM::BI__builtin_neon_vqnegq_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vqneg, Ty),
                        Ops, "vqneg");
  case ARM::BI__builtin_neon_vqrdmulh_v:
  case ARM::BI__builtin_neon_vqrdmulhq_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vqrdmulh, Ty),
                        Ops, "vqrdmulh");
  case ARM::BI__builtin_neon_vqrshl_v:
  case ARM::BI__builtin_neon_vqrshlq_v:
    Int = usgn ? Intrinsic::arm_neon_vqrshiftu : Intrinsic::arm_neon_vqrshifts;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vqrshl");
  case ARM::BI__builtin_neon_vqrshrn_n_v:
    Int =
      usgn ? Intrinsic::arm_neon_vqrshiftnu : Intrinsic::arm_neon_vqrshiftns;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vqrshrn_n",
                        1, true);
  case ARM::BI__builtin_neon_vqrshrun_n_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vqrshiftnsu, Ty),
                        Ops, "vqrshrun_n", 1, true);
  case ARM::BI__builtin_neon_vqshl_v:
  case ARM::BI__builtin_neon_vqshlq_v:
    Int = usgn ? Intrinsic::arm_neon_vqshiftu : Intrinsic::arm_neon_vqshifts;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vqshl");
  case ARM::BI__builtin_neon_vqshl_n_v:
  case ARM::BI__builtin_neon_vqshlq_n_v:
    Int = usgn ? Intrinsic::arm_neon_vqshiftu : Intrinsic::arm_neon_vqshifts;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vqshl_n",
                        1, false);
  case ARM::BI__builtin_neon_vqshlu_n_v:
  case ARM::BI__builtin_neon_vqshluq_n_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vqshiftsu, Ty),
                        Ops, "vqshlu", 1, false);
  case ARM::BI__builtin_neon_vqshrn_n_v:
    Int = usgn ? Intrinsic::arm_neon_vqshiftnu : Intrinsic::arm_neon_vqshiftns;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vqshrn_n",
                        1, true);
  case ARM::BI__builtin_neon_vqshrun_n_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vqshiftnsu, Ty),
                        Ops, "vqshrun_n", 1, true);
  case ARM::BI__builtin_neon_vqsub_v:
  case ARM::BI__builtin_neon_vqsubq_v:
    Int = usgn ? Intrinsic::arm_neon_vqsubu : Intrinsic::arm_neon_vqsubs;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vqsub");
  case ARM::BI__builtin_neon_vraddhn_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vraddhn, Ty),
                        Ops, "vraddhn");
  case ARM::BI__builtin_neon_vrecpe_v:
  case ARM::BI__builtin_neon_vrecpeq_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vrecpe, Ty),
                        Ops, "vrecpe");
  case ARM::BI__builtin_neon_vrecps_v:
  case ARM::BI__builtin_neon_vrecpsq_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vrecps, Ty),
                        Ops, "vrecps");
  case ARM::BI__builtin_neon_vrhadd_v:
  case ARM::BI__builtin_neon_vrhaddq_v:
    Int = usgn ? Intrinsic::arm_neon_vrhaddu : Intrinsic::arm_neon_vrhadds;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrhadd");
  case ARM::BI__builtin_neon_vrshl_v:
  case ARM::BI__builtin_neon_vrshlq_v:
    Int = usgn ? Intrinsic::arm_neon_vrshiftu : Intrinsic::arm_neon_vrshifts;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrshl");
  case ARM::BI__builtin_neon_vrshrn_n_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vrshiftn, Ty),
                        Ops, "vrshrn_n", 1, true);
  case ARM::BI__builtin_neon_vrshr_n_v:
  case ARM::BI__builtin_neon_vrshrq_n_v:
    Int = usgn ? Intrinsic::arm_neon_vrshiftu : Intrinsic::arm_neon_vrshifts;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vrshr_n", 1, true);
  case ARM::BI__builtin_neon_vrsqrte_v:
  case ARM::BI__builtin_neon_vrsqrteq_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vrsqrte, Ty),
                        Ops, "vrsqrte");
  case ARM::BI__builtin_neon_vrsqrts_v:
  case ARM::BI__builtin_neon_vrsqrtsq_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vrsqrts, Ty),
                        Ops, "vrsqrts");
  case ARM::BI__builtin_neon_vrsra_n_v:
  case ARM::BI__builtin_neon_vrsraq_n_v:
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[2] = EmitNeonShiftVector(Ops[2], Ty, true);
    Int = usgn ? Intrinsic::arm_neon_vrshiftu : Intrinsic::arm_neon_vrshifts;
    Ops[1] = Builder.CreateCall2(CGM.getIntrinsic(Int, Ty), Ops[1], Ops[2]);
    return Builder.CreateAdd(Ops[0], Ops[1], "vrsra_n");
  case ARM::BI__builtin_neon_vrsubhn_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vrsubhn, Ty),
                        Ops, "vrsubhn");
  case ARM::BI__builtin_neon_vshl_v:
  case ARM::BI__builtin_neon_vshlq_v:
    Int = usgn ? Intrinsic::arm_neon_vshiftu : Intrinsic::arm_neon_vshifts;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vshl");
  case ARM::BI__builtin_neon_vshll_n_v:
    Int = usgn ? Intrinsic::arm_neon_vshiftlu : Intrinsic::arm_neon_vshiftls;
    return EmitNeonCall(CGM.getIntrinsic(Int, Ty), Ops, "vshll", 1);
  case ARM::BI__builtin_neon_vshl_n_v:
  case ARM::BI__builtin_neon_vshlq_n_v:
    Ops[1] = EmitNeonShiftVector(Ops[1], Ty, false);
    return Builder.CreateShl(Builder.CreateBitCast(Ops[0],Ty), Ops[1],
                             "vshl_n");
  case ARM::BI__builtin_neon_vshrn_n_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vshiftn, Ty),
                        Ops, "vshrn_n", 1, true);
  case ARM::BI__builtin_neon_vshr_n_v:
  case ARM::BI__builtin_neon_vshrq_n_v:
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = EmitNeonShiftVector(Ops[1], Ty, false);
    if (usgn)
      return Builder.CreateLShr(Ops[0], Ops[1], "vshr_n");
    else
      return Builder.CreateAShr(Ops[0], Ops[1], "vshr_n");
  case ARM::BI__builtin_neon_vsri_n_v:
  case ARM::BI__builtin_neon_vsriq_n_v:
    rightShift = true;
  case ARM::BI__builtin_neon_vsli_n_v:
  case ARM::BI__builtin_neon_vsliq_n_v:
    Ops[2] = EmitNeonShiftVector(Ops[2], Ty, rightShift);
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vshiftins, Ty),
                        Ops, "vsli_n");
  case ARM::BI__builtin_neon_vsra_n_v:
  case ARM::BI__builtin_neon_vsraq_n_v:
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[2] = EmitNeonShiftVector(Ops[2], Ty, false);
    if (usgn)
      Ops[1] = Builder.CreateLShr(Ops[1], Ops[2], "vsra_n");
    else
      Ops[1] = Builder.CreateAShr(Ops[1], Ops[2], "vsra_n");
    return Builder.CreateAdd(Ops[0], Ops[1]);
  case ARM::BI__builtin_neon_vst1_v:
  case ARM::BI__builtin_neon_vst1q_v:
    Ops.push_back(Align);
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vst1, Ty),
                        Ops, "");
  case ARM::BI__builtin_neon_vst1q_lane_v:
    // Handle 64-bit integer elements as a special case.  Use a shuffle to get
    // a one-element vector and avoid poor code for i64 in the backend.
    if (VTy->getElementType()->isIntegerTy(64)) {
      Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
      Value *SV = llvm::ConstantVector::get(cast<llvm::Constant>(Ops[2]));
      Ops[1] = Builder.CreateShuffleVector(Ops[1], Ops[1], SV);
      Ops[2] = Align;
      return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::arm_neon_vst1,
                                                 Ops[1]->getType()), Ops);
    }
    // fall through
  case ARM::BI__builtin_neon_vst1_lane_v: {
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[1] = Builder.CreateExtractElement(Ops[1], Ops[2]);
    Ty = llvm::PointerType::getUnqual(Ops[1]->getType());
    StoreInst *St = Builder.CreateStore(Ops[1],
                                        Builder.CreateBitCast(Ops[0], Ty));
    St->setAlignment(cast<ConstantInt>(Align)->getZExtValue());
    return St;
  }
  case ARM::BI__builtin_neon_vst2_v:
  case ARM::BI__builtin_neon_vst2q_v:
    Ops.push_back(Align);
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vst2, Ty),
                        Ops, "");
  case ARM::BI__builtin_neon_vst2_lane_v:
  case ARM::BI__builtin_neon_vst2q_lane_v:
    Ops.push_back(Align);
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vst2lane, Ty),
                        Ops, "");
  case ARM::BI__builtin_neon_vst3_v:
  case ARM::BI__builtin_neon_vst3q_v:
    Ops.push_back(Align);
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vst3, Ty),
                        Ops, "");
  case ARM::BI__builtin_neon_vst3_lane_v:
  case ARM::BI__builtin_neon_vst3q_lane_v:
    Ops.push_back(Align);
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vst3lane, Ty),
                        Ops, "");
  case ARM::BI__builtin_neon_vst4_v:
  case ARM::BI__builtin_neon_vst4q_v:
    Ops.push_back(Align);
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vst4, Ty),
                        Ops, "");
  case ARM::BI__builtin_neon_vst4_lane_v:
  case ARM::BI__builtin_neon_vst4q_lane_v:
    Ops.push_back(Align);
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vst4lane, Ty),
                        Ops, "");
  case ARM::BI__builtin_neon_vsubhn_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vsubhn, Ty),
                        Ops, "vsubhn");
  case ARM::BI__builtin_neon_vtbl1_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vtbl1),
                        Ops, "vtbl1");
  case ARM::BI__builtin_neon_vtbl2_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vtbl2),
                        Ops, "vtbl2");
  case ARM::BI__builtin_neon_vtbl3_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vtbl3),
                        Ops, "vtbl3");
  case ARM::BI__builtin_neon_vtbl4_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vtbl4),
                        Ops, "vtbl4");
  case ARM::BI__builtin_neon_vtbx1_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vtbx1),
                        Ops, "vtbx1");
  case ARM::BI__builtin_neon_vtbx2_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vtbx2),
                        Ops, "vtbx2");
  case ARM::BI__builtin_neon_vtbx3_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vtbx3),
                        Ops, "vtbx3");
  case ARM::BI__builtin_neon_vtbx4_v:
    return EmitNeonCall(CGM.getIntrinsic(Intrinsic::arm_neon_vtbx4),
                        Ops, "vtbx4");
  case ARM::BI__builtin_neon_vtst_v:
  case ARM::BI__builtin_neon_vtstq_v: {
    Ops[0] = Builder.CreateBitCast(Ops[0], Ty);
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[0] = Builder.CreateAnd(Ops[0], Ops[1]);
    Ops[0] = Builder.CreateICmp(ICmpInst::ICMP_NE, Ops[0],
                                ConstantAggregateZero::get(Ty));
    return Builder.CreateSExt(Ops[0], Ty, "vtst");
  }
  case ARM::BI__builtin_neon_vtrn_v:
  case ARM::BI__builtin_neon_vtrnq_v: {
    Ops[0] = Builder.CreateBitCast(Ops[0], llvm::PointerType::getUnqual(Ty));
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);
    Value *SV = 0;

    for (unsigned vi = 0; vi != 2; ++vi) {
      SmallVector<Constant*, 16> Indices;
      for (unsigned i = 0, e = VTy->getNumElements(); i != e; i += 2) {
        Indices.push_back(Builder.getInt32(i+vi));
        Indices.push_back(Builder.getInt32(i+e+vi));
      }
      Value *Addr = Builder.CreateConstInBoundsGEP1_32(Ops[0], vi);
      SV = llvm::ConstantVector::get(Indices);
      SV = Builder.CreateShuffleVector(Ops[1], Ops[2], SV, "vtrn");
      SV = Builder.CreateStore(SV, Addr);
    }
    return SV;
  }
  case ARM::BI__builtin_neon_vuzp_v:
  case ARM::BI__builtin_neon_vuzpq_v: {
    Ops[0] = Builder.CreateBitCast(Ops[0], llvm::PointerType::getUnqual(Ty));
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);
    Value *SV = 0;

    for (unsigned vi = 0; vi != 2; ++vi) {
      SmallVector<Constant*, 16> Indices;
      for (unsigned i = 0, e = VTy->getNumElements(); i != e; ++i)
        Indices.push_back(ConstantInt::get(Int32Ty, 2*i+vi));

      Value *Addr = Builder.CreateConstInBoundsGEP1_32(Ops[0], vi);
      SV = llvm::ConstantVector::get(Indices);
      SV = Builder.CreateShuffleVector(Ops[1], Ops[2], SV, "vuzp");
      SV = Builder.CreateStore(SV, Addr);
    }
    return SV;
  }
  case ARM::BI__builtin_neon_vzip_v:
  case ARM::BI__builtin_neon_vzipq_v: {
    Ops[0] = Builder.CreateBitCast(Ops[0], llvm::PointerType::getUnqual(Ty));
    Ops[1] = Builder.CreateBitCast(Ops[1], Ty);
    Ops[2] = Builder.CreateBitCast(Ops[2], Ty);
    Value *SV = 0;

    for (unsigned vi = 0; vi != 2; ++vi) {
      SmallVector<Constant*, 16> Indices;
      for (unsigned i = 0, e = VTy->getNumElements(); i != e; i += 2) {
        Indices.push_back(ConstantInt::get(Int32Ty, (i + vi*e) >> 1));
        Indices.push_back(ConstantInt::get(Int32Ty, ((i + vi*e) >> 1)+e));
      }
      Value *Addr = Builder.CreateConstInBoundsGEP1_32(Ops[0], vi);
      SV = llvm::ConstantVector::get(Indices);
      SV = Builder.CreateShuffleVector(Ops[1], Ops[2], SV, "vzip");
      SV = Builder.CreateStore(SV, Addr);
    }
    return SV;
  }
  }
}

llvm::Value *CodeGenFunction::
BuildVector(ArrayRef<llvm::Value*> Ops) {
  assert((Ops.size() & (Ops.size() - 1)) == 0 &&
         "Not a power-of-two sized vector!");
  bool AllConstants = true;
  for (unsigned i = 0, e = Ops.size(); i != e && AllConstants; ++i)
    AllConstants &= isa<Constant>(Ops[i]);

  // If this is a constant vector, create a ConstantVector.
  if (AllConstants) {
    SmallVector<llvm::Constant*, 16> CstOps;
    for (unsigned i = 0, e = Ops.size(); i != e; ++i)
      CstOps.push_back(cast<Constant>(Ops[i]));
    return llvm::ConstantVector::get(CstOps);
  }

  // Otherwise, insertelement the values to build the vector.
  Value *Result =
    llvm::UndefValue::get(llvm::VectorType::get(Ops[0]->getType(), Ops.size()));

  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    Result = Builder.CreateInsertElement(Result, Ops[i], Builder.getInt32(i));

  return Result;
}

Value *CodeGenFunction::EmitX86BuiltinExpr(unsigned BuiltinID,
                                           const CallExpr *E) {
  SmallVector<Value*, 4> Ops;

  // Find out if any arguments are required to be integer constant expressions.
  unsigned ICEArguments = 0;
  ASTContext::GetBuiltinTypeError Error;
  getContext().GetBuiltinType(BuiltinID, Error, &ICEArguments);
  assert(Error == ASTContext::GE_None && "Should not codegen an error");

  for (unsigned i = 0, e = E->getNumArgs(); i != e; i++) {
    // If this is a normal argument, just emit it as a scalar.
    if ((ICEArguments & (1 << i)) == 0) {
      Ops.push_back(EmitScalarExpr(E->getArg(i)));
      continue;
    }

    // If this is required to be a constant, constant fold it so that we know
    // that the generated intrinsic gets a ConstantInt.
    llvm::APSInt Result;
    bool IsConst = E->getArg(i)->isIntegerConstantExpr(Result, getContext());
    assert(IsConst && "Constant arg isn't actually constant?"); (void)IsConst;
    Ops.push_back(llvm::ConstantInt::get(getLLVMContext(), Result));
  }

  switch (BuiltinID) {
  default: return 0;
  case X86::BI__builtin_ia32_vec_init_v8qi:
  case X86::BI__builtin_ia32_vec_init_v4hi:
  case X86::BI__builtin_ia32_vec_init_v2si:
    return Builder.CreateBitCast(BuildVector(Ops),
                                 llvm::Type::getX86_MMXTy(getLLVMContext()));
  case X86::BI__builtin_ia32_vec_ext_v2si:
    return Builder.CreateExtractElement(Ops[0],
                                  llvm::ConstantInt::get(Ops[1]->getType(), 0));
  case X86::BI__builtin_ia32_ldmxcsr: {
    llvm::Type *PtrTy = Int8PtrTy;
    Value *One = llvm::ConstantInt::get(Int32Ty, 1);
    Value *Tmp = Builder.CreateAlloca(Int32Ty, One);
    Builder.CreateStore(Ops[0], Tmp);
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::x86_sse_ldmxcsr),
                              Builder.CreateBitCast(Tmp, PtrTy));
  }
  case X86::BI__builtin_ia32_stmxcsr: {
    llvm::Type *PtrTy = Int8PtrTy;
    Value *One = llvm::ConstantInt::get(Int32Ty, 1);
    Value *Tmp = Builder.CreateAlloca(Int32Ty, One);
    Builder.CreateCall(CGM.getIntrinsic(Intrinsic::x86_sse_stmxcsr),
                       Builder.CreateBitCast(Tmp, PtrTy));
    return Builder.CreateLoad(Tmp, "stmxcsr");
  }
  case X86::BI__builtin_ia32_storehps:
  case X86::BI__builtin_ia32_storelps: {
    llvm::Type *PtrTy = llvm::PointerType::getUnqual(Int64Ty);
    llvm::Type *VecTy = llvm::VectorType::get(Int64Ty, 2);

    // cast val v2i64
    Ops[1] = Builder.CreateBitCast(Ops[1], VecTy, "cast");

    // extract (0, 1)
    unsigned Index = BuiltinID == X86::BI__builtin_ia32_storelps ? 0 : 1;
    llvm::Value *Idx = llvm::ConstantInt::get(Int32Ty, Index);
    Ops[1] = Builder.CreateExtractElement(Ops[1], Idx, "extract");

    // cast pointer to i64 & store
    Ops[0] = Builder.CreateBitCast(Ops[0], PtrTy);
    return Builder.CreateStore(Ops[1], Ops[0]);
  }
  case X86::BI__builtin_ia32_palignr: {
    unsigned shiftVal = cast<llvm::ConstantInt>(Ops[2])->getZExtValue();

    // If palignr is shifting the pair of input vectors less than 9 bytes,
    // emit a shuffle instruction.
    if (shiftVal <= 8) {
      SmallVector<llvm::Constant*, 8> Indices;
      for (unsigned i = 0; i != 8; ++i)
        Indices.push_back(llvm::ConstantInt::get(Int32Ty, shiftVal + i));

      Value* SV = llvm::ConstantVector::get(Indices);
      return Builder.CreateShuffleVector(Ops[1], Ops[0], SV, "palignr");
    }

    // If palignr is shifting the pair of input vectors more than 8 but less
    // than 16 bytes, emit a logical right shift of the destination.
    if (shiftVal < 16) {
      // MMX has these as 1 x i64 vectors for some odd optimization reasons.
      llvm::Type *VecTy = llvm::VectorType::get(Int64Ty, 1);

      Ops[0] = Builder.CreateBitCast(Ops[0], VecTy, "cast");
      Ops[1] = llvm::ConstantInt::get(VecTy, (shiftVal-8) * 8);

      // create i32 constant
      llvm::Function *F = CGM.getIntrinsic(Intrinsic::x86_mmx_psrl_q);
      return Builder.CreateCall(F, makeArrayRef(&Ops[0], 2), "palignr");
    }

    // If palignr is shifting the pair of vectors more than 16 bytes, emit zero.
    return llvm::Constant::getNullValue(ConvertType(E->getType()));
  }
  case X86::BI__builtin_ia32_palignr128: {
    unsigned shiftVal = cast<llvm::ConstantInt>(Ops[2])->getZExtValue();

    // If palignr is shifting the pair of input vectors less than 17 bytes,
    // emit a shuffle instruction.
    if (shiftVal <= 16) {
      SmallVector<llvm::Constant*, 16> Indices;
      for (unsigned i = 0; i != 16; ++i)
        Indices.push_back(llvm::ConstantInt::get(Int32Ty, shiftVal + i));

      Value* SV = llvm::ConstantVector::get(Indices);
      return Builder.CreateShuffleVector(Ops[1], Ops[0], SV, "palignr");
    }

    // If palignr is shifting the pair of input vectors more than 16 but less
    // than 32 bytes, emit a logical right shift of the destination.
    if (shiftVal < 32) {
      llvm::Type *VecTy = llvm::VectorType::get(Int64Ty, 2);

      Ops[0] = Builder.CreateBitCast(Ops[0], VecTy, "cast");
      Ops[1] = llvm::ConstantInt::get(Int32Ty, (shiftVal-16) * 8);

      // create i32 constant
      llvm::Function *F = CGM.getIntrinsic(Intrinsic::x86_sse2_psrl_dq);
      return Builder.CreateCall(F, makeArrayRef(&Ops[0], 2), "palignr");
    }

    // If palignr is shifting the pair of vectors more than 32 bytes, emit zero.
    return llvm::Constant::getNullValue(ConvertType(E->getType()));
  }
  case X86::BI__builtin_ia32_palignr256: {
    unsigned shiftVal = cast<llvm::ConstantInt>(Ops[2])->getZExtValue();

    // If palignr is shifting the pair of input vectors less than 17 bytes,
    // emit a shuffle instruction.
    if (shiftVal <= 16) {
      SmallVector<llvm::Constant*, 32> Indices;
      // 256-bit palignr operates on 128-bit lanes so we need to handle that
      for (unsigned l = 0; l != 2; ++l) {
        unsigned LaneStart = l * 16;
        unsigned LaneEnd = (l+1) * 16;
        for (unsigned i = 0; i != 16; ++i) {
          unsigned Idx = shiftVal + i + LaneStart;
          if (Idx >= LaneEnd) Idx += 16; // end of lane, switch operand
          Indices.push_back(llvm::ConstantInt::get(Int32Ty, Idx));
        }
      }

      Value* SV = llvm::ConstantVector::get(Indices);
      return Builder.CreateShuffleVector(Ops[1], Ops[0], SV, "palignr");
    }

    // If palignr is shifting the pair of input vectors more than 16 but less
    // than 32 bytes, emit a logical right shift of the destination.
    if (shiftVal < 32) {
      llvm::Type *VecTy = llvm::VectorType::get(Int64Ty, 4);

      Ops[0] = Builder.CreateBitCast(Ops[0], VecTy, "cast");
      Ops[1] = llvm::ConstantInt::get(Int32Ty, (shiftVal-16) * 8);

      // create i32 constant
      llvm::Function *F = CGM.getIntrinsic(Intrinsic::x86_avx2_psrl_dq);
      return Builder.CreateCall(F, makeArrayRef(&Ops[0], 2), "palignr");
    }

    // If palignr is shifting the pair of vectors more than 32 bytes, emit zero.
    return llvm::Constant::getNullValue(ConvertType(E->getType()));
  }
  case X86::BI__builtin_ia32_movntps:
  case X86::BI__builtin_ia32_movntps256:
  case X86::BI__builtin_ia32_movntpd:
  case X86::BI__builtin_ia32_movntpd256:
  case X86::BI__builtin_ia32_movntdq:
  case X86::BI__builtin_ia32_movntdq256:
  case X86::BI__builtin_ia32_movnti: {
    llvm::MDNode *Node = llvm::MDNode::get(getLLVMContext(),
                                           Builder.getInt32(1));

    // Convert the type of the pointer to a pointer to the stored type.
    Value *BC = Builder.CreateBitCast(Ops[0],
                                llvm::PointerType::getUnqual(Ops[1]->getType()),
                                      "cast");
    StoreInst *SI = Builder.CreateStore(Ops[1], BC);
    SI->setMetadata(CGM.getModule().getMDKindID("nontemporal"), Node);
    SI->setAlignment(16);
    return SI;
  }
  // 3DNow!
  case X86::BI__builtin_ia32_pswapdsf:
  case X86::BI__builtin_ia32_pswapdsi: {
    const char *name = 0;
    Intrinsic::ID ID = Intrinsic::not_intrinsic;
    switch(BuiltinID) {
    default: llvm_unreachable("Unsupported intrinsic!");
    case X86::BI__builtin_ia32_pswapdsf:
    case X86::BI__builtin_ia32_pswapdsi:
      name = "pswapd";
      ID = Intrinsic::x86_3dnowa_pswapd;
      break;
    }
    llvm::Type *MMXTy = llvm::Type::getX86_MMXTy(getLLVMContext());
    Ops[0] = Builder.CreateBitCast(Ops[0], MMXTy, "cast");
    llvm::Function *F = CGM.getIntrinsic(ID);
    return Builder.CreateCall(F, Ops, name);
  }
  case X86::BI__builtin_ia32_rdrand16_step:
  case X86::BI__builtin_ia32_rdrand32_step:
  case X86::BI__builtin_ia32_rdrand64_step: {
    Intrinsic::ID ID;
    switch (BuiltinID) {
    default: llvm_unreachable("Unsupported intrinsic!");
    case X86::BI__builtin_ia32_rdrand16_step:
      ID = Intrinsic::x86_rdrand_16;
      break;
    case X86::BI__builtin_ia32_rdrand32_step:
      ID = Intrinsic::x86_rdrand_32;
      break;
    case X86::BI__builtin_ia32_rdrand64_step:
      ID = Intrinsic::x86_rdrand_64;
      break;
    }

    Value *Call = Builder.CreateCall(CGM.getIntrinsic(ID));
    Builder.CreateStore(Builder.CreateExtractValue(Call, 0), Ops[0]);
    return Builder.CreateExtractValue(Call, 1);
  }
  }
}


Value *CodeGenFunction::EmitPPCBuiltinExpr(unsigned BuiltinID,
                                           const CallExpr *E) {
  SmallVector<Value*, 4> Ops;

  for (unsigned i = 0, e = E->getNumArgs(); i != e; i++)
    Ops.push_back(EmitScalarExpr(E->getArg(i)));

  Intrinsic::ID ID = Intrinsic::not_intrinsic;

  switch (BuiltinID) {
  default: return 0;

  // vec_ld, vec_lvsl, vec_lvsr
  case PPC::BI__builtin_altivec_lvx:
  case PPC::BI__builtin_altivec_lvxl:
  case PPC::BI__builtin_altivec_lvebx:
  case PPC::BI__builtin_altivec_lvehx:
  case PPC::BI__builtin_altivec_lvewx:
  case PPC::BI__builtin_altivec_lvsl:
  case PPC::BI__builtin_altivec_lvsr:
  {
    Ops[1] = Builder.CreateBitCast(Ops[1], Int8PtrTy);

    Ops[0] = Builder.CreateGEP(Ops[1], Ops[0]);
    Ops.pop_back();

    switch (BuiltinID) {
    default: llvm_unreachable("Unsupported ld/lvsl/lvsr intrinsic!");
    case PPC::BI__builtin_altivec_lvx:
      ID = Intrinsic::ppc_altivec_lvx;
      break;
    case PPC::BI__builtin_altivec_lvxl:
      ID = Intrinsic::ppc_altivec_lvxl;
      break;
    case PPC::BI__builtin_altivec_lvebx:
      ID = Intrinsic::ppc_altivec_lvebx;
      break;
    case PPC::BI__builtin_altivec_lvehx:
      ID = Intrinsic::ppc_altivec_lvehx;
      break;
    case PPC::BI__builtin_altivec_lvewx:
      ID = Intrinsic::ppc_altivec_lvewx;
      break;
    case PPC::BI__builtin_altivec_lvsl:
      ID = Intrinsic::ppc_altivec_lvsl;
      break;
    case PPC::BI__builtin_altivec_lvsr:
      ID = Intrinsic::ppc_altivec_lvsr;
      break;
    }
    llvm::Function *F = CGM.getIntrinsic(ID);
    return Builder.CreateCall(F, Ops, "");
  }

  // vec_st
  case PPC::BI__builtin_altivec_stvx:
  case PPC::BI__builtin_altivec_stvxl:
  case PPC::BI__builtin_altivec_stvebx:
  case PPC::BI__builtin_altivec_stvehx:
  case PPC::BI__builtin_altivec_stvewx:
  {
    Ops[2] = Builder.CreateBitCast(Ops[2], Int8PtrTy);
    Ops[1] = Builder.CreateGEP(Ops[2], Ops[1]);
    Ops.pop_back();

    switch (BuiltinID) {
    default: llvm_unreachable("Unsupported st intrinsic!");
    case PPC::BI__builtin_altivec_stvx:
      ID = Intrinsic::ppc_altivec_stvx;
      break;
    case PPC::BI__builtin_altivec_stvxl:
      ID = Intrinsic::ppc_altivec_stvxl;
      break;
    case PPC::BI__builtin_altivec_stvebx:
      ID = Intrinsic::ppc_altivec_stvebx;
      break;
    case PPC::BI__builtin_altivec_stvehx:
      ID = Intrinsic::ppc_altivec_stvehx;
      break;
    case PPC::BI__builtin_altivec_stvewx:
      ID = Intrinsic::ppc_altivec_stvewx;
      break;
    }
    llvm::Function *F = CGM.getIntrinsic(ID);
    return Builder.CreateCall(F, Ops, "");
  }
  }
}
