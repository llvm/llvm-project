//===---------- PPC.cpp - Emit LLVM Code for builtins ---------------------===//
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
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicsPowerPC.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm;

static llvm::Value *emitPPCLoadReserveIntrinsic(CodeGenFunction &CGF,
                                                unsigned BuiltinID,
                                                const CallExpr *E) {
  Value *Addr = CGF.EmitScalarExpr(E->getArg(0));

  SmallString<64> Asm;
  raw_svector_ostream AsmOS(Asm);
  llvm::IntegerType *RetType = CGF.Int32Ty;

  switch (BuiltinID) {
  case clang::PPC::BI__builtin_ppc_ldarx:
    AsmOS << "ldarx ";
    RetType = CGF.Int64Ty;
    break;
  case clang::PPC::BI__builtin_ppc_lwarx:
    AsmOS << "lwarx ";
    RetType = CGF.Int32Ty;
    break;
  case clang::PPC::BI__builtin_ppc_lharx:
    AsmOS << "lharx ";
    RetType = CGF.Int16Ty;
    break;
  case clang::PPC::BI__builtin_ppc_lbarx:
    AsmOS << "lbarx ";
    RetType = CGF.Int8Ty;
    break;
  default:
    llvm_unreachable("Expected only PowerPC load reserve intrinsics");
  }

  AsmOS << "$0, ${1:y}";

  std::string Constraints = "=r,*Z,~{memory}";
  std::string_view MachineClobbers = CGF.getTarget().getClobbers();
  if (!MachineClobbers.empty()) {
    Constraints += ',';
    Constraints += MachineClobbers;
  }

  llvm::Type *PtrType = CGF.DefaultPtrTy;
  llvm::FunctionType *FTy = llvm::FunctionType::get(RetType, {PtrType}, false);

  llvm::InlineAsm *IA =
      llvm::InlineAsm::get(FTy, Asm, Constraints, /*hasSideEffects=*/true);
  llvm::CallInst *CI = CGF.Builder.CreateCall(IA, {Addr});
  CI->addParamAttr(
      0, Attribute::get(CGF.getLLVMContext(), Attribute::ElementType, RetType));
  return CI;
}

Value *CodeGenFunction::EmitPPCBuiltinExpr(unsigned BuiltinID,
                                           const CallExpr *E) {
  // Do not emit the builtin arguments in the arguments of a function call,
  // because the evaluation order of function arguments is not specified in C++.
  // This is important when testing to ensure the arguments are emitted in the
  // same order every time. Eg:
  // Instead of:
  //   return Builder.CreateFDiv(EmitScalarExpr(E->getArg(0)),
  //                             EmitScalarExpr(E->getArg(1)), "swdiv");
  // Use:
  //   Value *Op0 = EmitScalarExpr(E->getArg(0));
  //   Value *Op1 = EmitScalarExpr(E->getArg(1));
  //   return Builder.CreateFDiv(Op0, Op1, "swdiv")

  Intrinsic::ID ID = Intrinsic::not_intrinsic;

#include "llvm/TargetParser/PPCTargetParser.def"
  auto GenAIXPPCBuiltinCpuExpr = [&](unsigned SupportMethod, unsigned FieldIdx,
                                     unsigned Mask, CmpInst::Predicate CompOp,
                                     unsigned OpValue) -> Value * {
    if (SupportMethod == BUILTIN_PPC_FALSE)
      return llvm::ConstantInt::getFalse(ConvertType(E->getType()));

    if (SupportMethod == BUILTIN_PPC_TRUE)
      return llvm::ConstantInt::getTrue(ConvertType(E->getType()));

    assert(SupportMethod <= SYS_CALL && "Invalid value for SupportMethod.");

    llvm::Value *FieldValue = nullptr;
    if (SupportMethod == USE_SYS_CONF) {
      llvm::Type *STy = llvm::StructType::get(PPC_SYSTEMCONFIG_TYPE);
      llvm::Constant *SysConf =
          CGM.CreateRuntimeVariable(STy, "_system_configuration");

      // Grab the appropriate field from _system_configuration.
      llvm::Value *Idxs[] = {ConstantInt::get(Int32Ty, 0),
                             ConstantInt::get(Int32Ty, FieldIdx)};

      FieldValue = Builder.CreateInBoundsGEP(STy, SysConf, Idxs);
      FieldValue = Builder.CreateAlignedLoad(Int32Ty, FieldValue,
                                             CharUnits::fromQuantity(4));
    } else if (SupportMethod == SYS_CALL) {
      llvm::FunctionType *FTy =
          llvm::FunctionType::get(Int64Ty, Int32Ty, false);
      llvm::FunctionCallee Func =
          CGM.CreateRuntimeFunction(FTy, "getsystemcfg");

      FieldValue =
          Builder.CreateCall(Func, {ConstantInt::get(Int32Ty, FieldIdx)});
    }
    assert(FieldValue &&
           "SupportMethod value is not defined in PPCTargetParser.def.");

    if (Mask)
      FieldValue = Builder.CreateAnd(FieldValue, Mask);

    llvm::Type *ValueType = FieldValue->getType();
    bool IsValueType64Bit = ValueType->isIntegerTy(64);
    assert(
        (IsValueType64Bit || ValueType->isIntegerTy(32)) &&
        "Only 32/64-bit integers are supported in GenAIXPPCBuiltinCpuExpr().");

    return Builder.CreateICmp(
        CompOp, FieldValue,
        ConstantInt::get(IsValueType64Bit ? Int64Ty : Int32Ty, OpValue));
  };

  switch (BuiltinID) {
  default: return nullptr;

  case Builtin::BI__builtin_cpu_is: {
    const Expr *CPUExpr = E->getArg(0)->IgnoreParenCasts();
    StringRef CPUStr = cast<clang::StringLiteral>(CPUExpr)->getString();
    llvm::Triple Triple = getTarget().getTriple();

    typedef std::tuple<unsigned, unsigned, unsigned, unsigned> CPUInfo;

    auto [LinuxSupportMethod, LinuxIDValue, AIXSupportMethod, AIXIDValue] =
        static_cast<CPUInfo>(StringSwitch<CPUInfo>(CPUStr)
#define PPC_CPU(NAME, Linux_SUPPORT_METHOD, LinuxID, AIX_SUPPORT_METHOD,       \
                AIXID)                                                         \
  .Case(NAME, {Linux_SUPPORT_METHOD, LinuxID, AIX_SUPPORT_METHOD, AIXID})
#include "llvm/TargetParser/PPCTargetParser.def"
                                 .Default({BUILTIN_PPC_UNSUPPORTED, 0,
                                           BUILTIN_PPC_UNSUPPORTED, 0}));

    if (Triple.isOSAIX()) {
      assert((AIXSupportMethod != BUILTIN_PPC_UNSUPPORTED) &&
             "Invalid CPU name. Missed by SemaChecking?");
      return GenAIXPPCBuiltinCpuExpr(AIXSupportMethod, AIX_SYSCON_IMPL_IDX, 0,
                                     ICmpInst::ICMP_EQ, AIXIDValue);
    }

    assert(Triple.isOSLinux() &&
           "__builtin_cpu_is() is only supported for AIX and Linux.");

    assert((LinuxSupportMethod != BUILTIN_PPC_UNSUPPORTED) &&
           "Invalid CPU name. Missed by SemaChecking?");

    if (LinuxSupportMethod == BUILTIN_PPC_FALSE)
      return llvm::ConstantInt::getFalse(ConvertType(E->getType()));

    Value *Op0 = llvm::ConstantInt::get(Int32Ty, PPC_FAWORD_CPUID);
    llvm::Function *F = CGM.getIntrinsic(Intrinsic::ppc_fixed_addr_ld);
    Value *TheCall = Builder.CreateCall(F, {Op0}, "cpu_is");
    return Builder.CreateICmpEQ(TheCall,
                                llvm::ConstantInt::get(Int32Ty, LinuxIDValue));
  }
  case Builtin::BI__builtin_cpu_supports: {
    llvm::Triple Triple = getTarget().getTriple();
    const Expr *CPUExpr = E->getArg(0)->IgnoreParenCasts();
    StringRef CPUStr = cast<clang::StringLiteral>(CPUExpr)->getString();
    if (Triple.isOSAIX()) {
      typedef std::tuple<unsigned, unsigned, unsigned, CmpInst::Predicate,
                         unsigned>
          CPUSupportType;
      auto [SupportMethod, FieldIdx, Mask, CompOp, Value] =
          static_cast<CPUSupportType>(StringSwitch<CPUSupportType>(CPUStr)
#define PPC_AIX_FEATURE(NAME, DESC, SUPPORT_METHOD, INDEX, MASK, COMP_OP,      \
                        VALUE)                                                 \
  .Case(NAME, {SUPPORT_METHOD, INDEX, MASK, COMP_OP, VALUE})
#include "llvm/TargetParser/PPCTargetParser.def"
                                          .Default({BUILTIN_PPC_FALSE, 0, 0,
                                                    CmpInst::Predicate(), 0}));
      return GenAIXPPCBuiltinCpuExpr(SupportMethod, FieldIdx, Mask, CompOp,
                                     Value);
    }

    assert(Triple.isOSLinux() &&
           "__builtin_cpu_supports() is only supported for AIX and Linux.");
    auto [FeatureWord, BitMask] =
        StringSwitch<std::pair<unsigned, unsigned>>(CPUStr)
#define PPC_LNX_FEATURE(Name, Description, EnumName, Bitmask, FA_WORD)         \
  .Case(Name, {FA_WORD, Bitmask})
#include "llvm/TargetParser/PPCTargetParser.def"
            .Default({0, 0});
    if (!BitMask)
      return Builder.getFalse();
    Value *Op0 = llvm::ConstantInt::get(Int32Ty, FeatureWord);
    llvm::Function *F = CGM.getIntrinsic(Intrinsic::ppc_fixed_addr_ld);
    Value *TheCall = Builder.CreateCall(F, {Op0}, "cpu_supports");
    Value *Mask =
        Builder.CreateAnd(TheCall, llvm::ConstantInt::get(Int32Ty, BitMask));
    return Builder.CreateICmpNE(Mask, llvm::Constant::getNullValue(Int32Ty));
#undef PPC_FAWORD_HWCAP
#undef PPC_FAWORD_HWCAP2
#undef PPC_FAWORD_CPUID
  }

  // __builtin_ppc_get_timebase is GCC 4.8+'s PowerPC-specific name for what we
  // call __builtin_readcyclecounter.
  case PPC::BI__builtin_ppc_get_timebase:
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::readcyclecounter));

  // vec_ld, vec_xl_be, vec_lvsl, vec_lvsr
  case PPC::BI__builtin_altivec_lvx:
  case PPC::BI__builtin_altivec_lvxl:
  case PPC::BI__builtin_altivec_lvebx:
  case PPC::BI__builtin_altivec_lvehx:
  case PPC::BI__builtin_altivec_lvewx:
  case PPC::BI__builtin_altivec_lvsl:
  case PPC::BI__builtin_altivec_lvsr:
  case PPC::BI__builtin_vsx_lxvd2x:
  case PPC::BI__builtin_vsx_lxvw4x:
  case PPC::BI__builtin_vsx_lxvd2x_be:
  case PPC::BI__builtin_vsx_lxvw4x_be:
  case PPC::BI__builtin_vsx_lxvl:
  case PPC::BI__builtin_vsx_lxvll:
  {
    SmallVector<Value *, 2> Ops;
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    if (!(BuiltinID == PPC::BI__builtin_vsx_lxvl ||
          BuiltinID == PPC::BI__builtin_vsx_lxvll)) {
      Ops[0] = Builder.CreateGEP(Int8Ty, Ops[1], Ops[0]);
      Ops.pop_back();
    }

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
    case PPC::BI__builtin_vsx_lxvd2x:
      ID = Intrinsic::ppc_vsx_lxvd2x;
      break;
    case PPC::BI__builtin_vsx_lxvw4x:
      ID = Intrinsic::ppc_vsx_lxvw4x;
      break;
    case PPC::BI__builtin_vsx_lxvd2x_be:
      ID = Intrinsic::ppc_vsx_lxvd2x_be;
      break;
    case PPC::BI__builtin_vsx_lxvw4x_be:
      ID = Intrinsic::ppc_vsx_lxvw4x_be;
      break;
    case PPC::BI__builtin_vsx_lxvl:
      ID = Intrinsic::ppc_vsx_lxvl;
      break;
    case PPC::BI__builtin_vsx_lxvll:
      ID = Intrinsic::ppc_vsx_lxvll;
      break;
    }
    llvm::Function *F = CGM.getIntrinsic(ID);
    return Builder.CreateCall(F, Ops, "");
  }

  // vec_st, vec_xst_be
  case PPC::BI__builtin_altivec_stvx:
  case PPC::BI__builtin_altivec_stvxl:
  case PPC::BI__builtin_altivec_stvebx:
  case PPC::BI__builtin_altivec_stvehx:
  case PPC::BI__builtin_altivec_stvewx:
  case PPC::BI__builtin_vsx_stxvd2x:
  case PPC::BI__builtin_vsx_stxvw4x:
  case PPC::BI__builtin_vsx_stxvd2x_be:
  case PPC::BI__builtin_vsx_stxvw4x_be:
  case PPC::BI__builtin_vsx_stxvl:
  case PPC::BI__builtin_vsx_stxvll:
  {
    SmallVector<Value *, 3> Ops;
    Ops.push_back(EmitScalarExpr(E->getArg(0)));
    Ops.push_back(EmitScalarExpr(E->getArg(1)));
    Ops.push_back(EmitScalarExpr(E->getArg(2)));
    if (!(BuiltinID == PPC::BI__builtin_vsx_stxvl ||
          BuiltinID == PPC::BI__builtin_vsx_stxvll)) {
      Ops[1] = Builder.CreateGEP(Int8Ty, Ops[2], Ops[1]);
      Ops.pop_back();
    }

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
    case PPC::BI__builtin_vsx_stxvd2x:
      ID = Intrinsic::ppc_vsx_stxvd2x;
      break;
    case PPC::BI__builtin_vsx_stxvw4x:
      ID = Intrinsic::ppc_vsx_stxvw4x;
      break;
    case PPC::BI__builtin_vsx_stxvd2x_be:
      ID = Intrinsic::ppc_vsx_stxvd2x_be;
      break;
    case PPC::BI__builtin_vsx_stxvw4x_be:
      ID = Intrinsic::ppc_vsx_stxvw4x_be;
      break;
    case PPC::BI__builtin_vsx_stxvl:
      ID = Intrinsic::ppc_vsx_stxvl;
      break;
    case PPC::BI__builtin_vsx_stxvll:
      ID = Intrinsic::ppc_vsx_stxvll;
      break;
    }
    llvm::Function *F = CGM.getIntrinsic(ID);
    return Builder.CreateCall(F, Ops, "");
  }
  case PPC::BI__builtin_vsx_ldrmb: {
    // Essentially boils down to performing an unaligned VMX load sequence so
    // as to avoid crossing a page boundary and then shuffling the elements
    // into the right side of the vector register.
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    int64_t NumBytes = cast<ConstantInt>(Op1)->getZExtValue();
    llvm::Type *ResTy = ConvertType(E->getType());
    bool IsLE = getTarget().isLittleEndian();

    // If the user wants the entire vector, just load the entire vector.
    if (NumBytes == 16) {
      Value *LD =
          Builder.CreateLoad(Address(Op0, ResTy, CharUnits::fromQuantity(1)));
      if (!IsLE)
        return LD;

      // Reverse the bytes on LE.
      SmallVector<int, 16> RevMask;
      for (int Idx = 0; Idx < 16; Idx++)
        RevMask.push_back(15 - Idx);
      return Builder.CreateShuffleVector(LD, LD, RevMask);
    }

    llvm::Function *Lvx = CGM.getIntrinsic(Intrinsic::ppc_altivec_lvx);
    llvm::Function *Lvs = CGM.getIntrinsic(IsLE ? Intrinsic::ppc_altivec_lvsr
                                                : Intrinsic::ppc_altivec_lvsl);
    llvm::Function *Vperm = CGM.getIntrinsic(Intrinsic::ppc_altivec_vperm);
    Value *HiMem = Builder.CreateGEP(
        Int8Ty, Op0, ConstantInt::get(Op1->getType(), NumBytes - 1));
    Value *LoLd = Builder.CreateCall(Lvx, Op0, "ld.lo");
    Value *HiLd = Builder.CreateCall(Lvx, HiMem, "ld.hi");
    Value *Mask1 = Builder.CreateCall(Lvs, Op0, "mask1");

    Op0 = IsLE ? HiLd : LoLd;
    Op1 = IsLE ? LoLd : HiLd;
    Value *AllElts = Builder.CreateCall(Vperm, {Op0, Op1, Mask1}, "shuffle1");
    Constant *Zero = llvm::Constant::getNullValue(IsLE ? ResTy : AllElts->getType());

    if (IsLE) {
      SmallVector<int, 16> Consts;
      for (int Idx = 0; Idx < 16; Idx++) {
        int Val = (NumBytes - Idx - 1 >= 0) ? (NumBytes - Idx - 1)
                                            : 16 - (NumBytes - Idx);
        Consts.push_back(Val);
      }
      return Builder.CreateShuffleVector(Builder.CreateBitCast(AllElts, ResTy),
                                         Zero, Consts);
    }
    SmallVector<Constant *, 16> Consts;
    for (int Idx = 0; Idx < 16; Idx++)
      Consts.push_back(Builder.getInt8(NumBytes + Idx));
    Value *Mask2 = ConstantVector::get(Consts);
    return Builder.CreateBitCast(
        Builder.CreateCall(Vperm, {Zero, AllElts, Mask2}, "shuffle2"), ResTy);
  }
  case PPC::BI__builtin_vsx_strmb: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Op2 = EmitScalarExpr(E->getArg(2));
    int64_t NumBytes = cast<ConstantInt>(Op1)->getZExtValue();
    bool IsLE = getTarget().isLittleEndian();
    auto StoreSubVec = [&](unsigned Width, unsigned Offset, unsigned EltNo) {
      // Storing the whole vector, simply store it on BE and reverse bytes and
      // store on LE.
      if (Width == 16) {
        Value *StVec = Op2;
        if (IsLE) {
          SmallVector<int, 16> RevMask;
          for (int Idx = 0; Idx < 16; Idx++)
            RevMask.push_back(15 - Idx);
          StVec = Builder.CreateShuffleVector(Op2, Op2, RevMask);
        }
        return Builder.CreateStore(
            StVec, Address(Op0, Op2->getType(), CharUnits::fromQuantity(1)));
      }
      auto *ConvTy = Int64Ty;
      unsigned NumElts = 0;
      switch (Width) {
      default:
        llvm_unreachable("width for stores must be a power of 2");
      case 8:
        ConvTy = Int64Ty;
        NumElts = 2;
        break;
      case 4:
        ConvTy = Int32Ty;
        NumElts = 4;
        break;
      case 2:
        ConvTy = Int16Ty;
        NumElts = 8;
        break;
      case 1:
        ConvTy = Int8Ty;
        NumElts = 16;
        break;
      }
      Value *Vec = Builder.CreateBitCast(
          Op2, llvm::FixedVectorType::get(ConvTy, NumElts));
      Value *Ptr =
          Builder.CreateGEP(Int8Ty, Op0, ConstantInt::get(Int64Ty, Offset));
      Value *Elt = Builder.CreateExtractElement(Vec, EltNo);
      if (IsLE && Width > 1) {
        Function *F = CGM.getIntrinsic(Intrinsic::bswap, ConvTy);
        Elt = Builder.CreateCall(F, Elt);
      }
      return Builder.CreateStore(
          Elt, Address(Ptr, ConvTy, CharUnits::fromQuantity(1)));
    };
    unsigned Stored = 0;
    unsigned RemainingBytes = NumBytes;
    Value *Result;
    if (NumBytes == 16)
      return StoreSubVec(16, 0, 0);
    if (NumBytes >= 8) {
      Result = StoreSubVec(8, NumBytes - 8, IsLE ? 0 : 1);
      RemainingBytes -= 8;
      Stored += 8;
    }
    if (RemainingBytes >= 4) {
      Result = StoreSubVec(4, NumBytes - Stored - 4,
                           IsLE ? (Stored >> 2) : 3 - (Stored >> 2));
      RemainingBytes -= 4;
      Stored += 4;
    }
    if (RemainingBytes >= 2) {
      Result = StoreSubVec(2, NumBytes - Stored - 2,
                           IsLE ? (Stored >> 1) : 7 - (Stored >> 1));
      RemainingBytes -= 2;
      Stored += 2;
    }
    if (RemainingBytes)
      Result =
          StoreSubVec(1, NumBytes - Stored - 1, IsLE ? Stored : 15 - Stored);
    return Result;
  }
  // Square root
  case PPC::BI__builtin_vsx_xvsqrtsp:
  case PPC::BI__builtin_vsx_xvsqrtdp: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *X = EmitScalarExpr(E->getArg(0));
    if (Builder.getIsFPConstrained()) {
      llvm::Function *F = CGM.getIntrinsic(
          Intrinsic::experimental_constrained_sqrt, ResultType);
      return Builder.CreateConstrainedFPCall(F, X);
    } else {
      llvm::Function *F = CGM.getIntrinsic(Intrinsic::sqrt, ResultType);
      return Builder.CreateCall(F, X);
    }
  }
  // Count leading zeros
  case PPC::BI__builtin_altivec_vclzb:
  case PPC::BI__builtin_altivec_vclzh:
  case PPC::BI__builtin_altivec_vclzw:
  case PPC::BI__builtin_altivec_vclzd: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *X = EmitScalarExpr(E->getArg(0));
    Value *Undef = ConstantInt::get(Builder.getInt1Ty(), false);
    Function *F = CGM.getIntrinsic(Intrinsic::ctlz, ResultType);
    return Builder.CreateCall(F, {X, Undef});
  }
  case PPC::BI__builtin_altivec_vctzb:
  case PPC::BI__builtin_altivec_vctzh:
  case PPC::BI__builtin_altivec_vctzw:
  case PPC::BI__builtin_altivec_vctzd: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *X = EmitScalarExpr(E->getArg(0));
    Value *Undef = ConstantInt::get(Builder.getInt1Ty(), false);
    Function *F = CGM.getIntrinsic(Intrinsic::cttz, ResultType);
    return Builder.CreateCall(F, {X, Undef});
  }
  case PPC::BI__builtin_altivec_vinsd:
  case PPC::BI__builtin_altivec_vinsw:
  case PPC::BI__builtin_altivec_vinsd_elt:
  case PPC::BI__builtin_altivec_vinsw_elt: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Op2 = EmitScalarExpr(E->getArg(2));

    bool IsUnaligned = (BuiltinID == PPC::BI__builtin_altivec_vinsw ||
                        BuiltinID == PPC::BI__builtin_altivec_vinsd);

    bool Is32bit = (BuiltinID == PPC::BI__builtin_altivec_vinsw ||
                    BuiltinID == PPC::BI__builtin_altivec_vinsw_elt);

    // The third argument must be a compile time constant.
    ConstantInt *ArgCI = dyn_cast<ConstantInt>(Op2);
    assert(ArgCI &&
           "Third Arg to vinsw/vinsd intrinsic must be a constant integer!");

    // Valid value for the third argument is dependent on the input type and
    // builtin called.
    int ValidMaxValue = 0;
    if (IsUnaligned)
      ValidMaxValue = (Is32bit) ? 12 : 8;
    else
      ValidMaxValue = (Is32bit) ? 3 : 1;

    // Get value of third argument.
    int64_t ConstArg = ArgCI->getSExtValue();

    // Compose range checking error message.
    std::string RangeErrMsg = IsUnaligned ? "byte" : "element";
    RangeErrMsg += " number " + llvm::to_string(ConstArg);
    RangeErrMsg += " is outside of the valid range [0, ";
    RangeErrMsg += llvm::to_string(ValidMaxValue) + "]";

    // Issue error if third argument is not within the valid range.
    if (ConstArg < 0 || ConstArg > ValidMaxValue)
      CGM.Error(E->getExprLoc(), RangeErrMsg);

    // Input to vec_replace_elt is an element index, convert to byte index.
    if (!IsUnaligned) {
      ConstArg *= Is32bit ? 4 : 8;
      // Fix the constant according to endianess.
      if (getTarget().isLittleEndian())
        ConstArg = (Is32bit ? 12 : 8) - ConstArg;
    }

    ID = Is32bit ? Intrinsic::ppc_altivec_vinsw : Intrinsic::ppc_altivec_vinsd;
    Op2 = ConstantInt::getSigned(Int32Ty, ConstArg);
    // Casting input to vector int as per intrinsic definition.
    Op0 =
        Is32bit
            ? Builder.CreateBitCast(Op0, llvm::FixedVectorType::get(Int32Ty, 4))
            : Builder.CreateBitCast(Op0,
                                    llvm::FixedVectorType::get(Int64Ty, 2));
    return Builder.CreateBitCast(
        Builder.CreateCall(CGM.getIntrinsic(ID), {Op0, Op1, Op2}), ResultType);
  }
  case PPC::BI__builtin_altivec_vadduqm:
  case PPC::BI__builtin_altivec_vsubuqm: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    llvm::Type *Int128Ty = llvm::IntegerType::get(getLLVMContext(), 128);
    Op0 = Builder.CreateBitCast(Op0, llvm::FixedVectorType::get(Int128Ty, 1));
    Op1 = Builder.CreateBitCast(Op1, llvm::FixedVectorType::get(Int128Ty, 1));
    if (BuiltinID == PPC::BI__builtin_altivec_vadduqm)
      return Builder.CreateAdd(Op0, Op1, "vadduqm");
    else
      return Builder.CreateSub(Op0, Op1, "vsubuqm");
  }
  case PPC::BI__builtin_altivec_vaddcuq_c:
  case PPC::BI__builtin_altivec_vsubcuq_c: {
    SmallVector<Value *, 2> Ops;
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    llvm::Type *V1I128Ty = llvm::FixedVectorType::get(
        llvm::IntegerType::get(getLLVMContext(), 128), 1);
    Ops.push_back(Builder.CreateBitCast(Op0, V1I128Ty));
    Ops.push_back(Builder.CreateBitCast(Op1, V1I128Ty));
    ID = (BuiltinID == PPC::BI__builtin_altivec_vaddcuq_c)
             ? Intrinsic::ppc_altivec_vaddcuq
             : Intrinsic::ppc_altivec_vsubcuq;
    return Builder.CreateCall(CGM.getIntrinsic(ID), Ops, "");
  }
  case PPC::BI__builtin_altivec_vaddeuqm_c:
  case PPC::BI__builtin_altivec_vaddecuq_c:
  case PPC::BI__builtin_altivec_vsubeuqm_c:
  case PPC::BI__builtin_altivec_vsubecuq_c: {
    SmallVector<Value *, 3> Ops;
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Op2 = EmitScalarExpr(E->getArg(2));
    llvm::Type *V1I128Ty = llvm::FixedVectorType::get(
        llvm::IntegerType::get(getLLVMContext(), 128), 1);
    Ops.push_back(Builder.CreateBitCast(Op0, V1I128Ty));
    Ops.push_back(Builder.CreateBitCast(Op1, V1I128Ty));
    Ops.push_back(Builder.CreateBitCast(Op2, V1I128Ty));
    switch (BuiltinID) {
    default:
      llvm_unreachable("Unsupported intrinsic!");
    case PPC::BI__builtin_altivec_vaddeuqm_c:
      ID = Intrinsic::ppc_altivec_vaddeuqm;
      break;
    case PPC::BI__builtin_altivec_vaddecuq_c:
      ID = Intrinsic::ppc_altivec_vaddecuq;
      break;
    case PPC::BI__builtin_altivec_vsubeuqm_c:
      ID = Intrinsic::ppc_altivec_vsubeuqm;
      break;
    case PPC::BI__builtin_altivec_vsubecuq_c:
      ID = Intrinsic::ppc_altivec_vsubecuq;
      break;
    }
    return Builder.CreateCall(CGM.getIntrinsic(ID), Ops, "");
  }
  case PPC::BI__builtin_ppc_rldimi:
  case PPC::BI__builtin_ppc_rlwimi: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Op2 = EmitScalarExpr(E->getArg(2));
    Value *Op3 = EmitScalarExpr(E->getArg(3));
    // rldimi is 64-bit instruction, expand the intrinsic before isel to
    // leverage peephole and avoid legalization efforts.
    if (BuiltinID == PPC::BI__builtin_ppc_rldimi &&
        !getTarget().getTriple().isPPC64()) {
      Function *F = CGM.getIntrinsic(Intrinsic::fshl, Op0->getType());
      Op2 = Builder.CreateZExt(Op2, Int64Ty);
      Value *Shift = Builder.CreateCall(F, {Op0, Op0, Op2});
      return Builder.CreateOr(Builder.CreateAnd(Shift, Op3),
                              Builder.CreateAnd(Op1, Builder.CreateNot(Op3)));
    }
    return Builder.CreateCall(
        CGM.getIntrinsic(BuiltinID == PPC::BI__builtin_ppc_rldimi
                             ? Intrinsic::ppc_rldimi
                             : Intrinsic::ppc_rlwimi),
        {Op0, Op1, Op2, Op3});
  }
  case PPC::BI__builtin_ppc_rlwnm: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Op2 = EmitScalarExpr(E->getArg(2));
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::ppc_rlwnm),
                              {Op0, Op1, Op2});
  }
  case PPC::BI__builtin_ppc_poppar4:
  case PPC::BI__builtin_ppc_poppar8: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    llvm::Type *ArgType = Op0->getType();
    Function *F = CGM.getIntrinsic(Intrinsic::ctpop, ArgType);
    Value *Tmp = Builder.CreateCall(F, Op0);

    llvm::Type *ResultType = ConvertType(E->getType());
    Value *Result = Builder.CreateAnd(Tmp, llvm::ConstantInt::get(ArgType, 1));
    if (Result->getType() != ResultType)
      Result = Builder.CreateIntCast(Result, ResultType, /*isSigned*/true,
                                     "cast");
    return Result;
  }
  case PPC::BI__builtin_ppc_cmpb: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    if (getTarget().getTriple().isPPC64()) {
      Function *F =
          CGM.getIntrinsic(Intrinsic::ppc_cmpb, {Int64Ty, Int64Ty, Int64Ty});
      return Builder.CreateCall(F, {Op0, Op1}, "cmpb");
    }
    // For 32 bit, emit the code as below:
    // %conv = trunc i64 %a to i32
    // %conv1 = trunc i64 %b to i32
    // %shr = lshr i64 %a, 32
    // %conv2 = trunc i64 %shr to i32
    // %shr3 = lshr i64 %b, 32
    // %conv4 = trunc i64 %shr3 to i32
    // %0 = tail call i32 @llvm.ppc.cmpb32(i32 %conv, i32 %conv1)
    // %conv5 = zext i32 %0 to i64
    // %1 = tail call i32 @llvm.ppc.cmpb32(i32 %conv2, i32 %conv4)
    // %conv614 = zext i32 %1 to i64
    // %shl = shl nuw i64 %conv614, 32
    // %or = or i64 %shl, %conv5
    // ret i64 %or
    Function *F =
        CGM.getIntrinsic(Intrinsic::ppc_cmpb, {Int32Ty, Int32Ty, Int32Ty});
    Value *ArgOneLo = Builder.CreateTrunc(Op0, Int32Ty);
    Value *ArgTwoLo = Builder.CreateTrunc(Op1, Int32Ty);
    Constant *ShiftAmt = ConstantInt::get(Int64Ty, 32);
    Value *ArgOneHi =
        Builder.CreateTrunc(Builder.CreateLShr(Op0, ShiftAmt), Int32Ty);
    Value *ArgTwoHi =
        Builder.CreateTrunc(Builder.CreateLShr(Op1, ShiftAmt), Int32Ty);
    Value *ResLo = Builder.CreateZExt(
        Builder.CreateCall(F, {ArgOneLo, ArgTwoLo}, "cmpb"), Int64Ty);
    Value *ResHiShift = Builder.CreateZExt(
        Builder.CreateCall(F, {ArgOneHi, ArgTwoHi}, "cmpb"), Int64Ty);
    Value *ResHi = Builder.CreateShl(ResHiShift, ShiftAmt);
    return Builder.CreateOr(ResLo, ResHi);
  }
  // Copy sign
  case PPC::BI__builtin_vsx_xvcpsgnsp:
  case PPC::BI__builtin_vsx_xvcpsgndp: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *X = EmitScalarExpr(E->getArg(0));
    Value *Y = EmitScalarExpr(E->getArg(1));
    ID = Intrinsic::copysign;
    llvm::Function *F = CGM.getIntrinsic(ID, ResultType);
    return Builder.CreateCall(F, {X, Y});
  }
  // Rounding/truncation
  case PPC::BI__builtin_vsx_xvrspip:
  case PPC::BI__builtin_vsx_xvrdpip:
  case PPC::BI__builtin_vsx_xvrdpim:
  case PPC::BI__builtin_vsx_xvrspim:
  case PPC::BI__builtin_vsx_xvrdpi:
  case PPC::BI__builtin_vsx_xvrspi:
  case PPC::BI__builtin_vsx_xvrdpic:
  case PPC::BI__builtin_vsx_xvrspic:
  case PPC::BI__builtin_vsx_xvrdpiz:
  case PPC::BI__builtin_vsx_xvrspiz: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *X = EmitScalarExpr(E->getArg(0));
    if (BuiltinID == PPC::BI__builtin_vsx_xvrdpim ||
        BuiltinID == PPC::BI__builtin_vsx_xvrspim)
      ID = Builder.getIsFPConstrained()
               ? Intrinsic::experimental_constrained_floor
               : Intrinsic::floor;
    else if (BuiltinID == PPC::BI__builtin_vsx_xvrdpi ||
             BuiltinID == PPC::BI__builtin_vsx_xvrspi)
      ID = Builder.getIsFPConstrained()
               ? Intrinsic::experimental_constrained_round
               : Intrinsic::round;
    else if (BuiltinID == PPC::BI__builtin_vsx_xvrdpic ||
             BuiltinID == PPC::BI__builtin_vsx_xvrspic)
      ID = Builder.getIsFPConstrained()
               ? Intrinsic::experimental_constrained_rint
               : Intrinsic::rint;
    else if (BuiltinID == PPC::BI__builtin_vsx_xvrdpip ||
             BuiltinID == PPC::BI__builtin_vsx_xvrspip)
      ID = Builder.getIsFPConstrained()
               ? Intrinsic::experimental_constrained_ceil
               : Intrinsic::ceil;
    else if (BuiltinID == PPC::BI__builtin_vsx_xvrdpiz ||
             BuiltinID == PPC::BI__builtin_vsx_xvrspiz)
      ID = Builder.getIsFPConstrained()
               ? Intrinsic::experimental_constrained_trunc
               : Intrinsic::trunc;
    llvm::Function *F = CGM.getIntrinsic(ID, ResultType);
    return Builder.getIsFPConstrained() ? Builder.CreateConstrainedFPCall(F, X)
                                        : Builder.CreateCall(F, X);
  }

  // Absolute value
  case PPC::BI__builtin_vsx_xvabsdp:
  case PPC::BI__builtin_vsx_xvabssp: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *X = EmitScalarExpr(E->getArg(0));
    llvm::Function *F = CGM.getIntrinsic(Intrinsic::fabs, ResultType);
    return Builder.CreateCall(F, X);
  }

  // Fastmath by default
  case PPC::BI__builtin_ppc_recipdivf:
  case PPC::BI__builtin_ppc_recipdivd:
  case PPC::BI__builtin_ppc_rsqrtf:
  case PPC::BI__builtin_ppc_rsqrtd: {
    FastMathFlags FMF = Builder.getFastMathFlags();
    Builder.getFastMathFlags().setFast();
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *X = EmitScalarExpr(E->getArg(0));

    if (BuiltinID == PPC::BI__builtin_ppc_recipdivf ||
        BuiltinID == PPC::BI__builtin_ppc_recipdivd) {
      Value *Y = EmitScalarExpr(E->getArg(1));
      Value *FDiv = Builder.CreateFDiv(X, Y, "recipdiv");
      Builder.getFastMathFlags() &= (FMF);
      return FDiv;
    }
    auto *One = ConstantFP::get(ResultType, 1.0);
    llvm::Function *F = CGM.getIntrinsic(Intrinsic::sqrt, ResultType);
    Value *FDiv = Builder.CreateFDiv(One, Builder.CreateCall(F, X), "rsqrt");
    Builder.getFastMathFlags() &= (FMF);
    return FDiv;
  }
  case PPC::BI__builtin_ppc_alignx: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    ConstantInt *AlignmentCI = cast<ConstantInt>(Op0);
    if (AlignmentCI->getValue().ugt(llvm::Value::MaximumAlignment))
      AlignmentCI = ConstantInt::get(AlignmentCI->getIntegerType(),
                                     llvm::Value::MaximumAlignment);

    emitAlignmentAssumption(Op1, E->getArg(1),
                            /*The expr loc is sufficient.*/ SourceLocation(),
                            AlignmentCI, nullptr);
    return Op1;
  }
  case PPC::BI__builtin_ppc_rdlam: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Op2 = EmitScalarExpr(E->getArg(2));
    llvm::Type *Ty = Op0->getType();
    Value *ShiftAmt = Builder.CreateIntCast(Op1, Ty, false);
    Function *F = CGM.getIntrinsic(Intrinsic::fshl, Ty);
    Value *Rotate = Builder.CreateCall(F, {Op0, Op0, ShiftAmt});
    return Builder.CreateAnd(Rotate, Op2);
  }
  case PPC::BI__builtin_ppc_load2r: {
    Function *F = CGM.getIntrinsic(Intrinsic::ppc_load2r);
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *LoadIntrinsic = Builder.CreateCall(F, {Op0});
    return Builder.CreateTrunc(LoadIntrinsic, Int16Ty);
  }
  // FMA variations
  case PPC::BI__builtin_ppc_fnmsub:
  case PPC::BI__builtin_ppc_fnmsubs:
  case PPC::BI__builtin_vsx_xvmaddadp:
  case PPC::BI__builtin_vsx_xvmaddasp:
  case PPC::BI__builtin_vsx_xvnmaddadp:
  case PPC::BI__builtin_vsx_xvnmaddasp:
  case PPC::BI__builtin_vsx_xvmsubadp:
  case PPC::BI__builtin_vsx_xvmsubasp:
  case PPC::BI__builtin_vsx_xvnmsubadp:
  case PPC::BI__builtin_vsx_xvnmsubasp: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *X = EmitScalarExpr(E->getArg(0));
    Value *Y = EmitScalarExpr(E->getArg(1));
    Value *Z = EmitScalarExpr(E->getArg(2));
    llvm::Function *F;
    if (Builder.getIsFPConstrained())
      F = CGM.getIntrinsic(Intrinsic::experimental_constrained_fma, ResultType);
    else
      F = CGM.getIntrinsic(Intrinsic::fma, ResultType);
    switch (BuiltinID) {
      case PPC::BI__builtin_vsx_xvmaddadp:
      case PPC::BI__builtin_vsx_xvmaddasp:
        if (Builder.getIsFPConstrained())
          return Builder.CreateConstrainedFPCall(F, {X, Y, Z});
        else
          return Builder.CreateCall(F, {X, Y, Z});
      case PPC::BI__builtin_vsx_xvnmaddadp:
      case PPC::BI__builtin_vsx_xvnmaddasp:
        if (Builder.getIsFPConstrained())
          return Builder.CreateFNeg(
              Builder.CreateConstrainedFPCall(F, {X, Y, Z}), "neg");
        else
          return Builder.CreateFNeg(Builder.CreateCall(F, {X, Y, Z}), "neg");
      case PPC::BI__builtin_vsx_xvmsubadp:
      case PPC::BI__builtin_vsx_xvmsubasp:
        if (Builder.getIsFPConstrained())
          return Builder.CreateConstrainedFPCall(
              F, {X, Y, Builder.CreateFNeg(Z, "neg")});
        else
          return Builder.CreateCall(F, {X, Y, Builder.CreateFNeg(Z, "neg")});
      case PPC::BI__builtin_ppc_fnmsub:
      case PPC::BI__builtin_ppc_fnmsubs:
      case PPC::BI__builtin_vsx_xvnmsubadp:
      case PPC::BI__builtin_vsx_xvnmsubasp:
        if (Builder.getIsFPConstrained())
          return Builder.CreateFNeg(
              Builder.CreateConstrainedFPCall(
                  F, {X, Y, Builder.CreateFNeg(Z, "neg")}),
              "neg");
        else
          return Builder.CreateCall(
              CGM.getIntrinsic(Intrinsic::ppc_fnmsub, ResultType), {X, Y, Z});
      }
    llvm_unreachable("Unknown FMA operation");
    return nullptr; // Suppress no-return warning
  }

  case PPC::BI__builtin_vsx_insertword: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Op2 = EmitScalarExpr(E->getArg(2));
    llvm::Function *F = CGM.getIntrinsic(Intrinsic::ppc_vsx_xxinsertw);

    // Third argument is a compile time constant int. It must be clamped to
    // to the range [0, 12].
    ConstantInt *ArgCI = dyn_cast<ConstantInt>(Op2);
    assert(ArgCI &&
           "Third arg to xxinsertw intrinsic must be constant integer");
    const int64_t MaxIndex = 12;
    int64_t Index = std::clamp(ArgCI->getSExtValue(), (int64_t)0, MaxIndex);

    // The builtin semantics don't exactly match the xxinsertw instructions
    // semantics (which ppc_vsx_xxinsertw follows). The builtin extracts the
    // word from the first argument, and inserts it in the second argument. The
    // instruction extracts the word from its second input register and inserts
    // it into its first input register, so swap the first and second arguments.
    std::swap(Op0, Op1);

    // Need to cast the second argument from a vector of unsigned int to a
    // vector of long long.
    Op1 = Builder.CreateBitCast(Op1, llvm::FixedVectorType::get(Int64Ty, 2));

    if (getTarget().isLittleEndian()) {
      // Reverse the double words in the vector we will extract from.
      Op0 = Builder.CreateBitCast(Op0, llvm::FixedVectorType::get(Int64Ty, 2));
      Op0 = Builder.CreateShuffleVector(Op0, Op0, {1, 0});

      // Reverse the index.
      Index = MaxIndex - Index;
    }

    // Intrinsic expects the first arg to be a vector of int.
    Op0 = Builder.CreateBitCast(Op0, llvm::FixedVectorType::get(Int32Ty, 4));
    Op2 = ConstantInt::getSigned(Int32Ty, Index);
    return Builder.CreateCall(F, {Op0, Op1, Op2});
  }

  case PPC::BI__builtin_vsx_extractuword: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    llvm::Function *F = CGM.getIntrinsic(Intrinsic::ppc_vsx_xxextractuw);

    // Intrinsic expects the first argument to be a vector of doublewords.
    Op0 = Builder.CreateBitCast(Op0, llvm::FixedVectorType::get(Int64Ty, 2));

    // The second argument is a compile time constant int that needs to
    // be clamped to the range [0, 12].
    ConstantInt *ArgCI = dyn_cast<ConstantInt>(Op1);
    assert(ArgCI &&
           "Second Arg to xxextractuw intrinsic must be a constant integer!");
    const int64_t MaxIndex = 12;
    int64_t Index = std::clamp(ArgCI->getSExtValue(), (int64_t)0, MaxIndex);

    if (getTarget().isLittleEndian()) {
      // Reverse the index.
      Index = MaxIndex - Index;
      Op1 = ConstantInt::getSigned(Int32Ty, Index);

      // Emit the call, then reverse the double words of the results vector.
      Value *Call = Builder.CreateCall(F, {Op0, Op1});

      Value *ShuffleCall =
          Builder.CreateShuffleVector(Call, Call, {1, 0});
      return ShuffleCall;
    } else {
      Op1 = ConstantInt::getSigned(Int32Ty, Index);
      return Builder.CreateCall(F, {Op0, Op1});
    }
  }

  case PPC::BI__builtin_vsx_xxpermdi: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Op2 = EmitScalarExpr(E->getArg(2));
    ConstantInt *ArgCI = dyn_cast<ConstantInt>(Op2);
    assert(ArgCI && "Third arg must be constant integer!");

    unsigned Index = ArgCI->getZExtValue();
    Op0 = Builder.CreateBitCast(Op0, llvm::FixedVectorType::get(Int64Ty, 2));
    Op1 = Builder.CreateBitCast(Op1, llvm::FixedVectorType::get(Int64Ty, 2));

    // Account for endianness by treating this as just a shuffle. So we use the
    // same indices for both LE and BE in order to produce expected results in
    // both cases.
    int ElemIdx0 = (Index & 2) >> 1;
    int ElemIdx1 = 2 + (Index & 1);

    int ShuffleElts[2] = {ElemIdx0, ElemIdx1};
    Value *ShuffleCall = Builder.CreateShuffleVector(Op0, Op1, ShuffleElts);
    QualType BIRetType = E->getType();
    auto RetTy = ConvertType(BIRetType);
    return Builder.CreateBitCast(ShuffleCall, RetTy);
  }

  case PPC::BI__builtin_vsx_xxsldwi: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Op2 = EmitScalarExpr(E->getArg(2));
    ConstantInt *ArgCI = dyn_cast<ConstantInt>(Op2);
    assert(ArgCI && "Third argument must be a compile time constant");
    unsigned Index = ArgCI->getZExtValue() & 0x3;
    Op0 = Builder.CreateBitCast(Op0, llvm::FixedVectorType::get(Int32Ty, 4));
    Op1 = Builder.CreateBitCast(Op1, llvm::FixedVectorType::get(Int32Ty, 4));

    // Create a shuffle mask
    int ElemIdx0;
    int ElemIdx1;
    int ElemIdx2;
    int ElemIdx3;
    if (getTarget().isLittleEndian()) {
      // Little endian element N comes from element 8+N-Index of the
      // concatenated wide vector (of course, using modulo arithmetic on
      // the total number of elements).
      ElemIdx0 = (8 - Index) % 8;
      ElemIdx1 = (9 - Index) % 8;
      ElemIdx2 = (10 - Index) % 8;
      ElemIdx3 = (11 - Index) % 8;
    } else {
      // Big endian ElemIdx<N> = Index + N
      ElemIdx0 = Index;
      ElemIdx1 = Index + 1;
      ElemIdx2 = Index + 2;
      ElemIdx3 = Index + 3;
    }

    int ShuffleElts[4] = {ElemIdx0, ElemIdx1, ElemIdx2, ElemIdx3};
    Value *ShuffleCall = Builder.CreateShuffleVector(Op0, Op1, ShuffleElts);
    QualType BIRetType = E->getType();
    auto RetTy = ConvertType(BIRetType);
    return Builder.CreateBitCast(ShuffleCall, RetTy);
  }

  case PPC::BI__builtin_pack_vector_int128: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    bool isLittleEndian = getTarget().isLittleEndian();
    Value *PoisonValue =
        llvm::PoisonValue::get(llvm::FixedVectorType::get(Op0->getType(), 2));
    Value *Res = Builder.CreateInsertElement(
        PoisonValue, Op0, (uint64_t)(isLittleEndian ? 1 : 0));
    Res = Builder.CreateInsertElement(Res, Op1,
                                      (uint64_t)(isLittleEndian ? 0 : 1));
    return Builder.CreateBitCast(Res, ConvertType(E->getType()));
  }

  case PPC::BI__builtin_unpack_vector_int128: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    ConstantInt *Index = cast<ConstantInt>(Op1);
    Value *Unpacked = Builder.CreateBitCast(
        Op0, llvm::FixedVectorType::get(ConvertType(E->getType()), 2));

    if (getTarget().isLittleEndian())
      Index =
          ConstantInt::get(Index->getIntegerType(), 1 - Index->getZExtValue());

    return Builder.CreateExtractElement(Unpacked, Index);
  }

  case PPC::BI__builtin_ppc_sthcx: {
    llvm::Function *F = CGM.getIntrinsic(Intrinsic::ppc_sthcx);
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = Builder.CreateSExt(EmitScalarExpr(E->getArg(1)), Int32Ty);
    return Builder.CreateCall(F, {Op0, Op1});
  }

  // The PPC MMA builtins take a pointer to a __vector_quad as an argument.
  // Some of the MMA instructions accumulate their result into an existing
  // accumulator whereas the others generate a new accumulator. So we need to
  // use custom code generation to expand a builtin call with a pointer to a
  // load (if the corresponding instruction accumulates its result) followed by
  // the call to the intrinsic and a store of the result.
#define CUSTOM_BUILTIN(Name, Intr, Types, Accumulate, Feature) \
  case PPC::BI__builtin_##Name:
#include "clang/Basic/BuiltinsPPC.def"
  {
    SmallVector<Value *, 4> Ops;
    for (unsigned i = 0, e = E->getNumArgs(); i != e; i++)
      if (E->getArg(i)->getType()->isArrayType())
        Ops.push_back(
            EmitArrayToPointerDecay(E->getArg(i)).emitRawPointer(*this));
      else
        Ops.push_back(EmitScalarExpr(E->getArg(i)));
    // The first argument of these two builtins is a pointer used to store their
    // result. However, the llvm intrinsics return their result in multiple
    // return values. So, here we emit code extracting these values from the
    // intrinsic results and storing them using that pointer.
    if (BuiltinID == PPC::BI__builtin_mma_disassemble_acc ||
        BuiltinID == PPC::BI__builtin_vsx_disassemble_pair ||
        BuiltinID == PPC::BI__builtin_mma_disassemble_pair) {
      unsigned NumVecs = 2;
      auto Intrinsic = Intrinsic::ppc_vsx_disassemble_pair;
      if (BuiltinID == PPC::BI__builtin_mma_disassemble_acc) {
        NumVecs = 4;
        Intrinsic = Intrinsic::ppc_mma_disassemble_acc;
      }
      llvm::Function *F = CGM.getIntrinsic(Intrinsic);
      Address Addr = EmitPointerWithAlignment(E->getArg(1));
      Value *Vec = Builder.CreateLoad(Addr);
      Value *Call = Builder.CreateCall(F, {Vec});
      llvm::Type *VTy = llvm::FixedVectorType::get(Int8Ty, 16);
      Value *Ptr = Ops[0];
      for (unsigned i=0; i<NumVecs; i++) {
        Value *Vec = Builder.CreateExtractValue(Call, i);
        llvm::ConstantInt* Index = llvm::ConstantInt::get(IntTy, i);
        Value *GEP = Builder.CreateInBoundsGEP(VTy, Ptr, Index);
        Builder.CreateAlignedStore(Vec, GEP, MaybeAlign(16));
      }
      return Call;
    }
    if (BuiltinID == PPC::BI__builtin_vsx_build_pair ||
        BuiltinID == PPC::BI__builtin_mma_build_acc) {
      // Reverse the order of the operands for LE, so the
      // same builtin call can be used on both LE and BE
      // without the need for the programmer to swap operands.
      // The operands are reversed starting from the second argument,
      // the first operand is the pointer to the pair/accumulator
      // that is being built.
      if (getTarget().isLittleEndian())
        std::reverse(Ops.begin() + 1, Ops.end());
    }
    bool Accumulate;
    switch (BuiltinID) {
  #define CUSTOM_BUILTIN(Name, Intr, Types, Acc, Feature) \
    case PPC::BI__builtin_##Name: \
      ID = Intrinsic::ppc_##Intr; \
      Accumulate = Acc; \
      break;
  #include "clang/Basic/BuiltinsPPC.def"
    }
    if (BuiltinID == PPC::BI__builtin_vsx_lxvp ||
        BuiltinID == PPC::BI__builtin_vsx_stxvp ||
        BuiltinID == PPC::BI__builtin_mma_lxvp ||
        BuiltinID == PPC::BI__builtin_mma_stxvp) {
      if (BuiltinID == PPC::BI__builtin_vsx_lxvp ||
          BuiltinID == PPC::BI__builtin_mma_lxvp) {
        Ops[0] = Builder.CreateGEP(Int8Ty, Ops[1], Ops[0]);
      } else {
        Ops[1] = Builder.CreateGEP(Int8Ty, Ops[2], Ops[1]);
      }
      Ops.pop_back();
      llvm::Function *F = CGM.getIntrinsic(ID);
      return Builder.CreateCall(F, Ops, "");
    }
    SmallVector<Value*, 4> CallOps;
    if (Accumulate) {
      Address Addr = EmitPointerWithAlignment(E->getArg(0));
      Value *Acc = Builder.CreateLoad(Addr);
      CallOps.push_back(Acc);
    }
    if (BuiltinID == PPC::BI__builtin_mma_dmmr ||
        BuiltinID == PPC::BI__builtin_mma_dmxor ||
        BuiltinID == PPC::BI__builtin_mma_disassemble_dmr ||
        BuiltinID == PPC::BI__builtin_mma_dmsha2hash) {
      Address Addr = EmitPointerWithAlignment(E->getArg(1));
      Ops[1] = Builder.CreateLoad(Addr);
    }
    if (BuiltinID == PPC::BI__builtin_mma_disassemble_dmr)
      return Builder.CreateAlignedStore(Ops[1], Ops[0], MaybeAlign());
    for (unsigned i=1; i<Ops.size(); i++)
      CallOps.push_back(Ops[i]);
    llvm::Function *F = CGM.getIntrinsic(ID);
    Value *Call = Builder.CreateCall(F, CallOps);
    return Builder.CreateAlignedStore(Call, Ops[0], MaybeAlign());
  }

  case PPC::BI__builtin_ppc_compare_and_swap:
  case PPC::BI__builtin_ppc_compare_and_swaplp: {
    Address Addr = EmitPointerWithAlignment(E->getArg(0));
    Address OldValAddr = EmitPointerWithAlignment(E->getArg(1));
    Value *OldVal = Builder.CreateLoad(OldValAddr);
    QualType AtomicTy = E->getArg(0)->getType()->getPointeeType();
    LValue LV = MakeAddrLValue(Addr, AtomicTy);
    Value *Op2 = EmitScalarExpr(E->getArg(2));
    auto Pair = EmitAtomicCompareExchange(
        LV, RValue::get(OldVal), RValue::get(Op2), E->getExprLoc(),
        llvm::AtomicOrdering::Monotonic, llvm::AtomicOrdering::Monotonic, true);
    // Unlike c11's atomic_compare_exchange, according to
    // https://www.ibm.com/docs/en/xl-c-and-cpp-aix/16.1?topic=functions-compare-swap-compare-swaplp
    // > In either case, the contents of the memory location specified by addr
    // > are copied into the memory location specified by old_val_addr.
    // But it hasn't specified storing to OldValAddr is atomic or not and
    // which order to use. Now following XL's codegen, treat it as a normal
    // store.
    Value *LoadedVal = Pair.first.getScalarVal();
    Builder.CreateStore(LoadedVal, OldValAddr);
    return Builder.CreateZExt(Pair.second, Builder.getInt32Ty());
  }
  case PPC::BI__builtin_ppc_fetch_and_add:
  case PPC::BI__builtin_ppc_fetch_and_addlp: {
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::Add, E,
                                 llvm::AtomicOrdering::Monotonic);
  }
  case PPC::BI__builtin_ppc_fetch_and_and:
  case PPC::BI__builtin_ppc_fetch_and_andlp: {
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::And, E,
                                 llvm::AtomicOrdering::Monotonic);
  }

  case PPC::BI__builtin_ppc_fetch_and_or:
  case PPC::BI__builtin_ppc_fetch_and_orlp: {
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::Or, E,
                                 llvm::AtomicOrdering::Monotonic);
  }
  case PPC::BI__builtin_ppc_fetch_and_swap:
  case PPC::BI__builtin_ppc_fetch_and_swaplp: {
    return MakeBinaryAtomicValue(*this, AtomicRMWInst::Xchg, E,
                                 llvm::AtomicOrdering::Monotonic);
  }
  case PPC::BI__builtin_ppc_ldarx:
  case PPC::BI__builtin_ppc_lwarx:
  case PPC::BI__builtin_ppc_lharx:
  case PPC::BI__builtin_ppc_lbarx:
    return emitPPCLoadReserveIntrinsic(*this, BuiltinID, E);
  case PPC::BI__builtin_ppc_mfspr: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    llvm::Type *RetType = CGM.getDataLayout().getTypeSizeInBits(VoidPtrTy) == 32
                              ? Int32Ty
                              : Int64Ty;
    Function *F = CGM.getIntrinsic(Intrinsic::ppc_mfspr, RetType);
    return Builder.CreateCall(F, {Op0});
  }
  case PPC::BI__builtin_ppc_mtspr: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    llvm::Type *RetType = CGM.getDataLayout().getTypeSizeInBits(VoidPtrTy) == 32
                              ? Int32Ty
                              : Int64Ty;
    Function *F = CGM.getIntrinsic(Intrinsic::ppc_mtspr, RetType);
    return Builder.CreateCall(F, {Op0, Op1});
  }
  case PPC::BI__builtin_ppc_popcntb: {
    Value *ArgValue = EmitScalarExpr(E->getArg(0));
    llvm::Type *ArgType = ArgValue->getType();
    Function *F = CGM.getIntrinsic(Intrinsic::ppc_popcntb, {ArgType, ArgType});
    return Builder.CreateCall(F, {ArgValue}, "popcntb");
  }
  case PPC::BI__builtin_ppc_mtfsf: {
    // The builtin takes a uint32 that needs to be cast to an
    // f64 to be passed to the intrinsic.
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Cast = Builder.CreateUIToFP(Op1, DoubleTy);
    llvm::Function *F = CGM.getIntrinsic(Intrinsic::ppc_mtfsf);
    return Builder.CreateCall(F, {Op0, Cast}, "");
  }

  case PPC::BI__builtin_ppc_swdiv_nochk:
  case PPC::BI__builtin_ppc_swdivs_nochk: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    FastMathFlags FMF = Builder.getFastMathFlags();
    Builder.getFastMathFlags().setFast();
    Value *FDiv = Builder.CreateFDiv(Op0, Op1, "swdiv_nochk");
    Builder.getFastMathFlags() &= (FMF);
    return FDiv;
  }
  case PPC::BI__builtin_ppc_fric:
    return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(
                           *this, E, Intrinsic::rint,
                           Intrinsic::experimental_constrained_rint))
        .getScalarVal();
  case PPC::BI__builtin_ppc_frim:
  case PPC::BI__builtin_ppc_frims:
    return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(
                           *this, E, Intrinsic::floor,
                           Intrinsic::experimental_constrained_floor))
        .getScalarVal();
  case PPC::BI__builtin_ppc_frin:
  case PPC::BI__builtin_ppc_frins:
    return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(
                           *this, E, Intrinsic::round,
                           Intrinsic::experimental_constrained_round))
        .getScalarVal();
  case PPC::BI__builtin_ppc_frip:
  case PPC::BI__builtin_ppc_frips:
    return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(
                           *this, E, Intrinsic::ceil,
                           Intrinsic::experimental_constrained_ceil))
        .getScalarVal();
  case PPC::BI__builtin_ppc_friz:
  case PPC::BI__builtin_ppc_frizs:
    return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(
                           *this, E, Intrinsic::trunc,
                           Intrinsic::experimental_constrained_trunc))
        .getScalarVal();
  case PPC::BI__builtin_ppc_fsqrt:
  case PPC::BI__builtin_ppc_fsqrts:
    return RValue::get(emitUnaryMaybeConstrainedFPBuiltin(
                           *this, E, Intrinsic::sqrt,
                           Intrinsic::experimental_constrained_sqrt))
        .getScalarVal();
  case PPC::BI__builtin_ppc_test_data_class: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    return Builder.CreateCall(
        CGM.getIntrinsic(Intrinsic::ppc_test_data_class, Op0->getType()),
        {Op0, Op1}, "test_data_class");
  }
  case PPC::BI__builtin_ppc_maxfe: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Op2 = EmitScalarExpr(E->getArg(2));
    Value *Op3 = EmitScalarExpr(E->getArg(3));
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::ppc_maxfe),
                              {Op0, Op1, Op2, Op3});
  }
  case PPC::BI__builtin_ppc_maxfl: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Op2 = EmitScalarExpr(E->getArg(2));
    Value *Op3 = EmitScalarExpr(E->getArg(3));
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::ppc_maxfl),
                              {Op0, Op1, Op2, Op3});
  }
  case PPC::BI__builtin_ppc_maxfs: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Op2 = EmitScalarExpr(E->getArg(2));
    Value *Op3 = EmitScalarExpr(E->getArg(3));
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::ppc_maxfs),
                              {Op0, Op1, Op2, Op3});
  }
  case PPC::BI__builtin_ppc_minfe: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Op2 = EmitScalarExpr(E->getArg(2));
    Value *Op3 = EmitScalarExpr(E->getArg(3));
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::ppc_minfe),
                              {Op0, Op1, Op2, Op3});
  }
  case PPC::BI__builtin_ppc_minfl: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Op2 = EmitScalarExpr(E->getArg(2));
    Value *Op3 = EmitScalarExpr(E->getArg(3));
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::ppc_minfl),
                              {Op0, Op1, Op2, Op3});
  }
  case PPC::BI__builtin_ppc_minfs: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    Value *Op2 = EmitScalarExpr(E->getArg(2));
    Value *Op3 = EmitScalarExpr(E->getArg(3));
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::ppc_minfs),
                              {Op0, Op1, Op2, Op3});
  }
  case PPC::BI__builtin_ppc_swdiv:
  case PPC::BI__builtin_ppc_swdivs: {
    Value *Op0 = EmitScalarExpr(E->getArg(0));
    Value *Op1 = EmitScalarExpr(E->getArg(1));
    return Builder.CreateFDiv(Op0, Op1, "swdiv");
  }
  case PPC::BI__builtin_ppc_set_fpscr_rn:
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::ppc_setrnd),
                              {EmitScalarExpr(E->getArg(0))});
  case PPC::BI__builtin_ppc_mffs:
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::ppc_readflm));
  }
}
