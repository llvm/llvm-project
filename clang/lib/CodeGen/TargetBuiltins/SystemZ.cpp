//===------ SystemZ.cpp - Emit LLVM Code for builtins ---------------------===//
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

#include "ABIInfo.h"
#include "CodeGenFunction.h"
#include "TargetInfo.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsS390.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm;

/// Handle a SystemZ function in which the final argument is a pointer
/// to an int that receives the post-instruction CC value.  At the LLVM level
/// this is represented as a function that returns a {result, cc} pair.
static Value *EmitSystemZIntrinsicWithCC(CodeGenFunction &CGF,
                                         unsigned IntrinsicID,
                                         const CallExpr *E) {
  unsigned NumArgs = E->getNumArgs() - 1;
  SmallVector<Value *, 8> Args(NumArgs);
  for (unsigned I = 0; I < NumArgs; ++I)
    Args[I] = CGF.EmitScalarExpr(E->getArg(I));
  Address CCPtr = CGF.EmitPointerWithAlignment(E->getArg(NumArgs));
  Function *F = CGF.CGM.getIntrinsic(IntrinsicID);
  Value *Call = CGF.Builder.CreateCall(F, Args);
  Value *CC = CGF.Builder.CreateExtractValue(Call, 1);
  CGF.Builder.CreateStore(CC, CCPtr);
  return CGF.Builder.CreateExtractValue(Call, 0);
}

Value *CodeGenFunction::EmitSystemZBuiltinExpr(unsigned BuiltinID,
                                               const CallExpr *E) {
  switch (BuiltinID) {
  case SystemZ::BI__builtin_tbegin: {
    Value *TDB = EmitScalarExpr(E->getArg(0));
    Value *Control = llvm::ConstantInt::get(Int32Ty, 0xff0c);
    Function *F = CGM.getIntrinsic(Intrinsic::s390_tbegin);
    return Builder.CreateCall(F, {TDB, Control});
  }
  case SystemZ::BI__builtin_tbegin_nofloat: {
    Value *TDB = EmitScalarExpr(E->getArg(0));
    Value *Control = llvm::ConstantInt::get(Int32Ty, 0xff0c);
    Function *F = CGM.getIntrinsic(Intrinsic::s390_tbegin_nofloat);
    return Builder.CreateCall(F, {TDB, Control});
  }
  case SystemZ::BI__builtin_tbeginc: {
    Value *TDB = llvm::ConstantPointerNull::get(Int8PtrTy);
    Value *Control = llvm::ConstantInt::get(Int32Ty, 0xff08);
    Function *F = CGM.getIntrinsic(Intrinsic::s390_tbeginc);
    return Builder.CreateCall(F, {TDB, Control});
  }
  case SystemZ::BI__builtin_tabort: {
    Value *Data = EmitScalarExpr(E->getArg(0));
    Function *F = CGM.getIntrinsic(Intrinsic::s390_tabort);
    return Builder.CreateCall(F, Builder.CreateSExt(Data, Int64Ty, "tabort"));
  }
  case SystemZ::BI__builtin_non_tx_store: {
    Value *Address = EmitScalarExpr(E->getArg(0));
    Value *Data = EmitScalarExpr(E->getArg(1));
    Function *F = CGM.getIntrinsic(Intrinsic::s390_ntstg);
    return Builder.CreateCall(F, {Data, Address});
  }

  // Vector builtins.  Note that most vector builtins are mapped automatically
  // to target-specific LLVM intrinsics.  The ones handled specially here can
  // be represented via standard LLVM IR, which is preferable to enable common
  // LLVM optimizations.

  case SystemZ::BI__builtin_s390_vclzb:
  case SystemZ::BI__builtin_s390_vclzh:
  case SystemZ::BI__builtin_s390_vclzf:
  case SystemZ::BI__builtin_s390_vclzg:
  case SystemZ::BI__builtin_s390_vclzq: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *X = EmitScalarExpr(E->getArg(0));
    Value *Undef = ConstantInt::get(Builder.getInt1Ty(), false);
    Function *F = CGM.getIntrinsic(Intrinsic::ctlz, ResultType);
    return Builder.CreateCall(F, {X, Undef});
  }

  case SystemZ::BI__builtin_s390_vctzb:
  case SystemZ::BI__builtin_s390_vctzh:
  case SystemZ::BI__builtin_s390_vctzf:
  case SystemZ::BI__builtin_s390_vctzg:
  case SystemZ::BI__builtin_s390_vctzq: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *X = EmitScalarExpr(E->getArg(0));
    Value *Undef = ConstantInt::get(Builder.getInt1Ty(), false);
    Function *F = CGM.getIntrinsic(Intrinsic::cttz, ResultType);
    return Builder.CreateCall(F, {X, Undef});
  }

  case SystemZ::BI__builtin_s390_verllb:
  case SystemZ::BI__builtin_s390_verllh:
  case SystemZ::BI__builtin_s390_verllf:
  case SystemZ::BI__builtin_s390_verllg: {
    llvm::Type *ResultType = ConvertType(E->getType());
    llvm::Value *Src = EmitScalarExpr(E->getArg(0));
    llvm::Value *Amt = EmitScalarExpr(E->getArg(1));
    // Splat scalar rotate amount to vector type.
    unsigned NumElts = cast<llvm::FixedVectorType>(ResultType)->getNumElements();
    Amt = Builder.CreateIntCast(Amt, ResultType->getScalarType(), false);
    Amt = Builder.CreateVectorSplat(NumElts, Amt);
    Function *F = CGM.getIntrinsic(Intrinsic::fshl, ResultType);
    return Builder.CreateCall(F, { Src, Src, Amt });
  }

  case SystemZ::BI__builtin_s390_verllvb:
  case SystemZ::BI__builtin_s390_verllvh:
  case SystemZ::BI__builtin_s390_verllvf:
  case SystemZ::BI__builtin_s390_verllvg: {
    llvm::Type *ResultType = ConvertType(E->getType());
    llvm::Value *Src = EmitScalarExpr(E->getArg(0));
    llvm::Value *Amt = EmitScalarExpr(E->getArg(1));
    Function *F = CGM.getIntrinsic(Intrinsic::fshl, ResultType);
    return Builder.CreateCall(F, { Src, Src, Amt });
  }

  case SystemZ::BI__builtin_s390_vfsqsb:
  case SystemZ::BI__builtin_s390_vfsqdb: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *X = EmitScalarExpr(E->getArg(0));
    if (Builder.getIsFPConstrained()) {
      Function *F = CGM.getIntrinsic(Intrinsic::experimental_constrained_sqrt, ResultType);
      return Builder.CreateConstrainedFPCall(F, { X });
    } else {
      Function *F = CGM.getIntrinsic(Intrinsic::sqrt, ResultType);
      return Builder.CreateCall(F, X);
    }
  }
  case SystemZ::BI__builtin_s390_vfmasb:
  case SystemZ::BI__builtin_s390_vfmadb: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *X = EmitScalarExpr(E->getArg(0));
    Value *Y = EmitScalarExpr(E->getArg(1));
    Value *Z = EmitScalarExpr(E->getArg(2));
    if (Builder.getIsFPConstrained()) {
      Function *F = CGM.getIntrinsic(Intrinsic::experimental_constrained_fma, ResultType);
      return Builder.CreateConstrainedFPCall(F, {X, Y, Z});
    } else {
      Function *F = CGM.getIntrinsic(Intrinsic::fma, ResultType);
      return Builder.CreateCall(F, {X, Y, Z});
    }
  }
  case SystemZ::BI__builtin_s390_vfmssb:
  case SystemZ::BI__builtin_s390_vfmsdb: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *X = EmitScalarExpr(E->getArg(0));
    Value *Y = EmitScalarExpr(E->getArg(1));
    Value *Z = EmitScalarExpr(E->getArg(2));
    if (Builder.getIsFPConstrained()) {
      Function *F = CGM.getIntrinsic(Intrinsic::experimental_constrained_fma, ResultType);
      return Builder.CreateConstrainedFPCall(F, {X, Y, Builder.CreateFNeg(Z, "neg")});
    } else {
      Function *F = CGM.getIntrinsic(Intrinsic::fma, ResultType);
      return Builder.CreateCall(F, {X, Y, Builder.CreateFNeg(Z, "neg")});
    }
  }
  case SystemZ::BI__builtin_s390_vfnmasb:
  case SystemZ::BI__builtin_s390_vfnmadb: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *X = EmitScalarExpr(E->getArg(0));
    Value *Y = EmitScalarExpr(E->getArg(1));
    Value *Z = EmitScalarExpr(E->getArg(2));
    if (Builder.getIsFPConstrained()) {
      Function *F = CGM.getIntrinsic(Intrinsic::experimental_constrained_fma, ResultType);
      return Builder.CreateFNeg(Builder.CreateConstrainedFPCall(F, {X, Y,  Z}), "neg");
    } else {
      Function *F = CGM.getIntrinsic(Intrinsic::fma, ResultType);
      return Builder.CreateFNeg(Builder.CreateCall(F, {X, Y, Z}), "neg");
    }
  }
  case SystemZ::BI__builtin_s390_vfnmssb:
  case SystemZ::BI__builtin_s390_vfnmsdb: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *X = EmitScalarExpr(E->getArg(0));
    Value *Y = EmitScalarExpr(E->getArg(1));
    Value *Z = EmitScalarExpr(E->getArg(2));
    if (Builder.getIsFPConstrained()) {
      Function *F = CGM.getIntrinsic(Intrinsic::experimental_constrained_fma, ResultType);
      Value *NegZ = Builder.CreateFNeg(Z, "sub");
      return Builder.CreateFNeg(Builder.CreateConstrainedFPCall(F, {X, Y, NegZ}));
    } else {
      Function *F = CGM.getIntrinsic(Intrinsic::fma, ResultType);
      Value *NegZ = Builder.CreateFNeg(Z, "neg");
      return Builder.CreateFNeg(Builder.CreateCall(F, {X, Y, NegZ}));
    }
  }
  case SystemZ::BI__builtin_s390_vflpsb:
  case SystemZ::BI__builtin_s390_vflpdb: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *X = EmitScalarExpr(E->getArg(0));
    Function *F = CGM.getIntrinsic(Intrinsic::fabs, ResultType);
    return Builder.CreateCall(F, X);
  }
  case SystemZ::BI__builtin_s390_vflnsb:
  case SystemZ::BI__builtin_s390_vflndb: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *X = EmitScalarExpr(E->getArg(0));
    Function *F = CGM.getIntrinsic(Intrinsic::fabs, ResultType);
    return Builder.CreateFNeg(Builder.CreateCall(F, X), "neg");
  }
  case SystemZ::BI__builtin_s390_vfisb:
  case SystemZ::BI__builtin_s390_vfidb: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *X = EmitScalarExpr(E->getArg(0));
    // Constant-fold the M4 and M5 mask arguments.
    llvm::APSInt M4 = *E->getArg(1)->getIntegerConstantExpr(getContext());
    llvm::APSInt M5 = *E->getArg(2)->getIntegerConstantExpr(getContext());
    // Check whether this instance can be represented via a LLVM standard
    // intrinsic.  We only support some combinations of M4 and M5.
    Intrinsic::ID ID = Intrinsic::not_intrinsic;
    Intrinsic::ID CI;
    switch (M4.getZExtValue()) {
    default: break;
    case 0:  // IEEE-inexact exception allowed
      switch (M5.getZExtValue()) {
      default: break;
      case 0: ID = Intrinsic::rint;
              CI = Intrinsic::experimental_constrained_rint; break;
      }
      break;
    case 4:  // IEEE-inexact exception suppressed
      switch (M5.getZExtValue()) {
      default: break;
      case 0: ID = Intrinsic::nearbyint;
              CI = Intrinsic::experimental_constrained_nearbyint; break;
      case 1: ID = Intrinsic::round;
              CI = Intrinsic::experimental_constrained_round; break;
      case 4: ID = Intrinsic::roundeven;
              CI = Intrinsic::experimental_constrained_roundeven; break;
      case 5: ID = Intrinsic::trunc;
              CI = Intrinsic::experimental_constrained_trunc; break;
      case 6: ID = Intrinsic::ceil;
              CI = Intrinsic::experimental_constrained_ceil; break;
      case 7: ID = Intrinsic::floor;
              CI = Intrinsic::experimental_constrained_floor; break;
      }
      break;
    }
    if (ID != Intrinsic::not_intrinsic) {
      if (Builder.getIsFPConstrained()) {
        Function *F = CGM.getIntrinsic(CI, ResultType);
        return Builder.CreateConstrainedFPCall(F, X);
      } else {
        Function *F = CGM.getIntrinsic(ID, ResultType);
        return Builder.CreateCall(F, X);
      }
    }
    switch (BuiltinID) { // FIXME: constrained version?
      case SystemZ::BI__builtin_s390_vfisb: ID = Intrinsic::s390_vfisb; break;
      case SystemZ::BI__builtin_s390_vfidb: ID = Intrinsic::s390_vfidb; break;
      default: llvm_unreachable("Unknown BuiltinID");
    }
    Function *F = CGM.getIntrinsic(ID);
    Value *M4Value = llvm::ConstantInt::get(getLLVMContext(), M4);
    Value *M5Value = llvm::ConstantInt::get(getLLVMContext(), M5);
    return Builder.CreateCall(F, {X, M4Value, M5Value});
  }
  case SystemZ::BI__builtin_s390_vfmaxsb:
  case SystemZ::BI__builtin_s390_vfmaxdb: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *X = EmitScalarExpr(E->getArg(0));
    Value *Y = EmitScalarExpr(E->getArg(1));
    // Constant-fold the M4 mask argument.
    llvm::APSInt M4 = *E->getArg(2)->getIntegerConstantExpr(getContext());
    // Check whether this instance can be represented via a LLVM standard
    // intrinsic.  We only support some values of M4.
    Intrinsic::ID ID = Intrinsic::not_intrinsic;
    Intrinsic::ID CI;
    switch (M4.getZExtValue()) {
    default: break;
    case 4: ID = Intrinsic::maxnum;
            CI = Intrinsic::experimental_constrained_maxnum; break;
    }
    if (ID != Intrinsic::not_intrinsic) {
      if (Builder.getIsFPConstrained()) {
        Function *F = CGM.getIntrinsic(CI, ResultType);
        return Builder.CreateConstrainedFPCall(F, {X, Y});
      } else {
        Function *F = CGM.getIntrinsic(ID, ResultType);
        return Builder.CreateCall(F, {X, Y});
      }
    }
    switch (BuiltinID) {
      case SystemZ::BI__builtin_s390_vfmaxsb: ID = Intrinsic::s390_vfmaxsb; break;
      case SystemZ::BI__builtin_s390_vfmaxdb: ID = Intrinsic::s390_vfmaxdb; break;
      default: llvm_unreachable("Unknown BuiltinID");
    }
    Function *F = CGM.getIntrinsic(ID);
    Value *M4Value = llvm::ConstantInt::get(getLLVMContext(), M4);
    return Builder.CreateCall(F, {X, Y, M4Value});
  }
  case SystemZ::BI__builtin_s390_vfminsb:
  case SystemZ::BI__builtin_s390_vfmindb: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *X = EmitScalarExpr(E->getArg(0));
    Value *Y = EmitScalarExpr(E->getArg(1));
    // Constant-fold the M4 mask argument.
    llvm::APSInt M4 = *E->getArg(2)->getIntegerConstantExpr(getContext());
    // Check whether this instance can be represented via a LLVM standard
    // intrinsic.  We only support some values of M4.
    Intrinsic::ID ID = Intrinsic::not_intrinsic;
    Intrinsic::ID CI;
    switch (M4.getZExtValue()) {
    default: break;
    case 4: ID = Intrinsic::minnum;
            CI = Intrinsic::experimental_constrained_minnum; break;
    }
    if (ID != Intrinsic::not_intrinsic) {
      if (Builder.getIsFPConstrained()) {
        Function *F = CGM.getIntrinsic(CI, ResultType);
        return Builder.CreateConstrainedFPCall(F, {X, Y});
      } else {
        Function *F = CGM.getIntrinsic(ID, ResultType);
        return Builder.CreateCall(F, {X, Y});
      }
    }
    switch (BuiltinID) {
      case SystemZ::BI__builtin_s390_vfminsb: ID = Intrinsic::s390_vfminsb; break;
      case SystemZ::BI__builtin_s390_vfmindb: ID = Intrinsic::s390_vfmindb; break;
      default: llvm_unreachable("Unknown BuiltinID");
    }
    Function *F = CGM.getIntrinsic(ID);
    Value *M4Value = llvm::ConstantInt::get(getLLVMContext(), M4);
    return Builder.CreateCall(F, {X, Y, M4Value});
  }

  case SystemZ::BI__builtin_s390_vlbrh:
  case SystemZ::BI__builtin_s390_vlbrf:
  case SystemZ::BI__builtin_s390_vlbrg:
  case SystemZ::BI__builtin_s390_vlbrq: {
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *X = EmitScalarExpr(E->getArg(0));
    Function *F = CGM.getIntrinsic(Intrinsic::bswap, ResultType);
    return Builder.CreateCall(F, X);
  }

  // Vector intrinsics that output the post-instruction CC value.

#define INTRINSIC_WITH_CC(NAME) \
    case SystemZ::BI__builtin_##NAME: \
      return EmitSystemZIntrinsicWithCC(*this, Intrinsic::NAME, E)

  INTRINSIC_WITH_CC(s390_vpkshs);
  INTRINSIC_WITH_CC(s390_vpksfs);
  INTRINSIC_WITH_CC(s390_vpksgs);

  INTRINSIC_WITH_CC(s390_vpklshs);
  INTRINSIC_WITH_CC(s390_vpklsfs);
  INTRINSIC_WITH_CC(s390_vpklsgs);

  INTRINSIC_WITH_CC(s390_vceqbs);
  INTRINSIC_WITH_CC(s390_vceqhs);
  INTRINSIC_WITH_CC(s390_vceqfs);
  INTRINSIC_WITH_CC(s390_vceqgs);
  INTRINSIC_WITH_CC(s390_vceqqs);

  INTRINSIC_WITH_CC(s390_vchbs);
  INTRINSIC_WITH_CC(s390_vchhs);
  INTRINSIC_WITH_CC(s390_vchfs);
  INTRINSIC_WITH_CC(s390_vchgs);
  INTRINSIC_WITH_CC(s390_vchqs);

  INTRINSIC_WITH_CC(s390_vchlbs);
  INTRINSIC_WITH_CC(s390_vchlhs);
  INTRINSIC_WITH_CC(s390_vchlfs);
  INTRINSIC_WITH_CC(s390_vchlgs);
  INTRINSIC_WITH_CC(s390_vchlqs);

  INTRINSIC_WITH_CC(s390_vfaebs);
  INTRINSIC_WITH_CC(s390_vfaehs);
  INTRINSIC_WITH_CC(s390_vfaefs);

  INTRINSIC_WITH_CC(s390_vfaezbs);
  INTRINSIC_WITH_CC(s390_vfaezhs);
  INTRINSIC_WITH_CC(s390_vfaezfs);

  INTRINSIC_WITH_CC(s390_vfeebs);
  INTRINSIC_WITH_CC(s390_vfeehs);
  INTRINSIC_WITH_CC(s390_vfeefs);

  INTRINSIC_WITH_CC(s390_vfeezbs);
  INTRINSIC_WITH_CC(s390_vfeezhs);
  INTRINSIC_WITH_CC(s390_vfeezfs);

  INTRINSIC_WITH_CC(s390_vfenebs);
  INTRINSIC_WITH_CC(s390_vfenehs);
  INTRINSIC_WITH_CC(s390_vfenefs);

  INTRINSIC_WITH_CC(s390_vfenezbs);
  INTRINSIC_WITH_CC(s390_vfenezhs);
  INTRINSIC_WITH_CC(s390_vfenezfs);

  INTRINSIC_WITH_CC(s390_vistrbs);
  INTRINSIC_WITH_CC(s390_vistrhs);
  INTRINSIC_WITH_CC(s390_vistrfs);

  INTRINSIC_WITH_CC(s390_vstrcbs);
  INTRINSIC_WITH_CC(s390_vstrchs);
  INTRINSIC_WITH_CC(s390_vstrcfs);

  INTRINSIC_WITH_CC(s390_vstrczbs);
  INTRINSIC_WITH_CC(s390_vstrczhs);
  INTRINSIC_WITH_CC(s390_vstrczfs);

  INTRINSIC_WITH_CC(s390_vfcesbs);
  INTRINSIC_WITH_CC(s390_vfcedbs);
  INTRINSIC_WITH_CC(s390_vfchsbs);
  INTRINSIC_WITH_CC(s390_vfchdbs);
  INTRINSIC_WITH_CC(s390_vfchesbs);
  INTRINSIC_WITH_CC(s390_vfchedbs);

  INTRINSIC_WITH_CC(s390_vftcisb);
  INTRINSIC_WITH_CC(s390_vftcidb);

  INTRINSIC_WITH_CC(s390_vstrsb);
  INTRINSIC_WITH_CC(s390_vstrsh);
  INTRINSIC_WITH_CC(s390_vstrsf);

  INTRINSIC_WITH_CC(s390_vstrszb);
  INTRINSIC_WITH_CC(s390_vstrszh);
  INTRINSIC_WITH_CC(s390_vstrszf);

#undef INTRINSIC_WITH_CC

  default:
    return nullptr;
  }
}
