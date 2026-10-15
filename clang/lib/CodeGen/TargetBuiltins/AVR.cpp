//===------ AVR.cpp - Emit LLVM Code for AVR builtins ---------------------===//
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
#include "llvm/IR/IntrinsicsAVR.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm;

Value *CodeGenFunction::EmitAVRBuiltinExpr(unsigned BuiltinID,
                                           const CallExpr *E) {
  switch (BuiltinID) {
  default:
    return nullptr;
  case AVR::BI__builtin_avr_nop:
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::avr_nop));
  case AVR::BI__builtin_avr_sei:
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::avr_sei));
  case AVR::BI__builtin_avr_cli:
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::avr_cli));
  case AVR::BI__builtin_avr_sleep:
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::avr_sleep));
  case AVR::BI__builtin_avr_wdr:
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::avr_wdr));
  case AVR::BI__builtin_avr_swap: {
    Value *Arg0 = EmitScalarExpr(E->getArg(0));
    return Builder.CreateCall(CGM.getIntrinsic(Intrinsic::avr_swap), Arg0);
  }
  }
}
