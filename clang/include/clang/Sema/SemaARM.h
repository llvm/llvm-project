//===----- SemaARM.h ------- ARM target-specific routines -----*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares semantic analysis functions specific to ARM.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMAARM_H
#define LLVM_CLANG_SEMA_SEMAARM_H

#include "clang/AST/Expr.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/SemaBase.h"
#include "llvm/ADT/SmallVector.h"
#include <tuple>

namespace clang {

class SemaARM : public SemaBase {
public:
  SemaARM(Sema &S);

  enum ArmStreamingType {
    ArmNonStreaming,
    ArmStreaming,
    ArmStreamingCompatible,
    ArmStreamingOrSVE2p1
  };

  bool CheckARMBuiltinExclusiveCall(unsigned BuiltinID, CallExpr *TheCall,
                                    unsigned MaxWidth);
  bool CheckNeonBuiltinFunctionCall(const TargetInfo &TI, unsigned BuiltinID,
                                    CallExpr *TheCall);
  bool CheckMVEBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);
  bool CheckSVEBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);
  bool
  ParseSVEImmChecks(CallExpr *TheCall,
                    llvm::SmallVector<std::tuple<int, int, int>, 3> &ImmChecks);
  bool CheckSMEBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);
  bool CheckCDEBuiltinFunctionCall(const TargetInfo &TI, unsigned BuiltinID,
                                   CallExpr *TheCall);
  bool CheckARMCoprocessorImmediate(const TargetInfo &TI, const Expr *CoprocArg,
                                    bool WantCDE);
  bool CheckARMBuiltinFunctionCall(const TargetInfo &TI, unsigned BuiltinID,
                                   CallExpr *TheCall);

  bool CheckAArch64BuiltinFunctionCall(const TargetInfo &TI, unsigned BuiltinID,
                                       CallExpr *TheCall);
  bool BuiltinARMSpecialReg(unsigned BuiltinID, CallExpr *TheCall, int ArgNum,
                            unsigned ExpectedFieldNum, bool AllowName);
  bool BuiltinARMMemoryTaggingCall(unsigned BuiltinID, CallExpr *TheCall);
};

SemaARM::ArmStreamingType getArmStreamingFnType(const FunctionDecl *FD);

} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMAARM_H
