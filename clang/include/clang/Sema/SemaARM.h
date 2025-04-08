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

#include "clang/AST/DeclBase.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/SemaBase.h"
#include "llvm/ADT/StringRef.h"
#include <tuple>

namespace llvm {
template <typename T, unsigned N> class SmallVector;
} // namespace llvm

namespace clang {
class ParsedAttr;
class TargetInfo;

class SemaARM : public SemaBase {
public:
  SemaARM(Sema &S);

  enum ArmStreamingType {
    ArmNonStreaming, /// Intrinsic is only available in normal mode
    ArmStreaming,    /// Intrinsic is only available in Streaming-SVE mode.
    ArmStreamingCompatible, /// Intrinsic is available both in normal and
                            /// Streaming-SVE mode.
    VerifyRuntimeMode       /// Intrinsic is available in normal mode with
                            /// SVE flags, or in Streaming-SVE mode with SME
                            /// flags. Do Sema checks for the runtime mode.
  };

  bool CheckImmediateArg(CallExpr *TheCall, unsigned CheckTy, unsigned ArgIdx,
                         unsigned EltBitWidth, unsigned VecBitWidth);
  bool CheckARMBuiltinExclusiveCall(unsigned BuiltinID, CallExpr *TheCall,
                                    unsigned MaxWidth);
  bool CheckNeonBuiltinFunctionCall(const TargetInfo &TI, unsigned BuiltinID,
                                    CallExpr *TheCall);
  bool PerformNeonImmChecks(
      CallExpr *TheCall,
      SmallVectorImpl<std::tuple<int, int, int, int>> &ImmChecks,
      int OverloadType = -1);
  bool
  PerformSVEImmChecks(CallExpr *TheCall,
                      SmallVectorImpl<std::tuple<int, int, int>> &ImmChecks);
  bool CheckMVEBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);
  bool CheckSVEBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall);
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

  bool MveAliasValid(unsigned BuiltinID, llvm::StringRef AliasName);
  bool CdeAliasValid(unsigned BuiltinID, llvm::StringRef AliasName);
  bool SveAliasValid(unsigned BuiltinID, llvm::StringRef AliasName);
  bool SmeAliasValid(unsigned BuiltinID, llvm::StringRef AliasName);
  void handleBuiltinAliasAttr(Decl *D, const ParsedAttr &AL);
  void handleNewAttr(Decl *D, const ParsedAttr &AL);
  void handleCmseNSEntryAttr(Decl *D, const ParsedAttr &AL);
  void handleInterruptAttr(Decl *D, const ParsedAttr &AL);

  void CheckSMEFunctionDefAttributes(const FunctionDecl *FD);
};

SemaARM::ArmStreamingType getArmStreamingFnType(const FunctionDecl *FD);

} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMAARM_H
