//===--- CIRToCIRArgMapping.cpp - Maps to ABI-specific arguments ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics the ClangToLLVMArgMapping class in
// clang/lib/CodeGen/CGCall.cpp. The queries are adapted to operate on the CIR
// dialect, however. This class was extracted into a separate file to resolve
// build issues.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRTOCIRARGMAPPING_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRTOCIRARGMAPPING_H

#include "CIRLowerContext.h"
#include "LowerFunctionInfo.h"
#include "clang/CIR/ABIArgInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace cir {

/// Encapsulates information about the way function arguments from
/// LoweringFunctionInfo should be passed to actual CIR function.
class CIRToCIRArgMapping {
  static const unsigned InvalidIndex = ~0U;
  unsigned TotalIRArgs;

  /// Arguments of CIR function corresponding to single CIR argument.
  /// NOTE(cir): We add an MLIR block argument here indicating the actual
  /// argument in the IR.
  struct IRArgs {
    unsigned PaddingArgIndex;
    // Argument is expanded to IR arguments at positions
    // [FirstArgIndex, FirstArgIndex + NumberOfArgs).
    unsigned FirstArgIndex;
    unsigned NumberOfArgs;

    IRArgs()
        : PaddingArgIndex(InvalidIndex), FirstArgIndex(InvalidIndex),
          NumberOfArgs(0) {}
  };

  llvm::SmallVector<IRArgs, 8> ArgInfo;

public:
  CIRToCIRArgMapping(const CIRLowerContext &context,
                     const LowerFunctionInfo &FI, bool onlyRequiredArgs = false)
      : ArgInfo(onlyRequiredArgs ? FI.getNumRequiredArgs() : FI.arg_size()) {
    construct(context, FI, onlyRequiredArgs);
  };

  unsigned totalIRArgs() const { return TotalIRArgs; }

  void construct(const CIRLowerContext &context, const LowerFunctionInfo &FI,
                 bool onlyRequiredArgs = false) {
    unsigned IRArgNo = 0;
    const ::cir::ABIArgInfo &RetAI = FI.getReturnInfo();

    if (RetAI.getKind() == ::cir::ABIArgInfo::Indirect) {
      llvm_unreachable("NYI");
    }

    unsigned ArgNo = 0;
    unsigned NumArgs =
        onlyRequiredArgs ? FI.getNumRequiredArgs() : FI.arg_size();
    for (LowerFunctionInfo::const_arg_iterator _ = FI.arg_begin();
         ArgNo < NumArgs; ++_, ++ArgNo) {
      llvm_unreachable("NYI");
    }
    assert(ArgNo == ArgInfo.size());

    if (::cir::MissingFeatures::inallocaArgs()) {
      llvm_unreachable("NYI");
    }

    TotalIRArgs = IRArgNo;
  }
};

} // namespace cir
} // namespace mlir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_CIRTOCIRARGMAPPING_H
