//===---- TargetInfo.h - Encapsulate target details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function
// definition used to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_TARGETINFO_H
#define LLVM_CLANG_LIB_CIR_TARGETINFO_H

#include "ABIInfo.h"
#include "CIRGenValue.h"
#include "mlir/IR/Types.h"

#include <memory>

namespace cir {

class CIRGenFunction;

/// This class organizes various target-specific codegeneration issues, like
/// target-specific attributes, builtins and so on.
/// Equivalent to LLVM's TargetCodeGenInfo.
class TargetCIRGenInfo {
  std::unique_ptr<ABIInfo> Info = nullptr;

public:
  TargetCIRGenInfo(std::unique_ptr<ABIInfo> Info) : Info(std::move(Info)) {}

  /// Returns ABI info helper for the target.
  const ABIInfo &getABIInfo() const { return *Info; }

  virtual bool isScalarizableAsmOperand(CIRGenFunction &CGF,
                                        mlir::Type Ty) const {
    return false;
  }

  /// Corrects the MLIR type for a given constraint and "usual"
  /// type.
  ///
  /// \returns A new MLIR type, possibly the same as the original
  /// on success
  virtual mlir::Type adjustInlineAsmType(CIRGenFunction &CGF,
                                         llvm::StringRef Constraint,
                                         mlir::Type Ty) const {
    return Ty;
  }

  virtual void
  addReturnRegisterOutputs(CIRGenFunction &CGF, LValue ReturnValue,
                           std::string &Constraints,
                           std::vector<mlir::Type> &ResultRegTypes,
                           std::vector<mlir::Type> &ResultTruncRegTypes,
                           std::vector<LValue> &ResultRegDests,
                           std::string &AsmString, unsigned NumOutputs) const {}

  virtual ~TargetCIRGenInfo() {}
};

} // namespace cir

#endif
