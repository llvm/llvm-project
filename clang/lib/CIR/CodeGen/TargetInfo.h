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
#include "clang/CIR/Dialect/IR/CIRAttrs.h"

#include <memory>

namespace cir {

class CIRGenFunction;
class CIRGenModule;

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

  /// Get the CIR address space for alloca.
  virtual mlir::cir::AddressSpaceAttr getCIRAllocaAddressSpace() const {
    // Return the null attribute, which means the target does not care about the
    // alloca address space.
    return {};
  }

  /// Perform address space cast of an expression of pointer type.
  /// \param V is the value to be casted to another address space.
  /// \param SrcAddr is the CIR address space of \p V.
  /// \param DestAddr is the targeted CIR address space.
  /// \param DestTy is the destination pointer type.
  /// \param IsNonNull is the flag indicating \p V is known to be non null.
  virtual mlir::Value performAddrSpaceCast(CIRGenFunction &CGF, mlir::Value V,
                                           mlir::cir::AddressSpaceAttr SrcAddr,
                                           mlir::cir::AddressSpaceAttr DestAddr,
                                           mlir::Type DestTy,
                                           bool IsNonNull = false) const;

  /// Get LLVM calling convention for OpenCL kernel.
  virtual unsigned getOpenCLKernelCallingConv() const {
    // OpenCL kernels are called via an explicit runtime API with arguments
    // set with clSetKernelArg(), not as normal sub-functions.
    // Return SPIR_KERNEL by default as the kernel calling convention to
    // ensure the fingerprint is fixed such way that each OpenCL argument
    // gets one matching argument in the produced kernel function argument
    // list to enable feasible implementation of clSetKernelArg() with
    // aggregates etc. In case we would use the default C calling conv here,
    // clSetKernelArg() might break depending on the target-specific
    // conventions; different targets might split structs passed as values
    // to multiple function arguments etc.
    return llvm::CallingConv::SPIR_KERNEL;
  }

  virtual ~TargetCIRGenInfo() {}
};

} // namespace cir

#endif
