//===--- LowerTypes.cpp - Type lowering for CIR dialect -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/CodeGenTypes.cpp. The queries
// are adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERTYPES_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERTYPES_H

#include "ABIInfo.h"
#include "CIRCXXABI.h"
#include "CIRLowerContext.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"

namespace cir {

// Forward declarations.
class LowerModule;

/// This class organizes lowering to ABI-specific types in CIR.
class LowerTypes {
  // FIXME(cir): This abstraction could likely be replaced by a MLIR interface
  // or direct queries to CIR types. It's here mostly for code parity.

private:
  LowerModule &lm;
  CIRLowerContext &context;
  const clang::TargetInfo &target;
  CIRCXXABI &CXXABI;

  // This should not be moved earlier, since its initialization depends on some
  // of the previous reference members being already initialized
  const ABIInfo &theABIInfo;

  // Used to build types and other MLIR operations.
  mlir::MLIRContext *mlirContext;

  cir::CIRDataLayout dataLayout;

public:
  LowerTypes(LowerModule &lm);
  ~LowerTypes() = default;

  LowerModule &getLm() const { return lm; }
//   CIRLowerContext &getContext() { return context; }
//   const clang::TargetInfo &getTarget() const { return target; }
//   const cir::CIRDataLayout &getDataLayout() const { return dataLayout; }
//   CIRCXXABI &getCXXABI() const { return CXXABI; }
//   mlir::MLIRContext *getMLIRContext() { return mlirContext; }
};

} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERTYPES_H
