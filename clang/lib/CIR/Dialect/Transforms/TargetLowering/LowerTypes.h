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

namespace mlir {
namespace cir {

// Forward declarations.
class LowerModule;

/// This class organizes lowering to ABI-specific types in CIR.
class LowerTypes {
  // FIXME(cir): This abstraction could likely be replaced by a MLIR interface
  // or direct queries to CIR types. It here mostly for code parity.

private:
  LowerModule &LM;
  CIRLowerContext &queries;
  const clang::TargetInfo &Target;
  CIRCXXABI &CXXABI;

  // This should not be moved earlier, since its initialization depends on some
  // of the previous reference members being already initialized
  const ABIInfo &TheABIInfo;

  // Used to build types and other MLIR operations.
  MLIRContext *mlirContext;

  ::cir::CIRDataLayout DL;

public:
  LowerTypes(LowerModule &LM, StringRef DLString);
  ~LowerTypes() = default;

  LowerModule &getLM() const { return LM; }
};

} // namespace cir
} // namespace mlir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_LOWERTYPES_H
