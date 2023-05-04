//===--- CIRDataLayout.h - CIR Data Layout Information ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Provides a LLVM-like API wrapper to DLTI and MLIR layout queries. This makes
// it easier to port some of LLVM codegen layout logic to CIR.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CIRDATALAYOUT_H
#define LLVM_CLANG_LIB_CIR_CIRDATALAYOUT_H

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/BuiltinOps.h"

namespace cir {

class CIRDataLayout {
  bool bigEndian = false;

public:
  mlir::DataLayout layout;

  CIRDataLayout(mlir::ModuleOp modOp);
  bool isBigEndian() { return bigEndian; }

  // `useABI` is `true` if not using prefered alignment.
  unsigned getAlignment(mlir::Type ty, bool useABI) const {
    return useABI ? layout.getTypeABIAlignment(ty)
                  : layout.getTypePreferredAlignment(ty);
  }
  unsigned getABITypeAlign(mlir::Type ty) const {
    return getAlignment(ty, true);
  }
};

} // namespace cir

#endif