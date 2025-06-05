//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Provides an LLVM-like API wrapper to DLTI and MLIR layout queries. This
// makes it easier to port some of LLVM codegen layout logic to CIR.
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIRDATALAYOUT_H
#define CLANG_CIR_DIALECT_IR_CIRDATALAYOUT_H

#include "mlir/IR/BuiltinOps.h"

namespace cir {

// TODO(cir): This might be replaced by a CIRDataLayout interface which can
// provide the same functionalities.
class CIRDataLayout {
  // This is starting with the minimum functionality needed for code that is
  // being upstreamed. Additional methods and members will be added as needed.
public:
  mlir::DataLayout layout;

  /// Constructs a DataLayout the module's data layout attribute.
  CIRDataLayout(mlir::ModuleOp modOp) : layout{modOp} {}
};

} // namespace cir

#endif // CLANG_CIR_DIALECT_IR_CIRDATALAYOUT_H
