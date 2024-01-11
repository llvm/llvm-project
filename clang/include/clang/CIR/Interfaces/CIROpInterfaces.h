//===- CIROpInterfaces.h - CIR Op Interfaces --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_CIR_OP_H_
#define MLIR_INTERFACES_CIR_OP_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"

#include "clang/AST/Attr.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Mangle.h"

namespace mlir {
namespace cir {} // namespace cir
} // namespace mlir

/// Include the generated interface declarations.
#include "clang/CIR/Interfaces/CIROpInterfaces.h.inc"

namespace mlir {
namespace cir {} // namespace cir
} // namespace mlir

#endif // MLIR_INTERFACES_CIR_OP_H_
