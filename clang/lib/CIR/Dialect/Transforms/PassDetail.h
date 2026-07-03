//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIR_DIALECT_TRANSFORMS_PASSDETAIL_H
#define CIR_DIALECT_TRANSFORMS_PASSDETAIL_H

#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"

namespace cir {
class CIRDialect;
} // namespace cir

namespace mlir {
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

#define GEN_PASS_DECL
#include "clang/CIR/Dialect/Passes.h.inc"

} // namespace mlir

#endif // CIR_DIALECT_TRANSFORMS_PASSDETAIL_H
