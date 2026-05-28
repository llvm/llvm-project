//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Centralised dialect registration for CIR tools.  Every tool that parses or
// transforms CIR (cir-opt, -fclangcir...)
// should call registerCIRDialects() instead of registering dialects and
// extensions individually, so the dialect surface presented to all tools is
// always consistent.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_CIRDIALECTREGISTRATION_H
#define CLANG_CIR_DIALECT_CIRDIALECTREGISTRATION_H

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace cir {

// Populate \p registry with every dialect and dialect extension required to
// parse, verify and transform CIR:
void registerCIRDialects(mlir::DialectRegistry &registry);

} // namespace cir

#endif // CLANG_CIR_DIALECT_CIRDIALECTREGISTRATION_H
