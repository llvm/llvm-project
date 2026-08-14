//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Centralised dialect registration for CIR tools. Every tool that parses or
// transforms CIR (cir-opt, cir-lsp-server, -fclangir...) should call
// registerAllDialects() instead of registering dialects and extensions
// individually, so the dialect surface presented to all tools is always
// consistent.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_INITALLDIALECTS_H
#define CLANG_CIR_INITALLDIALECTS_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
} // namespace mlir

namespace cir {

// Populate \p registry with every dialect and dialect extension required to
// parse, verify and transform CIR.
void registerAllDialects(mlir::DialectRegistry &registry);

// Convenience overload that registers the same dialects directly into \p
// context for lazy loading.
void registerAllDialects(mlir::MLIRContext &context);

} // namespace cir

#endif // CLANG_CIR_INITALLDIALECTS_H
