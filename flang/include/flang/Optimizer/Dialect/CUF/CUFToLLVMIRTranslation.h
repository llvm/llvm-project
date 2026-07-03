//===- CUFToLLVMIRTranslation.h - CUF Dialect to LLVM IR --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for GPU dialect to LLVM IR translation.
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_OPTIMIZER_DIALECT_CUF_GPUTOLLVMIRTRANSLATION_H_
#define FLANG_OPTIMIZER_DIALECT_CUF_GPUTOLLVMIRTRANSLATION_H_

namespace mlir {
class DialectRegistry;
class MLIRContext;
} // namespace mlir

namespace cuf {

/// Register the CUF dialect and the translation from it to the LLVM IR in
/// the given registry.
void registerCUFDialectTranslation(mlir::DialectRegistry &registry);

} // namespace cuf

#endif // FLANG_OPTIMIZER_DIALECT_CUF_GPUTOLLVMIRTRANSLATION_H_
