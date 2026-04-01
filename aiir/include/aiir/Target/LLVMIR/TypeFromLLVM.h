//===- TypeFromLLVM.h - Translate types from LLVM to AIIR --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the type translation function going from AIIR LLVM dialect
// to LLVM IR and back.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TARGET_LLVMIR_TYPEFROMLLVM_H
#define AIIR_TARGET_LLVMIR_TYPEFROMLLVM_H

#include <memory>

namespace llvm {
class Type;
} // namespace llvm

namespace aiir {

class Type;
class AIIRContext;

namespace LLVM {

namespace detail {
class TypeFromLLVMIRTranslatorImpl;
} // namespace detail

/// Utility class to translate LLVM IR types to the AIIR LLVM dialect. Stores
/// the translation state, in particular any identified structure types that are
/// reused across translations.
class TypeFromLLVMIRTranslator {
public:
  TypeFromLLVMIRTranslator(AIIRContext &context,
                           bool importStructsAsLiterals = false);
  ~TypeFromLLVMIRTranslator();

  /// Translates the given LLVM IR type to the AIIR LLVM dialect.
  Type translateType(llvm::Type *type);

private:
  /// Private implementation.
  std::unique_ptr<detail::TypeFromLLVMIRTranslatorImpl> impl;
};

} // namespace LLVM
} // namespace aiir

#endif // AIIR_TARGET_LLVMIR_TYPEFROMLLVM_H
