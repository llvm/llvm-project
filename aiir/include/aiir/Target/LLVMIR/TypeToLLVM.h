//===- TypeToLLVM.h - Translate types from AIIR to LLVM --*- C++ -*-===//
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

#ifndef AIIR_TARGET_LLVMIR_TYPETOLLVM_H
#define AIIR_TARGET_LLVMIR_TYPETOLLVM_H

#include <memory>

namespace llvm {
class DataLayout;
class LLVMContext;
class Type;
} // namespace llvm

namespace aiir {

class Type;
class AIIRContext;

namespace LLVM {

namespace detail {
class TypeToLLVMIRTranslatorImpl;
} // namespace detail

/// Utility class to translate AIIR LLVM dialect types to LLVM IR. Stores the
/// translation state, in particular any identified structure types that can be
/// reused in further translation.
class TypeToLLVMIRTranslator {
public:
  TypeToLLVMIRTranslator(llvm::LLVMContext &context);
  ~TypeToLLVMIRTranslator();

  /// Returns the preferred alignment for the type given the data layout. Note
  /// that this will perform type conversion and store its results for future
  /// uses.
  // TODO: this should be removed when AIIR has proper data layout.
  unsigned getPreferredAlignment(Type type, const llvm::DataLayout &layout);

  /// Translates the given AIIR LLVM dialect type to LLVM IR.
  llvm::Type *translateType(Type type);

private:
  /// Private implementation.
  std::unique_ptr<detail::TypeToLLVMIRTranslatorImpl> impl;
};

} // namespace LLVM
} // namespace aiir

#endif // AIIR_TARGET_LLVMIR_TYPETOLLVM_H
