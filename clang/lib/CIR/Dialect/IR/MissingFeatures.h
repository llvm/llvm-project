//===---- UnimplementedFeatureGuarding.h - Checks against NYI ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file introduces some helper classes to guard against features that
// CIR dialect supports that we do not have and also do not have great ways to
// assert against.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_IR_UFG
#define LLVM_CLANG_LIB_CIR_DIALECT_IR_UFG

namespace cir {

struct MissingFeatures {
  // C++ ABI support
  static bool cxxABI() { return false; }
  static bool setCallingConv() { return false; }

  // Address space related
  static bool addressSpace() { return false; }

  // Sanitizers
  static bool buildTypeCheck() { return false; }
};

} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_IR_UFG
