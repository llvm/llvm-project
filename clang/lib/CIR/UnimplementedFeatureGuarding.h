//===---- UnimplementedFeatureGuarding.h - Checks against NYI ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file introduces some helper classes to guard against features that
// CodeGen supports that we do not have and also do not have great ways to
// assert against.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_UFG
#define LLVM_CLANG_LIB_CIR_UFG

namespace cir {
struct UnimplementedFeature {
  // TODO(CIR): Implement the CIRGenFunction::buildTypeCheck method that handles
  // sanitizer related type check features
  static bool buildTypeCheck() { return false; }
};
} // namespace cir

#endif
