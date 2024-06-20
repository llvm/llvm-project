//===- CallingConv.h - CIR Calling Conventions ------------*- C++ -------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines CIR's set of calling conventions.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_CALLINGCONV_H
#define CLANG_CIR_CALLINGCONV_H

// TODO: This whole file needs translated to CIR

namespace cir {

/// CallingConv Namespace - This namespace contains an enum with a value for the
/// well-known calling conventions.
namespace CallingConv {

/// LLVM IR allows to use arbitrary numbers as calling convention identifiers.
/// TODO: What should we do for this for CIR
using ID = unsigned;

/// A set of enums which specify the assigned numeric values for known llvm
/// calling conventions.
/// LLVM Calling Convention Represetnation
enum {
  /// C - The default llvm calling convention, compatible with C. This
  /// convention is the only calling convention that supports varargs calls. As
  /// with typical C calling conventions, the callee/caller have to tolerate
  /// certain amounts of prototype mismatch.
  C = 0,

  /// Used for SPIR kernel functions. Inherits the restrictions of SPIR_FUNC,
  /// except it cannot have non-void return values, it cannot have variable
  /// arguments, it can also be called by the host or it is externally
  /// visible.
  SPIR_KERNEL = 76,
};

} // namespace CallingConv

} // namespace cir

#endif
