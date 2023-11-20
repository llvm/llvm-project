//===--- OpenACCKinds.h - OpenACC Enums -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines some OpenACC-specific enums and functions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_OPENACCKINDS_H
#define LLVM_CLANG_BASIC_OPENACCKINDS_H

namespace clang {
// Represents the Construct/Directive kind of a pragma directive. Note the
// OpenACC standard is inconsistent between calling these Construct vs
// Directive, but we're calling it a Directive to be consistent with OpenMP.
enum class OpenACCDirectiveKind {
  // Compute Constructs.
  Parallel,
  Serial,
  Kernels,

  // Data Environment.
  Data,
  // FIXME: 'enter data', 'exit data'.
  HostData,

  // Misc.
  Loop,
  // FIXME: 'cache'

  // Combined Constructs.
  ParallelLoop,
  SerialLoop,
  KernelsLoop,

  // FIXME: atomic Construct variants.

  // Declare Directive.
  Declare,

  // Executable Directives. "wait" is first referred to here, but ends up being
  // in its own section after "routine".
  Init,
  Shutdown,
  Set,
  Update,
  // FIXME: wait construct.

  // FIXME: routine construct.

  // Invalid.
  Invalid,
};
} // namespace clang

#endif // LLVM_CLANG_BASIC_OPENACCKINDS_H
