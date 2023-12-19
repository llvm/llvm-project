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

  // Data Environment. "enter data" and "exit data" are also referred to in the
  // Executable Directives section, but just as a back reference to the Data
  // Environment.
  Data,
  EnterData,
  ExitData,
  HostData,

  // Misc.
  Loop,
  Cache,

  // Combined Constructs.
  ParallelLoop,
  SerialLoop,
  KernelsLoop,

  // Atomic Construct.
  Atomic,

  // Declare Directive.
  Declare,

  // Executable Directives. "wait" is first referred to here, but ends up being
  // in its own section after "routine".
  Init,
  Shutdown,
  Set,
  Update,
  Wait,

  // Procedure Calls in Compute Regions.
  Routine,

  // Invalid.
  Invalid,
};

enum class OpenACCAtomicKind {
  Read,
  Write,
  Update,
  Capture,
  Invalid,
};

/// Represents the kind of an OpenACC clause.
enum class OpenACCClauseKind {
  // 'finalize' clause, allowed on 'exit data' directive.
  Finalize,
  // 'if_present' clause, allowed on 'host_data' and 'update' directives.
  IfPresent,
  // 'seq' clause, allowed on 'loop' and 'routine' directives.
  Seq,
  // 'independent' clause, allowed on 'loop' directives.
  Independent,
  // 'auto' clause, allowed on 'loop' directives.
  Auto,
  // 'worker' clause, allowed on 'loop' and 'routine' directives.
  Worker,
  // 'vector' clause, allowed on 'loop' and 'routine' directives. Takes no
  // arguments for 'routine', so the 'loop' version is not yet implemented
  // completely.
  Vector,
  // 'nohost' clause, allowed on 'routine' directives.
  NoHost,
  // Represents an invalid clause, for the purposes of parsing.
  Invalid,
};
} // namespace clang

#endif // LLVM_CLANG_BASIC_OPENACCKINDS_H
