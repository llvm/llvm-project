//===-- copyprof_reporting.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements the reporting mechanisms for CopyProf, generating
/// reports with stack traces when unnecessary object copies are destroyed.
///
//===----------------------------------------------------------------------===//

#include "copyprof_reporting.h"

#include "copyprof_internal.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

namespace __copyprof {
namespace {

// TODO: Make configurable via flags.
constexpr int kMaxNumStackFrames = 30;

}  // namespace

void LogCopyProfReport(uptr pc, uptr bp, uptr obj_size, bool did_allocate) {
  // FIXME: replace hard coded object size with flag.
  if (obj_size <= 16 || !did_allocate)
    return;
  UNINITIALIZED BufferedStackTrace stack_trace;
  stack_trace.Unwind(pc, bp, /*context=*/nullptr,
                     common_flags()->fast_unwind_on_fatal, kMaxNumStackFrames);
  InternalScopedString output;
  output.AppendF(
      "[copyprof] Destroyed unnecessary copy amounting to %zu bytes:\n",
      (usize)obj_size);
  stack_trace.PrintTo(&output);
  Printf("%s", output.data());
}
}  // namespace __copyprof
