//===-- copyprof_stack.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements stack unwinding utilities for CopyProf, enabling
/// stack trace collection for copy profiling reports.
///
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_stacktrace.h"

namespace __sanitizer {

void BufferedStackTrace::UnwindImpl(uptr pc, uptr bp, void* context,
                                    bool request_fast, u32 max_depth) {
  Unwind(max_depth, pc, bp, context, 0, 0,
         StackTrace::WillUseFastUnwind(request_fast));
}

}  // namespace __sanitizer
