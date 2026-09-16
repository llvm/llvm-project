//===-- copyprof_internal.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file declares internal runtime data structures, flags, and helper
/// functions shared within the CopyProf runtime library.
///
//===----------------------------------------------------------------------===//

#ifndef COPYPROF_INTERNAL_H
#define COPYPROF_INTERNAL_H

#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

using __sanitizer::u32;
using __sanitizer::u64;
using __sanitizer::uptr;
using __sanitizer::usize;

namespace __copyprof {

using __sanitizer::BufferedStackTrace;
extern bool copyprof_is_initialized;
extern bool copyprof_init_is_running;

}  // namespace __copyprof

#endif  // COPYPROF_INTERNAL_H
