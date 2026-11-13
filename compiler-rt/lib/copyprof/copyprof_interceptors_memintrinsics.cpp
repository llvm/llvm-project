//===-- copyprof_interceptors_memintrinsics.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements interceptors for memory intrinsic functions (memcpy,
/// memmove, memset) to correctly maintain shadow memory status.
///
//===----------------------------------------------------------------------===//

#define SANITIZER_COMMON_NO_REDEFINE_BUILTINS

#include "copyprof_interceptors.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_libc.h"

using namespace __copyprof;
using namespace __sanitizer;

// The default implementations fall back to the internal intrinsics until
// CopyProf is initialized (they may be called during `__copyprof_init_once`,
// e.g. from the internals of `Printf`), and only update shadow memory if
// `intercept_intrin` is set.
#include "sanitizer_common/sanitizer_common_interceptors_memintrinsics.inc"
