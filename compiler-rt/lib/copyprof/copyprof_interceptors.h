//===-- copyprof_interceptors.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file declares the standard library interceptor interface for CopyProf,
/// defining hooks for intercepted memory and string functions.
///
//===----------------------------------------------------------------------===//

#ifndef COPYPROF_INTERCEPTORS_H
#define COPYPROF_INTERCEPTORS_H

#include "interception/interception.h"

// Functions whose interceptor is defined in another TU.
DECLARE_REAL(void*, memset, void* block, int c, SIZE_T size)
DECLARE_REAL(void*, memcpy, void* to, const void* from, SIZE_T size)

// The macros below must be defined when using the interceptor infrastructure
// (i.e. sanitizer_common_interceptors*.inc files).
#define COPYPROF_INTERCEPT_FUNC(name)                            \
  do {                                                           \
    if (!INTERCEPT_FUNCTION(name))                               \
      VReport(1, "CopyProf: failed to intercept '%s'\n", #name); \
  } while (0)
#define COMMON_INTERCEPT_FUNCTION(name) COPYPROF_INTERCEPT_FUNC(name)

namespace __copyprof {

// Initializes the interception infrastructure, s.t. CopyProf can hook into
// library functions or syscalls. Must be called once during startup.
void InitializeInterceptors();

}  // namespace __copyprof

#endif  // COPYPROF_INTERCEPTORS_H
