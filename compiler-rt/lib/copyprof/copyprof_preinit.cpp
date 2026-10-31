//===-- copyprof_preinit.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file sets up early pre-initialization hooks for CopyProf, ensuring the
/// runtime initializes before any instrumented application code runs.
///
//===----------------------------------------------------------------------===//

#include "copyprof_interface_internal.h"

using namespace __copyprof;

// Make sure CopyProf initialization runs as early as possible as some
// applications use preinit_array on their own.
#if SANITIZER_CAN_USE_PREINIT_ARRAY
__attribute__((section(".preinit_array"), used)) static auto preinit =
    __copyprof_init_once;
#endif
