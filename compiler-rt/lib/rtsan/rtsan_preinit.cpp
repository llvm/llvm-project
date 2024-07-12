//===--- rtsan_preinit.cpp - Realtime Sanitizer -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_internal_defs.h"
#include <rtsan/rtsan.h>

#if SANITIZER_CAN_USE_PREINIT_ARRAY

// The symbol is called __local_rtsan_preinit, because it's not intended to be
// exported.
// This code is linked into the main executable when -fsanitize=realtime is in
// the link flags. It can only use exported interface functions.
__attribute__((section(".preinit_array"),
               used)) void (*__local_rtsan_preinit)(void) = __rtsan_init;

#endif
