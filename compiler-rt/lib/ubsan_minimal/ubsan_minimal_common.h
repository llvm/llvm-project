//===-- ubsan_minimal_common.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of the minimal UB sanitizer runtime.
//
//===----------------------------------------------------------------------===//

#ifndef UBSAN_MINIMAL_COMMON_H
#define UBSAN_MINIMAL_COMMON_H

#if defined(__UINTPTR_TYPE__) && defined(__SIZE_TYPE__)
typedef __UINTPTR_TYPE__ uintptr_t;
typedef __SIZE_TYPE__ size_t;

static_assert(sizeof(uintptr_t) == sizeof(void *),
              "uintptr_t must be the same size as void*");
#else // defined(__UINTPTR_TYPE__) && defined(__SIZE_TYPE__)
#include <stdint.h>
#endif // defined(__UINTPTR_TYPE__) && defined(__SIZE_TYPE__)

void __ubsan_message(const char *msg);
void __ubsan_message(const char *kind, uintptr_t caller);

[[noreturn]] void __ubsan_abort();
[[noreturn]] void __ubsan_abort_with_message(const char *kind,
                                             uintptr_t caller);

#endif // UBSAN_MINIMAL_COMMON_H
