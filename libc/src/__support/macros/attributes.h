//===-- Portable attributes -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This header file defines macros for declaring attributes for functions,
// types, and variables.
//
// These macros are used within llvm-libc and allow the compiler to optimize,
// where applicable, certain function calls.
//
// Most macros here are exposing GCC or Clang features, and are stubbed out for
// other compilers.

#ifndef LLVM_LIBC_SRC___SUPPORT_MACROS_ATTRIBUTES_H
#define LLVM_LIBC_SRC___SUPPORT_MACROS_ATTRIBUTES_H

#include "properties/architectures.h"

#ifndef __has_attribute
#define __has_attribute(x) 0
#endif

#define LIBC_INLINE inline
#define LIBC_INLINE_VAR inline
#define LIBC_INLINE_ASM __asm__ __volatile__
#define LIBC_UNUSED __attribute__((unused))

// Uses the platform specific specialization
#define LIBC_THREAD_MODE_PLATFORM 0

// Mutex guards nothing, used in single-threaded implementations
#define LIBC_THREAD_MODE_SINGLE 1

// Vendor provides implementation
#define LIBC_THREAD_MODE_EXTERNAL 2

// libcxx doesn't define LIBC_THREAD_MODE, unless that is passed in the command
// line in the CMake invocation. This defaults to the original implementation
// (before changes in https://github.com/llvm/llvm-project/pull/145358)
#ifndef LIBC_THREAD_MODE
#define LIBC_THREAD_MODE LIBC_THREAD_MODE_PLATFORM
#endif // LIBC_THREAD_MODE

#if LIBC_THREAD_MODE != LIBC_THREAD_MODE_PLATFORM &&                           \
    LIBC_THREAD_MODE != LIBC_THREAD_MODE_SINGLE &&                             \
    LIBC_THREAD_MODE != LIBC_THREAD_MODE_EXTERNAL
#error LIBC_THREAD_MODE must be one of the following values: \
LIBC_THREAD_MODE_PLATFORM, \
LIBC_THREAD_MODE_SINGLE, \
LIBC_THREAD_MODE_EXTERNAL.
#endif

#if LIBC_THREAD_MODE == LIBC_THREAD_MODE_SINGLE
#define LIBC_THREAD_LOCAL
#else
#define LIBC_THREAD_LOCAL thread_local
#endif

#if __cplusplus >= 202002L
#define LIBC_CONSTINIT constinit
#elif __has_attribute(__require_constant_initialization__)
#define LIBC_CONSTINIT __attribute__((__require_constant_initialization__))
#else
#define LIBC_CONSTINIT
#endif

#if defined(__clang__) && __has_attribute(preferred_type)
#define LIBC_PREFERED_TYPE(TYPE) [[clang::preferred_type(TYPE)]]
#else
#define LIBC_PREFERED_TYPE(TYPE)
#endif

#endif // LLVM_LIBC_SRC___SUPPORT_MACROS_ATTRIBUTES_H
