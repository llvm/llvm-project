//===-- Macro Override for SnMalloc ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// skip special headers
#define _LLVM_LIBC_COMMON_H
#define LLVM_LIBC_ERRNO_H

// define common macros
#define __BEGIN_C_DECLS namespace LIBC_NAMESPACE {
#define __END_C_DECLS                                                          \
  }                                                                            \
  using namespace LIBC_NAMESPACE;

#define _Noreturn [[noreturn]]
#define _Alignas alignas
#define _Static_assert static_assert
#define _Alignof alignof
#define _Thread_local thread_local

// Use empty definition to avoid spec mismatching
// We are building internally anyway, hence noexcept does not matter here
#define __NOEXCEPT

// TODO: define this in stdio.h
#define STDERR_FILENO 2

// When PThread destructor is used, snmalloc uses pthread_key_create to
// register its cleanup routine together with an atexit operation for main
// thread cleanup.
#define SNMALLOC_USE_PTHREAD_DESTRUCTORS

// Enforce internal errno implementation
#include "hdr/errno_macros.h"
#include "src/errno/libc_errno.h"
#define errno ::LIBC_NAMESPACE::libc_errno
