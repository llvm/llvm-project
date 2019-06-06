//===-- gwp_asan_definitions.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GWP_ASAN_DEFINITIONS_H_
#define GWP_ASAN_DEFINITIONS_H_

#define TLS_INITIAL_EXEC __thread __attribute__((tls_model("initial-exec")))

#ifdef LIKELY
# undef LIKELY
#endif // defined(LIKELY)
#define LIKELY(X) __builtin_expect(!!(X), 1)

#ifdef UNLIKELY
# undef UNLIKELY
#endif // defined(UNLIKELY)
#define UNLIKELY(X) __builtin_expect(!!(X), 0)

#ifdef ALWAYS_INLINE
# undef ALWAYS_INLINE
#endif // defined(ALWAYS_INLINE)
#define ALWAYS_INLINE inline __attribute__((always_inline))

#endif // GWP_ASAN_DEFINITIONS_H_
