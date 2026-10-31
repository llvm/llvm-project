//===-- Definition of macros to be used with Annex K functions ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This header intentionally has no include guard so that defining
// __STDC_WANT_LIB_EXT1__ after including another header can enable Annex K.

#if (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L) ||              \
    (defined(__cplusplus) && __cplusplus >= 201703L)

// TODO(bassiounix): Who should def this macro (clang vs libc)? Where?
// TODO(bassiounix): uncomment/move when Annex K is fully implemented.
// #define __STDC_LIB_EXT1__ 201112L

#if defined(__STDC_WANT_LIB_EXT1__) && __STDC_WANT_LIB_EXT1__ == 1

#ifndef LIBC_HAS_ANNEX_K
#define LIBC_HAS_ANNEX_K
#endif

#endif // defined(__STDC_WANT_LIB_EXT1__) && __STDC_WANT_LIB_EXT1__ == 1

#endif // (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L) ||
       // (defined(__cplusplus) && __cplusplus >= 201703L)
