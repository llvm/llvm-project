//===-- Wrapper for C standard string.h declarations on the GPU -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLANG_LLVM_LIBC_WRAPPERS_STRING_H__
#define __CLANG_LLVM_LIBC_WRAPPERS_STRING_H__

#if !defined(_OPENMP) && !defined(__HIP__) && !defined(__CUDA__)
#error "This file is for GPU offloading compilation only"
#endif

// The GNU headers provide non C-standard headers when in C++ mode. Manually
// undefine it here so that the definitions agree with the C standard for our
// purposes.
#ifdef __cplusplus
extern "C" {
#pragma push_macro("__cplusplus")
#undef __cplusplus
#endif

#include_next <string.h>

#pragma pop_macro("__cplusplus")
#ifdef __cplusplus
}
#endif

#if __has_include(<llvm-libc-decls/string.h>)

#if defined(__HIP__) || defined(__CUDA__)
#define __LIBC_ATTRS __attribute__((device))
#endif

#pragma omp begin declare target

#include <llvm-libc-decls/string.h>

#pragma omp end declare target

#undef __LIBC_ATTRS

#endif

#endif // __CLANG_LLVM_LIBC_WRAPPERS_STRING_H__
