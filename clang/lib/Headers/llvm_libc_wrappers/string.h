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

// The GNU headers provide C++ standard compliant headers when in C++ mode and
// the LLVM libc does not. We need to perform a pretty nasty hack to trick the
// GNU headers into emitting the C compatible definitions so we can use them.
#if defined(__cplusplus) && defined(__GLIBC__)

// We need to make sure that the GNU C library has done its setup before we mess
// with the expected macro values.
#if !defined(__GLIBC_INTERNAL_STARTING_HEADER_IMPLEMENTATION) &&               \
    __has_include(<bits/libc-header-start.h>)
#define __GLIBC_INTERNAL_STARTING_HEADER_IMPLEMENTATION
#include <bits/libc-header-start.h>
#endif

// Trick the GNU headers into thinking that this clang is too old for the C++
// definitions.
#pragma push_macro("__clang_major__")
#define __clang_major__ 3
#endif

#include_next <string.h>

// Resore the original macros if they were changed.
#if defined(__cplusplus) && defined(__GLIBC__)
#pragma pop_macro("__clang_major__")
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
