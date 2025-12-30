//===-- Wrapper for C standard stdlib.h declarations on the GPU -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLANG_LLVM_LIBC_WRAPPERS_STDLIB_H__
#define __CLANG_LLVM_LIBC_WRAPPERS_STDLIB_H__

#if !defined(_OPENMP) && !defined(__HIP__) && !defined(__CUDA__)
#error "This file is for GPU offloading compilation only"
#endif

#include_next <stdlib.h>

#if defined(__HIP__) || defined(__CUDA__)
#define __LIBC_ATTRS __attribute__((device))
#else
#define __LIBC_ATTRS
#endif

#pragma omp begin declare target

// TODO: Define these for CUDA / HIP.

#pragma omp end declare target

#undef __LIBC_ATTRS

#endif // __CLANG_LLVM_LIBC_WRAPPERS_STDLIB_H__
