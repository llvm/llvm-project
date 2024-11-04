//===-- Wrapper for C standard stdio.h declarations on the GPU ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLANG_LLVM_LIBC_WRAPPERS_STDIO_H__
#define __CLANG_LLVM_LIBC_WRAPPERS_STDIO_H__

#if !defined(_OPENMP) && !defined(__HIP__) && !defined(__CUDA__)
#error "This file is for GPU offloading compilation only"
#endif

#include_next <stdio.h>

#if __has_include(<llvm-libc-decls/stdio.h>)

#if defined(__HIP__) || defined(__CUDA__)
#define __LIBC_ATTRS __attribute__((device))
#endif

// Some headers provide these as macros. Temporarily undefine them so they do
// not conflict with any definitions for the GPU.

#pragma push_macro("stdout")
#pragma push_macro("stdin")
#pragma push_macro("stderr")

#undef stdout
#undef stderr
#undef stdin

#pragma omp begin declare target

#include <llvm-libc-decls/stdio.h>

#pragma omp end declare target

#undef __LIBC_ATTRS

// Restore the original macros when compiling on the host.
#if !defined(__NVPTX__) && !defined(__AMDGPU__)
#pragma pop_macro("stdout")
#pragma pop_macro("stderr")
#pragma pop_macro("stdin")
#endif

#endif

#endif // __CLANG_LLVM_LIBC_WRAPPERS_STDIO_H__
