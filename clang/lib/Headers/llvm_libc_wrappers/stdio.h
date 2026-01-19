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

#if defined(__HIP__) || defined(__CUDA__)
#define __LIBC_ATTRS __attribute__((device))
#else
#define __LIBC_ATTRS
#endif

// To turn off emissary print (and this macro) set -fno-use-emissary-print.
#ifdef OFFLOAD_ENABLE_EMISSARY_PRINT
#if defined(__NVPTX__) || defined(__AMDGCN__)
#include <EmissaryIds.h>
#define fprintf(...)                                                           \
  _emissary_exec(_PACK_EMIS_IDS(EMIS_ID_PRINT, _fprintf_idx, 0, 0),            \
                 __VA_ARGS__);
#define printf(...)                                                            \
  _emissary_exec(_PACK_EMIS_IDS(EMIS_ID_PRINT, _printf_idx, 0, 0), __VA_ARGS__);
#define fputc(c, stream) fprintf(stream, "%c", (unsigned char)(c))
#define putc(c, stream) fprintf(stream, "%c", (unsigned char)(c))
#define putchar(c) printf("%c", (char)(c))
#define fputs(str, stream) fprintf((stream), "%s", (str))
#define puts(str) fprintf(stdout, "%s", (str))
#endif
#endif

// Some headers provide these as macros. Temporarily undefine them so they do
// not conflict with any definitions for the GPU.


#undef __LIBC_ATTRS

#endif // __CLANG_LLVM_LIBC_WRAPPERS_STDIO_H__
