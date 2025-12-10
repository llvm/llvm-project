//===-- include/flang/Runtime/main.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_MAIN_H_
#define FORTRAN_RUNTIME_MAIN_H_

#include "flang/Runtime/c-or-cpp.h"
#include "flang/Runtime/entry-names.h"
#if (not defined(__AMDGPU__) && not defined(__NVPTX__)) || not defined(EMBED_FLANG_RT_GPU_LLVM_IR)
#include <thread>
#endif

struct EnvironmentDefaultList;

#if (not defined(__AMDGPU__) && not defined(__NVPTX__)) || not defined(EMBED_FLANG_RT_GPU_LLVM_IR)
std::thread::id RTNAME(GetMainThreadId)();
#endif

FORTRAN_EXTERN_C_BEGIN
void RTNAME(ProgramStart)(
    int, const char *[], const char *[], const struct EnvironmentDefaultList *);
void RTNAME(ByteswapOption)(void); // -byteswapio
FORTRAN_EXTERN_C_END

#endif // FORTRAN_RUNTIME_MAIN_H_
