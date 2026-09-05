//===---- cuda_runtime.h - Hermetic CUDA runtime shadow ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===

#ifndef __CLANG_CUDA_RUNTIME_SHADOW_H__
#define __CLANG_CUDA_RUNTIME_SHADOW_H__

#include <device_functions.h>

#if __has_include_next(<cuda_runtime.h>)
#include_next <cuda_runtime.h>
#endif

#endif // __CLANG_CUDA_RUNTIME_SHADOW_H__
