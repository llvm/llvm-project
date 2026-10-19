//===---- hip_runtime.h - Hermetic HIP runtime shadow ---------------------===
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===

#ifndef __CLANG_HIP_RUNTIME_SHADOW_H__
#define __CLANG_HIP_RUNTIME_SHADOW_H__

#include <__clang_gpu_runtime_wrapper.h>

#if __has_include_next(<hip/hip_runtime.h>)
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_RUNTIME_H
#include_next <hip/hip_runtime.h>
#endif

#include <__clang_gpu_builtin_vars.h>
#include <__clang_gpu_device_functions.h>
#include <__clang_gpu_intrinsics.h>

#endif // __CLANG_HIP_RUNTIME_SHADOW_H__
