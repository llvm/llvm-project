//===--- __clang_cuda_builtin_vars.h - Stub header for tests ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ___CLANG_CUDA_BUILTIN_VARS_H_
#define ___CLANG_CUDA_BUILTIN_VARS_H_

#define __CUDA_DEVICE_BUILTIN(FIELD) \
  static unsigned int FIELD;

struct __cuda_builtin_threadIdx_t {
  __CUDA_DEVICE_BUILTIN(x);
};

struct __cuda_builtin_blockIdx_t {
  __CUDA_DEVICE_BUILTIN(x);
};

struct __cuda_builtin_blockDim_t {
  __CUDA_DEVICE_BUILTIN(x);
};

struct __cuda_builtin_gridDim_t {
  __CUDA_DEVICE_BUILTIN(x);
};

__cuda_builtin_threadIdx_t threadIdx;
__cuda_builtin_blockIdx_t blockIdx;
__cuda_builtin_blockDim_t blockDim;
__cuda_builtin_gridDim_t gridDim;

#endif // ___CLANG_CUDA_BUILTIN_VARS_H_
