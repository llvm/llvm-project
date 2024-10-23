//===-- runtime/CUDA/registration.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/CUDA/registration.h"

#include "cuda_runtime.h"

namespace Fortran::runtime::cuda {

extern "C" {

extern void **__cudaRegisterFatBinary(void *);
extern void __cudaRegisterFatBinaryEnd(void *);
extern void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
    char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid,
    uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);

void *RTDECL(CUFRegisterModule)(void *data) {
  void **fatHandle{__cudaRegisterFatBinary(data)};
  __cudaRegisterFatBinaryEnd(fatHandle);
  return fatHandle;
}

void RTDEF(CUFRegisterFunction)(void **module, const char *fct) {
  __cudaRegisterFunction(module, fct, const_cast<char *>(fct), fct, -1,
      (uint3 *)0, (uint3 *)0, (dim3 *)0, (dim3 *)0, (int *)0);
}
}
} // namespace Fortran::runtime::cuda
