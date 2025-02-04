//===-- runtime/CUDA/registration.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/CUDA/registration.h"
#include "../terminator.h"
#include "flang/Runtime/CUDA/common.h"

#include "cuda_runtime.h"

namespace Fortran::runtime::cuda {

extern "C" {

extern void **__cudaRegisterFatBinary(void *);
extern void __cudaRegisterFatBinaryEnd(void *);
extern void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
    char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid,
    uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);
extern void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
    const char *deviceAddress, const char *deviceName, int ext, size_t size,
    int constant, int global);

void *RTDECL(CUFRegisterModule)(void *data) {
  void **fatHandle{__cudaRegisterFatBinary(data)};
  __cudaRegisterFatBinaryEnd(fatHandle);
  return fatHandle;
}

void RTDEF(CUFRegisterFunction)(
    void **module, const char *fctSym, char *fctName) {
  __cudaRegisterFunction(module, fctSym, fctName, fctName, -1, (uint3 *)0,
      (uint3 *)0, (dim3 *)0, (dim3 *)0, (int *)0);
}

void RTDEF(CUFRegisterVariable)(
    void **module, char *varSym, const char *varName, int64_t size) {
  __cudaRegisterVar(module, varSym, varName, varName, 0, size, 0, 0);
}

} // extern "C"

} // namespace Fortran::runtime::cuda
