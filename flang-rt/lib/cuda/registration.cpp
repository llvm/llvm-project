//===-- lib/cuda/registration.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/CUDA/registration.h"
#include "flang-rt/runtime/terminator.h"
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
extern void __cudaRegisterManagedVar(void **fatCubinHandle,
    void **hostVarPtrAddress, char *deviceAddress, const char *deviceName,
    int ext, size_t size, int constant, int global);
extern char __cudaInitModule(void **fatCubinHandle);

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

void RTDEF(CUFRegisterManagedVariable)(
    void **module, void **varSym, char *varName, int64_t size) {
  __cudaRegisterManagedVar(module, varSym, varName, varName, 0, size, 0, 0);
}

void RTDEF(CUFInitModule)(void **module) { __cudaInitModule(module); }

} // extern "C"

} // namespace Fortran::runtime::cuda
