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
#include <cstdint>
#include <unistd.h>

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
extern void __cudaRegisterHostVar(
    void **fatCubinHandle, const char *deviceName, char *hostVar, size_t size);
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

void RTDEF(CUFRegisterHostMemoryRange)(void *begin, void *end) {
  if (!begin || end <= begin)
    return;
  const auto pageSize{static_cast<std::uintptr_t>(sysconf(_SC_PAGESIZE))};
  const auto first{reinterpret_cast<std::uintptr_t>(begin) & ~(pageSize - 1)};
  const auto last{
      (reinterpret_cast<std::uintptr_t>(end) + pageSize - 1) & ~(pageSize - 1)};
  cudaError_t err{cudaHostRegister(reinterpret_cast<void *>(first),
      last - first, cudaHostRegisterPortable | cudaHostRegisterMapped)};
  if (err == cudaErrorHostMemoryAlreadyRegistered) {
    // Clear the error state left by the failed call.
    (void)cudaGetLastError();
    return;
  }
  if (err != cudaSuccess) {
    const char *name{cudaGetErrorName(err)};
    Terminator terminator{__FILE__, __LINE__};
    terminator.Crash("cudaHostRegister(%p, %zu) failed with '%s'",
        reinterpret_cast<void *>(first), static_cast<std::size_t>(last - first),
        name ? name : "<unknown>");
  }
}

} // extern "C"

} // namespace Fortran::runtime::cuda
