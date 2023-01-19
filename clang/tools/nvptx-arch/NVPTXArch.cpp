//===- NVPTXArch.cpp - list installed NVPTX devies ------*- C++ -*---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a tool for detecting name of CUDA gpus installed in the
// system.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <cstdio>
#include <memory>

#if DYNAMIC_CUDA
typedef enum cudaError_enum {
  CUDA_SUCCESS = 0,
  CUDA_ERROR_NO_DEVICE = 100,
} CUresult;

typedef enum CUdevice_attribute_enum {
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
  CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
} CUdevice_attribute;

typedef uint32_t CUdevice;

CUresult (*cuInit)(unsigned int);
CUresult (*cuDeviceGetCount)(int *);
CUresult (*cuGetErrorString)(CUresult, const char **);
CUresult (*cuDeviceGet)(CUdevice *, int);
CUresult (*cuDeviceGetAttribute)(int *, CUdevice_attribute, CUdevice);

constexpr const char *DynamicCudaPath = "libcuda.so";

llvm::Error loadCUDA() {
  std::string ErrMsg;
  auto DynlibHandle = std::make_unique<llvm::sys::DynamicLibrary>(
      llvm::sys::DynamicLibrary::getPermanentLibrary(DynamicCudaPath, &ErrMsg));
  if (!DynlibHandle->isValid()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to 'dlopen' %s\n", DynamicCudaPath);
  }
#define DYNAMIC_INIT(SYMBOL)                                                   \
  {                                                                            \
    void *SymbolPtr = DynlibHandle->getAddressOfSymbol(#SYMBOL);               \
    if (!SymbolPtr)                                                            \
      return llvm::createStringError(llvm::inconvertibleErrorCode(),           \
                                     "Failed to 'dlsym' " #SYMBOL);            \
    SYMBOL = reinterpret_cast<decltype(SYMBOL)>(SymbolPtr);                    \
  }
  DYNAMIC_INIT(cuInit);
  DYNAMIC_INIT(cuDeviceGetCount);
  DYNAMIC_INIT(cuGetErrorString);
  DYNAMIC_INIT(cuDeviceGet);
  DYNAMIC_INIT(cuDeviceGetAttribute);
#undef DYNAMIC_INIT
  return llvm::Error::success();
}
#else

#include "cuda.h"
llvm::Error loadCUDA() { return llvm::Error::success(); }

#endif

static int handleError(CUresult Err) {
  const char *ErrStr = nullptr;
  CUresult Result = cuGetErrorString(Err, &ErrStr);
  if (Result != CUDA_SUCCESS)
    return 1;
  fprintf(stderr, "CUDA error: %s\n", ErrStr);
  return 1;
}

int main(int argc, char *argv[]) {
  // Attempt to load the NVPTX driver runtime.
  if (llvm::Error Err = loadCUDA()) {
    logAllUnhandledErrors(std::move(Err), llvm::errs());
    return 1;
  }

  if (CUresult Err = cuInit(0)) {
    if (Err == CUDA_ERROR_NO_DEVICE)
      return 0;
    else
      return handleError(Err);
  }

  int Count = 0;
  if (CUresult Err = cuDeviceGetCount(&Count))
    return handleError(Err);
  if (Count == 0)
    return 0;
  for (int DeviceId = 0; DeviceId < Count; ++DeviceId) {
    CUdevice Device;
    if (CUresult Err = cuDeviceGet(&Device, DeviceId))
      return handleError(Err);

    int32_t Major, Minor;
    if (CUresult Err = cuDeviceGetAttribute(
            &Major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, Device))
      return handleError(Err);
    if (CUresult Err = cuDeviceGetAttribute(
            &Minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, Device))
      return handleError(Err);

    printf("sm_%d%d\n", Major, Minor);
  }
  return 0;
}
