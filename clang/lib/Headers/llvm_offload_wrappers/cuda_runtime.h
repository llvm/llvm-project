/*===- __cuda_runtime.h - LLVM/Offload wrappers for CUDA runtime API -------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __CUDA_RUNTIME_API__
#define __CUDA_RUNTIME_API__

#include <cstddef>
#include <cstdint>
#include <optional>

extern "C" {
int omp_get_initial_device(void);
void omp_target_free(void *Ptr, int Device);
void *omp_target_alloc(size_t Size, int Device);
int omp_target_memcpy(void *Dst, const void *Src, size_t Length,
                      size_t DstOffset, size_t SrcOffset, int DstDevice,
                      int SrcDevice);
void *omp_target_memset(void *Ptr, int C, size_t N, int DeviceNum);
int __tgt_target_synchronize_async_info_queue(void *Loc, int64_t DeviceNum,
                                              void *AsyncInfoQueue);
}

// TODO: There are many fields missing in this enumeration.
typedef enum cudaError {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorNoDevice = 100,
  cudaErrorInvalidDevice = 101,
  cudaErrorOTHER = -1,
} cudaError_t;

enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4
};

typedef void *cudaStream_t;

static thread_local cudaError_t __cudaomp_last_error = cudaSuccess;

// Returns the last error that has been produced and resets it to cudaSuccess.
inline cudaError_t cudaGetLastError() {
  cudaError_t TempError = __cudaomp_last_error;
  __cudaomp_last_error = cudaSuccess;
  return TempError;
}

// Returns the last error that has been produced without reseting it.
inline cudaError_t cudaPeekAtLastError() { return __cudaomp_last_error; }

inline cudaError_t cudaDeviceSynchronize() {
  int DeviceNum = 0;
  return __cudaomp_last_error =
             (cudaError_t)__tgt_target_synchronize_async_info_queue(
                 /*Loc=*/nullptr, DeviceNum, /*AsyncInfoQueue=*/nullptr);
}

inline cudaError_t __cudaMalloc(void **devPtr, size_t size) {
  int DeviceNum = 0;
  *devPtr = omp_target_alloc(size, DeviceNum);
  if (*devPtr == NULL)
    return __cudaomp_last_error = cudaErrorMemoryAllocation;

  return __cudaomp_last_error = cudaSuccess;
}

template <class T> cudaError_t cudaMalloc(T **devPtr, size_t size) {
  return __cudaMalloc((void **)devPtr, size);
}

inline cudaError_t __cudaFree(void *devPtr) {
  int DeviceNum = 0;
  omp_target_free(devPtr, DeviceNum);
  return __cudaomp_last_error = cudaSuccess;
}

template <class T> inline cudaError_t cudaFree(T *ptr) {
  return __cudaFree((void *)ptr);
}

inline cudaError_t __cudaMemcpy(void *dst, const void *src, size_t count,
                                cudaMemcpyKind kind) {
  // get the host device number (which is the inital device)
  int HostDeviceNum = omp_get_initial_device();

  // use the default device for gpu
  int GPUDeviceNum = 0;

  // default to copy from host to device
  int DstDeviceNum = GPUDeviceNum;
  int SrcDeviceNum = HostDeviceNum;

  if (kind == cudaMemcpyDeviceToHost)
    std::swap(DstDeviceNum, SrcDeviceNum);

  // omp_target_memcpy returns 0 on success and non-zero on failure
  if (omp_target_memcpy(dst, src, count, 0, 0, DstDeviceNum, SrcDeviceNum))
    return __cudaomp_last_error = cudaErrorInvalidValue;
  return __cudaomp_last_error = cudaSuccess;
}

template <class T>
inline cudaError_t cudaMemcpy(T *dst, const T *src, size_t count,
                              cudaMemcpyKind kind) {
  return __cudaMemcpy((void *)dst, (const void *)src, count, kind);
}

inline cudaError_t __cudaMemset(void *devPtr, int value, size_t count,
                                cudaStream_t stream = 0) {
  int DeviceNum = 0;
  if (!omp_target_memset(devPtr, value, count, DeviceNum))
    return __cudaomp_last_error = cudaErrorInvalidValue;
  return __cudaomp_last_error = cudaSuccess;
}

template <class T>
inline cudaError_t cudaMemset(T *devPtr, int value, size_t count) {
  return __cudaMemset((void *)devPtr, value, count);
}

inline cudaError_t cudaDeviceReset(void) {
  cudaDeviceSynchronize();
  // TODO: not implemented.
  return __cudaomp_last_error = cudaSuccess;
}

#endif
