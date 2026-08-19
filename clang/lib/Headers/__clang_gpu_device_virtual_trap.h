//===---- __clang_gpu_device_virtual_trap.h - Virtual Trap Functions --------===
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file provides device-side definitions of __cxa_pure_virtual() and
//  __cxa_deleted_virtual().
//
//===----------------------------------------------------------------------===//

#ifndef __CLANG_GPU_DEVICE_VIRTUAL_TRAP_H__
#define __CLANG_GPU_DEVICE_VIRTUAL_TRAP_H__

#if defined(__CUDA__) || defined(__HIP__)

#ifdef __cplusplus
extern "C" {
__attribute__((__visibility__("default")))
__attribute__((weak))
__attribute__((noreturn))
__device__ void __cxa_pure_virtual(void) {
  __builtin_trap();
}

__attribute__((__visibility__("default")))
__attribute__((weak))
__attribute__((noreturn))
__device__ void __cxa_deleted_virtual(void) {
  __builtin_trap();
}
} // extern "C"
#endif //__cplusplus

#endif // defined(__HIP__) || defined(__CUDA__)

#endif // __CLANG_GPU_DEVICE_VIRTUAL_TRAP_H__
