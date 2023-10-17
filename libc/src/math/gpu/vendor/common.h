//===-- Common interface for compiling the GPU math -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_GPU_COMMON_H
#define LLVM_LIBC_SRC_MATH_GPU_COMMON_H

#include "src/__support/macros/properties/architectures.h"

#if defined(LIBC_TARGET_ARCH_IS_AMDGPU)
#include "amdgpu/amdgpu.h"
#elif defined(LIBC_TARGET_ARCH_IS_NVPTX)
#include "nvptx/nvptx.h"
#else
#error "Unsupported platform"
#endif

#endif // LLVM_LIBC_SRC_MATH_GPU_COMMON_H
