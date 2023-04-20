//===---------------- Implementation of GPU utils ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_GPU_UTIL_H
#define LLVM_LIBC_SRC_SUPPORT_GPU_UTIL_H

#include "src/__support/macros/properties/architectures.h"

#if defined(LIBC_TARGET_ARCH_IS_AMDGPU)
#include "amdgpu/utils.h"
#elif defined(LIBC_TARGET_ARCH_IS_NVPTX)
#include "nvptx/utils.h"
#else
#include "generic/utils.h"
#endif

#endif // LLVM_LIBC_SRC_SUPPORT_OSUTIL_IO_H
