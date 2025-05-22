//===-- Generic utilities for GPU timing ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TIME_GPU_TIME_UTILS_H
#define LLVM_LIBC_SRC_TIME_GPU_TIME_UTILS_H

#include "src/__support/GPU/utils.h"

namespace LIBC_NAMESPACE {

#if defined(LIBC_TARGET_ARCH_IS_AMDGPU)
// AMDGPU does not have a single set frequency. Different architectures and
// cards can have different values. The actualy frequency needs to be read from
// the kernel driver and will be between 25 MHz and 100 MHz on most cards. All
// cards following the GFX9 ISAs use a 100 MHz clock so we will default to that.
constexpr uint64_t clock_freq = 100000000UL;

// We provide an externally visible symbol such that the runtime can set
// this to the correct value.
extern "C" [[gnu::visibility("protected")]] uint64_t
    [[clang::address_space(4)]] __llvm_libc_clock_freq;
#define GPU_CLOCKS_PER_SEC static_cast<clock_t>(__llvm_libc_clock_freq)

#elif defined(LIBC_TARGET_ARCH_IS_NVPTX)
// NPVTX uses a single 1 GHz fixed frequency clock for all target architectures.
#define GPU_CLOCKS_PER_SEC static_cast<clock_t>(1000000000UL)
#else
#error "Unsupported target"
#endif

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_TIME_GPU_TIME_UTILS_H
