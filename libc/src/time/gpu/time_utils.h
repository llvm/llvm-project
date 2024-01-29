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
// cards can have vary values. Here we default to a few known values, but for
// complete support the frequency needs to be read from the kernel driver.
#if defined(__GFX10__) || defined(__GFX11__) || defined(__GFX12__) ||          \
    defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
// These architectures use a 100 MHz fixed frequency clock.
constexpr uint64_t clock_freq = 100000000;
#elif defined(__GFX9__)
// These architectures use a 25 MHz fixed frequency clock expect for Vega 10
// which is actually 27 Mhz. We default to 25 MHz in all cases anyway.
constexpr uint64_t clock_freq = 25000000;
#else
// The frequency for these architecture is unknown. We simply default to zero.
constexpr uint64_t clock_freq = 0;
#endif

// We provide an externally visible symbol such that the runtime can set
// this to the correct value. If it is not set we try to default to the
// known values.
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
