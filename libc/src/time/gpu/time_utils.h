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

namespace __llvm_libc {

#if defined(LIBC_TARGET_ARCH_IS_AMDGPU)
// AMDGPU does not have a single set frequency. Different architectures and
// cards can have vary values. Here we default to a few known values, but for
// complete support the frequency needs to be read from the kernel driver.
#if defined(__gfx1010__) || defined(__gfx1011__) || defined(__gfx1012__) ||    \
    defined(__gfx1013__) || defined(__gfx1030__) || defined(__gfx1031__) ||    \
    defined(__gfx1032__) || defined(__gfx1033__) || defined(__gfx1034__) ||    \
    defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1100__) ||    \
    defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) ||    \
    defined(__gfx1150__) || defined(__gfx1151__)
// These architectures use a 100 MHz fixed frequency clock.
constexpr uint64_t clock_freq = 100000000;
#elif defined(__gfx900__) || defined(__gfx902__) || defined(__gfx904__) ||     \
    defined(__gfx906__) || defined(__gfx908__) || defined(__gfx909__) ||       \
    defined(__gfx90a__) || defined(__gfx90c__) || defined(__gfx940__)
// These architectures use a 25 MHz fixed frequency clock expect for Vega 10
// which is actually 27 Mhz. We default to 25 MHz in all cases anyway.
constexpr uint64_t clock_freq = 25000000;
#else
// The frequency for these architecture is unknown. We simply default to zero.
constexpr uint64_t clock_freq = 0;
#endif

// We provide an externally visible symbol such that the runtime can set this to
// the correct value. If it is not set we try to default to the known values.
extern "C" [[gnu::visibility("protected")]] uint64_t
    [[clang::address_space(4)]] __llvm_libc_clock_freq;
#define GPU_CLOCKS_PER_SEC static_cast<clock_t>(__llvm_libc_clock_freq)

#elif defined(LIBC_TARGET_ARCH_IS_NVPTX)
// NPVTX uses a single 1 GHz fixed frequency clock for all target architectures.
#define GPU_CLOCKS_PER_SEC static_cast<clock_t>(1000000000UL)
#else
#error "Unsupported target"
#endif

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_TIME_GPU_TIME_UTILS_H
