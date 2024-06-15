//===------------- AMDGPU implementation of timing utils --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_GPU_TIMING_AMDGPU
#define LLVM_LIBC_UTILS_GPU_TIMING_AMDGPU

#include "src/__support/GPU/utils.h"
#include "src/__support/common.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

#include <stdint.h>

namespace LIBC_NAMESPACE {

// Returns the overhead associated with calling the profiling region. This
// allows us to substract the constant-time overhead from the latency to
// obtain a true result. This can vary with system load.
[[gnu::noinline]] static LIBC_INLINE uint64_t overhead() {
  gpu::memory_fence();
  uint64_t start = gpu::processor_clock();
  uint32_t result = 0.0;
  asm("v_or_b32 %[v_reg], 0, %[v_reg]\n" ::[v_reg] "v"(result) :);
  asm("" ::"s"(start));
  uint64_t stop = gpu::processor_clock();
  return stop - start;
}

// Profile a simple function and obtain its latency in clock cycles on the
// system. This function cannot be inlined or else it will disturb the very
// delicate balance of hard-coded dependencies.
template <typename F, typename T>
[[gnu::noinline]] static LIBC_INLINE uint64_t latency(F f, T t) {
  // We need to store the input somewhere to guarantee that the compiler will
  // not constant propagate it and remove the profiling region.
  volatile uint32_t storage = t;
  float arg = storage;
  asm("" ::"s"(arg));

  // The AMDGPU architecture needs to wait on pending results.
  gpu::memory_fence();
  // Get the current timestamp from the clock.
  uint64_t start = gpu::processor_clock();

  // This forces the compiler to load the input argument and run the clock cycle
  // counter before the profiling region.
  asm("" ::"s"(arg), "s"(start));

  // Run the function under test and return its value.
  auto result = f(arg);

  // This inline assembly performs a no-op which forces the result to both be
  // used and prevents us from exiting this region before it's complete.
  asm("v_or_b32 %[v_reg], 0, %[v_reg]\n" ::[v_reg] "v"(result) :);

  // Obtain the current timestamp after running the calculation and force
  // ordering.
  uint64_t stop = gpu::processor_clock();
  asm("" ::"s"(stop));
  gpu::memory_fence();

  // Return the time elapsed.
  return stop - start;
}

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_UTILS_GPU_TIMING_AMDGPU
