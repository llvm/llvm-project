//===------------- NVPTX implementation of timing utils ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_GPU_TIMING_NVPTX
#define LLVM_LIBC_UTILS_GPU_TIMING_NVPTX

#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/algorithm.h"
#include "src/__support/CPP/array.h"
#include "src/__support/CPP/atomic.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/GPU/utils.h"
#include "src/__support/common.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// Returns the overhead associated with calling the profiling region. This
// allows us to substract the constant-time overhead from the latency to
// obtain a true result. This can vary with system load.
[[gnu::noinline]] static uint64_t overhead() {
  volatile uint32_t x = 1;
  uint32_t y = x;
  uint64_t start = gpu::processor_clock();
  asm("" ::"llr"(start));
  uint32_t result = y;
  asm("or.b32 %[v_reg], %[v_reg], 0;" ::[v_reg] "r"(result));
  uint64_t stop = gpu::processor_clock();
  volatile auto storage = result;
  return stop - start;
}

// Stimulate a simple function and obtain its latency in clock cycles on the
// system. This function cannot be inlined or else it will disturb the very
// delicate balance of hard-coded dependencies.
template <typename F, typename T>
[[gnu::noinline]] static LIBC_INLINE uint64_t latency(F f, T t) {
  // We need to store the input somewhere to guarantee that the compiler will
  // not constant propagate it and remove the profiling region.
  volatile T storage = t;
  T arg = storage;

  // Get the current timestamp from the clock.
  cpp::atomic_thread_fence(cpp::MemoryOrder::ACQ_REL);
  uint64_t start = gpu::processor_clock();

  // This forces the compiler to load the input argument and run the clock cycle
  // counter before the profiling region.
  asm("" ::"llr"(start));

  // Run the function under test and return its value.
  auto result = f(arg);

  // This inline assembly performs a no-op which forces the result to both be
  // used and prevents us from exiting this region before it's complete.
  asm("or.b32 %[v_reg], %[v_reg], 0;" ::[v_reg] "r"(result));

  // Obtain the current timestamp after running the calculation and force
  // ordering.
  uint64_t stop = gpu::processor_clock();
  cpp::atomic_thread_fence(cpp::MemoryOrder::ACQ_REL);
  asm("" ::"r"(stop));
  volatile T output = result;

  // Return the time elapsed.
  return stop - start;
}

template <typename F, typename T1, typename T2>
static LIBC_INLINE uint64_t latency(F f, T1 t1, T2 t2) {
  volatile T1 storage = t1;
  volatile T2 storage2 = t2;
  T1 arg = storage;
  T2 arg2 = storage2;

  cpp::atomic_thread_fence(cpp::MemoryOrder::ACQ_REL);
  uint64_t start = gpu::processor_clock();

  asm("" ::"llr"(start));

  auto result = f(arg, arg2);

  asm("or.b32 %[v_reg], %[v_reg], 0;" ::[v_reg] "r"(result));

  uint64_t stop = gpu::processor_clock();
  cpp::atomic_thread_fence(cpp::MemoryOrder::ACQ_REL);
  asm("" ::"r"(stop));
  volatile auto output = result;

  return stop - start;
}

// Provides the *baseline* for throughput: measures loop and measurement costs
// without calling the f function
template <typename T, size_t N>
static LIBC_INLINE uint64_t
throughput_baseline(const cpp::array<T, N> &inputs) {
  asm("" ::"r"(&inputs));

  cpp::atomic_thread_fence(cpp::MemoryOrder::ACQ_REL);
  uint64_t start = gpu::processor_clock();
  asm("" ::"llr"(start));

  T result{};
  for (auto input : inputs) {
    asm("" ::"r"(input));
    result = input;
    asm("" ::"r"(result));
  }

  uint64_t stop = gpu::processor_clock();
  asm("" ::"r"(stop));
  cpp::atomic_thread_fence(cpp::MemoryOrder::ACQ_REL);

  volatile auto output = result;

  return stop - start;
}

// Provides throughput benchmarking
template <typename F, typename T, size_t N>
static LIBC_INLINE uint64_t throughput(F f, const cpp::array<T, N> &inputs) {
  uint64_t baseline = UINT64_MAX;
  for (int i = 0; i < 5; ++i)
    baseline = cpp::min(baseline, throughput_baseline<T, N>(inputs));

  asm("" ::"r"(&inputs));

  cpp::atomic_thread_fence(cpp::MemoryOrder::ACQ_REL);
  uint64_t start = gpu::processor_clock();
  asm("" ::"llr"(start));

  T result{};
  for (auto input : inputs) {
    asm("" ::"r"(input));
    result = f(input);
    asm("" ::"r"(result));
  }

  uint64_t stop = gpu::processor_clock();
  asm("" ::"r"(stop));
  cpp::atomic_thread_fence(cpp::MemoryOrder::ACQ_REL);

  volatile auto output = result;

  const uint64_t measured = stop - start;
  return measured > baseline ? (measured - baseline) : 0;
}

// Provides the *baseline* for throughput with 2 arguments: measures loop and
// measurement costs without calling the f function
template <typename T, size_t N>
static LIBC_INLINE uint64_t throughput_baseline(
    const cpp::array<T, N> &inputs1, const cpp::array<T, N> &inputs2) {
  asm("" ::"r"(&inputs1), "r"(&inputs2));

  cpp::atomic_thread_fence(cpp::MemoryOrder::ACQ_REL);
  uint64_t start = gpu::processor_clock();
  asm("" ::"llr"(start));

  T result{};
  for (size_t i = 0; i < N; i++) {
    T x = inputs1[i];
    T y = inputs2[i];
    asm("" ::"r"(x), "r"(y));
    result = x;
    asm("" ::"r"(result));
  }

  uint64_t stop = gpu::processor_clock();
  asm("" ::"r"(stop));
  cpp::atomic_thread_fence(cpp::MemoryOrder::ACQ_REL);

  volatile auto output = result;

  return stop - start;
}

// Provides throughput benchmarking for 2 arguments (e.g. atan2())
template <typename F, typename T, size_t N>
static LIBC_INLINE uint64_t throughput(F f, const cpp::array<T, N> &inputs1,
                                       const cpp::array<T, N> &inputs2) {
  uint64_t baseline = UINT64_MAX;
  for (int i = 0; i < 5; ++i)
    baseline = cpp::min(baseline, throughput_baseline<T, N>(inputs1, inputs2));

  asm("" ::"r"(&inputs1), "r"(&inputs2));

  cpp::atomic_thread_fence(cpp::MemoryOrder::ACQ_REL);
  uint64_t start = gpu::processor_clock();
  asm("" ::"llr"(start));

  T result{};
  for (size_t i = 0; i < N; i++) {
    T x = inputs1[i];
    T y = inputs2[i];
    asm("" ::"r"(x), "r"(y));
    result = f(x, y);
    asm("" ::"r"(result));
  }

  uint64_t stop = gpu::processor_clock();
  asm("" ::"r"(stop));
  cpp::atomic_thread_fence(cpp::MemoryOrder::ACQ_REL);

  volatile auto output = result;

  const uint64_t measured = stop - start;
  return measured > baseline ? (measured - baseline) : 0;
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_UTILS_GPU_TIMING_NVPTX
