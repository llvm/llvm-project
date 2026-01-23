//===-- GPU benchmark for expf16 ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "benchmarks/gpu/LibcGpuBenchmark.h"
#include "benchmarks/gpu/Random.h"

#include "hdr/stdint_proxy.h"
#include "src/__support/macros/properties/types.h"
#include "src/math/expf16.h"

#if defined(NVPTX_MATH_FOUND) || defined(AMDGPU_MATH_FOUND)
#include "platform.h"
#endif

#define RANDOM_INPUT(T, Func, Dist, Min, Max, N)                               \
  [](uint32_t call_index) {                                                    \
    using namespace LIBC_NAMESPACE::benchmarks;                                \
                                                                               \
    const Dist<T> dist(Min, Max);                                              \
    return MathPerf<T>::template run_throughput<N>(Func, dist, call_index);    \
  }

#define BENCH(T, Name, Func, Dist, Min, Max)                                   \
  SINGLE_WAVE_BENCHMARK(LlvmLibcExpf16GpuBenchmark, Name##_1,                  \
                        RANDOM_INPUT(T, Func, Dist, Min, Max, 1));             \
  SINGLE_WAVE_BENCHMARK(LlvmLibcExpf16GpuBenchmark, Name##_128,                \
                        RANDOM_INPUT(T, Func, Dist, Min, Max, 128));           \
  SINGLE_WAVE_BENCHMARK(LlvmLibcExpf16GpuBenchmark, Name##_1024,               \
                        RANDOM_INPUT(T, Func, Dist, Min, Max, 1024));          \
  SINGLE_WAVE_BENCHMARK(LlvmLibcExpf16GpuBenchmark, Name##_4096,               \
                        RANDOM_INPUT(T, Func, Dist, Min, Max, 4096))

using LIBC_NAMESPACE::expf16;

BENCH(float16, Expf16Subnormal, expf16, UniformExponent, -14, -14);
BENCH(float16, Expf16CoreRange, expf16, UniformLinear, -10.0f16, 10.0f16);
BENCH(float16, Expf16Finite, expf16, UniformLinear, -16.0f16, 11.0f16);
BENCH(float16, Expf16Underflow, expf16, UniformLinear, -17.0f16, -16.0f16);
BENCH(float16, Expf16Overflow, expf16, UniformLinear, 11.0f16, 12.0f16);

#ifdef AMDGPU_MATH_FOUND
BENCH(float16, AmdExpf16Subnormal, __ocml_exp_f16, UniformExponent, -14, -14);
BENCH(float16, AmdExpf16CoreRange, __ocml_exp_f16, UniformLinear, -10.0f16,
      10.0f16);
BENCH(float16, AmdExpf16Finite, __ocml_exp_f16, UniformLinear, -16.0f16,
      11.0f16);
BENCH(float16, AmdExpf16Underflow, __ocml_exp_f16, UniformLinear, -17.0f16,
      -16.0f16);
BENCH(float16, AmdExpf16Overflow, __ocml_exp_f16, UniformLinear, 11.0f16,
      12.0f16);
#endif
