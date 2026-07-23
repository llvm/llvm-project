//===-- GPU benchmark for expf --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "benchmarks/gpu/LibcGpuBenchmark.h"
#include "benchmarks/gpu/Random.h"

#include "hdr/stdint_proxy.h"
#include "src/math/expf.h"

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
  SINGLE_WAVE_BENCHMARK(LlvmLibcExpfGpuBenchmark, Name##_1,                    \
                        RANDOM_INPUT(T, Func, Dist, Min, Max, 1));             \
  SINGLE_WAVE_BENCHMARK(LlvmLibcExpfGpuBenchmark, Name##_128,                  \
                        RANDOM_INPUT(T, Func, Dist, Min, Max, 128));           \
  SINGLE_WAVE_BENCHMARK(LlvmLibcExpfGpuBenchmark, Name##_1024,                 \
                        RANDOM_INPUT(T, Func, Dist, Min, Max, 1024));          \
  SINGLE_WAVE_BENCHMARK(LlvmLibcExpfGpuBenchmark, Name##_4096,                 \
                        RANDOM_INPUT(T, Func, Dist, Min, Max, 4096))

using LIBC_NAMESPACE::expf;

BENCH(float, ExpfSubnormal, expf, UniformExponent, -126, -126);
BENCH(float, ExpfCoreRange, expf, UniformLinear, -10.0f, 10.0f);
BENCH(float, ExpfFinite, expf, UniformLinear, -103.0f, 88.0f);
BENCH(float, ExpfUnderflow, expf, UniformLinear, -104.0f, -103.0f);
BENCH(float, ExpfOverflow, expf, UniformLinear, 88.0f, 89.0f);

#ifdef NVPTX_MATH_FOUND
BENCH(float, NvExpfSubnormal, __nv_expf, UniformExponent, -126, -126);
BENCH(float, NvExpfCoreRange, __nv_expf, UniformLinear, -10.0f, 10.0f);
BENCH(float, NvExpfFinite, __nv_expf, UniformLinear, -103.0f, 88.0f);
BENCH(float, NvExpfUnderflow, __nv_expf, UniformLinear, -104.0f, -103.0f);
BENCH(float, NvExpfOverflow, __nv_expf, UniformLinear, 88.0f, 89.0f);
#endif

#ifdef AMDGPU_MATH_FOUND
BENCH(float, AmdExpfSubnormal, __ocml_exp_f32, UniformExponent, -126, -126);
BENCH(float, AmdExpfCoreRange, __ocml_exp_f32, UniformLinear, -10.0f, 10.0f);
BENCH(float, AmdExpfFinite, __ocml_exp_f32, UniformLinear, -103.0f, 88.0f);
BENCH(float, AmdExpfUnderflow, __ocml_exp_f32, UniformLinear, -104.0f, -103.0f);
BENCH(float, AmdExpfOverflow, __ocml_exp_f32, UniformLinear, 88.0f, 89.0f);
#endif
