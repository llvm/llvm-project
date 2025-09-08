//===-- GPU benchmark for logf --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "benchmarks/gpu/LibcGpuBenchmark.h"
#include "benchmarks/gpu/Random.h"

#include "hdr/stdint_proxy.h"
#include "src/__support/sign.h"
#include "src/math/logf.h"

#if defined(NVPTX_MATH_FOUND) || defined(AMDGPU_MATH_FOUND)
#include "platform.h"
#endif

#define RANDOM_INPUT_UniformExponent(T, Func, Min, Max, N)                     \
  [](uint32_t call_index) {                                                    \
    using namespace LIBC_NAMESPACE::benchmarks;                                \
                                                                               \
    const UniformExponent<T> dist(Min, Max, LIBC_NAMESPACE::Sign::POS);        \
    return MathPerf<T>::template run_throughput<N>(Func, dist, call_index);    \
  }

#define RANDOM_INPUT_UniformLinear(T, Func, Min, Max, N)                       \
  [](uint32_t call_index) {                                                    \
    using namespace LIBC_NAMESPACE::benchmarks;                                \
                                                                               \
    const UniformLinear<T> dist(Min, Max);                                     \
    return MathPerf<T>::template run_throughput<N>(Func, dist, call_index);    \
  }

#define BENCH(T, Name, Func, Dist, Min, Max)                                   \
  SINGLE_WAVE_BENCHMARK(LlvmLibcLogfGpuBenchmark, Name##_1,                    \
                        RANDOM_INPUT_##Dist(T, Func, Min, Max, 1));            \
  SINGLE_WAVE_BENCHMARK(LlvmLibcLogfGpuBenchmark, Name##_128,                  \
                        RANDOM_INPUT_##Dist(T, Func, Min, Max, 128));          \
  SINGLE_WAVE_BENCHMARK(LlvmLibcLogfGpuBenchmark, Name##_1024,                 \
                        RANDOM_INPUT_##Dist(T, Func, Min, Max, 1024));         \
  SINGLE_WAVE_BENCHMARK(LlvmLibcLogfGpuBenchmark, Name##_4096,                 \
                        RANDOM_INPUT_##Dist(T, Func, Min, Max, 4096))

using LIBC_NAMESPACE::logf;

static constexpr float INV_E = 0x1.78b56362cef38p-2f; // exp(-1.0)
static constexpr float E = 0x1.5bf0a8b145769p+1f;     // exp(+1.0)

BENCH(float, LogfSubnormal, logf, UniformExponent, -126, -126);
BENCH(float, LogfAroundOne, logf, UniformLinear, INV_E, E);
BENCH(float, LogfMedMag, logf, UniformExponent, -10, 10);
BENCH(float, LogfNormal, logf, UniformExponent, -125, 127);

#ifdef NVPTX_MATH_FOUND
BENCH(float, NvLogfSubnormal, __nv_logf, UniformExponent, -126, -126);
BENCH(float, NvLogfAroundOne, __nv_logf, UniformLinear, INV_E, E);
BENCH(float, NvLogfMedMag, __nv_logf, UniformExponent, -10, 10);
BENCH(float, NvLogfNormal, __nv_logf, UniformExponent, -125, 127);
#endif

#ifdef AMDGPU_MATH_FOUND
BENCH(float, AmdLogfSubnormal, __ocml_log_f32, UniformExponent, -126, -126);
BENCH(float, AmdLogfAroundOne, __ocml_log_f32, UniformLinear, INV_E, E);
BENCH(float, AmdLogfMedMag, __ocml_log_f32, UniformExponent, -10, 10);
BENCH(float, AmdLogfNormal, __ocml_log_f32, UniformExponent, -125, 127);
#endif
