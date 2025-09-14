//===-- GPU benchmark for log ---------------------------------------------===//
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
#include "src/math/log.h"

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
  SINGLE_WAVE_BENCHMARK(LlvmLibcLogGpuBenchmark, Name##_1,                     \
                        RANDOM_INPUT_##Dist(T, Func, Min, Max, 1));            \
  SINGLE_WAVE_BENCHMARK(LlvmLibcLogGpuBenchmark, Name##_128,                   \
                        RANDOM_INPUT_##Dist(T, Func, Min, Max, 128));          \
  SINGLE_WAVE_BENCHMARK(LlvmLibcLogGpuBenchmark, Name##_1024,                  \
                        RANDOM_INPUT_##Dist(T, Func, Min, Max, 1024));         \
  SINGLE_WAVE_BENCHMARK(LlvmLibcLogGpuBenchmark, Name##_4096,                  \
                        RANDOM_INPUT_##Dist(T, Func, Min, Max, 4096))

using LIBC_NAMESPACE::log;

static constexpr double INV_E = 0x1.78b56362cef38p-2; // exp(-1.0)
static constexpr double E = 0x1.5bf0a8b145769p+1;     // exp(+1.0)

BENCH(double, LogSubnormal, log, UniformExponent, -1022, -1022);
BENCH(double, LogAroundOne, log, UniformLinear, INV_E, E);
BENCH(double, LogMedMag, log, UniformExponent, -10, 10);
BENCH(double, LogNormal, log, UniformExponent, -1021, 1023);

#ifdef NVPTX_MATH_FOUND
BENCH(double, NvLogSubnormal, __nv_log, UniformExponent, -1022, -1022);
BENCH(double, NvLogAroundOne, __nv_log, UniformLinear, INV_E, E);
BENCH(double, NvLogMedMag, __nv_log, UniformExponent, -10, 10);
BENCH(double, NvLogNormal, __nv_log, UniformExponent, -1021, 1023);
#endif

#ifdef AMDGPU_MATH_FOUND
BENCH(double, AmdLogSubnormal, __ocml_log_f64, UniformExponent, -1022, -1022);
BENCH(double, AmdLogAroundOne, __ocml_log_f64, UniformLinear, INV_E, E);
BENCH(double, AmdLogMedMag, __ocml_log_f64, UniformExponent, -10, 10);
BENCH(double, AmdLogNormal, __ocml_log_f64, UniformExponent, -1021, 1023);
#endif
