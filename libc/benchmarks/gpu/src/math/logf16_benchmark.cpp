//===-- GPU benchmark for logf16 ------------------------------------------===//
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
#include "src/__support/sign.h"
#include "src/math/logf16.h"

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
  SINGLE_WAVE_BENCHMARK(LlvmLibcLogf16GpuBenchmark, Name##_1,                  \
                        RANDOM_INPUT_##Dist(T, Func, Min, Max, 1));            \
  SINGLE_WAVE_BENCHMARK(LlvmLibcLogf16GpuBenchmark, Name##_128,                \
                        RANDOM_INPUT_##Dist(T, Func, Min, Max, 128));          \
  SINGLE_WAVE_BENCHMARK(LlvmLibcLogf16GpuBenchmark, Name##_1024,               \
                        RANDOM_INPUT_##Dist(T, Func, Min, Max, 1024));         \
  SINGLE_WAVE_BENCHMARK(LlvmLibcLogf16GpuBenchmark, Name##_4096,               \
                        RANDOM_INPUT_##Dist(T, Func, Min, Max, 4096))

using LIBC_NAMESPACE::logf16;

static constexpr float16 INV_E = 0x1.78b56362cef38p-2f16; // exp(-1.0)
static constexpr float16 E = 0x1.5bf0a8b145769p+1f16;     // exp(+1.0)

BENCH(float16, Logf16Subnormal, logf16, UniformExponent, -14, -14);
BENCH(float16, Logf16AroundOne, logf16, UniformLinear, INV_E, E);
BENCH(float16, Logf16MedMag, logf16, UniformExponent, -10, 10);
BENCH(float16, Logf16Normal, logf16, UniformExponent, -13, 15);

#ifdef AMDGPU_MATH_FOUND
BENCH(float16, AmdLogf16Subnormal, __ocml_log_f16, UniformExponent, -14, -14);
BENCH(float16, AmdLogf16AroundOne, __ocml_log_f16, UniformLinear, INV_E, E);
BENCH(float16, AmdLogf16MedMag, __ocml_log_f16, UniformExponent, -10, 10);
BENCH(float16, AmdLogf16Normal, __ocml_log_f16, UniformExponent, -13, 15);
#endif
