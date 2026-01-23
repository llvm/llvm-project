//===-- GPU benchmark for exp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "benchmarks/gpu/LibcGpuBenchmark.h"
#include "benchmarks/gpu/Random.h"

#include "hdr/stdint_proxy.h"
#include "src/math/exp.h"

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
  SINGLE_WAVE_BENCHMARK(LlvmLibcExpGpuBenchmark, Name##_1,                     \
                        RANDOM_INPUT(T, Func, Dist, Min, Max, 1));             \
  SINGLE_WAVE_BENCHMARK(LlvmLibcExpGpuBenchmark, Name##_128,                   \
                        RANDOM_INPUT(T, Func, Dist, Min, Max, 128));           \
  SINGLE_WAVE_BENCHMARK(LlvmLibcExpGpuBenchmark, Name##_1024,                  \
                        RANDOM_INPUT(T, Func, Dist, Min, Max, 1024));          \
  SINGLE_WAVE_BENCHMARK(LlvmLibcExpGpuBenchmark, Name##_4096,                  \
                        RANDOM_INPUT(T, Func, Dist, Min, Max, 4096))

using LIBC_NAMESPACE::exp;

BENCH(double, ExpSubnormal, exp, UniformExponent, -1022, -1022);
BENCH(double, ExpCoreRange, exp, UniformLinear, -10.0, 10.0);
BENCH(double, ExpFinite, exp, UniformLinear, -745.0, 709.0);
BENCH(double, ExpUnderflow, exp, UniformLinear, -746.0, -745.0);
BENCH(double, ExpOverflow, exp, UniformLinear, 709.0, 710.0);

#ifdef NVPTX_MATH_FOUND
BENCH(double, NvExpSubnormal, __nv_exp, UniformExponent, -1022, -1022);
BENCH(double, NvExpCoreRange, __nv_exp, UniformLinear, -10.0, 10.0);
BENCH(double, NvExpFinite, __nv_exp, UniformLinear, -745.0, 709.0);
BENCH(double, NvExpUnderflow, __nv_exp, UniformLinear, -746.0, -745.0);
BENCH(double, NvExpOverflow, __nv_exp, UniformLinear, 709.0, 710.0);
#endif

#ifdef AMDGPU_MATH_FOUND
BENCH(double, AmdExpSubnormal, __ocml_exp_f64, UniformExponent, -1022, -1022);
BENCH(double, AmdExpCoreRange, __ocml_exp_f64, UniformLinear, -10.0, 10.0);
BENCH(double, AmdExpFinite, __ocml_exp_f64, UniformLinear, -745.0, 709.0);
BENCH(double, AmdExpUnderflow, __ocml_exp_f64, UniformLinear, -746.0, -745.0);
BENCH(double, AmdExpOverflow, __ocml_exp_f64, UniformLinear, 709.0, 710.0);
#endif
