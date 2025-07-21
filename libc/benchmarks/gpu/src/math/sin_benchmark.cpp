#include "benchmarks/gpu/LibcGpuBenchmark.h"

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/functional.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/sin.h"
#include "src/math/sinf.h"
#include "src/stdlib/rand.h"

#if defined(NVPTX_MATH_FOUND) || defined(AMDGPU_MATH_FOUND)
#include "platform.h"
#endif

// BENCHMARK() expects a function that with no parameters that returns a
// uint64_t representing the latency. Defining each benchmark using macro that
// expands to a lambda to allow us to switch the implementation of `sin()` to
// easily register NVPTX benchmarks.
#define BM_RANDOM_INPUT(T, Func, MIN_EXP, MAX_EXP, N)                          \
  []() {                                                                       \
    return LIBC_NAMESPACE::benchmarks::MathPerf<T>::run_throughput_in_range<   \
        N>(Func, MIN_EXP, MAX_EXP);                                            \
  }

#define BENCH(T, Name, Func, MIN_EXP, MAX_EXP)                                 \
  SINGLE_WAVE_BENCHMARK(LlvmLibcSinGpuBenchmark, Name##_1,                     \
                        BM_RANDOM_INPUT(T, Func, MIN_EXP, MAX_EXP, 1));        \
  SINGLE_WAVE_BENCHMARK(LlvmLibcSinGpuBenchmark, Name##_128,                   \
                        BM_RANDOM_INPUT(T, Func, MIN_EXP, MAX_EXP, 128));      \
  SINGLE_WAVE_BENCHMARK(LlvmLibcSinGpuBenchmark, Name##_1024,                  \
                        BM_RANDOM_INPUT(T, Func, MIN_EXP, MAX_EXP, 1024));     \
  SINGLE_WAVE_BENCHMARK(LlvmLibcSinGpuBenchmark, Name##_4096,                  \
                        BM_RANDOM_INPUT(T, Func, MIN_EXP, MAX_EXP, 4096))

BENCH(double, Sin, LIBC_NAMESPACE::sin, -1023, 1023);
BENCH(double, SinTwoPi, LIBC_NAMESPACE::sin, -10, 3);
BENCH(double, SinTwoPow30, LIBC_NAMESPACE::sin, 0, 30);
BENCH(double, SinVeryLarge, LIBC_NAMESPACE::sin, 30, 1000);

#ifdef NVPTX_MATH_FOUND
BENCH(double, NvSin, __nv_sin, -1023, 1023);
BENCH(double, NvSinTwoPi, __nv_sin, -10, 3);
BENCH(double, NvSinTwoPow30, __nv_sin, 0, 30);
BENCH(double, NvSinVeryLarge, __nv_sin, 30, 1000);
#endif

#ifdef AMDGPU_MATH_FOUND
BENCH(double, AmdSin, __ocml_sin_f64, -1023, 1023);
BENCH(double, AmdSinTwoPi, __ocml_sin_f64, -10, 3);
BENCH(double, AmdSinTwoPow30, __ocml_sin_f64, 0, 30);
BENCH(double, AmdSinVeryLarge, __ocml_sin_f64, 30, 1000);
#endif

BENCH(float, Sinf, LIBC_NAMESPACE::sinf, -127, 128);
BENCH(float, SinfTwoPi, LIBC_NAMESPACE::sinf, -10, 3);
BENCH(float, SinfTwoPow30, LIBC_NAMESPACE::sinf, 0, 30);
BENCH(float, SinfVeryLarge, LIBC_NAMESPACE::sinf, 30, 120);

#ifdef NVPTX_MATH_FOUND
BENCH(float, NvSinf, __nv_sinf, -127, 128);
BENCH(float, NvSinfTwoPi, __nv_sinf, -10, 3);
BENCH(float, NvSinfTwoPow30, __nv_sinf, 0, 30);
BENCH(float, NvSinfVeryLarge, __nv_sinf, 30, 120);
#endif

#ifdef AMDGPU_MATH_FOUND
BENCH(float, AmdSinf, __ocml_sin_f32, -127, 128);
BENCH(float, AmdSinfTwoPi, __ocml_sin_f32, -10, 3);
BENCH(float, AmdSinfTwoPow30, __ocml_sin_f32, 0, 30);
BENCH(float, AmdSinfVeryLarge, __ocml_sin_f32, 30, 120);
#endif
