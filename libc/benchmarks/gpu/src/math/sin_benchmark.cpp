#include "benchmarks/gpu/LibcGpuBenchmark.h"

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/functional.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/sin.h"
#include "src/stdlib/rand.h"

#ifdef NVPTX_MATH_FOUND
#include "src/math/nvptx/declarations.h"
#endif

#ifdef AMDGPU_MATH_FOUND
#include "src/math/amdgpu/declarations.h"
#endif

// BENCHMARK() expects a function that with no parameters that returns a
// uint64_t representing the latency. Defining each benchmark using macro that
// expands to a lambda to allow us to switch the implementation of `sin()` to
// easily register NVPTX benchmarks.
#define BM_RANDOM_INPUT(Func, MIN_EXP, MAX_EXP, N)                             \
  []() {                                                                       \
    return LIBC_NAMESPACE::benchmarks::MathPerf<                               \
        double>::run_throughput_in_range<N>(Func, MIN_EXP, MAX_EXP);           \
  }

#define BENCH(Name, Func, MIN_EXP, MAX_EXP)                                    \
  SINGLE_WAVE_BENCHMARK(LlvmLibcSinGpuBenchmark, Name##_1,                     \
                        BM_RANDOM_INPUT(Func, MIN_EXP, MAX_EXP, 1));           \
  SINGLE_WAVE_BENCHMARK(LlvmLibcSinGpuBenchmark, Name##_128,                   \
                        BM_RANDOM_INPUT(Func, MIN_EXP, MAX_EXP, 128));         \
  SINGLE_WAVE_BENCHMARK(LlvmLibcSinGpuBenchmark, Name##_1024,                  \
                        BM_RANDOM_INPUT(Func, MIN_EXP, MAX_EXP, 1024));        \
  SINGLE_WAVE_BENCHMARK(LlvmLibcSinGpuBenchmark, Name##_4096,                  \
                        BM_RANDOM_INPUT(Func, MIN_EXP, MAX_EXP, 4096))

BENCH(Sin, LIBC_NAMESPACE::sin, -1023, 1023);
BENCH(SinTwoPi, LIBC_NAMESPACE::sin, -10, 3);
BENCH(SinTwoPow30, LIBC_NAMESPACE::sin, 0, 30);
BENCH(SinVeryLarge, LIBC_NAMESPACE::sin, 30, 1000);

#ifdef NVPTX_MATH_FOUND
BENCH(NvSin, LIBC_NAMESPACE::__nv_sin, -1023, 1023);
BENCH(NvSinTwoPi, LIBC_NAMESPACE::__nv_sin, -10, 3);
BENCH(NvSinTwoPow30, LIBC_NAMESPACE::__nv_sin, 0, 30);
BENCH(NvSinVeryLarge, LIBC_NAMESPACE::__nv_sin, 30, 1000);
#endif

#ifdef AMDGPU_MATH_FOUND
BENCH(AmdgpuSin, LIBC_NAMESPACE::__ocml_sin_f64, -1023, 1023);
BENCH(AmdgpuSinTwoPi, LIBC_NAMESPACE::__ocml_sin_f64, -10, 3);
BENCH(AmdgpuSinTwoPow30, LIBC_NAMESPACE::__ocml_sin_f64, 0, 30);
BENCH(AmdgpuSinVeryLarge, LIBC_NAMESPACE::__ocml_sin_f64, 30, 1000);
#endif
