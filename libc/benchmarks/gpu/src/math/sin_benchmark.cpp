#include "benchmarks/gpu/LibcGpuBenchmark.h"

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/functional.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/sin.h"
#include "src/stdlib/rand.h"
#include "src/stdlib/srand.h"

#ifdef NVPTX_MATH_FOUND
#include "src/math/nvptx/declarations.h"
#endif

constexpr double M_PI = 3.14159265358979323846;
uint64_t get_bits(double x) {
  return LIBC_NAMESPACE::cpp::bit_cast<uint64_t>(x);
}

// BENCHMARK() expects a function that with no parameters that returns a
// uint64_t representing the latency. Defining each benchmark using macro that
// expands to a lambda to allow us to switch the implementation of `sin()` to
// easily register NVPTX benchmarks.
#define BM_RANDOM_INPUT(Func)                                                  \
  []() {                                                                       \
    LIBC_NAMESPACE::srand(LIBC_NAMESPACE::gpu::processor_clock());             \
    double x = LIBC_NAMESPACE::benchmarks::get_rand_input<double>();           \
    return LIBC_NAMESPACE::latency(Func, x);                                   \
  }
BENCHMARK(LlvmLibcSinGpuBenchmark, Sin, BM_RANDOM_INPUT(LIBC_NAMESPACE::sin));

#define BM_TWO_PI(Func)                                                        \
  []() {                                                                       \
    return LIBC_NAMESPACE::benchmarks::MathPerf<double>::run_perf_in_range(    \
        Func, 0, get_bits(2 * M_PI), get_bits(M_PI / 64));                     \
  }
BENCHMARK(LlvmLibcSinGpuBenchmark, SinTwoPi, BM_TWO_PI(LIBC_NAMESPACE::sin));

#define BM_LARGE_INT(Func)                                                     \
  []() {                                                                       \
    return LIBC_NAMESPACE::benchmarks::MathPerf<double>::run_perf_in_range(    \
        Func, 0, get_bits(1 << 30), get_bits(1 << 4));                         \
  }
BENCHMARK(LlvmLibcSinGpuBenchmark, SinLargeInt,
          BM_LARGE_INT(LIBC_NAMESPACE::sin));

#ifdef NVPTX_MATH_FOUND
BENCHMARK(LlvmLibcSinGpuBenchmark, NvSin,
          BM_RANDOM_INPUT(LIBC_NAMESPACE::__nv_sin));
BENCHMARK(LlvmLibcSinGpuBenchmark, NvSinTwoPi,
          BM_TWO_PI(LIBC_NAMESPACE::__nv_sin));
BENCHMARK(LlvmLibcSinGpuBenchmark, NvSinLargeInt,
          BM_LARGE_INT(LIBC_NAMESPACE::__nv_sin));
#endif
