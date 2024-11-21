#include "benchmarks/gpu/LibcGpuBenchmark.h"

#include "src/math/atan2.h"
#include "src/stdlib/rand.h"

#ifdef NVPTX_MATH_FOUND
#include "src/math/nvptx/declarations.h"
#endif

#ifdef AMDGPU_MATH_FOUND
#include "src/math/amdgpu/declarations.h"
#endif

#define BM_TWO_RANDOM_INPUT(T, Func, MIN_EXP, MAX_EXP, N)                      \
  []() {                                                                       \
    return LIBC_NAMESPACE::benchmarks::MathPerf<T>::run_throughput_in_range<   \
        N>(Func, MIN_EXP, MAX_EXP, MIN_EXP, MAX_EXP);                          \
  }

#define BENCH(T, Name, Func, MIN_EXP, MAX_EXP)                                 \
  SINGLE_WAVE_BENCHMARK(LlvmLibcAtan2GpuBenchmark, Name##_1,                   \
                        BM_TWO_RANDOM_INPUT(T, Func, MIN_EXP, MAX_EXP, 1));    \
  SINGLE_WAVE_BENCHMARK(LlvmLibcAtan2GpuBenchmark, Name##_128,                 \
                        BM_TWO_RANDOM_INPUT(T, Func, MIN_EXP, MAX_EXP, 128));  \
  SINGLE_WAVE_BENCHMARK(LlvmLibcAtan2GpuBenchmark, Name##_1024,                \
                        BM_TWO_RANDOM_INPUT(T, Func, MIN_EXP, MAX_EXP, 1024)); \
  SINGLE_WAVE_BENCHMARK(LlvmLibcAtan2GpuBenchmark, Name##_4096,                \
                        BM_TWO_RANDOM_INPUT(T, Func, MIN_EXP, MAX_EXP, 4096))

BENCH(double, Atan2, LIBC_NAMESPACE::atan2, -1023, 1023);
BENCH(double, Atan2TwoPi, LIBC_NAMESPACE::atan2, -10, 3);
BENCH(double, Atan2TwoPow30, LIBC_NAMESPACE::atan2, 0, 30);
BENCH(double, Atan2Large, LIBC_NAMESPACE::atan2, 30, 1000);

#ifdef NVPTX_MATH_FOUND
BENCH(double, NvAtan2, LIBC_NAMESPACE::__nv_atan2, -1023, 1023);
BENCH(double, NvAtan2TwoPi, LIBC_NAMESPACE::__nv_atan2, -10, 3);
BENCH(double, NvAtan2TwoPow30, LIBC_NAMESPACE::__nv_atan2, 0, 30);
BENCH(double, NvAtan2Large, LIBC_NAMESPACE::__nv_atan2, 30, 1000);
#endif

#ifdef AMDGPU_MATH_FOUND
BENCH(double, AmdAtan2, LIBC_NAMESPACE::__ocml_atan2_f64, -1023, 1023);
BENCH(double, AmdAtan2TwoPi, LIBC_NAMESPACE::__ocml_atan2_f64, -10, 3);
BENCH(double, AmdAtan2TwoPow30, LIBC_NAMESPACE::__ocml_atan2_f64, 0, 30);
BENCH(double, AmdAtan2Large, LIBC_NAMESPACE::__ocml_atan2_f64, 30, 1000);
#endif
