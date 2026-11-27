#include "benchmarks/gpu/LibcGpuBenchmark.h"

#include "hdr/stdint_proxy.h"
#include "src/math/atan2.h"

#if defined(NVPTX_MATH_FOUND) || defined(AMDGPU_MATH_FOUND)
#include "platform.h"
#endif

#define BM_RANDOM_INPUTS(T, Func, MinExp, MaxExp, N)                           \
  [](uint32_t call_index) {                                                    \
    using namespace LIBC_NAMESPACE::benchmarks;                                \
                                                                               \
    const UniformExponent<T> dist(MinExp, MaxExp);                             \
    return MathPerf<T>::template run_throughput<N>(Func, dist, dist,           \
                                                   call_index);                \
  }

#define BENCH(T, Name, Func, MinExp, MaxExp)                                   \
  SINGLE_WAVE_BENCHMARK(LlvmLibcAtan2GpuBenchmark, Name##_1,                   \
                        BM_RANDOM_INPUTS(T, Func, MinExp, MaxExp, 1));         \
  SINGLE_WAVE_BENCHMARK(LlvmLibcAtan2GpuBenchmark, Name##_128,                 \
                        BM_RANDOM_INPUTS(T, Func, MinExp, MaxExp, 128));       \
  SINGLE_WAVE_BENCHMARK(LlvmLibcAtan2GpuBenchmark, Name##_1024,                \
                        BM_RANDOM_INPUTS(T, Func, MinExp, MaxExp, 1024));      \
  SINGLE_WAVE_BENCHMARK(LlvmLibcAtan2GpuBenchmark, Name##_4096,                \
                        BM_RANDOM_INPUTS(T, Func, MinExp, MaxExp, 4096))

BENCH(double, Atan2, LIBC_NAMESPACE::atan2, -1023, 1023);
BENCH(double, Atan2TwoPi, LIBC_NAMESPACE::atan2, -10, 3);
BENCH(double, Atan2TwoPow30, LIBC_NAMESPACE::atan2, 0, 30);
BENCH(double, Atan2Large, LIBC_NAMESPACE::atan2, 30, 1000);

#ifdef NVPTX_MATH_FOUND
BENCH(double, NvAtan2, __nv_atan2, -1023, 1023);
BENCH(double, NvAtan2TwoPi, __nv_atan2, -10, 3);
BENCH(double, NvAtan2TwoPow30, __nv_atan2, 0, 30);
BENCH(double, NvAtan2Large, __nv_atan2, 30, 1000);
#endif

#ifdef AMDGPU_MATH_FOUND
BENCH(double, AmdAtan2, __ocml_atan2_f64, -1023, 1023);
BENCH(double, AmdAtan2TwoPi, __ocml_atan2_f64, -10, 3);
BENCH(double, AmdAtan2TwoPow30, __ocml_atan2_f64, 0, 30);
BENCH(double, AmdAtan2Large, __ocml_atan2_f64, 30, 1000);
#endif
