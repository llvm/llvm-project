#include "benchmarks/gpu/LibcGpuBenchmark.h"

#include "hdr/stdint_proxy.h"
#include "src/math/sin.h"
#include "src/math/sinf.h"

#if defined(NVPTX_MATH_FOUND) || defined(AMDGPU_MATH_FOUND)
#include "platform.h"
#endif

#define BM_RANDOM_INPUT(T, Func, MinExp, MaxExp, N)                            \
  [](uint32_t call_index) {                                                    \
    return LIBC_NAMESPACE::benchmarks::MathPerf<T>::run_throughput_in_range<   \
        N>(Func, MinExp, MaxExp, call_index);                                  \
  }

#define BENCH(T, Name, Func, MinExp, MaxExp)                                   \
  SINGLE_WAVE_BENCHMARK(LlvmLibcSinGpuBenchmark, Name##_1,                     \
                        BM_RANDOM_INPUT(T, Func, MinExp, MaxExp, 1));          \
  SINGLE_WAVE_BENCHMARK(LlvmLibcSinGpuBenchmark, Name##_128,                   \
                        BM_RANDOM_INPUT(T, Func, MinExp, MaxExp, 128));        \
  SINGLE_WAVE_BENCHMARK(LlvmLibcSinGpuBenchmark, Name##_1024,                  \
                        BM_RANDOM_INPUT(T, Func, MinExp, MaxExp, 1024));       \
  SINGLE_WAVE_BENCHMARK(LlvmLibcSinGpuBenchmark, Name##_4096,                  \
                        BM_RANDOM_INPUT(T, Func, MinExp, MaxExp, 4096))

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
