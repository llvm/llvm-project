#include "benchmarks/gpu/LibcGpuBenchmark.h"

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/sin.h"
#include "src/stdlib/rand.h"
#include "src/stdlib/srand.h"

#ifdef NVPTX_MATH_FOUND
#include "src/math/nvptx/declarations.h"
#endif

// We want our values to be approximately
// |real value| <= 2^(max_exponent) * (1 + (random 52 bits) * 2^-52) <
// 2^(max_exponent + 1)
// The largest integer that can be stored in a double is 2^53
const int MAX_EXPONENT = 52;

double get_rand(int max_exponent) {
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<double>;
  uint64_t bits = LIBC_NAMESPACE::rand();
  double scale = 0.5 + max_exponent / 2048.0;
  FPBits fp(bits);
  fp.set_biased_exponent(
      static_cast<uint32_t>(fp.get_biased_exponent() * scale));
  return fp.get_val();
}

uint64_t BM_Sin() {
  LIBC_NAMESPACE::srand(LIBC_NAMESPACE::gpu::get_thread_id());
  double x = get_rand(MAX_EXPONENT);
  return LIBC_NAMESPACE::latency(LIBC_NAMESPACE::sin, x);
}
BENCHMARK(LlvmLibcSinGpuBenchmark, Sin, BM_Sin);
SINGLE_THREADED_BENCHMARK(LlvmLibcSinGpuBenchmark, SinSingleThread, BM_Sin);
SINGLE_WAVE_BENCHMARK(LlvmLibcSinGpuBenchmark, SinSingleWave, BM_Sin);

#ifdef NVPTX_MATH_FOUND
uint64_t BM_NvSin() {
  LIBC_NAMESPACE::srand(LIBC_NAMESPACE::gpu::get_thread_id());
  double x = get_rand(MAX_EXPONENT);
  return LIBC_NAMESPACE::latency(LIBC_NAMESPACE::__nv_sin, x);
}
BENCHMARK(LlvmLibcSinGpuBenchmark, NvSin, BM_NvSin);
SINGLE_THREADED_BENCHMARK(lvmLibcSinGpuBenchmark, NvSinSingleThread, BM_NvSin);
SINGLE_WAVE_BENCHMARK(LlvmLibcSinGpuBenchmark, NvSinSingleWave, BM_NvSin);

#endif
