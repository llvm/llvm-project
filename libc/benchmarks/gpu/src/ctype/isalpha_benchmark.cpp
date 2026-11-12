#include "benchmarks/gpu/LibcGpuBenchmark.h"

#include "src/ctype/isalpha.h"

uint64_t BM_IsAlpha() {
  char x = 'c';
  return LIBC_NAMESPACE::latency(LIBC_NAMESPACE::isalpha, x);
}
BENCHMARK(LlvmLibcIsAlphaGpuBenchmark, IsAlpha, BM_IsAlpha);
