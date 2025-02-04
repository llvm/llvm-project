#include "benchmarks/gpu/LibcGpuBenchmark.h"

#include "src/ctype/isalnum.h"

uint64_t BM_IsAlnum() {
  char x = 'c';
  return LIBC_NAMESPACE::latency(LIBC_NAMESPACE::isalnum, x);
}
BENCHMARK(LlvmLibcIsAlNumGpuBenchmark, IsAlnum, BM_IsAlnum);
SINGLE_THREADED_BENCHMARK(LlvmLibcIsAlNumGpuBenchmark, IsAlnumSingleThread,
                          BM_IsAlnum);
SINGLE_WAVE_BENCHMARK(LlvmLibcIsAlNumGpuBenchmark, IsAlnumSingleWave,
                      BM_IsAlnum);

uint64_t BM_IsAlnumCapital() {
  char x = 'A';
  return LIBC_NAMESPACE::latency(LIBC_NAMESPACE::isalnum, x);
}
BENCHMARK(LlvmLibcIsAlNumGpuBenchmark, IsAlnumCapital, BM_IsAlnumCapital);

uint64_t BM_IsAlnumNotAlnum() {
  char x = '{';
  return LIBC_NAMESPACE::latency(LIBC_NAMESPACE::isalnum, x);
}
BENCHMARK(LlvmLibcIsAlNumGpuBenchmark, IsAlnumNotAlnum, BM_IsAlnumNotAlnum);
