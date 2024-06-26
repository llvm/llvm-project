#include "benchmarks/gpu/LibcGpuBenchmark.h"

#include "src/ctype/isalnum.h"

uint64_t BM_IsAlnum() {
  char x = 'c';
  return LIBC_NAMESPACE::latency(LIBC_NAMESPACE::isalnum, x);
}
BENCHMARK(LlvmLibcIsAlNumGpuBenchmark, IsAlnumWrapper, BM_IsAlnum);
