#include "benchmarks/gpu/LibcGpuBenchmark.h"

#include "src/ctype/isalnum.h"

uint64_t BM_IsAlnum() {
  char x = 'c';
  return LIBC_NAMESPACE::latency(LIBC_NAMESPACE::isalnum, x);
}
BENCHMARK(LlvmLibcIsAlNumGpuBenchmark, IsAlnumWrapper, BM_IsAlnum);

[[gnu::noinline]] static uint64_t single_input_function(int x) {
  asm volatile("" ::"r"(x)); // prevent the compiler from optimizing out x
  return x;
}

uint64_t BM_IsAlnumWithOverhead() {
  char x = 'c';
  return LIBC_NAMESPACE::latency(LIBC_NAMESPACE::isalnum, x) -
         LIBC_NAMESPACE::latency(single_input_function, 0);
}
BENCHMARK(LlvmLibcIsAlNumGpuBenchmark, IsAlnumWithOverhead,
          BM_IsAlnumWithOverhead);
