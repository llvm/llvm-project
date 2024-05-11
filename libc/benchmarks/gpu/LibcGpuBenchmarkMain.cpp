#include "LibcGpuBenchmark.h"

extern "C" int main(int argc, char **argv, char **envp) {
  LIBC_NAMESPACE::libc_gpu_benchmarks::Benchmark::runBenchmarks();
  return 0;
}
