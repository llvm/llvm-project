#include "LibcGpuBenchmark.h"

extern "C" int main(int argc, char **argv, char **envp) {
  LIBC_NAMESPACE::libc_gpu_benchmarks::Benchmark::run_benchmarks();
  return 0;
}
