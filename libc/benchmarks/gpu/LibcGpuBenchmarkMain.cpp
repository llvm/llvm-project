#include "LibcGpuBenchmark.h"

extern "C" int main(int argc, char **argv, char **envp) {
  LIBC_NAMESPACE::benchmarks::Benchmark::run_benchmarks();
  return 0;
}
