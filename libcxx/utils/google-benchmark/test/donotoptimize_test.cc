#include "benchmark/benchmark.h"

#include <cstdint>

namespace {
#if defined(__GNUC__)
  std::uint64_t double_up(const std::uint64_t x) __attribute__ ((const));
#endif
  std::uint64_t double_up(const std::uint64_t x) {
    return x * 2;
  }
}

int main(int, char*[]) {

  // this test verifies compilation of DoNotOptimize() for some types

  char buffer8[8];
  benchmark::DoNotOptimize(buffer8);

  char buffer20[20];
  benchmark::DoNotOptimize(buffer20);

  char buffer1024[1024];
  benchmark::DoNotOptimize(buffer1024);
  benchmark::DoNotOptimize(&buffer1024[0]);

  int x = 123;
  benchmark::DoNotOptimize(x);
  benchmark::DoNotOptimize(&x);
  benchmark::DoNotOptimize(x += 42);

  benchmark::DoNotOptimize(double_up(x));

  return 0;
}
