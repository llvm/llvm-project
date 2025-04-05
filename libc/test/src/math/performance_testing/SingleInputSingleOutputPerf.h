//===-- Common utility class for differential analysis --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/algorithm.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/config.h"
#include "test/src/math/performance_testing/Timer.h"

#include <fstream>

namespace LIBC_NAMESPACE_DECL {
namespace testing {

template <typename T> class SingleInputSingleOutputPerf {
  using FPBits = fputil::FPBits<T>;
  using StorageType = typename FPBits::StorageType;
  static constexpr StorageType UIntMax =
      cpp::numeric_limits<StorageType>::max();

public:
  typedef T Func(T);

  static void runPerfInRange(Func FuncA, Func FuncB,
                             StorageType startingBit, StorageType endingBit,
                             size_t rounds, std::ofstream &log) {
    size_t n = 10'010'001;
    if (sizeof(StorageType) <= sizeof(size_t))
      n = cpp::min(n, static_cast<size_t>(endingBit - startingBit));

    auto runner = [=](Func func) {
      StorageType step = (endingBit - startingBit) / n;
      if (step == 0)
        step = 1;
      [[maybe_unused]] volatile T result;
      for (size_t i = 0; i < rounds; i++) {
        for (StorageType bits = startingBit; bits < endingBit; bits += step) {
          T x = FPBits(bits).get_val();
          result = func(x);
        }
      }
    };

    Timer timer;
    timer.start();
    runner(FuncA);
    timer.stop();

    double a_average = static_cast<double>(timer.nanoseconds()) / n / rounds;
    log << "-- Function A --\n";
    log << "     Total time      : " << timer.nanoseconds() << " ns \n";
    log << "     Average runtime : " << a_average << " ns/op \n";
    log << "     Ops per second  : "
        << static_cast<uint64_t>(1'000'000'000.0 / a_average) << " op/s \n";

    timer.start();
    runner(FuncB);
    timer.stop();

    double b_average = static_cast<double>(timer.nanoseconds()) / n / rounds;
    log << "-- Function B --\n";
    log << "     Total time      : " << timer.nanoseconds() << " ns \n";
    log << "     Average runtime : " << b_average << " ns/op \n";
    log << "     Ops per second  : "
        << static_cast<uint64_t>(1'000'000'000.0 / b_average) << " op/s \n";

    log << "-- Average ops per second ratio --\n";
    log << "     A / B  : " << b_average / a_average << " \n";
  }

  static void runPerf(Func FuncA, Func FuncB, size_t rounds, const char *name_a,
                      const char *name_b, const char *logFile) {
    std::ofstream log(logFile);
    log << "Function A - " << name_a << " Function B - " << name_b << "\n";
    log << " Performance tests with inputs in denormal range:\n";
    runPerfInRange(FuncA, FuncB, /* startingBit= */ StorageType(0),
                   /* endingBit= */ FPBits::max_subnormal().uintval(), rounds,
                   log);
    log << "\n Performance tests with inputs in normal range:\n";
    runPerfInRange(FuncA, FuncB,
                   /* startingBit= */ FPBits::min_normal().uintval(),
                   /* endingBit= */ FPBits::max_normal().uintval(), rounds,
                   log);
  }
};

} // namespace testing
} // namespace LIBC_NAMESPACE_DECL

#define SINGLE_INPUT_SINGLE_OUTPUT_PERF(T, FuncA, FuncB, filename)             \
  int main() {                                                                 \
    LIBC_NAMESPACE::testing::SingleInputSingleOutputPerf<T>::runPerf(          \
        &FuncA, &FuncB, 1, #FuncA, #FuncB, filename);                          \
    return 0;                                                                  \
  }

#define SINGLE_INPUT_SINGLE_OUTPUT_PERF_EX(T, FuncA, FuncB, rounds,            \
                                           filename)                           \
  {                                                                            \
    LIBC_NAMESPACE::testing::SingleInputSingleOutputPerf<T>::runPerf(          \
        &FuncA, &FuncB, rounds, #FuncA, #FuncB, filename);                     \
  }
