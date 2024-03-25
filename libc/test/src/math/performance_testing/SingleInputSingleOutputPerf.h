//===-- Common utility class for differential analysis --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "test/src/math/performance_testing/Timer.h"

#include <fstream>

namespace LIBC_NAMESPACE {
namespace testing {

template <typename T> class SingleInputSingleOutputPerf {
  using FPBits = fputil::FPBits<T>;
  using StorageType = typename FPBits::StorageType;
  static constexpr StorageType UIntMax =
      cpp::numeric_limits<StorageType>::max();

public:
  typedef T Func(T);

  static void runPerfInRange(Func myFunc, Func otherFunc,
                             StorageType startingBit, StorageType endingBit,
                             std::ofstream &log) {
    auto runner = [=](Func func) {
      constexpr StorageType N = 10'010'001;
      StorageType step = (endingBit - startingBit) / N;
      if (step == 0)
        step = 1;
      volatile T result;
      for (StorageType bits = startingBit; bits < endingBit; bits += step) {
        T x = FPBits(bits).get_val();
        result = func(x);
      }
    };

    Timer timer;
    timer.start();
    runner(myFunc);
    timer.stop();

    StorageType numberOfRuns = endingBit - startingBit + 1;
    double myAverage = static_cast<double>(timer.nanoseconds()) / numberOfRuns;
    log << "-- My function --\n";
    log << "     Total time      : " << timer.nanoseconds() << " ns \n";
    log << "     Average runtime : " << myAverage << " ns/op \n";
    log << "     Ops per second  : "
        << static_cast<uint64_t>(1'000'000'000.0 / myAverage) << " op/s \n";

    timer.start();
    runner(otherFunc);
    timer.stop();

    double otherAverage =
        static_cast<double>(timer.nanoseconds()) / numberOfRuns;
    log << "-- Other function --\n";
    log << "     Total time      : " << timer.nanoseconds() << " ns \n";
    log << "     Average runtime : " << otherAverage << " ns/op \n";
    log << "     Ops per second  : "
        << static_cast<uint64_t>(1'000'000'000.0 / otherAverage) << " op/s \n";

    log << "-- Average runtime ratio --\n";
    log << "     Mine / Other's  : " << myAverage / otherAverage << " \n";
  }

  static void runPerf(Func myFunc, Func otherFunc, const char *logFile) {
    std::ofstream log(logFile);
    log << " Performance tests with inputs in denormal range:\n";
    runPerfInRange(myFunc, otherFunc, /* startingBit= */ StorageType(0),
                   /* endingBit= */ FPBits::max_subnormal().uintval(), log);
    log << "\n Performance tests with inputs in normal range:\n";
    runPerfInRange(myFunc, otherFunc,
                   /* startingBit= */ FPBits::min_normal().uintval(),
                   /* endingBit= */ FPBits::max_normal().uintval(), log);
  }
};

} // namespace testing
} // namespace LIBC_NAMESPACE

#define SINGLE_INPUT_SINGLE_OUTPUT_PERF(T, myFunc, otherFunc, filename)        \
  int main() {                                                                 \
    LIBC_NAMESPACE::testing::SingleInputSingleOutputPerf<T>::runPerf(          \
        &myFunc, &otherFunc, filename);                                        \
    return 0;                                                                  \
  }
