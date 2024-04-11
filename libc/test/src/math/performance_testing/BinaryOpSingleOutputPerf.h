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

template <typename T> class BinaryOpSingleOutputPerf {
  using FPBits = fputil::FPBits<T>;
  using StorageType = typename FPBits::StorageType;
  static constexpr StorageType UIntMax =
      cpp::numeric_limits<StorageType>::max();

public:
  typedef T Func(T, T);

  static void run_perf_in_range(Func myFunc, Func otherFunc,
                                StorageType startingBit, StorageType endingBit,
                                StorageType N, std::ofstream &log) {
    auto runner = [=](Func func) {
      volatile T result;
      if (endingBit < startingBit) {
        return;
      }

      StorageType step = (endingBit - startingBit) / N;
      for (StorageType bitsX = startingBit, bitsY = endingBit;;
           bitsX += step, bitsY -= step) {
        T x = FPBits(bitsX).get_val();
        T y = FPBits(bitsY).get_val();
        result = func(x, y);
        if (endingBit - bitsX < step) {
          break;
        }
      }
    };

    Timer timer;
    timer.start();
    runner(myFunc);
    timer.stop();

    double my_average = static_cast<double>(timer.nanoseconds()) / N;
    log << "-- My function --\n";
    log << "     Total time      : " << timer.nanoseconds() << " ns \n";
    log << "     Average runtime : " << my_average << " ns/op \n";
    log << "     Ops per second  : "
        << static_cast<uint64_t>(1'000'000'000.0 / my_average) << " op/s \n";

    timer.start();
    runner(otherFunc);
    timer.stop();

    double other_average = static_cast<double>(timer.nanoseconds()) / N;
    log << "-- Other function --\n";
    log << "     Total time      : " << timer.nanoseconds() << " ns \n";
    log << "     Average runtime : " << other_average << " ns/op \n";
    log << "     Ops per second  : "
        << static_cast<uint64_t>(1'000'000'000.0 / other_average) << " op/s \n";

    log << "-- Average runtime ratio --\n";
    log << "     Mine / Other's  : " << my_average / other_average << " \n";
  }

  static void run_perf(Func myFunc, Func otherFunc, const char *logFile) {
    std::ofstream log(logFile);
    log << " Performance tests with inputs in denormal range:\n";
    run_perf_in_range(myFunc, otherFunc, /* startingBit= */ StorageType(0),
                      /* endingBit= */ FPBits::max_subnormal().uintval(),
                      10'000'001, log);
    log << "\n Performance tests with inputs in normal range:\n";
    run_perf_in_range(myFunc, otherFunc,
                      /* startingBit= */ FPBits::min_normal().uintval(),
                      /* endingBit= */ FPBits::max_normal().uintval(),
                      10'000'001, log);
    log << "\n Performance tests with inputs in normal range with exponents "
           "close to each other:\n";
    run_perf_in_range(
        myFunc, otherFunc, /* startingBit= */ FPBits(T(0x1.0p-10)).uintval(),
        /* endingBit= */ FPBits(T(0x1.0p+10)).uintval(), 1'001'001, log);
  }

  static void run_diff(Func myFunc, Func otherFunc, const char *logFile) {
    uint64_t diffCount = 0;
    std::ofstream log(logFile);
    log << " Diff tests with inputs in denormal range:\n";
    diffCount += run_diff_in_range(
        myFunc, otherFunc, /* startingBit= */ StorageType(0),
        /* endingBit= */ FPBits::max_subnormal().uintval(), 1'000'001, log);
    log << "\n Diff tests with inputs in normal range:\n";
    diffCount += run_diff_in_range(
        myFunc, otherFunc,
        /* startingBit= */ FPBits::min_normal().uintval(),
        /* endingBit= */ FPBits::max_normal().uintval(), 100'000'001, log);
    log << "\n Diff tests with inputs in normal range with exponents "
           "close to each other:\n";
    diffCount += run_diff_in_range(
        myFunc, otherFunc, /* startingBit= */ FPBits(T(0x1.0p-10)).uintval(),
        /* endingBit= */ FPBits(T(0x1.0p+10)).uintval(), 10'000'001, log);

    log << "Total number of differing results: " << diffCount << '\n';
  }
};

} // namespace testing
} // namespace LIBC_NAMESPACE

#define BINARY_OP_SINGLE_OUTPUT_PERF(T, myFunc, otherFunc, filename)           \
  int main() {                                                                 \
    LIBC_NAMESPACE::testing::BinaryOpSingleOutputPerf<T>::run_perf(            \
        &myFunc, &otherFunc, filename);                                        \
    return 0;                                                                  \
  }
