//===-- Common utility class for differential analysis --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "utils/testutils/StreamWrapper.h"
#include "utils/testutils/Timer.h"

namespace __llvm_libc {
namespace testing {

template <typename T> class BinaryOpSingleOutputDiff {
  using FPBits = fputil::FPBits<T>;
  using UIntType = typename FPBits::UIntType;
  static constexpr UIntType MSBIT = UIntType(1) << (8 * sizeof(UIntType) - 1);
  static constexpr UIntType UINTMAX = (MSBIT - 1) + MSBIT;

public:
  typedef T Func(T, T);

  static uint64_t run_diff_in_range(Func myFunc, Func otherFunc,
                                    UIntType startingBit, UIntType endingBit,
                                    UIntType N,
                                    testutils::OutputFileStream &log) {
    uint64_t result = 0;
    if (endingBit < startingBit) {
      return result;
    }

    UIntType step = (endingBit - startingBit) / N;
    for (UIntType bitsX = startingBit, bitsY = endingBit;;
         bitsX += step, bitsY -= step) {
      T x = T(FPBits(bitsX));
      T y = T(FPBits(bitsY));
      FPBits myBits = FPBits(myFunc(x, y));
      FPBits otherBits = FPBits(otherFunc(x, y));
      if (myBits.uintval() != otherBits.uintval()) {
        result++;
        log << "       Input: " << bitsX << ", " << bitsY << " (" << x << ", "
            << y << ")\n"
            << "   My result: " << myBits.uintval() << " (" << myBits.get_val()
            << ")\n"
            << "Other result: " << otherBits.uintval() << " ("
            << otherBits.get_val() << ")\n"
            << '\n';
      }

      if (endingBit - bitsX < step) {
        break;
      }
    }
    return result;
  }

  static void run_perf_in_range(Func myFunc, Func otherFunc,
                                UIntType startingBit, UIntType endingBit,
                                UIntType N, testutils::OutputFileStream &log) {
    auto runner = [=](Func func) {
      volatile T result;
      if (endingBit < startingBit) {
        return;
      }

      UIntType step = (endingBit - startingBit) / N;
      for (UIntType bitsX = startingBit, bitsY = endingBit;;
           bitsX += step, bitsY -= step) {
        T x = T(FPBits(bitsX));
        T y = T(FPBits(bitsY));
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
    testutils::OutputFileStream log(logFile);
    log << " Performance tests with inputs in denormal range:\n";
    run_perf_in_range(myFunc, otherFunc, /* startingBit= */ UIntType(0),
                      /* endingBit= */ FPBits::MAX_SUBNORMAL, 1'000'001, log);
    log << "\n Performance tests with inputs in normal range:\n";
    run_perf_in_range(myFunc, otherFunc, /* startingBit= */ FPBits::MIN_NORMAL,
                      /* endingBit= */ FPBits::MAX_NORMAL, 100'000'001, log);
    log << "\n Performance tests with inputs in normal range with exponents "
           "close to each other:\n";
    run_perf_in_range(
        myFunc, otherFunc, /* startingBit= */ FPBits(T(0x1.0p-10)).uintval(),
        /* endingBit= */ FPBits(T(0x1.0p+10)).uintval(), 10'000'001, log);
  }

  static void run_diff(Func myFunc, Func otherFunc, const char *logFile) {
    uint64_t diffCount = 0;
    testutils::OutputFileStream log(logFile);
    log << " Diff tests with inputs in denormal range:\n";
    diffCount += run_diff_in_range(
        myFunc, otherFunc, /* startingBit= */ UIntType(0),
        /* endingBit= */ FPBits::MAX_SUBNORMAL, 1'000'001, log);
    log << "\n Diff tests with inputs in normal range:\n";
    diffCount += run_diff_in_range(
        myFunc, otherFunc, /* startingBit= */ FPBits::MIN_NORMAL,
        /* endingBit= */ FPBits::MAX_NORMAL, 100'000'001, log);
    log << "\n Diff tests with inputs in normal range with exponents "
           "close to each other:\n";
    diffCount += run_diff_in_range(
        myFunc, otherFunc, /* startingBit= */ FPBits(T(0x1.0p-10)).uintval(),
        /* endingBit= */ FPBits(T(0x1.0p+10)).uintval(), 10'000'001, log);

    log << "Total number of differing results: " << diffCount << '\n';
  }
};

} // namespace testing
} // namespace __llvm_libc

#define BINARY_OP_SINGLE_OUTPUT_DIFF(T, myFunc, otherFunc, filename)           \
  int main() {                                                                 \
    __llvm_libc::testing::BinaryOpSingleOutputDiff<T>::run_diff(               \
        &myFunc, &otherFunc, filename);                                        \
    return 0;                                                                  \
  }

#define BINARY_OP_SINGLE_OUTPUT_PERF(T, myFunc, otherFunc, filename)           \
  int main() {                                                                 \
    __llvm_libc::testing::BinaryOpSingleOutputDiff<T>::run_perf(               \
        &myFunc, &otherFunc, filename);                                        \
    return 0;                                                                  \
  }
