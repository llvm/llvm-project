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

template <typename T> class SingleInputSingleOutputDiff {
  using FPBits = fputil::FPBits<T>;
  using UIntType = typename FPBits::UIntType;
  static constexpr UIntType MSBit = UIntType(1) << (8 * sizeof(UIntType) - 1);
  static constexpr UIntType UIntMax = (MSBit - 1) + MSBit;

public:
  typedef T Func(T);

  static void runDiff(Func myFunc, Func otherFunc, const char *logFile) {
    UIntType diffCount = 0;
    testutils::OutputFileStream log(logFile);
    log << "Starting diff for values from 0 to " << UIntMax << '\n'
        << "Only differing results will be logged.\n\n";
    for (UIntType bits = 0;; ++bits) {
      T x = T(FPBits(bits));
      T myResult = myFunc(x);
      T otherResult = otherFunc(x);
      UIntType myBits = FPBits(myResult).uintval();
      UIntType otherBits = FPBits(otherResult).uintval();
      if (myBits != otherBits) {
        ++diffCount;
        log << "       Input: " << bits << " (" << x << ")\n"
            << "   My result: " << myBits << " (" << myResult << ")\n"
            << "Other result: " << otherBits << " (" << otherResult << ")\n"
            << '\n';
      }
      if (bits == UIntMax)
        break;
    }
    log << "Total number of differing results: " << diffCount << '\n';
  }

  static void runPerf(Func myFunc, Func otherFunc, const char *logFile) {
    auto runner = [](Func func) {
      volatile T result;
      for (UIntType bits = 0;; ++bits) {
        T x = T(FPBits(bits));
        result = func(x);
        if (bits == UIntMax)
          break;
      }
    };

    testutils::OutputFileStream log(logFile);
    Timer timer;
    timer.start();
    runner(myFunc);
    timer.stop();
    log << "   Run time of my function: " << timer.nanoseconds() << " ns \n";

    timer.start();
    runner(otherFunc);
    timer.stop();
    log << "Run time of other function: " << timer.nanoseconds() << " ns \n";
  }
};

} // namespace testing
} // namespace __llvm_libc

#define SINGLE_INPUT_SINGLE_OUTPUT_DIFF(T, myFunc, otherFunc, filename)        \
  int main() {                                                                 \
    __llvm_libc::testing::SingleInputSingleOutputDiff<T>::runDiff(             \
        &myFunc, &otherFunc, filename);                                        \
    return 0;                                                                  \
  }

#define SINGLE_INPUT_SINGLE_OUTPUT_PERF(T, myFunc, otherFunc, filename)        \
  int main() {                                                                 \
    __llvm_libc::testing::SingleInputSingleOutputDiff<T>::runPerf(             \
        &myFunc, &otherFunc, filename);                                        \
    return 0;                                                                  \
  }
