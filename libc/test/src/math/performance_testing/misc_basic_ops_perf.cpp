//===-- Performance test for miscellaneous basic operations ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BinaryOpSingleOutputPerf.h"
#include "SingleInputSingleOutputPerf.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/math/copysignf.h"
#include "src/math/copysignf16.h"
#include "src/math/fabsf.h"
#include "src/math/fabsf16.h"
#include "src/math/fmaximum_numf.h"
#include "src/math/fmaximum_numf16.h"
#include "src/math/fminimum_numf.h"
#include "src/math/fminimum_numf16.h"
#include "src/math/frexpf16.h"
#include "test/src/math/performance_testing/Timer.h"

#include <algorithm>
#include <fstream>
#include <math.h>

namespace LIBC_NAMESPACE::testing {

template <typename T> class FrexpPerf {
  using FPBits = fputil::FPBits<T>;
  using StorageType = typename FPBits::StorageType;

public:
  typedef T Func(T, int *);

  static void run_perf_in_range(Func my_func, Func other_func,
                                StorageType starting_bit,
                                StorageType ending_bit, size_t rounds,
                                std::ofstream &log) {
    size_t n = 10'010'001;
    if (sizeof(StorageType) <= sizeof(size_t))
      n = std::min(n, static_cast<size_t>(ending_bit - starting_bit));

    auto runner = [=](Func func) {
      StorageType step = (ending_bit - starting_bit) / n;
      if (step == 0)
        step = 1;
      [[maybe_unused]] volatile T result;
      int result_exp;
      for (size_t i = 0; i < rounds; i++) {
        for (StorageType bits = starting_bit; bits < ending_bit; bits += step) {
          T x = FPBits(bits).get_val();
          result = func(x, &result_exp);
        }
      }
    };

    Timer timer;
    timer.start();
    runner(my_func);
    timer.stop();

    double my_average = static_cast<double>(timer.nanoseconds()) / n / rounds;
    log << "-- My function --\n";
    log << "     Total time      : " << timer.nanoseconds() << " ns \n";
    log << "     Average runtime : " << my_average << " ns/op \n";
    log << "     Ops per second  : "
        << static_cast<uint64_t>(1'000'000'000.0 / my_average) << " op/s \n";

    timer.start();
    runner(other_func);
    timer.stop();

    double other_average =
        static_cast<double>(timer.nanoseconds()) / n / rounds;
    log << "-- Other function --\n";
    log << "     Total time      : " << timer.nanoseconds() << " ns \n";
    log << "     Average runtime : " << other_average << " ns/op \n";
    log << "     Ops per second  : "
        << static_cast<uint64_t>(1'000'000'000.0 / other_average) << " op/s \n";

    log << "-- Average runtime ratio --\n";
    log << "     Mine / Other's  : " << my_average / other_average << " \n";
  }

  static void run_perf(Func my_func, Func other_func, size_t rounds,
                       const char *log_file) {
    std::ofstream log(log_file);
    log << " Performance tests with inputs in denormal range:\n";
    run_perf_in_range(my_func, other_func, /* startingBit= */ StorageType(0),
                      /* endingBit= */ FPBits::max_subnormal().uintval(),
                      rounds, log);
    log << "\n Performance tests with inputs in normal range:\n";
    run_perf_in_range(my_func, other_func,
                      /* startingBit= */ FPBits::min_normal().uintval(),
                      /* endingBit= */ FPBits::max_normal().uintval(), rounds,
                      log);
  }
};

} // namespace LIBC_NAMESPACE::testing

#define FREXP_PERF(T, my_func, other_func, rounds, filename)                   \
  {                                                                            \
    LIBC_NAMESPACE::testing::FrexpPerf<T>::run_perf(&my_func, &other_func,     \
                                                    rounds, filename);         \
    LIBC_NAMESPACE::testing::FrexpPerf<T>::run_perf(&my_func, &other_func,     \
                                                    rounds, filename);         \
  }

static constexpr size_t FLOAT16_ROUNDS = 20'000;
static constexpr size_t FLOAT_ROUNDS = 40;

// LLVM libc might be the only libc implementation with support for float16 math
// functions currently. We can't compare our float16 functions against the
// system libc, so we compare them against this placeholder function.
float16 placeholder_unaryf16(float16 x) { return x; }
float16 placeholder_binaryf16(float16 x, float16 y) { return x; }
float16 placeholder_frexpf16(float16 x, int *exp) { return x; }

// The system libc might not provide the f{max,min}imum_num* C23 math functions
// either.
float placeholder_binaryf(float x, float y) { return x; }

int main() {
  SINGLE_INPUT_SINGLE_OUTPUT_PERF_EX(float16, LIBC_NAMESPACE::fabsf16,
                                     placeholder_unaryf16, FLOAT16_ROUNDS,
                                     "fabsf16_perf.log")
  BINARY_OP_SINGLE_OUTPUT_PERF_EX(float16, LIBC_NAMESPACE::copysignf16,
                                  placeholder_binaryf16, FLOAT16_ROUNDS,
                                  "copysignf16_perf.log")
  BINARY_OP_SINGLE_OUTPUT_PERF_EX(float16, LIBC_NAMESPACE::fmaximum_numf16,
                                  placeholder_binaryf16, FLOAT16_ROUNDS,
                                  "fmaximum_numf16_perf.log")
  BINARY_OP_SINGLE_OUTPUT_PERF_EX(float16, LIBC_NAMESPACE::fminimum_numf16,
                                  placeholder_binaryf16, FLOAT16_ROUNDS,
                                  "fminimum_numf16_perf.log")
  FREXP_PERF(float16, LIBC_NAMESPACE::frexpf16, placeholder_frexpf16,
             FLOAT16_ROUNDS, "frexpf16_perf.log")

  SINGLE_INPUT_SINGLE_OUTPUT_PERF_EX(float, LIBC_NAMESPACE::fabsf, fabsf,
                                     FLOAT_ROUNDS, "fabsf_perf.log")
  BINARY_OP_SINGLE_OUTPUT_PERF_EX(float, LIBC_NAMESPACE::copysignf, copysignf,
                                  FLOAT_ROUNDS, "copysignf_perf.log")
  BINARY_OP_SINGLE_OUTPUT_PERF_EX(float, LIBC_NAMESPACE::fmaximum_numf,
                                  placeholder_binaryf, FLOAT_ROUNDS,
                                  "fmaximum_numf_perf.log")
  BINARY_OP_SINGLE_OUTPUT_PERF_EX(float, LIBC_NAMESPACE::fminimum_numf,
                                  placeholder_binaryf, FLOAT_ROUNDS,
                                  "fminimum_numf_perf.log")

  return 0;
}
