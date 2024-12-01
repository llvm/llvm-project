//===-- Performance test for nearest integer functions --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/FPBits.h"
#include "src/math/ceilf.h"
#include "src/math/ceilf16.h"
#include "src/math/floorf.h"
#include "src/math/floorf16.h"
#include "src/math/rintf.h"
#include "src/math/rintf16.h"
#include "src/math/roundevenf.h"
#include "src/math/roundevenf16.h"
#include "src/math/roundf.h"
#include "src/math/roundf16.h"
#include "src/math/truncf.h"
#include "src/math/truncf16.h"
#include "test/UnitTest/RoundingModeUtils.h"
#include "test/src/math/performance_testing/Timer.h"

#include <fstream>
#include <math.h>

using LIBC_NAMESPACE::fputil::testing::ForceRoundingMode;
using LIBC_NAMESPACE::fputil::testing::RoundingMode;

namespace LIBC_NAMESPACE::testing {

template <typename T> class NearestIntegerPerf {
  using FPBits = fputil::FPBits<T>;
  using StorageType = typename FPBits::StorageType;

public:
  typedef T Func(T);

  static void run_perf_in_range(Func my_func, Func other_func,
                                StorageType starting_bit,
                                StorageType ending_bit, StorageType step,
                                size_t rounds, std::ofstream &log) {
    auto runner = [=](Func func) {
      [[maybe_unused]] volatile T result;
      for (size_t i = 0; i < rounds; i++) {
        for (StorageType bits = starting_bit; bits <= ending_bit;
             bits += step) {
          T x = FPBits(bits).get_val();
          result = func(x);
        }
      }
    };

    Timer timer;
    timer.start();
    runner(my_func);
    timer.stop();

    size_t number_of_runs = (ending_bit - starting_bit) / step + 1;
    double my_average =
        static_cast<double>(timer.nanoseconds()) / number_of_runs / rounds;
    log << "-- My function --\n";
    log << "     Total time      : " << timer.nanoseconds() << " ns \n";
    log << "     Average runtime : " << my_average << " ns/op \n";
    log << "     Ops per second  : "
        << static_cast<uint64_t>(1'000'000'000.0 / my_average) << " op/s \n";

    timer.start();
    runner(other_func);
    timer.stop();

    double other_average =
        static_cast<double>(timer.nanoseconds()) / number_of_runs / rounds;
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
    log << "Performance tests with inputs in normal integral range:\n";
    run_perf_in_range(
        my_func, other_func,
        /*starting_bit=*/StorageType((FPBits::EXP_BIAS + 1) << FPBits::SIG_LEN),
        /*ending_bit=*/
        StorageType((FPBits::EXP_BIAS + FPBits::FRACTION_LEN - 1)
                    << FPBits::SIG_LEN),
        /*step=*/StorageType(1 << FPBits::SIG_LEN),
        rounds * FPBits::EXP_BIAS * FPBits::EXP_BIAS * 2, log);
    log << "\n Performance tests with inputs in low integral range:\n";
    run_perf_in_range(
        my_func, other_func,
        /*starting_bit=*/StorageType(1 << FPBits::SIG_LEN),
        /*ending_bit=*/StorageType((FPBits::EXP_BIAS - 1) << FPBits::SIG_LEN),
        /*step_bit=*/StorageType(1 << FPBits::SIG_LEN),
        rounds * FPBits::EXP_BIAS * FPBits::EXP_BIAS * 2, log);
    log << "\n Performance tests with inputs in high integral range:\n";
    run_perf_in_range(
        my_func, other_func,
        /*starting_bit=*/
        StorageType((FPBits::EXP_BIAS + FPBits::FRACTION_LEN)
                    << FPBits::SIG_LEN),
        /*ending_bit=*/
        StorageType(FPBits::MAX_BIASED_EXPONENT << FPBits::SIG_LEN),
        /*step=*/StorageType(1 << FPBits::SIG_LEN),
        rounds * FPBits::EXP_BIAS * FPBits::EXP_BIAS * 2, log);
    log << "\n Performance tests with inputs in normal fractional range:\n";
    run_perf_in_range(
        my_func, other_func,
        /*starting_bit=*/
        StorageType(((FPBits::EXP_BIAS + 1) << FPBits::SIG_LEN) + 1),
        /*ending_bit=*/
        StorageType(((FPBits::EXP_BIAS + 2) << FPBits::SIG_LEN) - 1),
        /*step=*/StorageType(1), rounds * 2, log);
    log << "\n Performance tests with inputs in subnormal fractional range:\n";
    run_perf_in_range(my_func, other_func, /*starting_bit=*/StorageType(1),
                      /*ending_bit=*/StorageType(FPBits::SIG_MASK),
                      /*step=*/StorageType(1), rounds, log);
  }
};

} // namespace LIBC_NAMESPACE::testing

#define NEAREST_INTEGER_PERF(T, my_func, other_func, rounds, filename)         \
  {                                                                            \
    LIBC_NAMESPACE::testing::NearestIntegerPerf<T>::run_perf(                  \
        &my_func, &other_func, rounds, filename);                              \
    LIBC_NAMESPACE::testing::NearestIntegerPerf<T>::run_perf(                  \
        &my_func, &other_func, rounds, filename);                              \
  }

static constexpr size_t FLOAT16_ROUNDS = 20'000;
static constexpr size_t FLOAT_ROUNDS = 40;

// LLVM libc might be the only libc implementation with support for float16 math
// functions currently. We can't compare our float16 functions against the
// system libc, so we compare them against this placeholder function.
float16 placeholderf16(float16 x) { return x; }

// The system libc might not provide the roundeven* C23 math functions either.
float placeholderf(float x) { return x; }

int main() {
  NEAREST_INTEGER_PERF(float16, LIBC_NAMESPACE::ceilf16, ::placeholderf16,
                       FLOAT16_ROUNDS, "ceilf16_perf.log")
  NEAREST_INTEGER_PERF(float16, LIBC_NAMESPACE::floorf16, ::placeholderf16,
                       FLOAT16_ROUNDS, "floorf16_perf.log")
  NEAREST_INTEGER_PERF(float16, LIBC_NAMESPACE::roundf16, ::placeholderf16,
                       FLOAT16_ROUNDS, "roundf16_perf.log")
  NEAREST_INTEGER_PERF(float16, LIBC_NAMESPACE::roundevenf16, ::placeholderf16,
                       FLOAT16_ROUNDS, "roundevenf16_perf.log")
  NEAREST_INTEGER_PERF(float16, LIBC_NAMESPACE::truncf16, ::placeholderf16,
                       FLOAT16_ROUNDS, "truncf16_perf.log")

  NEAREST_INTEGER_PERF(float, LIBC_NAMESPACE::ceilf, ::ceilf, FLOAT_ROUNDS,
                       "ceilf_perf.log")
  NEAREST_INTEGER_PERF(float, LIBC_NAMESPACE::floorf, ::floorf, FLOAT_ROUNDS,
                       "floorf_perf.log")
  NEAREST_INTEGER_PERF(float, LIBC_NAMESPACE::roundf, ::roundf, FLOAT_ROUNDS,
                       "roundf_perf.log")
  NEAREST_INTEGER_PERF(float, LIBC_NAMESPACE::roundevenf, ::placeholderf,
                       FLOAT_ROUNDS, "roundevenf_perf.log")
  NEAREST_INTEGER_PERF(float, LIBC_NAMESPACE::truncf, ::truncf, FLOAT_ROUNDS,
                       "truncf_perf.log")

  if (ForceRoundingMode r(RoundingMode::Upward); r.success) {
    NEAREST_INTEGER_PERF(float16, LIBC_NAMESPACE::rintf16, ::placeholderf16,
                         FLOAT16_ROUNDS, "rintf16_upward_perf.log")
    NEAREST_INTEGER_PERF(float, LIBC_NAMESPACE::rintf, ::rintf, FLOAT_ROUNDS,
                         "rintf_upward_perf.log")
  }
  if (ForceRoundingMode r(RoundingMode::Downward); r.success) {
    NEAREST_INTEGER_PERF(float16, LIBC_NAMESPACE::rintf16, ::placeholderf16,
                         FLOAT16_ROUNDS, "rintf16_downward_perf.log")
    NEAREST_INTEGER_PERF(float, LIBC_NAMESPACE::rintf, ::rintf, FLOAT_ROUNDS,
                         "rintf_downward_perf.log")
  }
  if (ForceRoundingMode r(RoundingMode::TowardZero); r.success) {
    NEAREST_INTEGER_PERF(float16, LIBC_NAMESPACE::rintf16, ::placeholderf16,
                         FLOAT16_ROUNDS, "rintf16_towardzero_perf.log")
    NEAREST_INTEGER_PERF(float, LIBC_NAMESPACE::rintf, ::rintf, FLOAT_ROUNDS,
                         "rintf_towardzero_perf.log")
  }
  if (ForceRoundingMode r(RoundingMode::Nearest); r.success) {
    NEAREST_INTEGER_PERF(float16, LIBC_NAMESPACE::rintf16, ::placeholderf16,
                         FLOAT16_ROUNDS, "rintf16_nearest_perf.log")
    NEAREST_INTEGER_PERF(float, LIBC_NAMESPACE::rintf, ::rintf, FLOAT_ROUNDS,
                         "rintf_nearest_perf.log")
  }

  return 0;
}
