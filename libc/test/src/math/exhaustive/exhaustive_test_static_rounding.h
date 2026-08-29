//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains exhaustive test template for statically rounded math
/// functions
///
//===----------------------------------------------------------------------===//

// This file is modeled after exhaustive_test.h, modified for testing statically
// rounded math functions.

#include "exhaustive_test.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/macros/properties/types.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/RoundingModeUtils.h"
#include "test/UnitTest/Test.h"
#include "test/UnitTest/TestLogger.h"

#include <atomic>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

template <typename OutType, typename InType = OutType>
using StaticallyRoundedUnaryOp = OutType(InType, int);

template <typename OutType, typename InType,
          UnaryOp<OutType, InType> BaselineFunc,
          StaticallyRoundedUnaryOp<OutType, InType> Func>
struct StaticallyRoundedUnaryOpChecker
    : public virtual LIBC_NAMESPACE::testing::Test {
  using FloatType = InType;
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<FloatType>;
  using StorageType = typename FPBits::StorageType;
  using RoundingMode = LIBC_NAMESPACE::fputil::testing::RoundingMode;
  using ForceRoundingMode = LIBC_NAMESPACE::fputil::testing::ForceRoundingMode;

  // Check in a range, return the number of failures.
  uint64_t check(StorageType start, StorageType stop, RoundingMode rounding) {
    ForceRoundingMode r(rounding);
    if (!r.success)
      return (stop > start);

    // ForceRoundingMode already checks for valid FE_* rounding mode
    using LIBC_NAMESPACE::fputil::testing::get_fe_rounding;
    const int fenv_rounding = get_fe_rounding(rounding);

    StorageType bits = start;
    uint64_t failed = 0;

    do {
      FPBits xbits(bits);
      FloatType x = xbits.get_val();
      bool correct = TEST_FP_EQ(BaselineFunc(x), Func(x, fenv_rounding));
      failed += (!correct);
      // Uncomment to print out failed values.
      if (!correct) {
        EXPECT_FP_EQ_ROUNDING_MODE(BaselineFunc(x), Func(x, fenv_rounding),
                                   rounding);
      }
    } while (bits++ < stop);

    return failed;
  }
};

// Modeled after `LlvmLibcExhaustiveMathTest`
//
// Checker class needs inherit from LIBC_NAMESPACE::testing::Test and provide
//   StorageType and check method.
template <typename Checker, size_t Increment = 1 << 20>
struct LlvmLibcExhaustiveStaticallyRoundedMathTest
    : public virtual LIBC_NAMESPACE::testing::Test,
      public Checker {
  using FloatType = typename Checker::FloatType;
  using FPBits = typename Checker::FPBits;
  using StorageType = typename Checker::StorageType;
  using RoundingMode = typename LIBC_NAMESPACE::fputil::testing::RoundingMode;

  void explain_failed_range(std::stringstream &msg, StorageType x_begin,
                            StorageType x_end) {
#ifdef LIBC_TYPES_HAS_FLOAT16
    using T = LIBC_NAMESPACE::cpp::conditional_t<
        LIBC_NAMESPACE::cpp::is_same_v<FloatType, float16>, float, FloatType>;
#else
    using T = FloatType;
#endif

    msg << x_begin << " to " << x_end << " [0x" << std::hex << x_begin << ", 0x"
        << x_end << "), [" << std::hexfloat
        << static_cast<T>(FPBits(x_begin).get_val()) << ", "
        << static_cast<T>(FPBits(x_end).get_val()) << ")";
  }

  void explain_failed_range(std::stringstream &msg, StorageType x_begin,
                            StorageType x_end, StorageType y_begin,
                            StorageType y_end) {
    msg << "x ";
    explain_failed_range(msg, x_begin, x_end);
    msg << ", y ";
    explain_failed_range(msg, y_begin, y_end);
  }

  // Break [start, stop) into `nthreads` subintervals and apply *check to each
  // subinterval in parallel.
  template <typename... T>
  void test_full_range(RoundingMode rounding, StorageType start,
                       StorageType stop, T... extra_range_bounds) {
    int n_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> thread_list;
    std::mutex mx_cur_val;
    int current_percent = -1;
    StorageType current_value = start;
    std::atomic<uint64_t> failed(0);

    for (int i = 0; i < n_threads; ++i) {
      thread_list.emplace_back([&, this]() {
        while (true) {
          StorageType range_begin, range_end;
          int new_percent = -1;
          {
            std::lock_guard<std::mutex> lock(mx_cur_val);
            if (current_value == stop)
              return;

            range_begin = current_value;
            if (stop >= Increment && stop - Increment >= current_value) {
              range_end = static_cast<StorageType>(current_value + Increment);
            } else {
              range_end = stop;
            }
            current_value = range_end;
            int pc =
                static_cast<int>(100.0 * (range_end - start) / (stop - start));
            if (current_percent != pc) {
              new_percent = pc;
              current_percent = pc;
            }
          }
          if (new_percent >= 0) {
            std::stringstream msg;
            msg << new_percent << "% is in process     \r";
            std::cout << msg.str() << std::flush;
          }

          uint64_t failed_in_range = Checker::check(
              range_begin, range_end, extra_range_bounds..., rounding);
          if (failed_in_range > 0) {
            std::stringstream msg;
            msg << "Test failed for " << std::dec << failed_in_range
                << " inputs in range: ";
            explain_failed_range(msg, range_begin, range_end,
                                 extra_range_bounds...);
            msg << "\n";
            std::cerr << msg.str() << std::flush;

            failed.fetch_add(failed_in_range);
          }
        }
      });
    }

    for (auto &thread : thread_list) {
      if (thread.joinable()) {
        thread.join();
      }
    }

    std::cout << std::endl;
    std::cout << "Test " << ((failed > 0) ? "FAILED" : "PASSED") << std::endl;
    ASSERT_EQ(failed.load(), uint64_t(0));
  }

  void test_full_range_all_roundings(StorageType start, StorageType stop) {
    std::cout << "-- Testing for FE_TONEAREST in range [0x" << std::hex << start
              << ", 0x" << stop << ") --" << std::dec << std::endl;
    test_full_range(RoundingMode::Nearest, start, stop);

    std::cout << "-- Testing for FE_UPWARD in range [0x" << std::hex << start
              << ", 0x" << stop << ") --" << std::dec << std::endl;
    test_full_range(RoundingMode::Upward, start, stop);

    std::cout << "-- Testing for FE_DOWNWARD in range [0x" << std::hex << start
              << ", 0x" << stop << ") --" << std::dec << std::endl;
    test_full_range(RoundingMode::Downward, start, stop);

    std::cout << "-- Testing for FE_TOWARDZERO in range [0x" << std::hex
              << start << ", 0x" << stop << ") --" << std::dec << std::endl;
    test_full_range(RoundingMode::TowardZero, start, stop);
  }

  void test_full_range_all_roundings(StorageType x_start, StorageType x_stop,
                                     StorageType y_start, StorageType y_stop) {
    std::cout << "-- Testing for FE_TONEAREST in x range [0x" << std::hex
              << x_start << ", 0x" << x_stop << "), y range [0x" << y_start
              << ", 0x" << y_stop << ") --" << std::dec << std::endl;
    test_full_range(RoundingMode::Nearest, x_start, x_stop, y_start, y_stop);

    std::cout << "-- Testing for FE_UPWARD in x range [0x" << std::hex
              << x_start << ", 0x" << x_stop << "), y range [0x" << y_start
              << ", 0x" << y_stop << ") --" << std::dec << std::endl;
    test_full_range(RoundingMode::Upward, x_start, x_stop, y_start, y_stop);

    std::cout << "-- Testing for FE_DOWNWARD in x range [0x" << std::hex
              << x_start << ", 0x" << x_stop << "), y range [0x" << y_start
              << ", 0x" << y_stop << ") --" << std::dec << std::endl;
    test_full_range(RoundingMode::Downward, x_start, x_stop, y_start, y_stop);

    std::cout << "-- Testing for FE_TOWARDZERO in x range [0x" << std::hex
              << x_start << ", 0x" << x_stop << "), y range [0x" << y_start
              << ", 0x" << y_stop << ") --" << std::dec << std::endl;
    test_full_range(RoundingMode::TowardZero, x_start, x_stop, y_start, y_stop);
  }
};

template <typename FloatType, UnaryOp<FloatType> BaselineFunc,
          StaticallyRoundedUnaryOp<FloatType> Func>
using LlvmLibcStaticallyRoundedUnaryOpExhaustiveMathTest =
    LlvmLibcExhaustiveStaticallyRoundedMathTest<StaticallyRoundedUnaryOpChecker<
        FloatType, FloatType, BaselineFunc, Func>>;
