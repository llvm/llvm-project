//===-- Exhaustive test template for math functions -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/type_traits.h"
#include "src/__support/FPUtil/FPBits.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

#include <atomic>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

// To test exhaustively for inputs in the range [start, stop) in parallel:
// 1. Define a Checker class with:
//    - FloatType: define floating point type to be used.
//    - FPBits: fputil::FPBits<FloatType>.
//    - UIntType: define bit type for the corresponding floating point type.
//    - uint64_t check(start, stop, rounding_mode): a method to test in given
//          range for a given rounding mode, which returns the number of
//          failures.
// 2. Use LlvmLibcExhaustiveMathTest<Checker> class
// 3. Call: test_full_range(start, stop, nthreads, rounding)
//       or test_full_range_all_roundings(start, stop).
// * For single input single output math function, use the convenient template:
//   LlvmLibcUnaryOpExhaustiveMathTest<FloatType, Op, Func>.
namespace mpfr = __llvm_libc::testing::mpfr;

template <typename T> using UnaryOp = T(T);

template <typename T, mpfr::Operation Op, UnaryOp<T> Func>
struct UnaryOpChecker : public virtual __llvm_libc::testing::Test {
  using FloatType = T;
  using FPBits = __llvm_libc::fputil::FPBits<FloatType>;
  using UIntType = typename FPBits::UIntType;

  static constexpr UnaryOp<FloatType> *FUNC = Func;

  // Check in a range, return the number of failures.
  uint64_t check(UIntType start, UIntType stop, mpfr::RoundingMode rounding) {
    mpfr::ForceRoundingMode r(rounding);
    if (!r.success)
      return (stop > start);
    UIntType bits = start;
    uint64_t failed = 0;
    do {
      FPBits xbits(bits);
      FloatType x = FloatType(xbits);
      bool correct =
          TEST_MPFR_MATCH_ROUNDING_SILENTLY(Op, x, FUNC(x), 0.5, rounding);
      failed += (!correct);
      // Uncomment to print out failed values.
      // if (!correct) {
      //   TEST_MPFR_MATCH(Op::Operation, x, Op::func(x), 0.5, rounding);
      // }
    } while (bits++ < stop);
    return failed;
  }
};

// Checker class needs inherit from __llvm_libc::testing::Test and provide
//   UIntType and check method.
template <typename Checker>
struct LlvmLibcExhaustiveMathTest : public virtual __llvm_libc::testing::Test,
                                    public Checker {
  using FloatType = typename Checker::FloatType;
  using FPBits = typename Checker::FPBits;
  using UIntType = typename Checker::UIntType;

  static constexpr UIntType INCREMENT = (1 << 20);

  // Break [start, stop) into `nthreads` subintervals and apply *check to each
  // subinterval in parallel.
  void test_full_range(UIntType start, UIntType stop,
                       mpfr::RoundingMode rounding) {
    int n_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> thread_list;
    std::mutex mx_cur_val;
    int current_percent = -1;
    UIntType current_value = start;
    std::atomic<uint64_t> failed(0);

    for (int i = 0; i < n_threads; ++i) {
      thread_list.emplace_back([&, this]() {
        while (true) {
          UIntType range_begin, range_end;
          int new_percent = -1;
          {
            std::lock_guard<std::mutex> lock(mx_cur_val);
            if (current_value == stop)
              return;

            range_begin = current_value;
            if (stop >= INCREMENT && stop - INCREMENT >= current_value) {
              range_end = current_value + INCREMENT;
            } else {
              range_end = stop;
            }
            current_value = range_end;
            int pc = 100.0 * (range_end - start) / (stop - start);
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

          uint64_t failed_in_range =
              Checker::check(range_begin, range_end, rounding);
          if (failed_in_range > 0) {
            std::stringstream msg;
            msg << "Test failed for " << std::dec << failed_in_range
                << " inputs in range: " << range_begin << " to " << range_end
                << " [0x" << std::hex << range_begin << ", 0x" << range_end
                << "), [" << std::hexfloat
                << static_cast<FloatType>(FPBits(range_begin)) << ", "
                << static_cast<FloatType>(FPBits(range_end)) << ")\n";
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

  void test_full_range_all_roundings(UIntType start, UIntType stop) {
    std::cout << "-- Testing for FE_TONEAREST in range [0x" << std::hex << start
              << ", 0x" << stop << ") --" << std::dec << std::endl;
    test_full_range(start, stop, mpfr::RoundingMode::Nearest);

    std::cout << "-- Testing for FE_UPWARD in range [0x" << std::hex << start
              << ", 0x" << stop << ") --" << std::dec << std::endl;
    test_full_range(start, stop, mpfr::RoundingMode::Upward);

    std::cout << "-- Testing for FE_DOWNWARD in range [0x" << std::hex << start
              << ", 0x" << stop << ") --" << std::dec << std::endl;
    test_full_range(start, stop, mpfr::RoundingMode::Downward);

    std::cout << "-- Testing for FE_TOWARDZERO in range [0x" << std::hex
              << start << ", 0x" << stop << ") --" << std::dec << std::endl;
    test_full_range(start, stop, mpfr::RoundingMode::TowardZero);
  };
};

template <typename FloatType, mpfr::Operation Op, UnaryOp<FloatType> Func>
using LlvmLibcUnaryOpExhaustiveMathTest =
    LlvmLibcExhaustiveMathTest<UnaryOpChecker<FloatType, Op, Func>>;
