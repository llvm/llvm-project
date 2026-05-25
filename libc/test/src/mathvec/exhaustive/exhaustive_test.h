//===-- Exhaustive tester for SIMD math functions -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/simd.h"
#include "test/UnitTest/FPMatcher.h"

#include <atomic>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

// To test SIMD math routines exhaustively against their scalar counterparts
// for inputs in the range [start, stop) in parallel:
// 1. Define a Checker class with:
//    - FloatType: define floating point type to be used.
//    - FPBits: fputil::FPBits<FloatType>.
//    - StorageType: define bit type for the corresponding floating point type.
//    - uint64_t check(start, stop, rounding_mode): a method to test in given
//          range for a given rounding mode, which returns the number of
//          failures.
// 2. Use LlvmLibcExhaustiveMathTest<Checker> class
// 3. Call: test_full_range_<RoundingMode>(start, stop)
//       or test_full_range_all_roundings(start, stop).
template <typename OutType, typename InType = OutType>
using ScalarUnaryOp = OutType(InType);

template <typename OutType, typename InType = OutType>
using VectorUnaryOp =
    LIBC_NAMESPACE::cpp::simd<OutType>(LIBC_NAMESPACE::cpp::simd<InType>);

template <typename OutType, typename InType,
          ScalarUnaryOp<OutType, InType> ScalarFunc,
          VectorUnaryOp<OutType, InType> VectorFunc>
struct UnaryOpChecker : public virtual LIBC_NAMESPACE::testing::Test {
  using FloatType = InType;
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<FloatType>;
  using StorageType = typename FPBits::StorageType;

  // Check in a range, return the number of failures.
  uint64_t check(StorageType start, StorageType stop,
                 LIBC_NAMESPACE::fputil::testing::RoundingMode rounding) {
    LIBC_NAMESPACE::fputil::testing::ForceRoundingMode r(rounding);
    if (!r.success)
      return (stop > start);

    StorageType bits = start;
    uint64_t failed = 0;
    do {
      FPBits xbits(bits);
      FloatType x = xbits.get_val();

      LIBC_NAMESPACE::cpp::simd<FloatType> vec_x(x);
      LIBC_NAMESPACE::cpp::simd<OutType> vec_result = VectorFunc(vec_x);
      OutType vec_res = vec_result[0];
      OutType scalar_result = ScalarFunc(x);
      bool correct = TEST_FP_EQ(scalar_result, vec_res);

      if (!correct) {
        EXPECT_FP_EQ(scalar_result, vec_res);
        failed++;
      }
    } while (bits++ < stop);
    return failed;
  }
};

// Checker class needs inherit from LIBC_NAMESPACE::testing::Test and provide
//   StorageType and check method.
template <typename Checker, size_t Increment = 1 << 20>
struct LlvmLibcExhaustiveMathvecTest
    : public virtual LIBC_NAMESPACE::testing::Test,
      public Checker {
  using FloatType = typename Checker::FloatType;
  using FPBits = typename Checker::FPBits;
  using StorageType = typename Checker::StorageType;

  void explain_failed_range(std::stringstream &msg, StorageType x_begin,
                            StorageType x_end) {
    msg << x_begin << " to " << x_end << " [0x" << std::hex << x_begin << ", 0x"
        << x_end << "), [" << std::hexfloat
        << static_cast<FloatType>(FPBits(x_begin).get_val()) << ", "
        << static_cast<FloatType>(FPBits(x_end).get_val()) << ")";
  }

  // Break [start, stop) into `nthreads` subintervals and apply *check to each
  // subinterval in parallel.
  template <typename... T>
  void test_full_range(LIBC_NAMESPACE::fputil::testing::RoundingMode rounding,
                       StorageType start, StorageType stop) {
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

          uint64_t failed_in_range =
              Checker::check(range_begin, range_end, rounding);
          if (failed_in_range > 0) {
            std::stringstream msg;
            msg << "Test failed for " << std::dec << failed_in_range
                << " inputs in range: ";
            explain_failed_range(msg, range_begin, range_end);
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

  void test_full_range_RN(StorageType start, StorageType stop) {
    std::cout << "-- Testing for FE_TONEAREST in range [0x" << std::hex << start
              << ", 0x" << stop << ") --" << std::dec << std::endl;
    test_full_range(LIBC_NAMESPACE::fputil::testing::RoundingMode::Nearest,
                    start, stop);
  }

  void test_full_range_RU(StorageType start, StorageType stop) {
    std::cout << "-- Testing for FE_UPWARD in range [0x" << std::hex << start
              << ", 0x" << stop << ") --" << std::dec << std::endl;
    test_full_range(LIBC_NAMESPACE::fputil::testing::RoundingMode::Upward,
                    start, stop);
  }

  void test_full_range_RD(StorageType start, StorageType stop) {
    std::cout << "-- Testing for FE_DOWNWARD in range [0x" << std::hex << start
              << ", 0x" << stop << ") --" << std::dec << std::endl;
    test_full_range(LIBC_NAMESPACE::fputil::testing::RoundingMode::Downward,
                    start, stop);
  }

  void test_full_range_RZ(StorageType start, StorageType stop) {
    std::cout << "-- Testing for FE_TOWARDZERO in range [0x" << std::hex
              << start << ", 0x" << stop << ") --" << std::dec << std::endl;
    test_full_range(LIBC_NAMESPACE::fputil::testing::RoundingMode::TowardZero,
                    start, stop);
  }

  void test_full_range_all_roundings(StorageType start, StorageType stop) {
    test_full_range_RN(start, stop);
    test_full_range_RU(start, stop);
    test_full_range_RD(start, stop);
    test_full_range_RZ(start, stop);
  }
};

template <typename FloatType, ScalarUnaryOp<FloatType> ScalarFunc,
          VectorUnaryOp<FloatType> VectorFunc>
using LlvmLibcUnaryOpExhaustiveMathvecTest = LlvmLibcExhaustiveMathvecTest<
    UnaryOpChecker<FloatType, FloatType, ScalarFunc, VectorFunc>>;
