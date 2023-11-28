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
namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

template <typename T> using UnaryOp = T(T);

template <typename T, mpfr::Operation Op, UnaryOp<T> Func>
struct UnaryOpChecker : public virtual LIBC_NAMESPACE::testing::Test {
  using FloatType = T;
  using FPBits = LIBC_NAMESPACE::fputil::FPBits<FloatType>;
  using UIntType = typename FPBits::UIntType;

  static constexpr UnaryOp<FloatType> *FUNC = Func;
  static constexpr mpfr::Operation OP = Op;

  // Check in a range, return the number of failures.
  bool check(FloatType in, FloatType out, mpfr::RoundingMode rounding) {
    mpfr::ForceRoundingMode r(rounding);
    if (!r.success)
      return true;

    bool correct = TEST_MPFR_MATCH_ROUNDING(Op, in, out, 0.5, rounding);
    return !correct;
  }
};

// Checker class needs inherit from LIBC_NAMESPACE::testing::Test and provide
// UIntType and check method.
template <typename Checker>
struct LlvmLibcExhaustiveMathTest
    : public virtual LIBC_NAMESPACE::testing::Test,
      public Checker {
  using FloatType = typename Checker::FloatType;
  using FPBits = typename Checker::FPBits;
  using UIntType = typename Checker::UIntType;

  static constexpr UIntType BLOCK_SIZE = (1 << 25);

  // Break [start, stop) into chunks and compare results on the GPU vs the CPU.
  void test_full_range(UIntType start, UIntType stop,
                       mpfr::RoundingMode rounding) {

    // TODO: We can run the GPU asynchronously to compute the next block.
    // However, the main bottleneck is MPFR on the CPU.
    uint64_t failed = 0;
    for (UIntType chunk = start; chunk <= stop; chunk += BLOCK_SIZE) {
      uint64_t percent = (static_cast<double>(chunk - start) /
                          static_cast<double>(stop - start)) *
                         100.0;
      std::cout << percent << "% is in process     \r" << std::flush;
      UIntType end = std::min(stop, chunk + BLOCK_SIZE);

      std::vector<FloatType> data(BLOCK_SIZE, FloatType(0));

      FloatType *ptr = data.data();
      // Fill the buffer with the computed results from the GPU.
#pragma omp target teams distribute parallel for map(from : ptr[0 : BLOCK_SIZE])
      for (UIntType begin = chunk; begin < end; ++begin) {
        UIntType idx = begin - chunk;

        FPBits xbits(begin);
        FloatType x = FloatType(xbits);

        ptr[idx] = Checker::FUNC(x);
      }

      std::atomic<uint64_t> failed_in_range = 0;
      // Check the GPU results against the MPFR library.
#pragma omp parallel for default(firstprivate) shared(failed_in_range)
      for (UIntType begin = chunk; begin < end; ++begin) {
        UIntType idx = begin - chunk;

        FPBits xbits(begin);
        FloatType x = FloatType(xbits);

        failed_in_range += Checker::check(x, data[idx], rounding);
      }

      if (failed_in_range > 0) {
        std::stringstream msg;
        msg << "Test failed for " << std::dec << failed_in_range
            << " inputs in range: " << chunk << " to " << end << " [0x"
            << std::hex << chunk << ", 0x" << end << "), [" << std::hexfloat
            << static_cast<FloatType>(FPBits(chunk)) << ", "
            << static_cast<FloatType>(FPBits(end)) << ")\n";
        std::cerr << msg.str() << std::flush;

        failed += failed_in_range.load();
      }

      // Check to make sure we don't overflow when updating the value.
      if (chunk > std::numeric_limits<UIntType>::max() - BLOCK_SIZE)
        chunk = std::numeric_limits<UIntType>::max();
    }

    std::cout << std::endl;
    std::cout << "Test " << ((failed > 0) ? "FAILED" : "PASSED") << std::endl;
    ASSERT_EQ(failed, uint64_t(0));
  }
};

template <typename FloatType, mpfr::Operation Op, UnaryOp<FloatType> Func>
using LlvmLibcUnaryOpExhaustiveMathTest =
    LlvmLibcExhaustiveMathTest<UnaryOpChecker<FloatType, Op, Func>>;
