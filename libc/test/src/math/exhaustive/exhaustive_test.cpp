//===-- Exhaustive test template for math functions -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <fenv.h>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "exhaustive_test.h"
#include "utils/UnitTest/Test.h"

template <typename T>
void LlvmLibcExhaustiveTest<T>::test_full_range(T start, T stop, int nthreads,
                                                mpfr::RoundingMode rounding) {
  std::vector<std::thread> thread_list(nthreads);
  T increment = (stop - start - 1) / nthreads + 1;
  T begin = start;
  T end = start + increment - 1;
  for (int i = 0; i < nthreads; ++i) {
    thread_list.emplace_back([this, begin, end, rounding]() {
      std::stringstream msg;
      msg << "-- Testing from " << begin << " to " << end << " [0x" << std::hex
          << begin << ", 0x" << end << ") ..." << std::endl;
      std::cout << msg.str();
      msg.str("");

      check(begin, end, rounding);

      msg << "** Finished testing from " << std::dec << begin << " to " << end
          << " [0x" << std::hex << begin << ", 0x" << end << ")" << std::endl;
      std::cout << msg.str();
    });
    begin += increment;
    end += increment;
    if (end > stop)
      end = stop;
  }
  for (auto &thread : thread_list) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

template void
LlvmLibcExhaustiveTest<uint32_t>::test_full_range(uint32_t, uint32_t, int,
                                                  mpfr::RoundingMode);
template void
LlvmLibcExhaustiveTest<uint64_t>::test_full_range(uint64_t, uint64_t, int,
                                                  mpfr::RoundingMode);
