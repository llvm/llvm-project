//===-- Exhaustive test template for math functions -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <atomic>
#include <fenv.h>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include "src/__support/FPUtil/FPBits.h"

#include "exhaustive_test.h"

template <typename T, typename FloatType>
void LlvmLibcExhaustiveTest<T, FloatType>::test_full_range(
    T start, T stop, mpfr::RoundingMode rounding) {
  int n_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> thread_list;
  std::mutex mx_cur_val;
  int current_percent = -1;
  T current_value = start;
  std::atomic<uint64_t> failed(0);
  for (int i = 0; i < n_threads; ++i) {
    thread_list.emplace_back([&, this]() {
      while (true) {
        T range_begin, range_end;
        int new_percent = -1;
        {
          std::lock_guard<std::mutex> lock(mx_cur_val);
          if (current_value == stop)
            return;

          range_begin = current_value;
          if (stop >= increment && stop - increment >= current_value) {
            range_end = current_value + increment;
          } else
            range_end = stop;
          current_value = range_end;
          int pc = 100.0 * double(range_end - start) / double(stop - start);
          if (current_percent != pc) {
            new_percent = pc;
            current_percent = pc;
          }
        }
        if (new_percent >= 0) {
          std::stringstream msg;
          msg << new_percent << "% is in process     \r";
          std::cout << msg.str() << std::flush;
          ;
        }

        bool check_passed = check(range_begin, range_end, rounding);
        if (!check_passed) {
          std::stringstream msg;
          msg << "Test failed in range: " << std::dec << range_begin << " to "
              << range_end << " [0x" << std::hex << range_begin << ", 0x"
              << range_end << "), [" << std::hexfloat
              << static_cast<FloatType>(__llvm_libc::fputil::FPBits<FloatType>(
                     static_cast<T>(range_begin)))
              << ", "
              << static_cast<FloatType>(
                     __llvm_libc::fputil::FPBits<FloatType>(range_end))
              << ") " << std::endl;
          std::cerr << msg.str() << std::flush;

          failed.fetch_add(1);
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
}

template void
    LlvmLibcExhaustiveTest<uint32_t>::test_full_range(uint32_t, uint32_t,
                                                      mpfr::RoundingMode);
template void LlvmLibcExhaustiveTest<uint64_t, double>::test_full_range(
    uint64_t, uint64_t, mpfr::RoundingMode);
