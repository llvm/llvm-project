//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <semaphore>

#include <semaphore>
#include <thread>
#include <chrono>
#include <cassert>
#include <iostream>
#include <iomanip>


#include "make_test_thread.h"
#include "test_macros.h"

void test(auto log_start){
  auto log = [log_start] ()-> auto& {
    using namespace std::chrono;

    auto elapsed = steady_clock::now() - log_start;

    auto hours = duration_cast<std::chrono::hours>(elapsed);
    elapsed -= hours;

    auto minutes = duration_cast<std::chrono::minutes>(elapsed);
    elapsed -= minutes;

    auto seconds = duration_cast<std::chrono::seconds>(elapsed);
    elapsed -= seconds;

    auto nanoseconds = duration_cast<std::chrono::nanoseconds>(elapsed);

    std::cerr << "["
              << std::setw(2) << std::setfill('0') << hours.count() << ":"
              << std::setw(2) << std::setfill('0') << minutes.count() << ":"
              << std::setw(2) << std::setfill('0') << seconds.count() << "."
              << std::setw(9) << std::setfill('0') << nanoseconds.count()
              << "] ";

    return std::cerr;
  };

  auto const start = std::chrono::steady_clock::now();
  std::counting_semaphore<> s(0);

  log() << "start: try_acquire_for: " << std::chrono::nanoseconds(1) << "\n";
  assert(!s.try_acquire_for(std::chrono::nanoseconds(1)));
  log() << "done:  try_acquire_for: " << std::chrono::nanoseconds(1) << "\n";

  log() << "start: try_acquire_for: " << std::chrono::microseconds(1) << "\n";
  assert(!s.try_acquire_for(std::chrono::microseconds(1)));
  log() << "done:  try_acquire_for: " << std::chrono::microseconds(1) << "\n";

  log() << "start: try_acquire_for: " << std::chrono::milliseconds(1) << "\n";
  assert(!s.try_acquire_for(std::chrono::milliseconds(1)));
  log() << "done:  try_acquire_for: " << std::chrono::milliseconds(1) << "\n";


  auto const end = std::chrono::steady_clock::now();
  assert(end - start < std::chrono::seconds(10));
}

int main(int, char**)
{
  auto const log_start = std::chrono::steady_clock::now();
  for (auto i = 0; i < 100; ++i) {
    std::cerr << "=== Iteration " << i << " ===\n";
    test(log_start);
  }

  return 0;

}
