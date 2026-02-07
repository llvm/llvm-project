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

int main(int, char**)
{
  auto const start = std::chrono::steady_clock::now();
  auto log = [start] ()-> auto& {
    using namespace std::chrono;

    auto elapsed = steady_clock::now() - start;

    auto hours = duration_cast<std::chrono::hours>(elapsed);
    elapsed -= hours;

    auto minutes = duration_cast<std::chrono::minutes>(elapsed);
    elapsed -= minutes;

    auto seconds = duration_cast<std::chrono::seconds>(elapsed);
    elapsed -= seconds;

    auto milliseconds = duration_cast<std::chrono::milliseconds>(elapsed);

    std::cerr << "["
              << std::setw(2) << std::setfill('0') << hours.count() << ":"
              << std::setw(2) << std::setfill('0') << minutes.count() << ":"
              << std::setw(2) << std::setfill('0') << seconds.count() << "."
              << std::setw(3) << std::setfill('0') << milliseconds.count()
              << "] ";

    return std::cerr;
  };

  std::counting_semaphore<> s(0);

  log() << "try_acquire_until: start + " <<  std::chrono::milliseconds(250)  << "\n";
  assert(!s.try_acquire_until(start + std::chrono::milliseconds(250)));
  log() << "done: try_acquire_until: start + " <<  std::chrono::milliseconds(250)  << "\n";

  log() << "try_acquire_for: " << std::chrono::milliseconds(250) << "\n";
  assert(!s.try_acquire_for(std::chrono::milliseconds(250)));
  log() << "done: try_acquire_for: " << std::chrono::milliseconds(250) << "\n";

  std::thread t = support::make_test_thread([&](){
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    s.release();
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    s.release();
  });

  log() << "try_acquire_until: start + " <<  std::chrono::seconds(2)  << "\n";
  assert(s.try_acquire_until(start + std::chrono::seconds(2)));
  log() << "done: try_acquire_until: start + " <<  std::chrono::seconds(2)  << "\n";

  log() << "try_acquire_for: " << std::chrono::seconds(2) << "\n";
  assert(s.try_acquire_for(std::chrono::seconds(2)));
  log() << "done: try_acquire_for: " << std::chrono::seconds(2) << "\n";
  t.join();

  auto const end = std::chrono::steady_clock::now();
  assert(end - start < std::chrono::seconds(10));

  return 1;
}
