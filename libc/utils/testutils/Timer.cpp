//===-- Timer.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Timer.h"

#include <chrono>
#include <fstream>

namespace __llvm_libc {
namespace testing {

struct TimerImplementation {
  std::chrono::high_resolution_clock::time_point Start;
  std::chrono::high_resolution_clock::time_point End;
};

Timer::Timer() : Impl(new TimerImplementation) {}

Timer::~Timer() { delete reinterpret_cast<TimerImplementation *>(Impl); }

void Timer::start() {
  auto T = reinterpret_cast<TimerImplementation *>(Impl);
  T->Start = std::chrono::high_resolution_clock::now();
}

void Timer::stop() {
  auto T = reinterpret_cast<TimerImplementation *>(Impl);
  T->End = std::chrono::high_resolution_clock::now();
}

uint64_t Timer::nanoseconds() const {
  auto T = reinterpret_cast<TimerImplementation *>(Impl);
  return std::chrono::nanoseconds(T->End - T->Start).count();
}

} // namespace testing
} // namespace __llvm_libc
