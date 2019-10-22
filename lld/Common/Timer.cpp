//===- Timer.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lld/Common/Timer.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/Support/Format.h"

using namespace lld;
using namespace llvm;

ScopedTimer::ScopedTimer(Timer &T) : T(&T) { T.start(); }

void ScopedTimer::stop() {
  if (!T)
    return;
  T->stop();
  T = nullptr;
}

ScopedTimer::~ScopedTimer() { stop(); }

Timer::Timer(llvm::StringRef Name) : Name(Name), Parent(nullptr) {}
Timer::Timer(llvm::StringRef Name, Timer &Parent)
    : Name(Name), Parent(&Parent) {}

void Timer::start() {
  if (Parent && Total.count() == 0)
    Parent->Children.push_back(this);
  StartTime = std::chrono::high_resolution_clock::now();
}

void Timer::stop() {
  Total += (std::chrono::high_resolution_clock::now() - StartTime);
}

Timer &Timer::root() {
  static Timer RootTimer("Total Link Time");
  return RootTimer;
}

void Timer::print() {
  double TotalDuration = static_cast<double>(root().millis());

  // We want to print the grand total under all the intermediate phases, so we
  // print all children first, then print the total under that.
  for (const auto &Child : Children)
    Child->print(1, TotalDuration);

  message(std::string(49, '-'));

  root().print(0, root().millis(), false);
}

double Timer::millis() const {
  return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
             Total)
      .count();
}

void Timer::print(int Depth, double TotalDuration, bool Recurse) const {
  double P = 100.0 * millis() / TotalDuration;

  SmallString<32> Str;
  llvm::raw_svector_ostream Stream(Str);
  std::string S = std::string(Depth * 2, ' ') + Name + std::string(":");
  Stream << format("%-30s%5d ms (%5.1f%%)", S.c_str(), (int)millis(), P);

  message(Str);

  if (Recurse) {
    for (const auto &Child : Children)
      Child->print(Depth + 1, TotalDuration);
  }
}
