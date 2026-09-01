// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++23 -emit-module-interface %t/runner.cppm -o %t/runner.pcm
// RUN: %clang_cc1 -std=c++23 -fmodule-file=runner=%t/runner.pcm -fsyntax-only -verify %t/main.cpp
//
// Test again with reduced BMI.
// RUN: %clang_cc1 -std=c++23 -emit-reduced-module-interface %t/runner.cppm -o %t/runner.pcm
// RUN: %clang_cc1 -std=c++23 -fmodule-file=runner=%t/runner.pcm -fsyntax-only -verify %t/main.cpp

//--- shared.hpp
#pragma once

class Runner {
public:
  template <typename V>
    requires requires { []() {}(); }
  static void run(const V &value) { (void)value; }

  template <typename V>
    requires requires { []() {}(); }
  void runNonStatic(const V &value) { (void)value; }
};

//--- runner.cppm
module;
#include "shared.hpp"
export module runner;
export using ::Runner;

//--- main.cpp
// expected-no-diagnostics
#include "shared.hpp"
import runner;

void test() {
  Runner::run(42);
  Runner{}.runNonStatic(42);
}
