// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBCPP_TEST_STD_UTILITIES_MEMORY_SPECIALIZED_ALGORITHMS_COUNTED_H
#define LIBCPP_TEST_STD_UTILITIES_MEMORY_SPECIALIZED_ALGORITHMS_COUNTED_H

#include "test_macros.h"

struct Counted {
  static int current_objects;
  static int total_objects;
  static int throw_on;

  int value;

  explicit Counted() {
    check_throw();
    increase_counters();
  }

  explicit Counted(int v) : value(v) {
    check_throw();
    increase_counters();
  }

  ~Counted() { --current_objects; }

  static void reset() {
    current_objects = total_objects = 0;
    throw_on = -1;
  }

  Counted(const Counted& rhs) : value(rhs.value) {
    check_throw();
    increase_counters();
  }

  friend void operator&(Counted) = delete;

private:
  void check_throw() {
    if (throw_on == total_objects) {
      TEST_THROW(1);
    }
  }

  void increase_counters() {
    ++current_objects;
    ++total_objects;
  }
};
int Counted::current_objects = 0;
int Counted::total_objects = 0;
int Counted::throw_on = -1;

#endif // LIBCPP_TEST_STD_UTILITIES_MEMORY_SPECIALIZED_ALGORITHMS_COUNTED_H
