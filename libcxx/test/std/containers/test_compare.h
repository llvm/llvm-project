//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_COMPARE_H
#define TEST_COMPARE_H

#include "test_macros.h"

template <class T>
struct test_equal_to {
  int data_;
  TEST_CONSTEXPR explicit test_equal_to() : data_(0) {}
  TEST_CONSTEXPR explicit test_equal_to(int data) : data_(data) {}
  TEST_CONSTEXPR bool operator()(const T& a, const T& b) const { return a == b; }
  TEST_CONSTEXPR friend bool operator==(const test_equal_to& a, const test_equal_to& b) { return a.data_ == b.data_; }
};

template <class T>
struct test_less {
  int data_;
  TEST_CONSTEXPR explicit test_less() : data_(0) {}
  TEST_CONSTEXPR explicit test_less(int data) : data_(data) {}
  TEST_CONSTEXPR bool operator()(const T& a, const T& b) const { return a < b; }
  TEST_CONSTEXPR friend bool operator==(const test_less& a, const test_less& b) { return a.data_ == b.data_; }
};

#endif // TEST_COMPARE_H
