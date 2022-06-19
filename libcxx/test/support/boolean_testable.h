//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_TEST_SUPPORT_BOOLEAN_TESTABLE_H
#define LIBCXX_TEST_SUPPORT_BOOLEAN_TESTABLE_H

#include "test_macros.h"

#if TEST_STD_VER > 17

class BooleanTestable {
public:
  constexpr operator bool() const {
    return value_;
  }

  friend constexpr BooleanTestable operator==(const BooleanTestable& lhs, const BooleanTestable& rhs) {
    return lhs.value_ == rhs.value_;
  }

  friend constexpr BooleanTestable operator!=(const BooleanTestable& lhs, const BooleanTestable& rhs) {
    return !(lhs == rhs);
  }

  constexpr BooleanTestable operator!() {
    return BooleanTestable{!value_};
  }

  // this class should behave like a bool, so the constructor shouldn't be explicit
  constexpr BooleanTestable(bool value) : value_{value} {}
  constexpr BooleanTestable(const BooleanTestable&) = delete;
  constexpr BooleanTestable(BooleanTestable&&) = delete;

private:
  bool value_;
};

template <class T>
class StrictComparable {
public:
  // this shouldn't be explicit to make it easier to initlaize inside arrays (which it almost always is)
  constexpr StrictComparable(T value) : value_{value} {}

  friend constexpr BooleanTestable operator==(const StrictComparable& lhs, const StrictComparable& rhs) {
    return (lhs.value_ == rhs.value_);
  }

  friend constexpr BooleanTestable operator!=(const StrictComparable& lhs, const StrictComparable& rhs) {
    return !(lhs == rhs);
  }

private:
  T value_;
};

#endif // TEST_STD_VER > 17

#endif // LIBCXX_TEST_SUPPORT_BOOLEAN_TESTABLE_H
