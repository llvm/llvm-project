//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COPYCONSTRUCTIBLE_H
#define COPYCONSTRUCTIBLE_H

#include "test_macros.h"

#include <cstddef>
#include <functional>
#include <type_traits>

class CopyConstructible {
  int data_;

public:
  TEST_CONSTEXPR CopyConstructible(int data = 1) : data_(data) {}

  CopyConstructible(const CopyConstructible&)            = default;
  CopyConstructible& operator=(const CopyConstructible&) = default;

  TEST_CONSTEXPR_CXX14 CopyConstructible(CopyConstructible&& x) TEST_NOEXCEPT : data_(x.data_) { x.data_ = 0; }
  TEST_CONSTEXPR_CXX14 CopyConstructible& operator=(CopyConstructible&& x) {
    data_   = x.data_;
    x.data_ = 0;
    return *this;
  }

  TEST_CONSTEXPR int get() const { return data_; }

  friend TEST_CONSTEXPR bool operator==(const CopyConstructible& x, const CopyConstructible& y) {
    return x.data_ == y.data_;
  }
  friend TEST_CONSTEXPR bool operator!=(const CopyConstructible& x, const CopyConstructible& y) {
    return x.data_ != y.data_;
  }
  friend TEST_CONSTEXPR bool operator<(const CopyConstructible& x, const CopyConstructible& y) {
    return x.data_ < y.data_;
  }
  friend TEST_CONSTEXPR bool operator<=(const CopyConstructible& x, const CopyConstructible& y) {
    return x.data_ <= y.data_;
  }
  friend TEST_CONSTEXPR bool operator>(const CopyConstructible& x, const CopyConstructible& y) {
    return x.data_ > y.data_;
  }
  friend TEST_CONSTEXPR bool operator>=(const CopyConstructible& x, const CopyConstructible& y) {
    return x.data_ >= y.data_;
  }

#if TEST_STD_VER > 17
  friend constexpr auto operator<=>(const CopyConstructible&, const CopyConstructible&) = default;
#endif // TEST_STD_VER > 17

  TEST_CONSTEXPR_CXX14 CopyConstructible operator+(const CopyConstructible& x) const {
    return CopyConstructible(data_ + x.data_);
  }
  TEST_CONSTEXPR_CXX14 CopyConstructible operator*(const CopyConstructible& x) const {
    return CopyConstructible(data_ * x.data_);
  }

  template <class T>
  friend void operator,(CopyConstructible const&, T) = delete;

  template <class T>
  friend void operator,(T, CopyConstructible const&) = delete;
};
static_assert(std::is_copy_constructible<CopyConstructible>::value, "Needs to be copy-constructible");

#endif