//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: no-exceptions

// <inplace_vector>

// constexpr reference push_back(const T& x);
// constexpr reference push_back(T&& x);
// template<class... Args>
//   constexpr reference emplace_back(Args&&... args);

#include <cassert>
#include <inplace_vector>

struct ThrowingCopy {
  int value_;
  static bool throw_now_;

  explicit ThrowingCopy(int value) : value_(value) {}
  ThrowingCopy(const ThrowingCopy& other) : value_(other.value_) {
    if (throw_now_)
      throw 1;
  }
  ThrowingCopy(ThrowingCopy&& other) : value_(other.value_) {
    if (throw_now_)
      throw 1;
  }
  ThrowingCopy& operator=(const ThrowingCopy&) = default;
  ThrowingCopy& operator=(ThrowingCopy&&)      = default;
};

bool ThrowingCopy::throw_now_ = false;

struct ThrowingCtor {
  explicit ThrowingCtor(int) { throw 1; }
};

int main(int, char**) {
  {
    std::inplace_vector<ThrowingCopy, 3> c;
    c.emplace_back(1);
    ThrowingCopy value(2);
    ThrowingCopy::throw_now_ = true;
    try {
      c.push_back(value);
      assert(false);
    } catch (int) {
      assert(c.size() == 1);
      assert(c[0].value_ == 1);
    }
    ThrowingCopy::throw_now_ = false;
  }
  {
    std::inplace_vector<ThrowingCtor, 3> c;
    try {
      c.emplace_back(1);
      assert(false);
    } catch (int) {
      assert(c.empty());
    }
  }

  return 0;
}
