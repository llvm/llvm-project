//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// pointer data() noexcept;

#include <inplace_vector>
#include <concepts>
#include <memory>
#include <cassert>

#include "test_macros.h"

struct Nasty {
  constexpr Nasty() : i_(0) {}
  constexpr Nasty(int i) : i_(i) {}
  constexpr Nasty(const Nasty& other) : i_(other.i_) {}
  constexpr Nasty& operator=(const Nasty& other) {
    i_ = other.i_;
    return *this;
  }
  constexpr ~Nasty() {}

  Nasty* operator&() const {
    assert(false);
    return nullptr;
  }
  int i_;
};

std::inplace_vector<Nasty, 10> glob{123};

constexpr bool tests() {
  {
    std::inplace_vector<int, 0> v;
    ASSERT_NOEXCEPT(v.data());
    assert(v.data() == nullptr);
    static_assert(v.data() == nullptr);
  }
  {
    std::inplace_vector<int, 10> v;
    ASSERT_NOEXCEPT(v.data());
    assert(v.data() == std::to_address(v.begin()));
    std::same_as<int*> decltype(auto) data = v.data();
    v.push_back(4);
    assert(data == std::addressof(v.front()));
    assert(v.data() == data);
    assert(data[0] == 4);
    data[0] = 3;
    assert(v.front() == 3);
  }
  {
    std::inplace_vector<Nasty, 0> v;
    ASSERT_NOEXCEPT(v.data());
    assert(v.data() == nullptr);
    static_assert(v.data() == nullptr);
  }
  if !consteval {
    std::inplace_vector<Nasty, 10> v;
    ASSERT_NOEXCEPT(v.data());
    assert(v.data() == std::to_address(v.begin()));
    std::same_as<Nasty*> decltype(auto) data = v.data();
    v.push_back(4);
    assert(data == std::addressof(v.front()));
    assert(v.data() == data);
    assert(data[0].i_ == 4);
    data[0].i_ = 3;
    assert(v.front().i_ == 3);
  }
  if !consteval {
    TEST_DIAGNOSTIC_PUSH
    TEST_GCC_DIAGNOSTIC_IGNORED("-Waddress")
    static_assert(glob.data() != nullptr);
    TEST_DIAGNOSTIC_POP
    assert(glob.data()[0].i_ == 123);
    glob.data()[0].i_ = 321;
    assert(glob.front().i_ == 321);
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
