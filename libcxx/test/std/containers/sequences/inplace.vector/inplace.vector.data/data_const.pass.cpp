//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// const_pointer data() const noexcept;

#include <inplace_vector>
#include <concepts>
#include <memory>
#include <utility>
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
    ASSERT_NOEXCEPT(std::as_const(v).data());
    assert(std::as_const(v).data() == nullptr);
    static_assert(std::as_const(v).data() == nullptr);
  }
  {
    std::inplace_vector<int, 10> v;
    ASSERT_NOEXCEPT(std::as_const(v).data());
    assert(std::as_const(v).data() == v.data());
    assert(std::as_const(v).data() == std::to_address(std::as_const(v).begin()));
    std::same_as<const int*> decltype(auto) data = std::as_const(v).data();
    v.push_back(4);
    assert(data == std::addressof(v.front()));
    assert(std::as_const(v).data() == data);
    assert(data[0] == 4);
    const_cast<int*>(data)[0] = 3;
    assert(v.front() == 3);
  }
  {
    std::inplace_vector<Nasty, 0> v;
    ASSERT_NOEXCEPT(std::as_const(v).data());
    assert(std::as_const(v).data() == nullptr);
    static_assert(std::as_const(v).data() == nullptr);
  }
  if !consteval {
    std::inplace_vector<Nasty, 10> v;
    ASSERT_NOEXCEPT(std::as_const(v).data());
    assert(std::as_const(v).data() == v.data());
    assert(std::as_const(v).data() == std::to_address(v.begin()));
    std::same_as<const Nasty*> decltype(auto) data = std::as_const(v).data();
    v.push_back(4);
    assert(data == std::addressof(v.front()));
    assert(std::as_const(v).data() == data);
    assert(data[0].i_ == 4);
    const_cast<Nasty*>(data)[0].i_ = 3;
    assert(v.front().i_ == 3);
  }
  if !consteval {
    TEST_DIAGNOSTIC_PUSH
    TEST_GCC_DIAGNOSTIC_IGNORED("-Waddress")
    static_assert(std::as_const(glob).data() != nullptr);
    TEST_DIAGNOSTIC_POP
    assert(std::as_const(glob).data()[0].i_ == 123);
    const_cast<Nasty*>(std::as_const(glob).data())[0].i_ = 321;
    assert(glob.front().i_ == 321);
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
