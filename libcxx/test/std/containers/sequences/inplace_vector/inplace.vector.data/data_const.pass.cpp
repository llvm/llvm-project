//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// constexpr T* inplace_vector<T,N>::data() const noexcept;
// constexpr T* inplace_vector<T,0>::data() const noexcept;

#include <cassert>
#include <inplace_vector>
#include <memory>

#include "../common.h"
#include "test_macros.h"

struct Nasty {
  constexpr Nasty() : i_(0) {}
  constexpr Nasty(int i) : i_(i) {}
  constexpr ~Nasty() {}
  Nasty* operator&() const {
    assert(false);
    return nullptr;
  }
  int i_;
};

constexpr bool tests() {
  {
    const std::inplace_vector<int, 4> v;
    ASSERT_SAME_TYPE(decltype(v.data()), const int*);
    ASSERT_NOEXCEPT(v.data());
    assert(v.data() == v.data());
  }
  {
    const std::inplace_vector<int, 100> v(100);
    assert(v.data() == std::addressof(v.front()));
  }
  if (!TEST_IS_CONSTANT_EVALUATED || TEST_INPLACE_VECTOR_NONTRIVIAL_CONSTEXPR) {
    const std::inplace_vector<Nasty, 100> v(100);
    assert(v.data() == std::addressof(v.front()));
  }
  {
    const std::inplace_vector<int, 0> v;
    assert(v.data() == v.data());
    assert(v.data() == nullptr);
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());

  return 0;
}
