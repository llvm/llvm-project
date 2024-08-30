//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// inplace_vector

#include <inplace_vector>
#include <type_traits>

struct NonTrivial {
  int i = 0;
};
struct VeryNonTrivial {
  VeryNonTrivial();
  VeryNonTrivial(const VeryNonTrivial&);
  VeryNonTrivial& operator=(const VeryNonTrivial&);
  ~VeryNonTrivial();
};
struct MoreNonTrivial : virtual VeryNonTrivial {
  virtual void f();
};

static_assert(std::is_empty_v<std::inplace_vector<int, 0>>);
static_assert(std::is_empty_v<std::inplace_vector<NonTrivial, 0>>);
static_assert(std::is_empty_v<std::inplace_vector<VeryNonTrivial, 0>>);
static_assert(std::is_empty_v<std::inplace_vector<MoreNonTrivial, 0>>);
