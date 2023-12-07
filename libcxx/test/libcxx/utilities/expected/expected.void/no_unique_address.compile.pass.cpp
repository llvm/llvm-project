//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// clang-cl and cl currently don't support [[no_unique_address]]
// XFAIL: msvc

// test [[no_unique_address]] is applied to the union

#include <expected>

struct Empty {};

struct A {
  int x_;
  int y_;
};

struct B : public A {
  int z_;
  virtual ~B() = default;
};

static_assert(sizeof(std::expected<void, Empty>) == sizeof(bool));
static_assert(sizeof(std::expected<void, A>) == 2 * sizeof(int) + alignof(std::expected<void, A>));
static_assert(sizeof(std::expected<void, B>) == sizeof(B) + alignof(std::expected<void, B>));
