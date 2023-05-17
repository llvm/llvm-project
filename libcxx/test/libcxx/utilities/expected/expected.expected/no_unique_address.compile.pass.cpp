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

static_assert(sizeof(std::expected<Empty, Empty>) == sizeof(bool));
static_assert(sizeof(std::expected<Empty, A>) == 2 * sizeof(int) + alignof(std::expected<Empty, A>));
static_assert(sizeof(std::expected<Empty, B>) == sizeof(B) + alignof(std::expected<Empty, B>));
static_assert(sizeof(std::expected<A, Empty>) == 2 * sizeof(int) + alignof(std::expected<A, Empty>));
static_assert(sizeof(std::expected<A, A>) == 2 * sizeof(int) + alignof(std::expected<A, A>));
static_assert(sizeof(std::expected<B, Empty>) == sizeof(B) + alignof(std::expected<B, Empty>));
static_assert(sizeof(std::expected<B, B>) == sizeof(B) + alignof(std::expected<B, B>));
