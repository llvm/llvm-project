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

#include <__type_traits/datasizeof.h>
#include <expected>
#include <optional>
#include <memory>

struct Empty {};

struct A {
  int x_;
  int y_;
};

struct B : public A {
  int z_;
  short z2_;
  virtual ~B() = default;
};

struct BoolWithPadding {
  explicit operator bool() { return b; }

private:
  alignas(1024) bool b = false;
};

static_assert(sizeof(std::expected<Empty, Empty>) == sizeof(bool));
static_assert(sizeof(std::expected<Empty, A>) == 2 * sizeof(int) + alignof(std::expected<Empty, A>));
static_assert(sizeof(std::expected<Empty, B>) == sizeof(B));
static_assert(sizeof(std::expected<A, Empty>) == 2 * sizeof(int) + alignof(std::expected<A, Empty>));
static_assert(sizeof(std::expected<A, A>) == 2 * sizeof(int) + alignof(std::expected<A, A>));
static_assert(sizeof(std::expected<B, Empty>) == sizeof(B));
static_assert(sizeof(std::expected<B, B>) == sizeof(B));

// Check that `expected`'s datasize is large enough for the parameter type(s).
static_assert(sizeof(std::expected<BoolWithPadding, Empty>) ==
              std::__libcpp_datasizeof<std::expected<BoolWithPadding, Empty>>::value);
static_assert(sizeof(std::expected<Empty, BoolWithPadding>) ==
              std::__libcpp_datasizeof<std::expected<Empty, BoolWithPadding>>::value);

// In this case, there should be tail padding in the `expected` because `A`
// itself does _not_ have tail padding.
static_assert(sizeof(std::expected<A, A>) > std::__libcpp_datasizeof<std::expected<A, A>>::value);

// Test with some real types.
static_assert(sizeof(std::expected<std::optional<int>, int>) == 8);
static_assert(std::__libcpp_datasizeof<std::expected<std::optional<int>, int>>::value == 8);

static_assert(sizeof(std::expected<int, std::optional<int>>) == 8);
static_assert(std::__libcpp_datasizeof<std::expected<int, std::optional<int>>>::value == 8);

static_assert(sizeof(std::expected<int, int>) == 8);
static_assert(std::__libcpp_datasizeof<std::expected<int, int>>::value == 5);

// clang-format off
static_assert(std::__libcpp_datasizeof<int>::value == 4);
static_assert(std::__libcpp_datasizeof<std::expected<int, int>>::value == 5);
static_assert(std::__libcpp_datasizeof<std::expected<std::expected<int, int>, int>>::value == 8);
static_assert(std::__libcpp_datasizeof<std::expected<std::expected<std::expected<int, int>, int>, int>>::value == 9);
static_assert(std::__libcpp_datasizeof<std::expected<std::expected<std::expected<std::expected<int, int>, int>, int>, int>>::value == 12);
// clang-format on
