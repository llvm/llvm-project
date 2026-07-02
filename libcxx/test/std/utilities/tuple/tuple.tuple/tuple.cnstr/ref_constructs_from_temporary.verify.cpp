//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <tuple>

// tuple(UTypes&&... u);
// tuple(tuple<UTypes...>& u);
// tuple(const tuple<UTypes...>& u);
// tuple(tuple<UTypes...>&& u);
// tuple(const tuple<UTypes...>&& u);
// tuple(pair<U1, U2>& u);
// tuple(const pair<U1, U2>& u);
// tuple(pair<U1, U2>&& u);
// tuple(const pair<U1, U2>&& u);
//
// The constructors above are defined as deleted if reference_constructs_from_temporary_v
// is true for one of the corresponding tuple elements.

#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

struct X {
  X(int);
};

static_assert(std::is_constructible_v<std::tuple<const X&>, X&>);
static_assert(!std::is_constructible_v<std::tuple<const X&>, int>);
static_assert(!std::is_constructible_v<std::tuple<const X&>, std::tuple<int>&>);
static_assert(!std::is_constructible_v<std::tuple<const X&>, const std::tuple<int>&>);
static_assert(!std::is_constructible_v<std::tuple<const X&>, std::tuple<int>&&>);
static_assert(!std::is_constructible_v<std::tuple<const X&>, const std::tuple<int>&&>);
static_assert(!std::is_constructible_v<std::tuple<const X&, int>, std::pair<int, int>&>);
static_assert(!std::is_constructible_v<std::tuple<const X&, int>, const std::pair<int, int>&>);
static_assert(!std::is_constructible_v<std::tuple<const X&, int>, std::pair<int, int>&&>);
static_assert(!std::is_constructible_v<std::tuple<const X&, int>, const std::pair<int, int>&&>);

void test() {
  std::allocator<int> alloc;
  std::tuple<int> t(1);
  const std::tuple<int> ct(1);
  std::pair<int, int> p(1, 2);
  const std::pair<int, int> cp(1, 2);

  // expected-error-re@*:* 18 {{call to deleted constructor of 'std::tuple<{{.*}}>'}}
  std::tuple<const X&> utypes(1);
  std::tuple<const X&> alloc_utypes(std::allocator_arg, alloc, 1);

  std::tuple<const X&> tuple_lvalue(t);
  std::tuple<const X&> alloc_tuple_lvalue(std::allocator_arg, alloc, t);
  std::tuple<const X&> const_tuple_lvalue(ct);
  std::tuple<const X&> alloc_const_tuple_lvalue(std::allocator_arg, alloc, ct);
  std::tuple<const X&> tuple_rvalue(std::tuple<int>{1});
  std::tuple<const X&> alloc_tuple_rvalue(std::allocator_arg, alloc, std::tuple<int>{1});
  std::tuple<const X&> const_tuple_rvalue(std::move(ct));
  std::tuple<const X&> alloc_const_tuple_rvalue(std::allocator_arg, alloc, std::move(ct));

  std::tuple<const X&, int> pair_lvalue(p);
  std::tuple<const X&, int> alloc_pair_lvalue(std::allocator_arg, alloc, p);
  std::tuple<const X&, int> const_pair_lvalue(cp);
  std::tuple<const X&, int> alloc_const_pair_lvalue(std::allocator_arg, alloc, cp);
  std::tuple<const X&, int> pair_rvalue(std::pair<int, int>{1, 2});
  std::tuple<const X&, int> alloc_pair_rvalue(std::allocator_arg, alloc, std::pair<int, int>{1, 2});
  std::tuple<const X&, int> const_pair_rvalue(std::move(cp));
  std::tuple<const X&, int> alloc_const_pair_rvalue(std::allocator_arg, alloc, std::move(cp));
}
