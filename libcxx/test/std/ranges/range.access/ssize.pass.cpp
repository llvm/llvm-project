//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::ranges::ssize

#include <ranges>

#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"

using RangeSSizeT = decltype(std::ranges::ssize);

static_assert(!std::is_invocable_v<RangeSSizeT, int[]>);
static_assert( std::is_invocable_v<RangeSSizeT, int[1]>);
static_assert( std::is_invocable_v<RangeSSizeT, int (&&)[1]>);
static_assert( std::is_invocable_v<RangeSSizeT, int (&)[1]>);

struct SizeMember {
  constexpr size_t size() { return 42; }
};
static_assert(!std::is_invocable_v<decltype(std::ranges::ssize), const SizeMember&>);

struct SizeFunction {
  friend constexpr size_t size(SizeFunction) { return 42; }
};

struct SizeFunctionSigned {
  friend constexpr std::ptrdiff_t size(SizeFunctionSigned) { return 42; }
};

struct SizedSentinelRange {
  int data_[2] = {};
  constexpr int *begin() { return data_; }
  constexpr auto end() { return sized_sentinel<int*>(data_ + 2); }
};

struct ShortUnsignedReturnType {
  constexpr unsigned short size() { return 42; }
};

// size_t changes depending on the platform.
using SignedSizeT = std::make_signed_t<size_t>;

constexpr bool test() {
  int a[4];

  assert(std::ranges::ssize(a) == 4);
  ASSERT_SAME_TYPE(decltype(std::ranges::ssize(a)), SignedSizeT);

  assert(std::ranges::ssize(SizeMember()) == 42);
  ASSERT_SAME_TYPE(decltype(std::ranges::ssize(SizeMember())), SignedSizeT);

  assert(std::ranges::ssize(SizeFunction()) == 42);
  ASSERT_SAME_TYPE(decltype(std::ranges::ssize(SizeFunction())), SignedSizeT);

  assert(std::ranges::ssize(SizeFunctionSigned()) == 42);
  ASSERT_SAME_TYPE(decltype(std::ranges::ssize(SizeFunctionSigned())), std::ptrdiff_t);

  SizedSentinelRange b;
  assert(std::ranges::ssize(b) == 2);
  ASSERT_SAME_TYPE(decltype(std::ranges::ssize(b)), std::ptrdiff_t);

  // This gets converted to ptrdiff_t because it's wider.
  ShortUnsignedReturnType c;
  assert(std::ranges::ssize(c) == 42);
  ASSERT_SAME_TYPE(decltype(std::ranges::ssize(c)), ptrdiff_t);

  return true;
}

// Test ADL-proofing.
struct Incomplete;
template<class T> struct Holder { T t; };
static_assert(!std::is_invocable_v<RangeSSizeT, Holder<Incomplete>*>);
static_assert(!std::is_invocable_v<RangeSSizeT, Holder<Incomplete>*&>);

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
