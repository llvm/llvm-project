//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// std::ranges::begin

#include <ranges>

#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"

using RangeBeginT = decltype(std::ranges::begin)&;
using RangeCBeginT = decltype(std::ranges::cbegin)&;

static int globalBuff[8];

struct Incomplete;

static_assert(!std::is_invocable_v<RangeBeginT, int (&&)[10]>);
static_assert( std::is_invocable_v<RangeBeginT, int (&)[10]>);
static_assert(!std::is_invocable_v<RangeBeginT, int (&&)[]>);
static_assert( std::is_invocable_v<RangeBeginT, int (&)[]>);

struct BeginMember {
  int x;
  constexpr const int *begin() const { return &x; }
};

// Ensure that we can't call with rvalues with borrowing disabled.
static_assert( std::is_invocable_v<RangeBeginT, BeginMember &>);
static_assert(!std::is_invocable_v<RangeBeginT, BeginMember &&>);
static_assert( std::is_invocable_v<RangeBeginT, BeginMember const&>);
static_assert(!std::is_invocable_v<RangeBeginT, BeginMember const&&>);
static_assert( std::is_invocable_v<RangeCBeginT, BeginMember &>);
static_assert(!std::is_invocable_v<RangeCBeginT, BeginMember &&>);
static_assert( std::is_invocable_v<RangeCBeginT, BeginMember const&>);
static_assert( std::is_invocable_v<RangeCBeginT, BeginMember const&&>);

constexpr bool testArray() {
  int a[2];
  assert(std::ranges::begin(a) == a);
  assert(std::ranges::cbegin(a) == a);

  int b[2][2];
  assert(std::ranges::begin(b) == b);
  assert(std::ranges::cbegin(b) == b);

  BeginMember c[2];
  assert(std::ranges::begin(c) == c);
  assert(std::ranges::cbegin(c) == c);

  return true;
}

struct BeginMemberFunction {
  int x;
  constexpr const int *begin() const { return &x; }
  friend int *begin(BeginMemberFunction const&);
};

struct BeginMemberReturnsInt {
  int begin() const;
};
static_assert(!std::is_invocable_v<RangeBeginT, BeginMemberReturnsInt const&>);

struct BeginMemberReturnsVoidPtr {
  const void *begin() const;
};
static_assert(!std::is_invocable_v<RangeBeginT, BeginMemberReturnsVoidPtr const&>);

struct EmptyBeginMember {
  struct iterator {};
  iterator begin() const;
};
static_assert(!std::is_invocable_v<RangeBeginT, EmptyBeginMember const&>);

struct EmptyPtrBeginMember {
  struct Empty {};
  Empty x;
  constexpr const Empty *begin() const { return &x; }
};

struct PtrConvertibleBeginMember {
  struct iterator { operator int*() const; };
  iterator begin() const;
};
static_assert(!std::is_invocable_v<RangeBeginT, PtrConvertibleBeginMember const&>);

struct NonConstBeginMember {
  int x;
  constexpr int *begin() { return &x; }
};
static_assert( std::is_invocable_v<RangeBeginT,  NonConstBeginMember &>);
static_assert(!std::is_invocable_v<RangeBeginT,  NonConstBeginMember const&>);
static_assert(!std::is_invocable_v<RangeCBeginT, NonConstBeginMember &>);
static_assert(!std::is_invocable_v<RangeCBeginT, NonConstBeginMember const&>);

struct EnabledBorrowingBeginMember {
  constexpr int *begin() const { return &globalBuff[0]; }
};
template<>
inline constexpr bool std::ranges::enable_borrowed_range<EnabledBorrowingBeginMember> = true;

constexpr bool testBeginMember() {
  BeginMember a;
  assert(std::ranges::begin(a) == &a.x);
  assert(std::ranges::cbegin(a) == &a.x);

  NonConstBeginMember b;
  assert(std::ranges::begin(b) == &b.x);

  EnabledBorrowingBeginMember c;
  assert(std::ranges::begin(std::move(c)) == &globalBuff[0]);

  BeginMemberFunction d;
  assert(std::ranges::begin(d) == &d.x);
  assert(std::ranges::cbegin(d) == &d.x);

  EmptyPtrBeginMember e;
  assert(std::ranges::begin(e) == &e.x);
  assert(std::ranges::cbegin(e) == &e.x);

  return true;
}


struct BeginFunction {
  int x;
  friend constexpr const int *begin(BeginFunction const& bf) { return &bf.x; }
};
static_assert( std::is_invocable_v<RangeBeginT,  BeginFunction const&>);
static_assert(!std::is_invocable_v<RangeBeginT,  BeginFunction &&>);
static_assert(!std::is_invocable_v<RangeBeginT,  BeginFunction &>);
static_assert( std::is_invocable_v<RangeCBeginT, BeginFunction const&>);
static_assert( std::is_invocable_v<RangeCBeginT, BeginFunction &>);

struct BeginFunctionWithDataMember {
  int x;
  int begin;
  friend constexpr const int *begin(BeginFunctionWithDataMember const& bf) { return &bf.x; }
};

struct BeginFunctionWithPrivateBeginMember {
  int y;
  friend constexpr const int *begin(BeginFunctionWithPrivateBeginMember const& bf) { return &bf.y; }
private:
  const int *begin() const;
};

struct BeginFunctionReturnsEmptyPtr {
  struct Empty {};
  Empty x;
  friend constexpr const Empty *begin(BeginFunctionReturnsEmptyPtr const& bf) { return &bf.x; }
};

struct BeginFunctionByValue {
  friend constexpr int *begin(BeginFunctionByValue) { return &globalBuff[1]; }
};
static_assert(!std::is_invocable_v<RangeCBeginT, BeginFunctionByValue>);

struct BeginFunctionEnabledBorrowing {
  friend constexpr int *begin(BeginFunctionEnabledBorrowing) { return &globalBuff[2]; }
};
template<>
inline constexpr bool std::ranges::enable_borrowed_range<BeginFunctionEnabledBorrowing> = true;

struct BeginFunctionReturnsInt {
  friend int begin(BeginFunctionReturnsInt const&);
};
static_assert(!std::is_invocable_v<RangeBeginT, BeginFunctionReturnsInt const&>);

struct BeginFunctionReturnsVoidPtr {
  friend void *begin(BeginFunctionReturnsVoidPtr const&);
};
static_assert(!std::is_invocable_v<RangeBeginT, BeginFunctionReturnsVoidPtr const&>);

struct BeginFunctionReturnsEmpty {
  struct Empty {};
  friend Empty begin(BeginFunctionReturnsEmpty const&);
};
static_assert(!std::is_invocable_v<RangeBeginT, BeginFunctionReturnsEmpty const&>);

struct BeginFunctionReturnsPtrConvertible {
  struct iterator { operator int*() const; };
  friend iterator begin(BeginFunctionReturnsPtrConvertible const&);
};
static_assert(!std::is_invocable_v<RangeBeginT, BeginFunctionReturnsPtrConvertible const&>);

constexpr bool testBeginFunction() {
  BeginFunction a{};
  const BeginFunction aa{};
  static_assert(!std::invocable<decltype(std::ranges::begin), decltype((a))>);
  assert(std::ranges::begin(aa) == &aa.x);
  assert(std::ranges::cbegin(a) == &a.x);
  assert(std::ranges::cbegin(aa) == &aa.x);

  BeginFunctionByValue b{};
  const BeginFunctionByValue bb{};
  assert(std::ranges::begin(b) == &globalBuff[1]);
  assert(std::ranges::begin(bb) == &globalBuff[1]);
  assert(std::ranges::cbegin(b) == &globalBuff[1]);
  assert(std::ranges::cbegin(bb) == &globalBuff[1]);

  BeginFunctionEnabledBorrowing c{};
  const BeginFunctionEnabledBorrowing cc{};
  assert(std::ranges::begin(std::move(c)) == &globalBuff[2]);
  static_assert(!std::invocable<decltype(std::ranges::cbegin), decltype(std::move(c))>);
  assert(std::ranges::begin(std::move(cc)) == &globalBuff[2]);
  assert(std::ranges::cbegin(std::move(cc)) == &globalBuff[2]);

  BeginFunctionReturnsEmptyPtr d{};
  const BeginFunctionReturnsEmptyPtr dd{};
  static_assert(!std::invocable<decltype(std::ranges::begin), decltype((d))>);
  assert(std::ranges::begin(dd) == &dd.x);
  assert(std::ranges::cbegin(d) == &d.x);
  assert(std::ranges::cbegin(dd) == &dd.x);

  BeginFunctionWithDataMember e{};
  const BeginFunctionWithDataMember ee{};
  static_assert(!std::invocable<decltype(std::ranges::begin), decltype((e))>);
  assert(std::ranges::begin(ee) == &ee.x);
  assert(std::ranges::cbegin(e) == &e.x);
  assert(std::ranges::cbegin(ee) == &ee.x);

  BeginFunctionWithPrivateBeginMember f{};
  const BeginFunctionWithPrivateBeginMember ff{};
  static_assert(!std::invocable<decltype(std::ranges::begin), decltype((f))>);
  assert(std::ranges::begin(ff) == &ff.y);
  assert(std::ranges::cbegin(f) == &f.y);
  assert(std::ranges::cbegin(ff) == &ff.y);

  return true;
}


ASSERT_NOEXCEPT(std::ranges::begin(std::declval<int (&)[10]>()));
ASSERT_NOEXCEPT(std::ranges::cbegin(std::declval<int (&)[10]>()));

template<class T>
struct NoThrowMemberBegin {
  T begin() const noexcept;
};
ASSERT_NOEXCEPT(std::ranges::begin(std::declval<NoThrowMemberBegin<int*>&>()));
ASSERT_NOEXCEPT(std::ranges::cbegin(std::declval<NoThrowMemberBegin<int*>&>()));
ASSERT_NOT_NOEXCEPT(std::ranges::begin(std::declval<NoThrowMemberBegin<ThrowingIterator<int>>&>()));
ASSERT_NOT_NOEXCEPT(std::ranges::cbegin(std::declval<NoThrowMemberBegin<ThrowingIterator<int>>&>()));

template<class T>
struct NoThrowADLBegin {
  friend T begin(NoThrowADLBegin&) noexcept { return T{}; }
  friend T begin(NoThrowADLBegin const&) noexcept { return T{}; }
};
ASSERT_NOEXCEPT(std::ranges::begin(std::declval<NoThrowADLBegin<int*>&>()));
ASSERT_NOEXCEPT(std::ranges::cbegin(std::declval<NoThrowADLBegin<int*>&>()));
ASSERT_NOT_NOEXCEPT(std::ranges::begin(std::declval<NoThrowADLBegin<ThrowingIterator<int>>&>()));
ASSERT_NOT_NOEXCEPT(std::ranges::cbegin(std::declval<NoThrowADLBegin<ThrowingIterator<int>>&>()));


int main(int, char**) {
  testArray();
  static_assert(testArray());

  testBeginMember();
  static_assert(testBeginMember());

  testBeginFunction();
  static_assert(testBeginFunction());

  return 0;
}
