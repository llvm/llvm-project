//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::ranges::begin
// std::ranges::cbegin // until C++23

#include <ranges>

#include <cassert>
#include <utility>
#include "test_macros.h"
#include "test_iterators.h"

using RangeBeginT = decltype(std::ranges::begin);
#if TEST_STD_VER < 23
using RangeCBeginT = decltype(std::ranges::cbegin);
#endif // TEST_STD_VER < 23

static int globalBuff[8];

static_assert(!std::is_invocable_v<RangeBeginT, int (&&)[10]>);
static_assert( std::is_invocable_v<RangeBeginT, int (&)[10]>);
static_assert(!std::is_invocable_v<RangeBeginT, int (&&)[]>);
static_assert( std::is_invocable_v<RangeBeginT, int (&)[]>);
#if TEST_STD_VER < 23
static_assert(!std::is_invocable_v<RangeCBeginT, int (&&)[10]>);
static_assert( std::is_invocable_v<RangeCBeginT, int (&)[10]>);
static_assert(!std::is_invocable_v<RangeCBeginT, int (&&)[]>);
static_assert( std::is_invocable_v<RangeCBeginT, int (&)[]>);
#endif // TEST_STD_VER < 23

struct Incomplete;
static_assert(!std::is_invocable_v<RangeBeginT, Incomplete(&&)[]>);
static_assert(!std::is_invocable_v<RangeBeginT, const Incomplete(&&)[]>);
#if TEST_STD_VER < 23
static_assert(!std::is_invocable_v<RangeCBeginT, Incomplete(&&)[]>);
static_assert(!std::is_invocable_v<RangeCBeginT, const Incomplete(&&)[]>);
#endif // TEST_STD_VER < 23

static_assert(!std::is_invocable_v<RangeBeginT, Incomplete(&&)[10]>);
static_assert(!std::is_invocable_v<RangeBeginT, const Incomplete(&&)[10]>);
#if TEST_STD_VER < 23
static_assert(!std::is_invocable_v<RangeCBeginT, Incomplete(&&)[10]>);
static_assert(!std::is_invocable_v<RangeCBeginT, const Incomplete(&&)[10]>);
#endif // TEST_STD_VER < 23

// This case is IFNDR; we handle it SFINAE-friendly.
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeBeginT, Incomplete(&)[]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeBeginT, const Incomplete(&)[]>);
#if TEST_STD_VER < 23
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCBeginT, Incomplete(&)[]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCBeginT, const Incomplete(&)[]>);
#endif // TEST_STD_VER < 23

// This case is IFNDR; we handle it SFINAE-friendly.
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeBeginT, Incomplete(&)[10]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeBeginT, const Incomplete(&)[10]>);
#if TEST_STD_VER < 23
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCBeginT, Incomplete(&)[10]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCBeginT, const Incomplete(&)[10]>);
#endif // TEST_STD_VER < 23

struct BeginMember {
  int x;
  constexpr const int *begin() const { return &x; }
};

// Ensure that we can't call with rvalues with borrowing disabled.
static_assert( std::is_invocable_v<RangeBeginT, BeginMember &>);
static_assert(!std::is_invocable_v<RangeBeginT, BeginMember &&>);
static_assert( std::is_invocable_v<RangeBeginT, BeginMember const&>);
static_assert(!std::is_invocable_v<RangeBeginT, BeginMember const&&>);
#if TEST_STD_VER < 23
static_assert( std::is_invocable_v<RangeCBeginT, BeginMember &>);
static_assert(!std::is_invocable_v<RangeCBeginT, BeginMember &&>);
static_assert( std::is_invocable_v<RangeCBeginT, BeginMember const&>);
static_assert(!std::is_invocable_v<RangeCBeginT, BeginMember const&&>);
#endif // TEST_STD_VER < 23

constexpr bool testReturnTypes() {
  int* a[2];
  int b[2][2];
  struct Different {
    char*& begin();
    short*& begin() const;
  } c;

  ASSERT_SAME_TYPE(decltype(std::ranges::begin(a)), int**);
  ASSERT_SAME_TYPE(decltype(std::ranges::begin(b)), int(*)[2]);
  ASSERT_SAME_TYPE(decltype(std::ranges::begin(c)), char*);

#if TEST_STD_VER < 23
  ASSERT_SAME_TYPE(decltype(std::ranges::cbegin(a)), int* const*);
  ASSERT_SAME_TYPE(decltype(std::ranges::cbegin(b)), const int(*)[2]);
  ASSERT_SAME_TYPE(decltype(std::ranges::cbegin(c)), short*);
#endif // TEST_STD_VER < 23

  return true;
}

constexpr bool testArray() {
  int a[2];
  int b[2][2];
  BeginMember c[2];

  assert(std::ranges::begin(a) == a);
  assert(std::ranges::begin(b) == b);
  assert(std::ranges::begin(c) == c);

#if TEST_STD_VER < 23
  assert(std::ranges::cbegin(a) == a);
  assert(std::ranges::cbegin(b) == b);
  assert(std::ranges::cbegin(c) == c);
#endif // TEST_STD_VER < 23

  return true;
}

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
#if TEST_STD_VER < 23
static_assert(!std::is_invocable_v<RangeCBeginT, NonConstBeginMember &>);
static_assert(!std::is_invocable_v<RangeCBeginT, NonConstBeginMember const&>);
#endif // TEST_STD_VER < 23

struct EnabledBorrowingBeginMember {
  constexpr int *begin() const { return &globalBuff[0]; }
};
template<>
inline constexpr bool std::ranges::enable_borrowed_range<EnabledBorrowingBeginMember> = true;

struct BeginMemberFunction {
  int x;
  constexpr const int *begin() const { return &x; }
  friend int *begin(BeginMemberFunction const&);
};

struct EmptyPtrBeginMember {
  struct Empty {};
  Empty x;
  constexpr const Empty *begin() const { return &x; }
};

constexpr bool testBeginMember() {
  BeginMember a;
  NonConstBeginMember b;
  EnabledBorrowingBeginMember c;
  BeginMemberFunction d;
  EmptyPtrBeginMember e;

  assert(std::ranges::begin(a) == &a.x);
  static_assert(!std::is_invocable_v<RangeBeginT, BeginMember&&>);
  assert(std::ranges::begin(b) == &b.x);
  assert(std::ranges::begin(c) == &globalBuff[0]);
  assert(std::ranges::begin(std::move(c)) == &globalBuff[0]);
  assert(std::ranges::begin(d) == &d.x);
  assert(std::ranges::begin(e) == &e.x);

#if TEST_STD_VER < 23
  assert(std::ranges::cbegin(a) == &a.x);
  static_assert(!std::is_invocable_v<RangeCBeginT, BeginMember&&>);
  static_assert(!std::is_invocable_v<RangeCBeginT, NonConstBeginMember&>);
  assert(std::ranges::cbegin(c) == &globalBuff[0]);
  assert(std::ranges::cbegin(std::move(c)) == &globalBuff[0]);
  assert(std::ranges::cbegin(d) == &d.x);
  assert(std::ranges::cbegin(e) == &e.x);
#endif // TEST_STD_VER < 23

  return true;
}


struct BeginFunction {
  int x;
  friend constexpr const int *begin(BeginFunction const& bf) { return &bf.x; }
};
static_assert( std::is_invocable_v<RangeBeginT,  BeginFunction const&>);
static_assert(!std::is_invocable_v<RangeBeginT,  BeginFunction &&>);
static_assert(std::is_invocable_v<RangeBeginT, BeginFunction&>); // Ill-formed before P2602R2 Poison Pills are Too Toxic
#if TEST_STD_VER < 23
static_assert( std::is_invocable_v<RangeCBeginT, BeginFunction const&>);
static_assert( std::is_invocable_v<RangeCBeginT, BeginFunction &>);
#endif // TEST_STD_VER < 23

struct BeginFunctionReturnsInt {
  friend int begin(BeginFunctionReturnsInt const&);
};
static_assert(!std::is_invocable_v<RangeBeginT, BeginFunctionReturnsInt const&>);

struct BeginFunctionReturnsVoidPtr {
  friend void *begin(BeginFunctionReturnsVoidPtr const&);
};
static_assert(!std::is_invocable_v<RangeBeginT, BeginFunctionReturnsVoidPtr const&>);

struct BeginFunctionReturnsPtrConvertible {
  struct iterator { operator int*() const; };
  friend iterator begin(BeginFunctionReturnsPtrConvertible const&);
};
static_assert(!std::is_invocable_v<RangeBeginT, BeginFunctionReturnsPtrConvertible const&>);

struct BeginFunctionByValue {
  friend constexpr int *begin(BeginFunctionByValue) { return &globalBuff[1]; }
};
#if TEST_STD_VER < 23
static_assert(!std::is_invocable_v<RangeCBeginT, BeginFunctionByValue>);
#endif // TEST_STD_VER < 23

struct BeginFunctionEnabledBorrowing {
  friend constexpr int *begin(BeginFunctionEnabledBorrowing) { return &globalBuff[2]; }
};
template<>
inline constexpr bool std::ranges::enable_borrowed_range<BeginFunctionEnabledBorrowing> = true;

struct BeginFunctionReturnsEmptyPtr {
  struct Empty {};
  Empty x;
  friend constexpr const Empty *begin(BeginFunctionReturnsEmptyPtr const& bf) { return &bf.x; }
};

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

constexpr bool testBeginFunction() {
  BeginFunction a{};
  const BeginFunction aa{};
  BeginFunctionByValue b{};
  const BeginFunctionByValue bb{};
  BeginFunctionEnabledBorrowing c{};
  const BeginFunctionEnabledBorrowing cc{};
  BeginFunctionReturnsEmptyPtr d{};
  const BeginFunctionReturnsEmptyPtr dd{};
  BeginFunctionWithDataMember e{};
  const BeginFunctionWithDataMember ee{};
  BeginFunctionWithPrivateBeginMember f{};
  const BeginFunctionWithPrivateBeginMember ff{};

  assert(std::ranges::begin(a) == &a.x); // Ill-formed before P2602R2 Poison Pills are Too Toxic
  assert(std::ranges::begin(aa) == &aa.x);
  assert(std::ranges::begin(b) == &globalBuff[1]);
  assert(std::ranges::begin(bb) == &globalBuff[1]);
  assert(std::ranges::begin(std::move(c)) == &globalBuff[2]);
  assert(std::ranges::begin(std::move(cc)) == &globalBuff[2]);
  assert(std::ranges::begin(d) == &d.x); // Ill-formed before P2602R2 Poison Pills are Too Toxic
  assert(std::ranges::begin(dd) == &dd.x);
  assert(std::ranges::begin(e) == &e.x); // Ill-formed before P2602R2 Poison Pills are Too Toxic
  assert(std::ranges::begin(ee) == &ee.x);
  assert(std::ranges::begin(f) == &f.y); // Ill-formed before P2602R2 Poison Pills are Too Toxic
  assert(std::ranges::begin(ff) == &ff.y);

#if TEST_STD_VER < 23
  assert(std::ranges::cbegin(a) == &a.x);
  assert(std::ranges::cbegin(aa) == &aa.x);
  assert(std::ranges::cbegin(b) == &globalBuff[1]);
  assert(std::ranges::cbegin(bb) == &globalBuff[1]);
  assert(std::ranges::cbegin(std::move(c)) == &globalBuff[2]);
  assert(std::ranges::cbegin(std::move(cc)) == &globalBuff[2]);
  assert(std::ranges::cbegin(d) == &d.x);
  assert(std::ranges::cbegin(dd) == &dd.x);
  assert(std::ranges::cbegin(e) == &e.x);
  assert(std::ranges::cbegin(ee) == &ee.x);
  assert(std::ranges::cbegin(f) == &f.y);
  assert(std::ranges::cbegin(ff) == &ff.y);
#endif // TEST_STD_VER < 23

  return true;
}


ASSERT_NOEXCEPT(std::ranges::begin(std::declval<int (&)[10]>()));
#if TEST_STD_VER < 23
ASSERT_NOEXCEPT(std::ranges::cbegin(std::declval<int (&)[10]>()));
#endif // TEST_STD_VER < 23

struct NoThrowMemberBegin {
  ThrowingIterator<int> begin() const noexcept; // auto(t.begin()) doesn't throw
} ntmb;
static_assert(noexcept(std::ranges::begin(ntmb)));
#if TEST_STD_VER < 23
static_assert(noexcept(std::ranges::cbegin(ntmb)));
#endif // TEST_STD_VER < 23

struct NoThrowADLBegin {
  friend ThrowingIterator<int> begin(NoThrowADLBegin&) noexcept;  // auto(begin(t)) doesn't throw
  friend ThrowingIterator<int> begin(const NoThrowADLBegin&) noexcept;
} ntab;
static_assert(noexcept(std::ranges::begin(ntab)));
#if TEST_STD_VER < 23
static_assert(noexcept(std::ranges::cbegin(ntab)));
#endif // TEST_STD_VER < 23

struct NoThrowMemberBeginReturnsRef {
  ThrowingIterator<int>& begin() const noexcept; // auto(t.begin()) may throw
} ntmbrr;
static_assert(!noexcept(std::ranges::begin(ntmbrr)));
#if TEST_STD_VER < 23
static_assert(!noexcept(std::ranges::cbegin(ntmbrr)));
#endif // TEST_STD_VER < 23

struct BeginReturnsArrayRef {
    auto begin() const noexcept -> int(&)[10];
} brar;
static_assert(noexcept(std::ranges::begin(brar)));
#if TEST_STD_VER < 23
static_assert(noexcept(std::ranges::cbegin(brar)));
#endif // TEST_STD_VER < 23

// Test ADL-proofing.
struct Incomplete;
template<class T> struct Holder { T t; };
static_assert(!std::is_invocable_v<RangeBeginT, Holder<Incomplete>*>);
static_assert(!std::is_invocable_v<RangeBeginT, Holder<Incomplete>*&>);
#if TEST_STD_VER < 23
static_assert(!std::is_invocable_v<RangeCBeginT, Holder<Incomplete>*>);
static_assert(!std::is_invocable_v<RangeCBeginT, Holder<Incomplete>*&>);
#endif // TEST_STD_VER < 23

int main(int, char**) {
  static_assert(testReturnTypes());

  testArray();
  static_assert(testArray());

  testBeginMember();
  static_assert(testBeginMember());

  testBeginFunction();
  static_assert(testBeginFunction());

  return 0;
}
