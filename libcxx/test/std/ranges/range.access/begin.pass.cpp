//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// std::ranges::begin
// std::ranges::cbegin

#include <ranges>

#include <cassert>
#include <utility>
#include "test_macros.h"
#include "test_iterators.h"

using RangeBeginT = decltype(std::ranges::begin);
using RangeCBeginT = decltype(std::ranges::cbegin);

static int globalBuff[8];

static_assert(!std::is_invocable_v<RangeBeginT, int (&&)[10]>);
static_assert( std::is_invocable_v<RangeBeginT, int (&)[10]>);
static_assert(!std::is_invocable_v<RangeBeginT, int (&&)[]>);
static_assert( std::is_invocable_v<RangeBeginT, int (&)[]>);
static_assert(!std::is_invocable_v<RangeCBeginT, int (&&)[10]>);
static_assert( std::is_invocable_v<RangeCBeginT, int (&)[10]>);
static_assert(!std::is_invocable_v<RangeCBeginT, int (&&)[]>);
static_assert( std::is_invocable_v<RangeCBeginT, int (&)[]>);

struct Incomplete;
static_assert(!std::is_invocable_v<RangeBeginT, Incomplete(&&)[]>);
static_assert(!std::is_invocable_v<RangeBeginT, const Incomplete(&&)[]>);
static_assert(!std::is_invocable_v<RangeCBeginT, Incomplete(&&)[]>);
static_assert(!std::is_invocable_v<RangeCBeginT, const Incomplete(&&)[]>);

static_assert(!std::is_invocable_v<RangeBeginT, Incomplete(&&)[10]>);
static_assert(!std::is_invocable_v<RangeBeginT, const Incomplete(&&)[10]>);
static_assert(!std::is_invocable_v<RangeCBeginT, Incomplete(&&)[10]>);
static_assert(!std::is_invocable_v<RangeCBeginT, const Incomplete(&&)[10]>);

// This case is IFNDR; we handle it SFINAE-friendly.
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeBeginT, Incomplete(&)[]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeBeginT, const Incomplete(&)[]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCBeginT, Incomplete(&)[]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCBeginT, const Incomplete(&)[]>);

// This case is IFNDR; we handle it SFINAE-friendly.
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeBeginT, Incomplete(&)[10]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeBeginT, const Incomplete(&)[10]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCBeginT, Incomplete(&)[10]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCBeginT, const Incomplete(&)[10]>);

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
static_assert(!std::is_invocable_v<RangeCBeginT, BeginMember const&&>);

constexpr bool testReturnTypes() {
  {
    int *x[2];
    ASSERT_SAME_TYPE(decltype(std::ranges::begin(x)), int**);
    ASSERT_SAME_TYPE(decltype(std::ranges::cbegin(x)), int* const*);
  }
  {
    int x[2][2];
    ASSERT_SAME_TYPE(decltype(std::ranges::begin(x)), int(*)[2]);
    ASSERT_SAME_TYPE(decltype(std::ranges::cbegin(x)), const int(*)[2]);
  }
  {
    struct Different {
      char*& begin();
      short*& begin() const;
    } x;
    ASSERT_SAME_TYPE(decltype(std::ranges::begin(x)), char*);
    ASSERT_SAME_TYPE(decltype(std::ranges::cbegin(x)), short*);
  }
  return true;
}

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
static_assert(!std::is_invocable_v<RangeCBeginT, NonConstBeginMember &>);
static_assert(!std::is_invocable_v<RangeCBeginT, NonConstBeginMember const&>);

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
  assert(std::ranges::begin(a) == &a.x);
  assert(std::ranges::cbegin(a) == &a.x);
  static_assert(!std::is_invocable_v<RangeBeginT, BeginMember&&>);
  static_assert(!std::is_invocable_v<RangeCBeginT, BeginMember&&>);

  NonConstBeginMember b;
  assert(std::ranges::begin(b) == &b.x);
  static_assert(!std::is_invocable_v<RangeCBeginT, NonConstBeginMember&>);

  EnabledBorrowingBeginMember c;
  assert(std::ranges::begin(c) == &globalBuff[0]);
  assert(std::ranges::cbegin(c) == &globalBuff[0]);
  assert(std::ranges::begin(std::move(c)) == &globalBuff[0]);
  assert(std::ranges::cbegin(std::move(c)) == &globalBuff[0]);

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
static_assert(!std::is_invocable_v<RangeCBeginT, BeginFunctionByValue>);

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
  static_assert(!std::invocable<RangeBeginT, decltype((a))>);
  assert(std::ranges::cbegin(a) == &a.x);
  assert(std::ranges::begin(aa) == &aa.x);
  assert(std::ranges::cbegin(aa) == &aa.x);

  BeginFunctionByValue b{};
  const BeginFunctionByValue bb{};
  assert(std::ranges::begin(b) == &globalBuff[1]);
  assert(std::ranges::cbegin(b) == &globalBuff[1]);
  assert(std::ranges::begin(bb) == &globalBuff[1]);
  assert(std::ranges::cbegin(bb) == &globalBuff[1]);

  BeginFunctionEnabledBorrowing c{};
  const BeginFunctionEnabledBorrowing cc{};
  assert(std::ranges::begin(std::move(c)) == &globalBuff[2]);
  assert(std::ranges::cbegin(std::move(c)) == &globalBuff[2]);
  assert(std::ranges::begin(std::move(cc)) == &globalBuff[2]);
  assert(std::ranges::cbegin(std::move(cc)) == &globalBuff[2]);

  BeginFunctionReturnsEmptyPtr d{};
  const BeginFunctionReturnsEmptyPtr dd{};
  static_assert(!std::invocable<RangeBeginT, decltype((d))>);
  assert(std::ranges::cbegin(d) == &d.x);
  assert(std::ranges::begin(dd) == &dd.x);
  assert(std::ranges::cbegin(dd) == &dd.x);

  BeginFunctionWithDataMember e{};
  const BeginFunctionWithDataMember ee{};
  static_assert(!std::invocable<RangeBeginT, decltype((e))>);
  assert(std::ranges::begin(ee) == &ee.x);
  assert(std::ranges::cbegin(e) == &e.x);
  assert(std::ranges::cbegin(ee) == &ee.x);

  BeginFunctionWithPrivateBeginMember f{};
  const BeginFunctionWithPrivateBeginMember ff{};
  static_assert(!std::invocable<RangeBeginT, decltype((f))>);
  assert(std::ranges::cbegin(f) == &f.y);
  assert(std::ranges::begin(ff) == &ff.y);
  assert(std::ranges::cbegin(ff) == &ff.y);

  return true;
}


ASSERT_NOEXCEPT(std::ranges::begin(std::declval<int (&)[10]>()));
ASSERT_NOEXCEPT(std::ranges::cbegin(std::declval<int (&)[10]>()));

struct NoThrowMemberBegin {
  ThrowingIterator<int> begin() const noexcept; // auto(t.begin()) doesn't throw
} ntmb;
static_assert(noexcept(std::ranges::begin(ntmb)));
static_assert(noexcept(std::ranges::cbegin(ntmb)));

struct NoThrowADLBegin {
  friend ThrowingIterator<int> begin(NoThrowADLBegin&) noexcept;  // auto(begin(t)) doesn't throw
  friend ThrowingIterator<int> begin(const NoThrowADLBegin&) noexcept;
} ntab;
static_assert(noexcept(std::ranges::begin(ntab)));
static_assert(noexcept(std::ranges::cbegin(ntab)));

struct NoThrowMemberBeginReturnsRef {
  ThrowingIterator<int>& begin() const noexcept; // auto(t.begin()) may throw
} ntmbrr;
static_assert(!noexcept(std::ranges::begin(ntmbrr)));
static_assert(!noexcept(std::ranges::cbegin(ntmbrr)));

struct BeginReturnsArrayRef {
    auto begin() const noexcept -> int(&)[10];
} brar;
static_assert(noexcept(std::ranges::begin(brar)));
static_assert(noexcept(std::ranges::cbegin(brar)));

// Test ADL-proofing.
struct Incomplete;
template<class T> struct Holder { T t; };
static_assert(!std::is_invocable_v<RangeBeginT, Holder<Incomplete>*>);
static_assert(!std::is_invocable_v<RangeBeginT, Holder<Incomplete>*&>);
static_assert(!std::is_invocable_v<RangeCBeginT, Holder<Incomplete>*>);
static_assert(!std::is_invocable_v<RangeCBeginT, Holder<Incomplete>*&>);

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
