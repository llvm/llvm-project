//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::ranges::rbegin
// std::ranges::crbegin // until C++23

#include <ranges>

#include <cassert>
#include <utility>
#include "test_macros.h"
#include "test_iterators.h"

using RangeRBeginT = decltype(std::ranges::rbegin);
#if TEST_STD_VER < 23
using RangeCRBeginT = decltype(std::ranges::crbegin);
#endif // TEST_STD_VER < 23

static int globalBuff[8];

static_assert(!std::is_invocable_v<RangeRBeginT, int (&&)[10]>);
static_assert( std::is_invocable_v<RangeRBeginT, int (&)[10]>);
static_assert(!std::is_invocable_v<RangeRBeginT, int (&&)[]>);
static_assert(!std::is_invocable_v<RangeRBeginT, int (&)[]>);
#if TEST_STD_VER < 23
static_assert(!std::is_invocable_v<RangeCRBeginT, int (&&)[10]>);
static_assert( std::is_invocable_v<RangeCRBeginT, int (&)[10]>);
static_assert(!std::is_invocable_v<RangeCRBeginT, int (&&)[]>);
static_assert(!std::is_invocable_v<RangeCRBeginT, int (&)[]>);
#endif // TEST_STD_VER < 23

struct Incomplete;

static_assert(!std::is_invocable_v<RangeRBeginT, Incomplete(&&)[]>);
static_assert(!std::is_invocable_v<RangeRBeginT, const Incomplete (&&)[]>);
static_assert(!std::is_invocable_v<RangeRBeginT, Incomplete(&&)[10]>);
static_assert(!std::is_invocable_v<RangeRBeginT, const Incomplete(&&)[10]>);

#if TEST_STD_VER < 23
static_assert(!std::is_invocable_v<RangeCRBeginT, Incomplete (&&)[]>);
static_assert(!std::is_invocable_v<RangeCRBeginT, const Incomplete (&&)[]>);
static_assert(!std::is_invocable_v<RangeCRBeginT, Incomplete(&&)[10]>);
static_assert(!std::is_invocable_v<RangeCRBeginT, const Incomplete(&&)[10]>);
#endif // TEST_STD_VER < 23

// This case is IFNDR; we handle it SFINAE-friendly.
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeRBeginT, Incomplete(&)[]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeRBeginT, const Incomplete(&)[]>);
#if TEST_STD_VER < 23
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCRBeginT, Incomplete(&)[]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCRBeginT, const Incomplete(&)[]>);
#endif // TEST_STD_VER < 23

// This case is IFNDR; we handle it SFINAE-friendly.
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeRBeginT, Incomplete(&)[10]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeRBeginT, const Incomplete(&)[10]>);
#if TEST_STD_VER < 23
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCRBeginT, Incomplete(&)[10]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCRBeginT, const Incomplete(&)[10]>);
#endif // TEST_STD_VER < 23

struct RBeginMember {
  int x;
  constexpr const int *rbegin() const { return &x; }
};

// Ensure that we can't call with rvalues with borrowing disabled.
static_assert( std::is_invocable_v<RangeRBeginT, RBeginMember &>);
static_assert(!std::is_invocable_v<RangeRBeginT, RBeginMember &&>);
static_assert( std::is_invocable_v<RangeRBeginT, RBeginMember const&>);
static_assert(!std::is_invocable_v<RangeRBeginT, RBeginMember const&&>);
#if TEST_STD_VER < 23
static_assert( std::is_invocable_v<RangeCRBeginT, RBeginMember &>);
static_assert(!std::is_invocable_v<RangeCRBeginT, RBeginMember &&>);
static_assert( std::is_invocable_v<RangeCRBeginT, RBeginMember const&>);
static_assert(!std::is_invocable_v<RangeCRBeginT, RBeginMember const&&>);
#endif // TEST_STD_VER < 23

constexpr bool testReturnTypes() {
  int* a[2];
  int b[2][2];
  struct Different {
    char*& rbegin();
    short*& rbegin() const;
  } c;

  ASSERT_SAME_TYPE(decltype(std::ranges::rbegin(a)), std::reverse_iterator<int**>);
  ASSERT_SAME_TYPE(decltype(std::ranges::rbegin(b)), std::reverse_iterator<int(*)[2]>);
  ASSERT_SAME_TYPE(decltype(std::ranges::rbegin(c)), char*);

#if TEST_STD_VER < 23
  ASSERT_SAME_TYPE(decltype(std::ranges::crbegin(a)), std::reverse_iterator<int* const*>);
  ASSERT_SAME_TYPE(decltype(std::ranges::crbegin(b)), std::reverse_iterator<const int(*)[2]>);
  ASSERT_SAME_TYPE(decltype(std::ranges::crbegin(c)), short*);
#endif // TEST_STD_VER < 23

  return true;
}

constexpr bool testArray() {
  int a[2];
  int b[2][2];
  RBeginMember c[2];

  assert(std::ranges::rbegin(a).base() == a + 2);
  assert(std::ranges::rbegin(b).base() == b + 2);
  assert(std::ranges::rbegin(c).base() == c + 2);

#if TEST_STD_VER < 23
  assert(std::ranges::crbegin(a).base() == a + 2);
  assert(std::ranges::crbegin(b).base() == b + 2);
  assert(std::ranges::crbegin(c).base() == c + 2);
#endif // TEST_STD_VER < 23

  return true;
}

struct RBeginMemberReturnsInt {
  int rbegin() const;
};
static_assert(!std::is_invocable_v<RangeRBeginT, RBeginMemberReturnsInt const&>);

struct RBeginMemberReturnsVoidPtr {
  const void *rbegin() const;
};
static_assert(!std::is_invocable_v<RangeRBeginT, RBeginMemberReturnsVoidPtr const&>);

struct PtrConvertibleRBeginMember {
  struct iterator { operator int*() const; };
  iterator rbegin() const;
};
static_assert(!std::is_invocable_v<RangeRBeginT, PtrConvertibleRBeginMember const&>);

struct NonConstRBeginMember {
  int x;
  constexpr int* rbegin() { return &x; }
};
static_assert( std::is_invocable_v<RangeRBeginT,  NonConstRBeginMember &>);
static_assert(!std::is_invocable_v<RangeRBeginT,  NonConstRBeginMember const&>);
#if TEST_STD_VER < 23
static_assert(!std::is_invocable_v<RangeCRBeginT, NonConstRBeginMember &>);
static_assert(!std::is_invocable_v<RangeCRBeginT, NonConstRBeginMember const&>);
#endif // TEST_STD_VER < 23

struct EnabledBorrowingRBeginMember {
  constexpr int *rbegin() const { return globalBuff; }
};
template<>
inline constexpr bool std::ranges::enable_borrowed_range<EnabledBorrowingRBeginMember> = true;

struct RBeginMemberFunction {
  int x;
  constexpr const int *rbegin() const { return &x; }
  friend int* rbegin(RBeginMemberFunction const&);
};

struct EmptyPtrRBeginMember {
  struct Empty {};
  Empty x;
  constexpr const Empty* rbegin() const { return &x; }
};

constexpr bool testRBeginMember() {
  RBeginMember a;
  NonConstRBeginMember b;
  EnabledBorrowingRBeginMember c;
  RBeginMemberFunction d;
  EmptyPtrRBeginMember e;

  assert(std::ranges::rbegin(a) == &a.x);
  static_assert(!std::is_invocable_v<RangeRBeginT, RBeginMember&&>);
  assert(std::ranges::rbegin(b) == &b.x);
  assert(std::ranges::rbegin(c) == globalBuff);
  assert(std::ranges::rbegin(std::move(c)) == globalBuff);
  assert(std::ranges::rbegin(d) == &d.x);
  assert(std::ranges::rbegin(e) == &e.x);

#if TEST_STD_VER < 23
  assert(std::ranges::crbegin(a) == &a.x);
  static_assert(!std::is_invocable_v<RangeCRBeginT, RBeginMember&&>);
  static_assert(!std::is_invocable_v<RangeCRBeginT, NonConstRBeginMember&>);
  assert(std::ranges::crbegin(c) == globalBuff);
  assert(std::ranges::crbegin(std::move(c)) == globalBuff);
  assert(std::ranges::crbegin(d) == &d.x);
  assert(std::ranges::crbegin(e) == &e.x);
#endif // TEST_STD_VER < 23

  return true;
}


struct RBeginFunction {
  int x;
  friend constexpr const int* rbegin(RBeginFunction const& bf) { return &bf.x; }
};
static_assert( std::is_invocable_v<RangeRBeginT,  RBeginFunction const&>);
static_assert(!std::is_invocable_v<RangeRBeginT,  RBeginFunction &&>);
static_assert(
    std::is_invocable_v<RangeRBeginT, RBeginFunction&>); // Ill-formed before P2602R2 Poison Pills are Too Toxic
#if TEST_STD_VER < 23
static_assert( std::is_invocable_v<RangeCRBeginT, RBeginFunction const&>);
static_assert( std::is_invocable_v<RangeCRBeginT, RBeginFunction &>);
#endif // TEST_STD_VER < 23

struct RBeginFunctionReturnsInt {
  friend int rbegin(RBeginFunctionReturnsInt const&);
};
static_assert(!std::is_invocable_v<RangeRBeginT, RBeginFunctionReturnsInt const&>);

struct RBeginFunctionReturnsVoidPtr {
  friend void *rbegin(RBeginFunctionReturnsVoidPtr const&);
};
static_assert(!std::is_invocable_v<RangeRBeginT, RBeginFunctionReturnsVoidPtr const&>);

struct RBeginFunctionReturnsEmpty {
  struct Empty {};
  friend Empty rbegin(RBeginFunctionReturnsEmpty const&);
};
static_assert(!std::is_invocable_v<RangeRBeginT, RBeginFunctionReturnsEmpty const&>);

struct RBeginFunctionReturnsPtrConvertible {
  struct iterator { operator int*() const; };
  friend iterator rbegin(RBeginFunctionReturnsPtrConvertible const&);
};
static_assert(!std::is_invocable_v<RangeRBeginT, RBeginFunctionReturnsPtrConvertible const&>);

struct RBeginFunctionByValue {
  friend constexpr int *rbegin(RBeginFunctionByValue) { return globalBuff + 1; }
};
#if TEST_STD_VER < 23
static_assert(!std::is_invocable_v<RangeCRBeginT, RBeginFunctionByValue>);
#endif // TEST_STD_VER < 23

struct RBeginFunctionEnabledBorrowing {
  friend constexpr int *rbegin(RBeginFunctionEnabledBorrowing) { return globalBuff + 2; }
};
template<>
inline constexpr bool std::ranges::enable_borrowed_range<RBeginFunctionEnabledBorrowing> = true;

struct RBeginFunctionReturnsEmptyPtr {
  struct Empty {};
  Empty x;
  friend constexpr const Empty *rbegin(RBeginFunctionReturnsEmptyPtr const& bf) { return &bf.x; }
};

struct RBeginFunctionWithDataMember {
  int x;
  int rbegin;
  friend constexpr const int *rbegin(RBeginFunctionWithDataMember const& bf) { return &bf.x; }
};

struct RBeginFunctionWithPrivateBeginMember {
  int y;
  friend constexpr const int *rbegin(RBeginFunctionWithPrivateBeginMember const& bf) { return &bf.y; }
private:
  const int *rbegin() const;
};

constexpr bool testRBeginFunction() {
  RBeginFunction a{};
  const RBeginFunction aa{};
  RBeginFunctionByValue b{};
  const RBeginFunctionByValue bb{};
  RBeginFunctionEnabledBorrowing c{};
  const RBeginFunctionEnabledBorrowing cc{};
  RBeginFunctionReturnsEmptyPtr d{};
  const RBeginFunctionReturnsEmptyPtr dd{};
  RBeginFunctionWithDataMember e{};
  const RBeginFunctionWithDataMember ee{};
  RBeginFunctionWithPrivateBeginMember f{};
  const RBeginFunctionWithPrivateBeginMember ff{};

  assert(std::ranges::rbegin(a) == &a.x); // Ill-formed before P2602R2 Poison Pills are Too Toxic
  assert(std::ranges::rbegin(aa) == &aa.x);
  assert(std::ranges::rbegin(b) == globalBuff + 1);
  assert(std::ranges::rbegin(bb) == globalBuff + 1);
  assert(std::ranges::rbegin(std::move(c)) == globalBuff + 2);
  assert(std::ranges::rbegin(std::move(cc)) == globalBuff + 2);
  assert(std::ranges::rbegin(d) == &d.x); // Ill-formed before P2602R2 Poison Pills are Too Toxic
  assert(std::ranges::rbegin(dd) == &dd.x);
  assert(std::ranges::rbegin(e) == &e.x); // Ill-formed before P2602R2 Poison Pills are Too Toxic
  assert(std::ranges::rbegin(ee) == &ee.x);
  assert(std::ranges::rbegin(f) == &f.y); // Ill-formed before P2602R2 Poison Pills are Too Toxic
  assert(std::ranges::rbegin(ff) == &ff.y);

#if TEST_STD_VER < 23
  assert(std::ranges::crbegin(a) == &a.x);
  assert(std::ranges::crbegin(aa) == &aa.x);
  assert(std::ranges::crbegin(b) == globalBuff + 1);
  assert(std::ranges::crbegin(bb) == globalBuff + 1);
  assert(std::ranges::crbegin(std::move(c)) == globalBuff + 2);
  assert(std::ranges::crbegin(std::move(cc)) == globalBuff + 2);
  assert(std::ranges::crbegin(d) == &d.x);
  assert(std::ranges::crbegin(dd) == &dd.x);
  assert(std::ranges::crbegin(e) == &e.x);
  assert(std::ranges::crbegin(ee) == &ee.x);
  assert(std::ranges::crbegin(f) == &f.y);
  assert(std::ranges::crbegin(ff) == &ff.y);
#endif // TEST_STD_VER < 23

  return true;
}


struct MemberBeginEnd {
  int b, e;
  char cb, ce;
  constexpr bidirectional_iterator<int*> begin() { return bidirectional_iterator<int*>(&b); }
  constexpr bidirectional_iterator<int*> end() { return bidirectional_iterator<int*>(&e); }
  constexpr bidirectional_iterator<const char*> begin() const { return bidirectional_iterator<const char*>(&cb); }
  constexpr bidirectional_iterator<const char*> end() const { return bidirectional_iterator<const char*>(&ce); }
};
static_assert( std::is_invocable_v<RangeRBeginT, MemberBeginEnd&>);
static_assert( std::is_invocable_v<RangeRBeginT, MemberBeginEnd const&>);
#if TEST_STD_VER < 23
static_assert( std::is_invocable_v<RangeCRBeginT, MemberBeginEnd const&>);
#endif // TEST_STD_VER < 23

struct FunctionBeginEnd {
  int b, e;
  char cb, ce;
  friend constexpr bidirectional_iterator<int*> begin(FunctionBeginEnd& v) {
    return bidirectional_iterator<int*>(&v.b);
  }
  friend constexpr bidirectional_iterator<int*> end(FunctionBeginEnd& v) { return bidirectional_iterator<int*>(&v.e); }
  friend constexpr bidirectional_iterator<const char*> begin(const FunctionBeginEnd& v) {
    return bidirectional_iterator<const char*>(&v.cb);
  }
  friend constexpr bidirectional_iterator<const char*> end(const FunctionBeginEnd& v) {
    return bidirectional_iterator<const char*>(&v.ce);
  }
};
static_assert( std::is_invocable_v<RangeRBeginT, FunctionBeginEnd&>);
static_assert( std::is_invocable_v<RangeRBeginT, FunctionBeginEnd const&>);
#if TEST_STD_VER < 23
static_assert( std::is_invocable_v<RangeCRBeginT, FunctionBeginEnd const&>);
#endif // TEST_STD_VER < 23

struct MemberBeginFunctionEnd {
  int b, e;
  char cb, ce;
  constexpr bidirectional_iterator<int*> begin() { return bidirectional_iterator<int*>(&b); }
  friend constexpr bidirectional_iterator<int*> end(MemberBeginFunctionEnd& v) {
    return bidirectional_iterator<int*>(&v.e);
  }
  constexpr bidirectional_iterator<const char*> begin() const { return bidirectional_iterator<const char*>(&cb); }
  friend constexpr bidirectional_iterator<const char*> end(const MemberBeginFunctionEnd& v) {
    return bidirectional_iterator<const char*>(&v.ce);
  }
};
static_assert( std::is_invocable_v<RangeRBeginT, MemberBeginFunctionEnd&>);
static_assert( std::is_invocable_v<RangeRBeginT, MemberBeginFunctionEnd const&>);
#if TEST_STD_VER < 23
static_assert( std::is_invocable_v<RangeCRBeginT, MemberBeginFunctionEnd const&>);
#endif // TEST_STD_VER < 23

struct FunctionBeginMemberEnd {
  int b, e;
  char cb, ce;
  friend constexpr bidirectional_iterator<int*> begin(FunctionBeginMemberEnd& v) {
    return bidirectional_iterator<int*>(&v.b);
  }
  constexpr bidirectional_iterator<int*> end() { return bidirectional_iterator<int*>(&e); }
  friend constexpr bidirectional_iterator<const char*> begin(const FunctionBeginMemberEnd& v) {
    return bidirectional_iterator<const char*>(&v.cb);
  }
  constexpr bidirectional_iterator<const char*> end() const { return bidirectional_iterator<const char*>(&ce); }
};
static_assert( std::is_invocable_v<RangeRBeginT, FunctionBeginMemberEnd&>);
static_assert( std::is_invocable_v<RangeRBeginT, FunctionBeginMemberEnd const&>);
#if TEST_STD_VER < 23
static_assert( std::is_invocable_v<RangeCRBeginT, FunctionBeginMemberEnd const&>);
#endif // TEST_STD_VER < 23

struct MemberBeginEndDifferentTypes {
  bidirectional_iterator<int*> begin();
  bidirectional_iterator<const int*> end();
};
static_assert(!std::is_invocable_v<RangeRBeginT, MemberBeginEndDifferentTypes&>);
#if TEST_STD_VER < 23
static_assert(!std::is_invocable_v<RangeCRBeginT, MemberBeginEndDifferentTypes&>);
#endif // TEST_STD_VER < 23

struct FunctionBeginEndDifferentTypes {
  friend bidirectional_iterator<int*> begin(FunctionBeginEndDifferentTypes&);
  friend bidirectional_iterator<const int*> end(FunctionBeginEndDifferentTypes&);
};
static_assert(!std::is_invocable_v<RangeRBeginT, FunctionBeginEndDifferentTypes&>);
#if TEST_STD_VER < 23
static_assert(!std::is_invocable_v<RangeCRBeginT, FunctionBeginEndDifferentTypes&>);
#endif // TEST_STD_VER < 23

struct MemberBeginEndForwardIterators {
  forward_iterator<int*> begin();
  forward_iterator<int*> end();
};
static_assert(!std::is_invocable_v<RangeRBeginT, MemberBeginEndForwardIterators&>);
#if TEST_STD_VER < 23
static_assert(!std::is_invocable_v<RangeCRBeginT, MemberBeginEndForwardIterators&>);
#endif // TEST_STD_VER < 23

struct FunctionBeginEndForwardIterators {
  friend forward_iterator<int*> begin(FunctionBeginEndForwardIterators&);
  friend forward_iterator<int*> end(FunctionBeginEndForwardIterators&);
};
static_assert(!std::is_invocable_v<RangeRBeginT, FunctionBeginEndForwardIterators&>);
#if TEST_STD_VER < 23
static_assert(!std::is_invocable_v<RangeCRBeginT, FunctionBeginEndForwardIterators&>);
#endif // TEST_STD_VER < 23

struct MemberBeginOnly {
  bidirectional_iterator<int*> begin() const;
};
static_assert(!std::is_invocable_v<RangeRBeginT, MemberBeginOnly&>);
#if TEST_STD_VER < 23
static_assert(!std::is_invocable_v<RangeCRBeginT, MemberBeginOnly&>);
#endif // TEST_STD_VER < 23

struct FunctionBeginOnly {
  friend bidirectional_iterator<int*> begin(FunctionBeginOnly&);
};
static_assert(!std::is_invocable_v<RangeRBeginT, FunctionBeginOnly&>);
#if TEST_STD_VER < 23
static_assert(!std::is_invocable_v<RangeCRBeginT, FunctionBeginOnly&>);
#endif // TEST_STD_VER < 23

struct MemberEndOnly {
  bidirectional_iterator<int*> end() const;
};
static_assert(!std::is_invocable_v<RangeRBeginT, MemberEndOnly&>);
#if TEST_STD_VER < 23
static_assert(!std::is_invocable_v<RangeCRBeginT, MemberEndOnly&>);
#endif // TEST_STD_VER < 23

struct FunctionEndOnly {
  friend bidirectional_iterator<int*> end(FunctionEndOnly&);
};
static_assert(!std::is_invocable_v<RangeRBeginT, FunctionEndOnly&>);
#if TEST_STD_VER < 23
static_assert(!std::is_invocable_v<RangeCRBeginT, FunctionEndOnly&>);
#endif // TEST_STD_VER < 23

// Make sure there is no clash between the following cases:
// - the case that handles classes defining member `rbegin` and `rend` functions;
// - the case that handles classes defining `begin` and `end` functions returning reversible iterators.
struct MemberBeginAndRBegin {
  int* begin() const;
  int* end() const;
  int* rbegin() const;
  int* rend() const;
};
static_assert(std::is_invocable_v<RangeRBeginT, MemberBeginAndRBegin&>);
static_assert( std::same_as<std::invoke_result_t<RangeRBeginT, MemberBeginAndRBegin&>, int*>);
#if TEST_STD_VER < 23
static_assert(std::is_invocable_v<RangeCRBeginT, MemberBeginAndRBegin&>);
static_assert( std::same_as<std::invoke_result_t<RangeCRBeginT, MemberBeginAndRBegin&>, int*>);
#endif // TEST_STD_VER < 23

constexpr bool testBeginEnd() {
  MemberBeginEnd a{};
  const MemberBeginEnd aa{};
  FunctionBeginEnd b{};
  const FunctionBeginEnd bb{};
  MemberBeginFunctionEnd c{};
  const MemberBeginFunctionEnd cc{};
  FunctionBeginMemberEnd d{};
  const FunctionBeginMemberEnd dd{};

  assert(base(std::ranges::rbegin(a).base()) == &a.e);
  assert(base(std::ranges::rbegin(aa).base()) == &aa.ce);
  assert(base(std::ranges::rbegin(b).base()) == &b.e);
  assert(base(std::ranges::rbegin(bb).base()) == &bb.ce);
  assert(base(std::ranges::rbegin(c).base()) == &c.e);
  assert(base(std::ranges::rbegin(cc).base()) == &cc.ce);
  assert(base(std::ranges::rbegin(d).base()) == &d.e);
  assert(base(std::ranges::rbegin(dd).base()) == &dd.ce);

#if TEST_STD_VER < 23
  assert(base(std::ranges::crbegin(a).base()) == &a.ce);
  assert(base(std::ranges::crbegin(aa).base()) == &aa.ce);
  assert(base(std::ranges::crbegin(b).base()) == &b.ce);
  assert(base(std::ranges::crbegin(bb).base()) == &bb.ce);
  assert(base(std::ranges::crbegin(c).base()) == &c.ce);
  assert(base(std::ranges::crbegin(cc).base()) == &cc.ce);
  assert(base(std::ranges::crbegin(d).base()) == &d.ce);
  assert(base(std::ranges::crbegin(dd).base()) == &dd.ce);
#endif // TEST_STD_VER < 23

  return true;
}


ASSERT_NOEXCEPT(std::ranges::rbegin(std::declval<int (&)[10]>()));
#if TEST_STD_VER < 23
ASSERT_NOEXCEPT(std::ranges::crbegin(std::declval<int (&)[10]>()));
#endif // TEST_STD_VER < 23

struct NoThrowMemberRBegin {
  ThrowingIterator<int> rbegin() const noexcept; // auto(t.rbegin()) doesn't throw
} ntmb;
static_assert(noexcept(std::ranges::rbegin(ntmb)));
#if TEST_STD_VER < 23
static_assert(noexcept(std::ranges::crbegin(ntmb)));
#endif // TEST_STD_VER < 23

struct NoThrowADLRBegin {
  friend ThrowingIterator<int> rbegin(NoThrowADLRBegin&) noexcept;  // auto(rbegin(t)) doesn't throw
  friend ThrowingIterator<int> rbegin(const NoThrowADLRBegin&) noexcept;
} ntab;
static_assert(noexcept(std::ranges::rbegin(ntab)));
#if TEST_STD_VER < 23
static_assert(noexcept(std::ranges::crbegin(ntab)));
#endif // TEST_STD_VER < 23

struct NoThrowMemberRBeginReturnsRef {
  ThrowingIterator<int>& rbegin() const noexcept; // auto(t.rbegin()) may throw
} ntmbrr;
static_assert(!noexcept(std::ranges::rbegin(ntmbrr)));
#if TEST_STD_VER < 23
static_assert(!noexcept(std::ranges::crbegin(ntmbrr)));
#endif // TEST_STD_VER < 23

struct RBeginReturnsArrayRef {
    auto rbegin() const noexcept -> int(&)[10];
} brar;
static_assert(noexcept(std::ranges::rbegin(brar)));
#if TEST_STD_VER < 23
static_assert(noexcept(std::ranges::crbegin(brar)));
#endif // TEST_STD_VER < 23

struct NoThrowBeginThrowingEnd {
  int* begin() const noexcept;
  int* end() const;
} ntbte;
static_assert(!noexcept(std::ranges::rbegin(ntbte)));
#if TEST_STD_VER < 23
static_assert(!noexcept(std::ranges::crbegin(ntbte)));
#endif // TEST_STD_VER < 23

struct NoThrowEndThrowingBegin {
  int* begin() const;
  int* end() const noexcept;
} ntetb;
static_assert(noexcept(std::ranges::rbegin(ntetb)));
#if TEST_STD_VER < 23
static_assert(noexcept(std::ranges::crbegin(ntetb)));
#endif // TEST_STD_VER < 23

// Test ADL-proofing.
struct Incomplete;
template<class T> struct Holder { T t; };
static_assert(!std::is_invocable_v<RangeRBeginT, Holder<Incomplete>*>);
static_assert(!std::is_invocable_v<RangeRBeginT, Holder<Incomplete>*&>);
#if TEST_STD_VER < 23
static_assert(!std::is_invocable_v<RangeCRBeginT, Holder<Incomplete>*>);
static_assert(!std::is_invocable_v<RangeCRBeginT, Holder<Incomplete>*&>);
#endif // TEST_STD_VER < 23

int main(int, char**) {
  static_assert(testReturnTypes());

  testArray();
  static_assert(testArray());

  testRBeginMember();
  static_assert(testRBeginMember());

  testRBeginFunction();
  static_assert(testRBeginFunction());

  testBeginEnd();
  static_assert(testBeginEnd());

  return 0;
}
