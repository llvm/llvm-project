//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::ranges::rbegin
// std::ranges::crbegin

#include <ranges>

#include <cassert>
#include <utility>
#include "test_macros.h"
#include "test_iterators.h"

using RangeRBeginT = decltype(std::ranges::rbegin);
using RangeCRBeginT = decltype(std::ranges::crbegin);

static int globalBuff[8];

static_assert(!std::is_invocable_v<RangeRBeginT, int (&&)[10]>);
static_assert( std::is_invocable_v<RangeRBeginT, int (&)[10]>);
static_assert(!std::is_invocable_v<RangeRBeginT, int (&&)[]>);
static_assert(!std::is_invocable_v<RangeRBeginT, int (&)[]>);
static_assert(!std::is_invocable_v<RangeCRBeginT, int (&&)[10]>);
static_assert( std::is_invocable_v<RangeCRBeginT, int (&)[10]>);
static_assert(!std::is_invocable_v<RangeCRBeginT, int (&&)[]>);
static_assert(!std::is_invocable_v<RangeCRBeginT, int (&)[]>);

struct Incomplete;

static_assert(!std::is_invocable_v<RangeRBeginT, Incomplete(&&)[]>);
static_assert(!std::is_invocable_v<RangeRBeginT, const Incomplete(&&)[]>);
static_assert(!std::is_invocable_v<RangeCRBeginT, Incomplete(&&)[]>);
static_assert(!std::is_invocable_v<RangeCRBeginT, const Incomplete(&&)[]>);

static_assert(!std::is_invocable_v<RangeRBeginT, Incomplete(&&)[10]>);
static_assert(!std::is_invocable_v<RangeRBeginT, const Incomplete(&&)[10]>);
static_assert(!std::is_invocable_v<RangeCRBeginT, Incomplete(&&)[10]>);
static_assert(!std::is_invocable_v<RangeCRBeginT, const Incomplete(&&)[10]>);

// This case is IFNDR; we handle it SFINAE-friendly.
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeRBeginT, Incomplete(&)[]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeRBeginT, const Incomplete(&)[]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCRBeginT, Incomplete(&)[]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCRBeginT, const Incomplete(&)[]>);

// This case is IFNDR; we handle it SFINAE-friendly.
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeRBeginT, Incomplete(&)[10]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeRBeginT, const Incomplete(&)[10]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCRBeginT, Incomplete(&)[10]>);
LIBCPP_STATIC_ASSERT(!std::is_invocable_v<RangeCRBeginT, const Incomplete(&)[10]>);

struct RBeginMember {
  int x;
  constexpr const int *rbegin() const { return &x; }
};

// Ensure that we can't call with rvalues with borrowing disabled.
static_assert( std::is_invocable_v<RangeRBeginT, RBeginMember &>);
static_assert(!std::is_invocable_v<RangeRBeginT, RBeginMember &&>);
static_assert( std::is_invocable_v<RangeRBeginT, RBeginMember const&>);
static_assert(!std::is_invocable_v<RangeRBeginT, RBeginMember const&&>);
static_assert( std::is_invocable_v<RangeCRBeginT, RBeginMember &>);
static_assert(!std::is_invocable_v<RangeCRBeginT, RBeginMember &&>);
static_assert( std::is_invocable_v<RangeCRBeginT, RBeginMember const&>);
static_assert(!std::is_invocable_v<RangeCRBeginT, RBeginMember const&&>);

constexpr bool testReturnTypes() {
  {
    int *x[2];
    ASSERT_SAME_TYPE(decltype(std::ranges::rbegin(x)), std::reverse_iterator<int**>);
    ASSERT_SAME_TYPE(decltype(std::ranges::crbegin(x)), std::reverse_iterator<int* const*>);
  }
  {
    int x[2][2];
    ASSERT_SAME_TYPE(decltype(std::ranges::rbegin(x)), std::reverse_iterator<int(*)[2]>);
    ASSERT_SAME_TYPE(decltype(std::ranges::crbegin(x)), std::reverse_iterator<const int(*)[2]>);
  }
  {
    struct Different {
      char*& rbegin();
      short*& rbegin() const;
    } x;
    ASSERT_SAME_TYPE(decltype(std::ranges::rbegin(x)), char*);
    ASSERT_SAME_TYPE(decltype(std::ranges::crbegin(x)), short*);
  }
  return true;
}

constexpr bool testArray() {
  int a[2];
  assert(std::ranges::rbegin(a).base() == a + 2);
  assert(std::ranges::crbegin(a).base() == a + 2);

  int b[2][2];
  assert(std::ranges::rbegin(b).base() == b + 2);
  assert(std::ranges::crbegin(b).base() == b + 2);

  RBeginMember c[2];
  assert(std::ranges::rbegin(c).base() == c + 2);
  assert(std::ranges::crbegin(c).base() == c + 2);

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
static_assert(!std::is_invocable_v<RangeCRBeginT, NonConstRBeginMember &>);
static_assert(!std::is_invocable_v<RangeCRBeginT, NonConstRBeginMember const&>);

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
  assert(std::ranges::rbegin(a) == &a.x);
  assert(std::ranges::crbegin(a) == &a.x);
  static_assert(!std::is_invocable_v<RangeRBeginT, RBeginMember&&>);
  static_assert(!std::is_invocable_v<RangeCRBeginT, RBeginMember&&>);

  NonConstRBeginMember b;
  assert(std::ranges::rbegin(b) == &b.x);
  static_assert(!std::is_invocable_v<RangeCRBeginT, NonConstRBeginMember&>);

  EnabledBorrowingRBeginMember c;
  assert(std::ranges::rbegin(c) == globalBuff);
  assert(std::ranges::crbegin(c) == globalBuff);
  assert(std::ranges::rbegin(std::move(c)) == globalBuff);
  assert(std::ranges::crbegin(std::move(c)) == globalBuff);

  RBeginMemberFunction d;
  assert(std::ranges::rbegin(d) == &d.x);
  assert(std::ranges::crbegin(d) == &d.x);

  EmptyPtrRBeginMember e;
  assert(std::ranges::rbegin(e) == &e.x);
  assert(std::ranges::crbegin(e) == &e.x);

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
static_assert( std::is_invocable_v<RangeCRBeginT, RBeginFunction const&>);
static_assert( std::is_invocable_v<RangeCRBeginT, RBeginFunction &>);

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
static_assert(!std::is_invocable_v<RangeCRBeginT, RBeginFunctionByValue>);

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
  assert(std::ranges::rbegin(a) == &a.x); // Ill-formed before P2602R2 Poison Pills are Too Toxic
  assert(std::ranges::crbegin(a) == &a.x);
  assert(std::ranges::rbegin(aa) == &aa.x);
  assert(std::ranges::crbegin(aa) == &aa.x);

  RBeginFunctionByValue b{};
  const RBeginFunctionByValue bb{};
  assert(std::ranges::rbegin(b) == globalBuff + 1);
  assert(std::ranges::crbegin(b) == globalBuff + 1);
  assert(std::ranges::rbegin(bb) == globalBuff + 1);
  assert(std::ranges::crbegin(bb) == globalBuff + 1);

  RBeginFunctionEnabledBorrowing c{};
  const RBeginFunctionEnabledBorrowing cc{};
  assert(std::ranges::rbegin(std::move(c)) == globalBuff + 2);
  assert(std::ranges::crbegin(std::move(c)) == globalBuff + 2);
  assert(std::ranges::rbegin(std::move(cc)) == globalBuff + 2);
  assert(std::ranges::crbegin(std::move(cc)) == globalBuff + 2);

  RBeginFunctionReturnsEmptyPtr d{};
  const RBeginFunctionReturnsEmptyPtr dd{};
  assert(std::ranges::rbegin(d) == &d.x); // Ill-formed before P2602R2 Poison Pills are Too Toxic
  assert(std::ranges::crbegin(d) == &d.x);
  assert(std::ranges::rbegin(dd) == &dd.x);
  assert(std::ranges::crbegin(dd) == &dd.x);

  RBeginFunctionWithDataMember e{};
  const RBeginFunctionWithDataMember ee{};
  assert(std::ranges::rbegin(e) == &e.x); // Ill-formed before P2602R2 Poison Pills are Too Toxic
  assert(std::ranges::rbegin(ee) == &ee.x);
  assert(std::ranges::crbegin(e) == &e.x);
  assert(std::ranges::crbegin(ee) == &ee.x);

  RBeginFunctionWithPrivateBeginMember f{};
  const RBeginFunctionWithPrivateBeginMember ff{};
  assert(std::ranges::rbegin(f) == &f.y); // Ill-formed before P2602R2 Poison Pills are Too Toxic
  assert(std::ranges::crbegin(f) == &f.y);
  assert(std::ranges::rbegin(ff) == &ff.y);
  assert(std::ranges::crbegin(ff) == &ff.y);

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
static_assert( std::is_invocable_v<RangeCRBeginT, MemberBeginEnd const&>);

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
static_assert( std::is_invocable_v<RangeCRBeginT, FunctionBeginEnd const&>);

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
static_assert( std::is_invocable_v<RangeCRBeginT, MemberBeginFunctionEnd const&>);

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
static_assert( std::is_invocable_v<RangeCRBeginT, FunctionBeginMemberEnd const&>);

struct MemberBeginEndDifferentTypes {
  bidirectional_iterator<int*> begin();
  bidirectional_iterator<const int*> end();
};
static_assert(!std::is_invocable_v<RangeRBeginT, MemberBeginEndDifferentTypes&>);
static_assert(!std::is_invocable_v<RangeCRBeginT, MemberBeginEndDifferentTypes&>);

struct FunctionBeginEndDifferentTypes {
  friend bidirectional_iterator<int*> begin(FunctionBeginEndDifferentTypes&);
  friend bidirectional_iterator<const int*> end(FunctionBeginEndDifferentTypes&);
};
static_assert(!std::is_invocable_v<RangeRBeginT, FunctionBeginEndDifferentTypes&>);
static_assert(!std::is_invocable_v<RangeCRBeginT, FunctionBeginEndDifferentTypes&>);

struct MemberBeginEndForwardIterators {
  forward_iterator<int*> begin();
  forward_iterator<int*> end();
};
static_assert(!std::is_invocable_v<RangeRBeginT, MemberBeginEndForwardIterators&>);
static_assert(!std::is_invocable_v<RangeCRBeginT, MemberBeginEndForwardIterators&>);

struct FunctionBeginEndForwardIterators {
  friend forward_iterator<int*> begin(FunctionBeginEndForwardIterators&);
  friend forward_iterator<int*> end(FunctionBeginEndForwardIterators&);
};
static_assert(!std::is_invocable_v<RangeRBeginT, FunctionBeginEndForwardIterators&>);
static_assert(!std::is_invocable_v<RangeCRBeginT, FunctionBeginEndForwardIterators&>);

struct MemberBeginOnly {
  bidirectional_iterator<int*> begin() const;
};
static_assert(!std::is_invocable_v<RangeRBeginT, MemberBeginOnly&>);
static_assert(!std::is_invocable_v<RangeCRBeginT, MemberBeginOnly&>);

struct FunctionBeginOnly {
  friend bidirectional_iterator<int*> begin(FunctionBeginOnly&);
};
static_assert(!std::is_invocable_v<RangeRBeginT, FunctionBeginOnly&>);
static_assert(!std::is_invocable_v<RangeCRBeginT, FunctionBeginOnly&>);

struct MemberEndOnly {
  bidirectional_iterator<int*> end() const;
};
static_assert(!std::is_invocable_v<RangeRBeginT, MemberEndOnly&>);
static_assert(!std::is_invocable_v<RangeCRBeginT, MemberEndOnly&>);

struct FunctionEndOnly {
  friend bidirectional_iterator<int*> end(FunctionEndOnly&);
};
static_assert(!std::is_invocable_v<RangeRBeginT, FunctionEndOnly&>);
static_assert(!std::is_invocable_v<RangeCRBeginT, FunctionEndOnly&>);

// Make sure there is no clash between the following cases:
// - the case that handles classes defining member `rbegin` and `rend` functions;
// - the case that handles classes defining `begin` and `end` functions returning reversible iterators.
struct MemberBeginAndRBegin {
  int* begin() const;
  int* end() const;
  int* rbegin() const;
  int* rend() const;
};
static_assert( std::is_invocable_v<RangeRBeginT, MemberBeginAndRBegin&>);
static_assert( std::is_invocable_v<RangeCRBeginT, MemberBeginAndRBegin&>);
static_assert( std::same_as<std::invoke_result_t<RangeRBeginT, MemberBeginAndRBegin&>, int*>);
static_assert( std::same_as<std::invoke_result_t<RangeCRBeginT, MemberBeginAndRBegin&>, int*>);

constexpr bool testBeginEnd() {
  MemberBeginEnd a{};
  const MemberBeginEnd aa{};
  assert(base(std::ranges::rbegin(a).base()) == &a.e);
  assert(base(std::ranges::crbegin(a).base()) == &a.ce);
  assert(base(std::ranges::rbegin(aa).base()) == &aa.ce);
  assert(base(std::ranges::crbegin(aa).base()) == &aa.ce);

  FunctionBeginEnd b{};
  const FunctionBeginEnd bb{};
  assert(base(std::ranges::rbegin(b).base()) == &b.e);
  assert(base(std::ranges::crbegin(b).base()) == &b.ce);
  assert(base(std::ranges::rbegin(bb).base()) == &bb.ce);
  assert(base(std::ranges::crbegin(bb).base()) == &bb.ce);

  MemberBeginFunctionEnd c{};
  const MemberBeginFunctionEnd cc{};
  assert(base(std::ranges::rbegin(c).base()) == &c.e);
  assert(base(std::ranges::crbegin(c).base()) == &c.ce);
  assert(base(std::ranges::rbegin(cc).base()) == &cc.ce);
  assert(base(std::ranges::crbegin(cc).base()) == &cc.ce);

  FunctionBeginMemberEnd d{};
  const FunctionBeginMemberEnd dd{};
  assert(base(std::ranges::rbegin(d).base()) == &d.e);
  assert(base(std::ranges::crbegin(d).base()) == &d.ce);
  assert(base(std::ranges::rbegin(dd).base()) == &dd.ce);
  assert(base(std::ranges::crbegin(dd).base()) == &dd.ce);

  return true;
}


ASSERT_NOEXCEPT(std::ranges::rbegin(std::declval<int (&)[10]>()));
ASSERT_NOEXCEPT(std::ranges::crbegin(std::declval<int (&)[10]>()));

struct NoThrowMemberRBegin {
  ThrowingIterator<int> rbegin() const noexcept; // auto(t.rbegin()) doesn't throw
} ntmb;
static_assert(noexcept(std::ranges::rbegin(ntmb)));
static_assert(noexcept(std::ranges::crbegin(ntmb)));

struct NoThrowADLRBegin {
  friend ThrowingIterator<int> rbegin(NoThrowADLRBegin&) noexcept;  // auto(rbegin(t)) doesn't throw
  friend ThrowingIterator<int> rbegin(const NoThrowADLRBegin&) noexcept;
} ntab;
static_assert(noexcept(std::ranges::rbegin(ntab)));
static_assert(noexcept(std::ranges::crbegin(ntab)));

struct NoThrowMemberRBeginReturnsRef {
  ThrowingIterator<int>& rbegin() const noexcept; // auto(t.rbegin()) may throw
} ntmbrr;
static_assert(!noexcept(std::ranges::rbegin(ntmbrr)));
static_assert(!noexcept(std::ranges::crbegin(ntmbrr)));

struct RBeginReturnsArrayRef {
    auto rbegin() const noexcept -> int(&)[10];
} brar;
static_assert(noexcept(std::ranges::rbegin(brar)));
static_assert(noexcept(std::ranges::crbegin(brar)));

struct NoThrowBeginThrowingEnd {
  int* begin() const noexcept;
  int* end() const;
} ntbte;
static_assert(!noexcept(std::ranges::rbegin(ntbte)));
static_assert(!noexcept(std::ranges::crbegin(ntbte)));

struct NoThrowEndThrowingBegin {
  int* begin() const;
  int* end() const noexcept;
} ntetb;
static_assert(noexcept(std::ranges::rbegin(ntetb)));
static_assert(noexcept(std::ranges::crbegin(ntetb)));

// Test ADL-proofing.
struct Incomplete;
template<class T> struct Holder { T t; };
static_assert(!std::is_invocable_v<RangeRBeginT, Holder<Incomplete>*>);
static_assert(!std::is_invocable_v<RangeRBeginT, Holder<Incomplete>*&>);
static_assert(!std::is_invocable_v<RangeCRBeginT, Holder<Incomplete>*>);
static_assert(!std::is_invocable_v<RangeCRBeginT, Holder<Incomplete>*&>);

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
