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

// std::ranges::rend
// std::ranges::crend

#include <ranges>

#include <cassert>
#include <utility>
#include "test_macros.h"
#include "test_iterators.h"

using RangeREndT = decltype(std::ranges::rend);
using RangeCREndT = decltype(std::ranges::crend);

static int globalBuff[8];

static_assert(!std::is_invocable_v<RangeREndT, int (&&)[]>);
static_assert(!std::is_invocable_v<RangeREndT, int (&)[]>);
static_assert(!std::is_invocable_v<RangeREndT, int (&&)[10]>);
static_assert( std::is_invocable_v<RangeREndT, int (&)[10]>);
static_assert(!std::is_invocable_v<RangeCREndT, int (&&)[]>);
static_assert(!std::is_invocable_v<RangeCREndT, int (&)[]>);
static_assert(!std::is_invocable_v<RangeCREndT, int (&&)[10]>);
static_assert( std::is_invocable_v<RangeCREndT, int (&)[10]>);

struct Incomplete;
static_assert(!std::is_invocable_v<RangeREndT, Incomplete(&&)[]>);
static_assert(!std::is_invocable_v<RangeREndT, Incomplete(&&)[42]>);
static_assert(!std::is_invocable_v<RangeCREndT, Incomplete(&&)[]>);
static_assert(!std::is_invocable_v<RangeCREndT, Incomplete(&&)[42]>);

struct REndMember {
  int x;
  const int* rbegin() const;
  constexpr const int* rend() const { return &x; }
};

// Ensure that we can't call with rvalues with borrowing disabled.
static_assert( std::is_invocable_v<RangeREndT, REndMember&>);
static_assert(!std::is_invocable_v<RangeREndT, REndMember &&>);
static_assert( std::is_invocable_v<RangeREndT, REndMember const&>);
static_assert(!std::is_invocable_v<RangeREndT, REndMember const&&>);
static_assert( std::is_invocable_v<RangeCREndT, REndMember &>);
static_assert(!std::is_invocable_v<RangeCREndT, REndMember &&>);
static_assert( std::is_invocable_v<RangeCREndT, REndMember const&>);
static_assert(!std::is_invocable_v<RangeCREndT, REndMember const&&>);

constexpr bool testReturnTypes() {
  {
    int *x[2];
    ASSERT_SAME_TYPE(decltype(std::ranges::rend(x)), std::reverse_iterator<int**>);
    ASSERT_SAME_TYPE(decltype(std::ranges::crend(x)), std::reverse_iterator<int* const*>);
  }

  {
    int x[2][2];
    ASSERT_SAME_TYPE(decltype(std::ranges::rend(x)), std::reverse_iterator<int(*)[2]>);
    ASSERT_SAME_TYPE(decltype(std::ranges::crend(x)), std::reverse_iterator<const int(*)[2]>);
  }

  {
    struct Different {
      char* rbegin();
      sentinel_wrapper<char*>& rend();
      short* rbegin() const;
      sentinel_wrapper<short*>& rend() const;
    } x;
    ASSERT_SAME_TYPE(decltype(std::ranges::rend(x)), sentinel_wrapper<char*>);
    ASSERT_SAME_TYPE(decltype(std::ranges::crend(x)), sentinel_wrapper<short*>);
  }

  return true;
}

constexpr bool testArray() {
  int a[2];
  assert(std::ranges::rend(a).base() == a);
  assert(std::ranges::crend(a).base() == a);

  int b[2][2];
  assert(std::ranges::rend(b).base() == b);
  assert(std::ranges::crend(b).base() == b);

  REndMember c[2];
  assert(std::ranges::rend(c).base() == c);
  assert(std::ranges::crend(c).base() == c);

  return true;
}

struct REndMemberReturnsInt {
  int rbegin() const;
  int rend() const;
};
static_assert(!std::is_invocable_v<RangeREndT, REndMemberReturnsInt const&>);

struct REndMemberReturnsVoidPtr {
  const void *rbegin() const;
  const void *rend() const;
};
static_assert(!std::is_invocable_v<RangeREndT, REndMemberReturnsVoidPtr const&>);

struct PtrConvertible {
  operator int*() const;
};
struct PtrConvertibleREndMember {
  PtrConvertible rbegin() const;
  PtrConvertible rend() const;
};
static_assert(!std::is_invocable_v<RangeREndT, PtrConvertibleREndMember const&>);

struct NoRBeginMember {
  constexpr const int* rend();
};
static_assert(!std::is_invocable_v<RangeREndT, NoRBeginMember const&>);

struct NonConstREndMember {
  int x;
  constexpr int* rbegin() { return nullptr; }
  constexpr int* rend() { return &x; }
};
static_assert( std::is_invocable_v<RangeREndT,  NonConstREndMember &>);
static_assert(!std::is_invocable_v<RangeREndT,  NonConstREndMember const&>);
static_assert(!std::is_invocable_v<RangeCREndT, NonConstREndMember &>);
static_assert(!std::is_invocable_v<RangeCREndT, NonConstREndMember const&>);

struct EnabledBorrowingREndMember {
  constexpr int* rbegin() const { return nullptr; }
  constexpr int* rend() const { return &globalBuff[0]; }
};

template <>
inline constexpr bool std::ranges::enable_borrowed_range<EnabledBorrowingREndMember> = true;

struct REndMemberFunction {
  int x;
  constexpr const int* rbegin() const { return nullptr; }
  constexpr const int* rend() const { return &x; }
  friend constexpr int* rend(REndMemberFunction const&);
};

struct Empty { };
struct EmptyEndMember {
  Empty rbegin() const;
  Empty rend() const;
};
static_assert(!std::is_invocable_v<RangeREndT, EmptyEndMember const&>);

struct EmptyPtrREndMember {
  Empty x;
  constexpr const Empty* rbegin() const { return nullptr; }
  constexpr const Empty* rend() const { return &x; }
};

constexpr bool testREndMember() {
  REndMember a;
  assert(std::ranges::rend(a) == &a.x);
  assert(std::ranges::crend(a) == &a.x);

  NonConstREndMember b;
  assert(std::ranges::rend(b) == &b.x);
  static_assert(!std::is_invocable_v<RangeCREndT, decltype((b))>);

  EnabledBorrowingREndMember c;
  assert(std::ranges::rend(std::move(c)) == &globalBuff[0]);
  assert(std::ranges::crend(std::move(c)) == &globalBuff[0]);

  REndMemberFunction d;
  assert(std::ranges::rend(d) == &d.x);
  assert(std::ranges::crend(d) == &d.x);

  EmptyPtrREndMember e;
  assert(std::ranges::rend(e) == &e.x);
  assert(std::ranges::crend(e) == &e.x);

  return true;
}

struct REndFunction {
  int x;
  friend constexpr const int* rbegin(REndFunction const&) { return nullptr; }
  friend constexpr const int* rend(REndFunction const& bf) { return &bf.x; }
};

static_assert( std::is_invocable_v<RangeREndT, REndFunction const&>);
static_assert(!std::is_invocable_v<RangeREndT, REndFunction &&>);

static_assert( std::is_invocable_v<RangeREndT,  REndFunction const&>);
static_assert(!std::is_invocable_v<RangeREndT,  REndFunction &&>);
static_assert(!std::is_invocable_v<RangeREndT,  REndFunction &>);
static_assert( std::is_invocable_v<RangeCREndT, REndFunction const&>);
static_assert( std::is_invocable_v<RangeCREndT, REndFunction &>);

struct REndFunctionReturnsInt {
  friend constexpr int rbegin(REndFunctionReturnsInt const&);
  friend constexpr int rend(REndFunctionReturnsInt const&);
};
static_assert(!std::is_invocable_v<RangeREndT, REndFunctionReturnsInt const&>);

struct REndFunctionReturnsVoidPtr {
  friend constexpr void* rbegin(REndFunctionReturnsVoidPtr const&);
  friend constexpr void* rend(REndFunctionReturnsVoidPtr const&);
};
static_assert(!std::is_invocable_v<RangeREndT, REndFunctionReturnsVoidPtr const&>);

struct REndFunctionReturnsEmpty {
  friend constexpr Empty rbegin(REndFunctionReturnsEmpty const&);
  friend constexpr Empty rend(REndFunctionReturnsEmpty const&);
};
static_assert(!std::is_invocable_v<RangeREndT, REndFunctionReturnsEmpty const&>);

struct REndFunctionReturnsPtrConvertible {
  friend constexpr PtrConvertible rbegin(REndFunctionReturnsPtrConvertible const&);
  friend constexpr PtrConvertible rend(REndFunctionReturnsPtrConvertible const&);
};
static_assert(!std::is_invocable_v<RangeREndT, REndFunctionReturnsPtrConvertible const&>);

struct NoRBeginFunction {
  friend constexpr const int* rend(NoRBeginFunction const&);
};
static_assert(!std::is_invocable_v<RangeREndT, NoRBeginFunction const&>);

struct REndFunctionByValue {
  friend constexpr int* rbegin(REndFunctionByValue) { return nullptr; }
  friend constexpr int* rend(REndFunctionByValue) { return &globalBuff[1]; }
};
static_assert(!std::is_invocable_v<RangeCREndT, REndFunctionByValue>);

struct REndFunctionEnabledBorrowing {
  friend constexpr int* rbegin(REndFunctionEnabledBorrowing) { return nullptr; }
  friend constexpr int* rend(REndFunctionEnabledBorrowing) { return &globalBuff[2]; }
};
template<>
inline constexpr bool std::ranges::enable_borrowed_range<REndFunctionEnabledBorrowing> = true;

struct REndFunctionReturnsEmptyPtr {
  Empty x;
  friend constexpr const Empty* rbegin(REndFunctionReturnsEmptyPtr const&) { return nullptr; }
  friend constexpr const Empty* rend(REndFunctionReturnsEmptyPtr const& bf) { return &bf.x; }
};

struct REndFunctionWithDataMember {
  int x;
  int rend;
  friend constexpr const int* rbegin(REndFunctionWithDataMember const&) { return nullptr; }
  friend constexpr const int* rend(REndFunctionWithDataMember const& bf) { return &bf.x; }
};

struct REndFunctionWithPrivateEndMember : private REndMember {
  int y;
  friend constexpr const int* rbegin(REndFunctionWithPrivateEndMember const&) { return nullptr; }
  friend constexpr const int* rend(REndFunctionWithPrivateEndMember const& bf) { return &bf.y; }
};

struct RBeginMemberEndFunction {
  int x;
  constexpr const int* rbegin() const { return nullptr; }
  friend constexpr const int* rend(RBeginMemberEndFunction const& bf) { return &bf.x; }
};

constexpr bool testREndFunction() {
  const REndFunction a{};
  assert(std::ranges::rend(a) == &a.x);
  assert(std::ranges::crend(a) == &a.x);
  REndFunction aa{};
  static_assert(!std::is_invocable_v<RangeREndT, decltype((aa))>);
  assert(std::ranges::crend(aa) == &aa.x);

  REndFunctionByValue b;
  assert(std::ranges::rend(b) == &globalBuff[1]);
  assert(std::ranges::crend(b) == &globalBuff[1]);

  REndFunctionEnabledBorrowing c;
  assert(std::ranges::rend(std::move(c)) == &globalBuff[2]);
  assert(std::ranges::crend(std::move(c)) == &globalBuff[2]);

  const REndFunctionReturnsEmptyPtr d{};
  assert(std::ranges::rend(d) == &d.x);
  assert(std::ranges::crend(d) == &d.x);
  REndFunctionReturnsEmptyPtr dd{};
  static_assert(!std::is_invocable_v<RangeREndT, decltype((dd))>);
  assert(std::ranges::crend(dd) == &dd.x);

  const REndFunctionWithDataMember e{};
  assert(std::ranges::rend(e) == &e.x);
  assert(std::ranges::crend(e) == &e.x);
  REndFunctionWithDataMember ee{};
  static_assert(!std::is_invocable_v<RangeREndT, decltype((ee))>);
  assert(std::ranges::crend(ee) == &ee.x);

  const REndFunctionWithPrivateEndMember f{};
  assert(std::ranges::rend(f) == &f.y);
  assert(std::ranges::crend(f) == &f.y);
  REndFunctionWithPrivateEndMember ff{};
  static_assert(!std::is_invocable_v<RangeREndT, decltype((ff))>);
  assert(std::ranges::crend(ff) == &ff.y);

  const RBeginMemberEndFunction g{};
  assert(std::ranges::rend(g) == &g.x);
  assert(std::ranges::crend(g) == &g.x);
  RBeginMemberEndFunction gg{};
  static_assert(!std::is_invocable_v<RangeREndT, decltype((gg))>);
  assert(std::ranges::crend(gg) == &gg.x);

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
static_assert( std::is_invocable_v<RangeREndT, MemberBeginEnd&>);
static_assert( std::is_invocable_v<RangeREndT, MemberBeginEnd const&>);
static_assert( std::is_invocable_v<RangeCREndT, MemberBeginEnd const&>);

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
static_assert( std::is_invocable_v<RangeREndT, FunctionBeginEnd&>);
static_assert( std::is_invocable_v<RangeREndT, FunctionBeginEnd const&>);
static_assert( std::is_invocable_v<RangeCREndT, FunctionBeginEnd const&>);

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
static_assert( std::is_invocable_v<RangeREndT, MemberBeginFunctionEnd&>);
static_assert( std::is_invocable_v<RangeREndT, MemberBeginFunctionEnd const&>);
static_assert( std::is_invocable_v<RangeCREndT, MemberBeginFunctionEnd const&>);

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
static_assert( std::is_invocable_v<RangeREndT, FunctionBeginMemberEnd&>);
static_assert( std::is_invocable_v<RangeREndT, FunctionBeginMemberEnd const&>);
static_assert( std::is_invocable_v<RangeCREndT, FunctionBeginMemberEnd const&>);

struct MemberBeginEndDifferentTypes {
  bidirectional_iterator<int*> begin();
  bidirectional_iterator<const int*> end();
};
static_assert(!std::is_invocable_v<RangeREndT, MemberBeginEndDifferentTypes&>);
static_assert(!std::is_invocable_v<RangeCREndT, MemberBeginEndDifferentTypes&>);

struct FunctionBeginEndDifferentTypes {
  friend bidirectional_iterator<int*> begin(FunctionBeginEndDifferentTypes&);
  friend bidirectional_iterator<const int*> end(FunctionBeginEndDifferentTypes&);
};
static_assert(!std::is_invocable_v<RangeREndT, FunctionBeginEndDifferentTypes&>);
static_assert(!std::is_invocable_v<RangeCREndT, FunctionBeginEndDifferentTypes&>);

struct MemberBeginEndForwardIterators {
  forward_iterator<int*> begin();
  forward_iterator<int*> end();
};
static_assert(!std::is_invocable_v<RangeREndT, MemberBeginEndForwardIterators&>);
static_assert(!std::is_invocable_v<RangeCREndT, MemberBeginEndForwardIterators&>);

struct FunctionBeginEndForwardIterators {
  friend forward_iterator<int*> begin(FunctionBeginEndForwardIterators&);
  friend forward_iterator<int*> end(FunctionBeginEndForwardIterators&);
};
static_assert(!std::is_invocable_v<RangeREndT, FunctionBeginEndForwardIterators&>);
static_assert(!std::is_invocable_v<RangeCREndT, FunctionBeginEndForwardIterators&>);

struct MemberBeginOnly {
  bidirectional_iterator<int*> begin() const;
};
static_assert(!std::is_invocable_v<RangeREndT, MemberBeginOnly&>);
static_assert(!std::is_invocable_v<RangeCREndT, MemberBeginOnly&>);

struct FunctionBeginOnly {
  friend bidirectional_iterator<int*> begin(FunctionBeginOnly&);
};
static_assert(!std::is_invocable_v<RangeREndT, FunctionBeginOnly&>);
static_assert(!std::is_invocable_v<RangeCREndT, FunctionBeginOnly&>);

struct MemberEndOnly {
  bidirectional_iterator<int*> end() const;
};
static_assert(!std::is_invocable_v<RangeREndT, MemberEndOnly&>);
static_assert(!std::is_invocable_v<RangeCREndT, MemberEndOnly&>);

struct FunctionEndOnly {
  friend bidirectional_iterator<int*> end(FunctionEndOnly&);
};
static_assert(!std::is_invocable_v<RangeREndT, FunctionEndOnly&>);
static_assert(!std::is_invocable_v<RangeCREndT, FunctionEndOnly&>);

// Make sure there is no clash between the following cases:
// - the case that handles classes defining member `rbegin` and `rend` functions;
// - the case that handles classes defining `begin` and `end` functions returning reversible iterators.
struct MemberBeginAndRBegin {
  int* begin() const;
  int* end() const;
  int* rbegin() const;
  int* rend() const;
};
static_assert( std::is_invocable_v<RangeREndT, MemberBeginAndRBegin&>);
static_assert( std::is_invocable_v<RangeCREndT, MemberBeginAndRBegin&>);
static_assert( std::same_as<std::invoke_result_t<RangeREndT, MemberBeginAndRBegin&>, int*>);
static_assert( std::same_as<std::invoke_result_t<RangeCREndT, MemberBeginAndRBegin&>, int*>);

constexpr bool testBeginEnd() {
  MemberBeginEnd a{};
  const MemberBeginEnd aa{};
  assert(base(std::ranges::rend(a).base()) == &a.b);
  assert(base(std::ranges::crend(a).base()) == &a.cb);
  assert(base(std::ranges::rend(aa).base()) == &aa.cb);
  assert(base(std::ranges::crend(aa).base()) == &aa.cb);

  FunctionBeginEnd b{};
  const FunctionBeginEnd bb{};
  assert(base(std::ranges::rend(b).base()) == &b.b);
  assert(base(std::ranges::crend(b).base()) == &b.cb);
  assert(base(std::ranges::rend(bb).base()) == &bb.cb);
  assert(base(std::ranges::crend(bb).base()) == &bb.cb);

  MemberBeginFunctionEnd c{};
  const MemberBeginFunctionEnd cc{};
  assert(base(std::ranges::rend(c).base()) == &c.b);
  assert(base(std::ranges::crend(c).base()) == &c.cb);
  assert(base(std::ranges::rend(cc).base()) == &cc.cb);
  assert(base(std::ranges::crend(cc).base()) == &cc.cb);

  FunctionBeginMemberEnd d{};
  const FunctionBeginMemberEnd dd{};
  assert(base(std::ranges::rend(d).base()) == &d.b);
  assert(base(std::ranges::crend(d).base()) == &d.cb);
  assert(base(std::ranges::rend(dd).base()) == &dd.cb);
  assert(base(std::ranges::crend(dd).base()) == &dd.cb);

  return true;
}


ASSERT_NOEXCEPT(std::ranges::rend(std::declval<int (&)[10]>()));
ASSERT_NOEXCEPT(std::ranges::crend(std::declval<int (&)[10]>()));

struct NoThrowMemberREnd {
  ThrowingIterator<int> rbegin() const;
  ThrowingIterator<int> rend() const noexcept; // auto(t.rend()) doesn't throw
} ntmre;
static_assert(noexcept(std::ranges::rend(ntmre)));
static_assert(noexcept(std::ranges::crend(ntmre)));

struct NoThrowADLREnd {
  ThrowingIterator<int> rbegin() const;
  friend ThrowingIterator<int> rend(NoThrowADLREnd&) noexcept;  // auto(rend(t)) doesn't throw
  friend ThrowingIterator<int> rend(const NoThrowADLREnd&) noexcept;
} ntare;
static_assert(noexcept(std::ranges::rend(ntare)));
static_assert(noexcept(std::ranges::crend(ntare)));

struct NoThrowMemberREndReturnsRef {
  ThrowingIterator<int> rbegin() const;
  ThrowingIterator<int>& rend() const noexcept; // auto(t.rend()) may throw
} ntmrerr;
static_assert(!noexcept(std::ranges::rend(ntmrerr)));
static_assert(!noexcept(std::ranges::crend(ntmrerr)));

struct REndReturnsArrayRef {
    auto rbegin() const noexcept -> int(&)[10];
    auto rend() const noexcept -> int(&)[10];
} rerar;
static_assert(noexcept(std::ranges::rend(rerar)));
static_assert(noexcept(std::ranges::crend(rerar)));

struct NoThrowBeginThrowingEnd {
  int* begin() const noexcept;
  int* end() const;
} ntbte;
static_assert(noexcept(std::ranges::rend(ntbte)));
static_assert(noexcept(std::ranges::crend(ntbte)));

struct NoThrowEndThrowingBegin {
  int* begin() const;
  int* end() const noexcept;
} ntetb;
static_assert(!noexcept(std::ranges::rend(ntetb)));
static_assert(!noexcept(std::ranges::crend(ntetb)));

// Test ADL-proofing.
struct Incomplete;
template<class T> struct Holder { T t; };
static_assert(!std::is_invocable_v<RangeREndT, Holder<Incomplete>*>);
static_assert(!std::is_invocable_v<RangeREndT, Holder<Incomplete>*&>);
static_assert(!std::is_invocable_v<RangeCREndT, Holder<Incomplete>*>);
static_assert(!std::is_invocable_v<RangeCREndT, Holder<Incomplete>*&>);

int main(int, char**) {
  static_assert(testReturnTypes());

  testArray();
  static_assert(testArray());

  testREndMember();
  static_assert(testREndMember());

  testREndFunction();
  static_assert(testREndFunction());

  testBeginEnd();
  static_assert(testBeginEnd());

  return 0;
}
