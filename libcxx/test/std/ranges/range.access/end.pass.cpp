//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::ranges::end
// std::ranges::cend // until C++23

#include <ranges>

#include <cassert>
#include <utility>
#include "test_macros.h"
#include "test_iterators.h"

using RangeEndT = decltype(std::ranges::end);
#if TEST_STD_VER < 23
using RangeCEndT = decltype(std::ranges::cend);
#endif // TEST_STD_VER < 23

static int globalBuff[8];

static_assert(!std::is_invocable_v<RangeEndT, int (&&)[]>);
static_assert(!std::is_invocable_v<RangeEndT, int (&)[]>);
static_assert(!std::is_invocable_v<RangeEndT, int (&&)[10]>);
static_assert( std::is_invocable_v<RangeEndT, int (&)[10]>);
#if TEST_STD_VER < 23
static_assert(!std::is_invocable_v<RangeCEndT, int (&&)[]>);
static_assert(!std::is_invocable_v<RangeCEndT, int (&)[]>);
static_assert(!std::is_invocable_v<RangeCEndT, int (&&)[10]>);
static_assert( std::is_invocable_v<RangeCEndT, int (&)[10]>);
#endif // TEST_STD_VER < 23

struct Incomplete;
static_assert(!std::is_invocable_v<RangeEndT, Incomplete(&&)[]>);
static_assert(!std::is_invocable_v<RangeEndT, Incomplete(&&)[42]>);
#if TEST_STD_VER < 23
static_assert(!std::is_invocable_v<RangeCEndT, Incomplete(&&)[]>);
static_assert(!std::is_invocable_v<RangeCEndT, Incomplete(&&)[42]>);
#endif // TEST_STD_VER < 23

struct EndMember {
  int x;
  const int *begin() const;
  constexpr const int *end() const { return &x; }
};

// Ensure that we can't call with rvalues with borrowing disabled.
static_assert( std::is_invocable_v<RangeEndT, EndMember &>);
static_assert(!std::is_invocable_v<RangeEndT, EndMember &&>);
static_assert( std::is_invocable_v<RangeEndT, EndMember const&>);
static_assert(!std::is_invocable_v<RangeEndT, EndMember const&&>);
#if TEST_STD_VER < 23
static_assert( std::is_invocable_v<RangeCEndT, EndMember &>);
static_assert(!std::is_invocable_v<RangeCEndT, EndMember &&>);
static_assert( std::is_invocable_v<RangeCEndT, EndMember const&>);
static_assert(!std::is_invocable_v<RangeCEndT, EndMember const&&>);
#endif // TEST_STD_VER < 23

constexpr bool testReturnTypes() {
  int* a[2];
  int b[2][2];
  struct Different {
    char* begin();
    sentinel_wrapper<char*>& end();
    short* begin() const;
    sentinel_wrapper<short*>& end() const;
  } c;

  ASSERT_SAME_TYPE(decltype(std::ranges::end(a)), int**);
  ASSERT_SAME_TYPE(decltype(std::ranges::end(b)), int(*)[2]);
  ASSERT_SAME_TYPE(decltype(std::ranges::end(c)), sentinel_wrapper<char*>);

#if TEST_STD_VER < 23
  ASSERT_SAME_TYPE(decltype(std::ranges::cend(a)), int* const*);
  ASSERT_SAME_TYPE(decltype(std::ranges::cend(b)), const int(*)[2]);
  ASSERT_SAME_TYPE(decltype(std::ranges::cend(c)), sentinel_wrapper<short*>);
#endif // TEST_STD_VER < 23

  return true;
}

constexpr bool testArray() {
  int a[2];
  int b[2][2];
  EndMember c[2];

  assert(std::ranges::end(a) == a + 2);
  assert(std::ranges::end(b) == b + 2);
  assert(std::ranges::end(c) == c + 2);

#if TEST_STD_VER < 23
  assert(std::ranges::cend(a) == a + 2);
  assert(std::ranges::cend(b) == b + 2);
  assert(std::ranges::cend(c) == c + 2);
#endif // TEST_STD_VER < 23

  return true;
}

struct EndMemberReturnsInt {
  int begin() const;
  int end() const;
};
static_assert(!std::is_invocable_v<RangeEndT, EndMemberReturnsInt const&>);

struct EndMemberReturnsVoidPtr {
  const void *begin() const;
  const void *end() const;
};
static_assert(!std::is_invocable_v<RangeEndT, EndMemberReturnsVoidPtr const&>);

struct PtrConvertible {
  operator int*() const;
};
struct PtrConvertibleEndMember {
  PtrConvertible begin() const;
  PtrConvertible end() const;
};
static_assert(!std::is_invocable_v<RangeEndT, PtrConvertibleEndMember const&>);

struct NoBeginMember {
  constexpr const int *end();
};
static_assert(!std::is_invocable_v<RangeEndT, NoBeginMember const&>);

struct NonConstEndMember {
  int x;
  constexpr int *begin() { return nullptr; }
  constexpr int *end() { return &x; }
};
static_assert( std::is_invocable_v<RangeEndT,  NonConstEndMember &>);
static_assert(!std::is_invocable_v<RangeEndT,  NonConstEndMember const&>);
#if TEST_STD_VER < 23
static_assert(!std::is_invocable_v<RangeCEndT, NonConstEndMember &>);
static_assert(!std::is_invocable_v<RangeCEndT, NonConstEndMember const&>);
#endif // TEST_STD_VER < 23

struct EnabledBorrowingEndMember {
  constexpr int *begin() const { return nullptr; }
  constexpr int *end() const { return &globalBuff[0]; }
};

template<>
inline constexpr bool std::ranges::enable_borrowed_range<EnabledBorrowingEndMember> = true;

struct EndMemberFunction {
  int x;
  constexpr const int *begin() const { return nullptr; }
  constexpr const int *end() const { return &x; }
  friend constexpr int *end(EndMemberFunction const&);
};

struct Empty { };
struct EmptyEndMember {
  Empty begin() const;
  Empty end() const;
};
static_assert(!std::is_invocable_v<RangeEndT, EmptyEndMember const&>);

struct EmptyPtrEndMember {
  Empty x;
  constexpr const Empty *begin() const { return nullptr; }
  constexpr const Empty *end() const { return &x; }
};

constexpr bool testEndMember() {
  EndMember a;
  NonConstEndMember b;
  EnabledBorrowingEndMember c;
  EndMemberFunction d;
  EmptyPtrEndMember e;

  assert(std::ranges::end(a) == &a.x);
  assert(std::ranges::end(b) == &b.x);
  assert(std::ranges::end(std::move(c)) == &globalBuff[0]);
  assert(std::ranges::end(d) == &d.x);
  assert(std::ranges::end(e) == &e.x);

#if TEST_STD_VER < 23
  assert(std::ranges::cend(a) == &a.x);
  static_assert(!std::is_invocable_v<RangeCEndT, decltype((b))>);
  assert(std::ranges::cend(std::move(c)) == &globalBuff[0]);
  assert(std::ranges::cend(d) == &d.x);
  assert(std::ranges::cend(e) == &e.x);
#endif // TEST_STD_VER < 23

  return true;
}

struct EndFunction {
  int x;
  friend constexpr const int *begin(EndFunction const&) { return nullptr; }
  friend constexpr const int *end(EndFunction const& bf) { return &bf.x; }
};

static_assert( std::is_invocable_v<RangeEndT, EndFunction const&>);
static_assert(!std::is_invocable_v<RangeEndT, EndFunction &&>);

static_assert( std::is_invocable_v<RangeEndT,  EndFunction const&>);
static_assert(!std::is_invocable_v<RangeEndT,  EndFunction &&>);
static_assert(std::is_invocable_v<RangeEndT, EndFunction&>); // Ill-formed before P2602R2 Poison Pills are Too Toxic
#if TEST_STD_VER < 23
static_assert( std::is_invocable_v<RangeCEndT, EndFunction const&>);
static_assert( std::is_invocable_v<RangeCEndT, EndFunction &>);
#endif // TEST_STD_VER < 23

struct EndFunctionReturnsInt {
  friend constexpr int begin(EndFunctionReturnsInt const&);
  friend constexpr int end(EndFunctionReturnsInt const&);
};
static_assert(!std::is_invocable_v<RangeEndT, EndFunctionReturnsInt const&>);

struct EndFunctionReturnsVoidPtr {
  friend constexpr void *begin(EndFunctionReturnsVoidPtr const&);
  friend constexpr void *end(EndFunctionReturnsVoidPtr const&);
};
static_assert(!std::is_invocable_v<RangeEndT, EndFunctionReturnsVoidPtr const&>);

struct EndFunctionReturnsEmpty {
  friend constexpr Empty begin(EndFunctionReturnsEmpty const&);
  friend constexpr Empty end(EndFunctionReturnsEmpty const&);
};
static_assert(!std::is_invocable_v<RangeEndT, EndFunctionReturnsEmpty const&>);

struct EndFunctionReturnsPtrConvertible {
  friend constexpr PtrConvertible begin(EndFunctionReturnsPtrConvertible const&);
  friend constexpr PtrConvertible end(EndFunctionReturnsPtrConvertible const&);
};
static_assert(!std::is_invocable_v<RangeEndT, EndFunctionReturnsPtrConvertible const&>);

struct NoBeginFunction {
  friend constexpr const int *end(NoBeginFunction const&);
};
static_assert(!std::is_invocable_v<RangeEndT, NoBeginFunction const&>);

struct EndFunctionByValue {
  friend constexpr int *begin(EndFunctionByValue) { return nullptr; }
  friend constexpr int *end(EndFunctionByValue) { return &globalBuff[1]; }
};
#if TEST_STD_VER < 23
static_assert(!std::is_invocable_v<RangeCEndT, EndFunctionByValue>);
#endif // TEST_STD_VER < 23

struct EndFunctionEnabledBorrowing {
  friend constexpr int *begin(EndFunctionEnabledBorrowing) { return nullptr; }
  friend constexpr int *end(EndFunctionEnabledBorrowing) { return &globalBuff[2]; }
};
template<>
inline constexpr bool std::ranges::enable_borrowed_range<EndFunctionEnabledBorrowing> = true;

struct EndFunctionReturnsEmptyPtr {
  Empty x;
  friend constexpr const Empty *begin(EndFunctionReturnsEmptyPtr const&) { return nullptr; }
  friend constexpr const Empty *end(EndFunctionReturnsEmptyPtr const& bf) { return &bf.x; }
};

struct EndFunctionWithDataMember {
  int x;
  int end;
  friend constexpr const int *begin(EndFunctionWithDataMember const&) { return nullptr; }
  friend constexpr const int *end(EndFunctionWithDataMember const& bf) { return &bf.x; }
};

struct EndFunctionWithPrivateEndMember {
  int y;
  friend constexpr const int *begin(EndFunctionWithPrivateEndMember const&) { return nullptr; }
  friend constexpr const int *end(EndFunctionWithPrivateEndMember const& bf) { return &bf.y; }
private:
  const int *end() const;
};

struct BeginMemberEndFunction {
  int x;
  constexpr const int *begin() const { return nullptr; }
  friend constexpr const int *end(BeginMemberEndFunction const& bf) { return &bf.x; }
};

constexpr bool testEndFunction() {
  const EndFunction a{};
  EndFunction aa{};
  EndFunctionByValue b;
  EndFunctionEnabledBorrowing c;
  const EndFunctionReturnsEmptyPtr d{};
  EndFunctionReturnsEmptyPtr dd{};
  const EndFunctionWithDataMember e{};
  EndFunctionWithDataMember ee{};
  const EndFunctionWithPrivateEndMember f{};
  EndFunctionWithPrivateEndMember ff{};
  const BeginMemberEndFunction g{};
  BeginMemberEndFunction gg{};

  assert(std::ranges::end(a) == &a.x);
  assert(std::ranges::end(aa) == &aa.x); // Ill-formed before P2602R2 Poison Pills are Too Toxic
  assert(std::ranges::end(b) == &globalBuff[1]);
  assert(std::ranges::end(std::move(c)) == &globalBuff[2]);
  assert(std::ranges::end(d) == &d.x);
  assert(std::ranges::end(dd) == &dd.x); // Ill-formed before P2602R2 Poison Pills are Too Toxic
  assert(std::ranges::end(e) == &e.x);
  assert(std::ranges::end(ee) == &ee.x); // Ill-formed before P2602R2 Poison Pills are Too Toxic
  assert(std::ranges::end(f) == &f.y);
  assert(std::ranges::end(ff) == &ff.y); // Ill-formed before P2602R2 Poison Pills are Too Toxic
  assert(std::ranges::end(g) == &g.x);
  assert(std::ranges::end(gg) == &gg.x); // Ill-formed before P2602R2 Poison Pills are Too Toxic

#if TEST_STD_VER < 23
  assert(std::ranges::cend(a) == &a.x);
  assert(std::ranges::cend(aa) == &aa.x);
  assert(std::ranges::cend(b) == &globalBuff[1]);
  assert(std::ranges::cend(std::move(c)) == &globalBuff[2]);
  assert(std::ranges::cend(d) == &d.x);
  assert(std::ranges::cend(dd) == &dd.x);
  assert(std::ranges::cend(e) == &e.x);
  assert(std::ranges::cend(ee) == &ee.x);
  assert(std::ranges::cend(f) == &f.y);
  assert(std::ranges::cend(ff) == &ff.y);
  assert(std::ranges::cend(g) == &g.x);
  assert(std::ranges::cend(gg) == &gg.x);
#endif // TEST_STD_VER < 23

  return true;
}


ASSERT_NOEXCEPT(std::ranges::end(std::declval<int (&)[10]>()));
#if TEST_STD_VER < 23
ASSERT_NOEXCEPT(std::ranges::cend(std::declval<int (&)[10]>()));
#endif // TEST_STD_VER < 23

struct NoThrowMemberEnd {
  ThrowingIterator<int> begin() const;
  ThrowingIterator<int> end() const noexcept; // auto(t.end()) doesn't throw
} ntme;
static_assert(noexcept(std::ranges::end(ntme)));
#if TEST_STD_VER < 23
static_assert(noexcept(std::ranges::cend(ntme)));
#endif // TEST_STD_VER < 23

struct NoThrowADLEnd {
  ThrowingIterator<int> begin() const;
  friend ThrowingIterator<int> end(NoThrowADLEnd&) noexcept;  // auto(end(t)) doesn't throw
  friend ThrowingIterator<int> end(const NoThrowADLEnd&) noexcept;
} ntae;
static_assert(noexcept(std::ranges::end(ntae)));
#if TEST_STD_VER < 23
static_assert(noexcept(std::ranges::cend(ntae)));
#endif // TEST_STD_VER < 23

struct NoThrowMemberEndReturnsRef {
  ThrowingIterator<int> begin() const;
  ThrowingIterator<int>& end() const noexcept; // auto(t.end()) may throw
} ntmerr;
static_assert(!noexcept(std::ranges::end(ntmerr)));
#if TEST_STD_VER < 23
static_assert(!noexcept(std::ranges::cend(ntmerr)));
#endif // TEST_STD_VER < 23

struct EndReturnsArrayRef {
    auto begin() const noexcept -> int(&)[10];
    auto end() const noexcept -> int(&)[10];
} erar;
static_assert(noexcept(std::ranges::end(erar)));
#if TEST_STD_VER < 23
static_assert(noexcept(std::ranges::cend(erar)));
#endif // TEST_STD_VER < 23

// Test ADL-proofing.
struct Incomplete;
template<class T> struct Holder { T t; };
static_assert(!std::is_invocable_v<RangeEndT, Holder<Incomplete>*>);
static_assert(!std::is_invocable_v<RangeEndT, Holder<Incomplete>*&>);
#if TEST_STD_VER < 23
static_assert(!std::is_invocable_v<RangeCEndT, Holder<Incomplete>*>);
static_assert(!std::is_invocable_v<RangeCEndT, Holder<Incomplete>*&>);
#endif // TEST_STD_VER < 23

int main(int, char**) {
  static_assert(testReturnTypes());

  testArray();
  static_assert(testArray());

  testEndMember();
  static_assert(testEndMember());

  testEndFunction();
  static_assert(testEndFunction());

  return 0;
}
