//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::ranges::size

#include <ranges>

#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"

using RangeSizeT = decltype(std::ranges::size);

static_assert(!std::is_invocable_v<RangeSizeT, int[]>);
static_assert( std::is_invocable_v<RangeSizeT, int[1]>);
static_assert( std::is_invocable_v<RangeSizeT, int (&&)[1]>);
static_assert( std::is_invocable_v<RangeSizeT, int (&)[1]>);

struct Incomplete;
static_assert(!std::is_invocable_v<RangeSizeT, Incomplete[]>);
static_assert(!std::is_invocable_v<RangeSizeT, Incomplete(&)[]>);
static_assert(!std::is_invocable_v<RangeSizeT, Incomplete(&&)[]>);

extern Incomplete array_of_incomplete[42];
static_assert(std::ranges::size(array_of_incomplete) == 42);
static_assert(std::ranges::size(std::move(array_of_incomplete)) == 42);
static_assert(std::ranges::size(std::as_const(array_of_incomplete)) == 42);
static_assert(std::ranges::size(static_cast<const Incomplete(&&)[42]>(array_of_incomplete)) == 42);

struct SizeMember {
  constexpr size_t size() { return 42; }
};

struct StaticSizeMember {
  constexpr static size_t size() { return 42; }
};

static_assert(!std::is_invocable_v<RangeSizeT, const SizeMember>);

struct SizeFunction {
  friend constexpr size_t size(SizeFunction) { return 42; }
};

// Make sure the size member is preferred.
struct SizeMemberAndFunction {
  constexpr size_t size() { return 42; }
  friend constexpr size_t size(SizeMemberAndFunction) { return 0; }
};

bool constexpr testArrayType() {
  int a[4];
  int b[1];
  SizeMember c[4];
  SizeFunction d[4];

  assert(std::ranges::size(a) == 4);
  ASSERT_SAME_TYPE(decltype(std::ranges::size(a)), size_t);
  assert(std::ranges::size(b) == 1);
  ASSERT_SAME_TYPE(decltype(std::ranges::size(b)), size_t);
  assert(std::ranges::size(c) == 4);
  ASSERT_SAME_TYPE(decltype(std::ranges::size(c)), size_t);
  assert(std::ranges::size(d) == 4);
  ASSERT_SAME_TYPE(decltype(std::ranges::size(d)), size_t);

  return true;
}

struct SizeMemberConst {
  constexpr size_t size() const { return 42; }
};

struct SizeMemberSigned {
  constexpr long size() { return 42; }
};

bool constexpr testHasSizeMember() {
  assert(std::ranges::size(SizeMember()) == 42);
  ASSERT_SAME_TYPE(decltype(std::ranges::size(SizeMember())), size_t);

  const SizeMemberConst sizeMemberConst;
  assert(std::ranges::size(sizeMemberConst) == 42);

  assert(std::ranges::size(SizeMemberAndFunction()) == 42);

  assert(std::ranges::size(SizeMemberSigned()) == 42);
  ASSERT_SAME_TYPE(decltype(std::ranges::size(SizeMemberSigned())), long);

  assert(std::ranges::size(StaticSizeMember()) == 42);
  ASSERT_SAME_TYPE(decltype(std::ranges::size(StaticSizeMember())), size_t);

  return true;
}

struct MoveOnlySizeFunction {
  MoveOnlySizeFunction() = default;
  MoveOnlySizeFunction(MoveOnlySizeFunction &&) = default;
  MoveOnlySizeFunction(MoveOnlySizeFunction const&) = delete;

  friend constexpr size_t size(MoveOnlySizeFunction) { return 42; }
};

enum EnumSizeFunction {
  a, b
};

constexpr size_t size(EnumSizeFunction) { return 42; }

struct SizeFunctionConst {
  friend constexpr size_t size(const SizeFunctionConst) { return 42; }
};

struct SizeFunctionRef {
  friend constexpr size_t size(SizeFunctionRef&) { return 42; }
};

struct SizeFunctionConstRef {
  friend constexpr size_t size(SizeFunctionConstRef const&) { return 42; }
};

struct SizeFunctionSigned {
  friend constexpr long size(SizeFunctionSigned) { return 42; }
};

bool constexpr testHasSizeFunction() {
  assert(std::ranges::size(SizeFunction()) == 42);
  ASSERT_SAME_TYPE(decltype(std::ranges::size(SizeFunction())), size_t);
  static_assert(!std::is_invocable_v<RangeSizeT, MoveOnlySizeFunction>);
  assert(std::ranges::size(EnumSizeFunction()) == 42);
  assert(std::ranges::size(SizeFunctionConst()) == 42);

  SizeFunctionRef a;
  assert(std::ranges::size(a) == 42);

  const SizeFunctionConstRef b;
  assert(std::ranges::size(b) == 42);

  assert(std::ranges::size(SizeFunctionSigned()) == 42);
  ASSERT_SAME_TYPE(decltype(std::ranges::size(SizeFunctionSigned())), long);

  return true;
}

struct Empty { };
static_assert(!std::is_invocable_v<RangeSizeT, Empty>);

struct InvalidReturnTypeMember {
  Empty size();
};

struct InvalidReturnTypeFunction {
  friend Empty size(InvalidReturnTypeFunction);
};

struct Convertible {
  operator size_t();
};

struct ConvertibleReturnTypeMember {
  Convertible size();
};

struct ConvertibleReturnTypeFunction {
  friend Convertible size(ConvertibleReturnTypeFunction);
};

struct BoolReturnTypeMember {
  bool size() const;
};

struct BoolReturnTypeFunction {
  friend bool size(BoolReturnTypeFunction const&);
};

static_assert(!std::is_invocable_v<RangeSizeT, InvalidReturnTypeMember>);
static_assert(!std::is_invocable_v<RangeSizeT, InvalidReturnTypeFunction>);
static_assert( std::is_invocable_v<RangeSizeT, InvalidReturnTypeMember (&)[4]>);
static_assert( std::is_invocable_v<RangeSizeT, InvalidReturnTypeFunction (&)[4]>);
static_assert(!std::is_invocable_v<RangeSizeT, ConvertibleReturnTypeMember>);
static_assert(!std::is_invocable_v<RangeSizeT, ConvertibleReturnTypeFunction>);
static_assert(!std::is_invocable_v<RangeSizeT, BoolReturnTypeMember const&>);
static_assert(!std::is_invocable_v<RangeSizeT, BoolReturnTypeFunction const&>);

struct SizeMemberDisabled {
  size_t size() { return 42; }
};

template <>
inline constexpr bool std::ranges::disable_sized_range<SizeMemberDisabled> = true;

struct ImproperlyDisabledMember {
  size_t size() const { return 42; }
};

// Intentionally disabling "const ConstSizeMemberDisabled". This doesn't disable anything
// because T is always uncvrefed before being checked.
template <>
inline constexpr bool std::ranges::disable_sized_range<const ImproperlyDisabledMember> = true;

struct SizeFunctionDisabled {
  friend size_t size(SizeFunctionDisabled) { return 42; }
};

template <>
inline constexpr bool std::ranges::disable_sized_range<SizeFunctionDisabled> = true;

struct ImproperlyDisabledFunction {
  friend size_t size(ImproperlyDisabledFunction const&) { return 42; }
};

template <>
inline constexpr bool std::ranges::disable_sized_range<const ImproperlyDisabledFunction> = true;

static_assert( std::is_invocable_v<RangeSizeT, ImproperlyDisabledMember&>);
static_assert( std::is_invocable_v<RangeSizeT, const ImproperlyDisabledMember&>);
static_assert(!std::is_invocable_v<RangeSizeT, ImproperlyDisabledFunction&>);
static_assert( std::is_invocable_v<RangeSizeT, const ImproperlyDisabledFunction&>);

// No begin end.
struct HasMinusOperator {
  friend constexpr size_t operator-(HasMinusOperator, HasMinusOperator) { return 2; }
};
static_assert(!std::is_invocable_v<RangeSizeT, HasMinusOperator>);

struct HasMinusBeginEnd {
  struct sentinel {
    friend bool operator==(sentinel, forward_iterator<int*>);
    friend constexpr std::ptrdiff_t operator-(const sentinel, const forward_iterator<int*>) { return 2; }
    friend constexpr std::ptrdiff_t operator-(const forward_iterator<int*>, const sentinel) { return 2; }
  };

  friend constexpr forward_iterator<int*> begin(HasMinusBeginEnd) { return {}; }
  friend constexpr sentinel end(HasMinusBeginEnd) { return {}; }
};

struct other_forward_iterator : forward_iterator<int*> { };

struct InvalidMinusBeginEnd {
  struct sentinel {
    friend bool operator==(sentinel, other_forward_iterator);
    friend constexpr std::ptrdiff_t operator-(const sentinel, const other_forward_iterator) { return 2; }
    friend constexpr std::ptrdiff_t operator-(const other_forward_iterator, const sentinel) { return 2; }
  };

  friend constexpr other_forward_iterator begin(InvalidMinusBeginEnd) { return {}; }
  friend constexpr sentinel end(InvalidMinusBeginEnd) { return {}; }
};

// short is integer-like, but it is not other_forward_iterator's difference_type.
static_assert(!std::same_as<other_forward_iterator::difference_type, short>);
static_assert(!std::is_invocable_v<RangeSizeT, InvalidMinusBeginEnd>);

struct RandomAccessRange {
  struct sentinel {
    friend bool operator==(sentinel, random_access_iterator<int*>);
    friend constexpr std::ptrdiff_t operator-(const sentinel, const random_access_iterator<int*>) { return 2; }
    friend constexpr std::ptrdiff_t operator-(const random_access_iterator<int*>, const sentinel) { return 2; }
  };

  constexpr random_access_iterator<int*> begin() { return {}; }
  constexpr sentinel end() { return {}; }
};

struct IntPtrBeginAndEnd {
  int buff[8];
  constexpr int* begin() { return buff; }
  constexpr int* end() { return buff + 8; }
};

struct DisabledSizeRangeWithBeginEnd {
  int buff[8];
  constexpr int* begin() { return buff; }
  constexpr int* end() { return buff + 8; }
  constexpr size_t size() { return 1; }
};

template <>
inline constexpr bool std::ranges::disable_sized_range<DisabledSizeRangeWithBeginEnd> = true;

struct SizeBeginAndEndMembers {
  int buff[8];
  constexpr int* begin() { return buff; }
  constexpr int* end() { return buff + 8; }
  constexpr size_t size() { return 1; }
};

constexpr bool testRanges() {
  HasMinusBeginEnd a;
  assert(std::ranges::size(a) == 2);
  // Ensure that this is converted to an *unsigned* type.
  ASSERT_SAME_TYPE(decltype(std::ranges::size(a)), size_t);

  IntPtrBeginAndEnd b;
  assert(std::ranges::size(b) == 8);

  DisabledSizeRangeWithBeginEnd c;
  assert(std::ranges::size(c) == 8);

  RandomAccessRange d;
  assert(std::ranges::size(d) == 2);
  ASSERT_SAME_TYPE(decltype(std::ranges::size(d)), size_t);

  SizeBeginAndEndMembers e;
  assert(std::ranges::size(e) == 1);

  return true;
}

// Test ADL-proofing.
struct Incomplete;
template<class T> struct Holder { T t; };
static_assert(!std::is_invocable_v<RangeSizeT, Holder<Incomplete>*>);
static_assert(!std::is_invocable_v<RangeSizeT, Holder<Incomplete>*&>);

int main(int, char**) {
  testArrayType();
  static_assert(testArrayType());

  testHasSizeMember();
  static_assert(testHasSizeMember());

  testHasSizeFunction();
  static_assert(testHasSizeFunction());

  testRanges();
  static_assert(testRanges());

  return 0;
}
