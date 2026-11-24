//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for cpp::span.
///
//===----------------------------------------------------------------------===//

#include "hdr/stdint_proxy.h"
#include "src/__support/CPP/array.h"
#include "src/__support/CPP/cstddef.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/type_traits.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::array;
using LIBC_NAMESPACE::cpp::as_bytes;
using LIBC_NAMESPACE::cpp::as_writable_bytes;
using LIBC_NAMESPACE::cpp::byte;
using LIBC_NAMESPACE::cpp::is_same_v;
using LIBC_NAMESPACE::cpp::span;
using LIBC_NAMESPACE::cpp::to_integer;

namespace {

template <typename T, typename = void>
struct HasWritableBytes : LIBC_NAMESPACE::cpp::false_type {};

template <typename T>
struct HasWritableBytes<
    T, LIBC_NAMESPACE::cpp::void_t<decltype(as_writable_bytes(span<T>()))>>
    : LIBC_NAMESPACE::cpp::true_type {};

static_assert(HasWritableBytes<int>::value);
static_assert(HasWritableBytes<byte>::value);
static_assert(!HasWritableBytes<const int>::value);
static_assert(!HasWritableBytes<const byte>::value);

struct Pair {
  int first;
  int second;
};

constexpr bool test_constexpr_span() {
  constexpr int CONST_ARR[] = {10, 20, 30, 40};
  span const_s(CONST_ARR);
  if (const_s.size() != 4 || const_s.size_bytes() != 4 * sizeof(int))
    return false;
  if (const_s.front() != 10 || const_s.back() != 40 || const_s[2] != 30)
    return false;

  int mut_arr[] = {1, 2, 3, 4, 5};
  span mut_s(mut_arr);
  mut_s[0] = 11;
  if (mut_s.front() != 11)
    return false;

  span range_s(mut_arr + 1, mut_arr + 4);
  if (range_s.size() != 3 || range_s[0] != 2 || range_s.back() != 4)
    return false;

  span empty_null(static_cast<int *>(nullptr), static_cast<int *>(nullptr));
  if (!empty_null.empty() || empty_null.size() != 0)
    return false;

  span zero_count(mut_arr, 0);
  if (!zero_count.empty())
    return false;

  // Forward and const iterators.
  int sum = 0;
  for (auto it = mut_s.cbegin(); it != mut_s.cend(); ++it)
    sum += *it;
  if (sum != (11 + 2 + 3 + 4 + 5))
    return false;

  // Reverse iterators.
  int rev_vals[5] = {};
  size_t idx = 0;
  for (auto rit = mut_s.rbegin(); rit != mut_s.rend(); ++rit)
    rev_vals[idx++] = *rit;
  if (rev_vals[0] != 5 || rev_vals[4] != 11)
    return false;

  auto crit = mut_s.crbegin();
  if (*crit != 5 || crit[4] != 11 || (mut_s.crend() - crit) != 5)
    return false;
  ++crit;
  --crit;
  if (*crit != 5)
    return false;

  // Runtime and compile-time subviews.
  const auto FIRST2 = mut_s.first(2);
  const auto FIRST2_CT = mut_s.first<2>();
  if (FIRST2.size() != 2 || FIRST2[0] != 11 || FIRST2_CT[1] != 2)
    return false;

  const auto LAST2 = mut_s.last(2);
  const auto LAST2_CT = mut_s.last<2>();
  if (LAST2.size() != 2 || LAST2[0] != 4 || LAST2_CT[1] != 5)
    return false;

  const auto SUB = mut_s.subspan(1, 3);
  const auto SUB_CT = mut_s.subspan<1, 3>();
  const auto SUB_TAIL_CT = mut_s.subspan<3>();
  if (SUB.size() != 3 || SUB[0] != 2 || SUB_CT[2] != 4)
    return false;
  if (SUB_TAIL_CT.size() != 2 || SUB_TAIL_CT[0] != 4 || SUB_TAIL_CT[1] != 5)
    return false;

  return true;
}

static_assert(test_constexpr_span());

} // namespace

TEST(LlvmLibcSpanTest, InitializeEmpty) {
  span<int> s;
  ASSERT_EQ(s.size(), size_t(0));
  ASSERT_EQ(s.size_bytes(), size_t(0));
  ASSERT_TRUE(s.empty());
  ASSERT_EQ(s.data(), nullptr);
  ASSERT_EQ(s.begin(), s.end());
  ASSERT_EQ(s.cbegin(), s.cend());
  ASSERT_EQ(s.rbegin().base(), s.rend().base());
  ASSERT_EQ(s.crbegin().base(), s.crend().base());
  ASSERT_TRUE(s.rbegin() == s.rend());
  ASSERT_TRUE(s.crbegin() == s.crend());
}

TEST(LlvmLibcSpanTest, InitializeSingleton) {
  int a = 42;
  span<int> s(&a, 1);
  ASSERT_EQ(s.size(), size_t(1));
  ASSERT_EQ(s.size_bytes(), sizeof(int));
  ASSERT_FALSE(s.empty());
  ASSERT_EQ(s.data(), &a);
  ASSERT_EQ(s.front(), 42);
  ASSERT_EQ(s.back(), 42);
}

TEST(LlvmLibcSpanTest, InitializePointerRange) {
  int a[] = {10, 20, 30, 40};
  span<int> s(a + 1, a + 3);
  ASSERT_EQ(s.size(), size_t(2));
  ASSERT_EQ(s.data(), &a[1]);
  ASSERT_EQ(s[0], 20);
  ASSERT_EQ(s[1], 30);

  span<const int> cs(a, a + 4);
  ASSERT_EQ(cs.size(), size_t(4));
  ASSERT_EQ(cs.front(), 10);
  ASSERT_EQ(cs.back(), 40);

  span<int> empty_same(a + 2, a + 2);
  ASSERT_TRUE(empty_same.empty());
  ASSERT_EQ(empty_same.data(), &a[2]);

  span<int> empty_null(nullptr, nullptr);
  ASSERT_TRUE(empty_null.empty());
  ASSERT_EQ(empty_null.data(), nullptr);

  // Verify literal 0 resolves unambiguously to (pointer, size_type).
  span<int> empty_zero(a, 0);
  ASSERT_TRUE(empty_zero.empty());
  ASSERT_EQ(empty_zero.data(), &a[0]);
}

TEST(LlvmLibcSpanTest, InitializeCArray) {
  int a[] = {1, 2, 3};
  span<int> s(a);
  ASSERT_EQ(s.size(), size_t(3));
  ASSERT_FALSE(s.empty());
  ASSERT_EQ(s.data(), &a[0]);
  ASSERT_EQ(s.front(), 1);
  ASSERT_EQ(s.back(), 3);
  ASSERT_EQ(s[0], 1);
  ASSERT_EQ(s[1], 2);
  ASSERT_EQ(s[2], 3);
}

TEST(LlvmLibcSpanTest, InitializeArray) {
  array<int, 3> a = {1, 2, 3};
  span<int> s(a);
  ASSERT_EQ(s.size(), size_t(3));
  ASSERT_FALSE(s.empty());
  ASSERT_EQ(s.data(), &a[0]);
  ASSERT_EQ(s.front(), 1);
  ASSERT_EQ(s.back(), 3);
  ASSERT_EQ(s[0], 1);
  ASSERT_EQ(s[1], 2);
  ASSERT_EQ(s[2], 3);
}

TEST(LlvmLibcSpanTest, InitializeViewFormMutableSingleton) {
  int a = 42;
  span<const int> s(&a, 1);
  ASSERT_EQ(s.size(), size_t(1));
  ASSERT_EQ(s.data(), static_cast<const int *>(&a));
}

TEST(LlvmLibcSpanTest, InitializeViewFormMutableCArray) {
  int a[] = {1, 2, 3};
  span<const int> s(a);
  ASSERT_EQ(s.size(), size_t(3));
  ASSERT_EQ(s[0], 1);
  ASSERT_EQ(s[1], 2);
  ASSERT_EQ(s[2], 3);
}

TEST(LlvmLibcSpanTest, InitializeViewFormMutableArray) {
  array<int, 3> a = {1, 2, 3};
  span<const int> s(a);
  ASSERT_EQ(s.size(), size_t(3));
  ASSERT_EQ(s[0], 1);
  ASSERT_EQ(s[1], 2);
  ASSERT_EQ(s[2], 3);
}

TEST(LlvmLibcSpanTest, InitializeViewFromConstArray) {
  const array<int, 3> A = {1, 2, 3};
  span<const int> s(A);
  ASSERT_EQ(s.size(), size_t(3));
  ASSERT_EQ(s[0], 1);
  ASSERT_EQ(s[1], 2);
  ASSERT_EQ(s[2], 3);
}

TEST(LlvmLibcSpanTest, InitializeFromMutable) {
  int a[] = {1, 2, 3};
  span<int> s(a);
  span<const int> view(s);
  ASSERT_EQ(view.size(), size_t(3));
  ASSERT_EQ(view.data(), static_cast<const int *>(&a[0]));
  ASSERT_EQ(view[0], 1);
  ASSERT_EQ(view[1], 2);
  ASSERT_EQ(view[2], 3);

  const span<int> CONST_S(a);
  span<const int> view_from_const(CONST_S);
  ASSERT_EQ(view_from_const.size(), size_t(3));
  ASSERT_EQ(view_from_const.data(), static_cast<const int *>(&a[0]));

  span<const int> view_from_rvalue(s.subspan(1));
  ASSERT_EQ(view_from_rvalue.size(), size_t(2));
  ASSERT_EQ(view_from_rvalue.data(), static_cast<const int *>(&a[1]));
  ASSERT_EQ(view_from_rvalue[0], 2);
  ASSERT_EQ(view_from_rvalue[1], 3);
}

TEST(LlvmLibcSpanTest, DeductionGuides) {
  int c_arr[] = {1, 2, 3};
  span s_c_arr(c_arr);
  static_assert(is_same_v<decltype(s_c_arr), span<int>>);
  ASSERT_EQ(s_c_arr.size(), size_t(3));

  const int CONST_C_ARR[] = {4, 5};
  span s_const_c_arr(CONST_C_ARR);
  static_assert(is_same_v<decltype(s_const_c_arr), span<const int>>);
  ASSERT_EQ(s_const_c_arr.size(), size_t(2));

  array<int, 4> cpp_arr = {1, 2, 3, 4};
  span s_cpp_arr(cpp_arr);
  static_assert(is_same_v<decltype(s_cpp_arr), span<int>>);
  ASSERT_EQ(s_cpp_arr.size(), size_t(4));

  const array<int, 2> CONST_CPP_ARR = {7, 8};
  span s_const_cpp_arr(CONST_CPP_ARR);
  static_assert(is_same_v<decltype(s_const_cpp_arr), span<const int>>);
  ASSERT_EQ(s_const_cpp_arr.size(), size_t(2));

  int *ptr = c_arr;
  span s_ptr_len(ptr, size_t(2));
  static_assert(is_same_v<decltype(s_ptr_len), span<int>>);
  ASSERT_EQ(s_ptr_len.size(), size_t(2));

  span s_ptr_zero(ptr, 0);
  static_assert(is_same_v<decltype(s_ptr_zero), span<int>>);
  ASSERT_TRUE(s_ptr_zero.empty());

  const int *const_ptr = CONST_C_ARR;
  span s_ptr_range(const_ptr, const_ptr + 2);
  static_assert(is_same_v<decltype(s_ptr_range), span<const int>>);
  ASSERT_EQ(s_ptr_range.size(), size_t(2));
}

TEST(LlvmLibcSpanTest, Assign) {
  span<int> s;
  span<int> other;
  other = s;
  ASSERT_TRUE(other.empty());
}

TEST(LlvmLibcSpanTest, CopyAssignment) {
  int a[] = {1, 2, 3};
  span<int> s1(a);
  span<int> s2;
  s2 = s1;
  ASSERT_EQ(s2.size(), size_t(3));
  ASSERT_EQ(s2.data(), &a[0]);
  ASSERT_EQ(s2[0], 1);
  ASSERT_EQ(s2[1], 2);
  ASSERT_EQ(s2[2], 3);
}

TEST(LlvmLibcSpanTest, ReassignFromRValueTemporary) {
  int a[] = {1, 2, 3, 4};
  span<int> s(a);
  s = s.subspan(1);
  ASSERT_EQ(s.size(), size_t(3));
  ASSERT_EQ(s.data(), &a[1]);
  ASSERT_EQ(s[0], 2);
  ASSERT_EQ(s[1], 3);
  ASSERT_EQ(s[2], 4);

  s = span<int>();
  ASSERT_EQ(s.size(), size_t(0));
  ASSERT_TRUE(s.empty());
  ASSERT_EQ(s.data(), nullptr);
}

TEST(LlvmLibcSpanTest, AssignFromMutable) {
  span<int> s;
  span<const int> view;
  view = s;
  ASSERT_TRUE(view.empty());
}

TEST(LlvmLibcSpanTest, ConvertingAssignment) {
  int a[] = {1, 2, 3};
  span<int> s(a);
  span<const int> view;
  view = s;
  ASSERT_EQ(view.size(), size_t(3));
  ASSERT_EQ(view.data(), static_cast<const int *>(&a[0]));
  ASSERT_EQ(view[0], 1);
  ASSERT_EQ(view[1], 2);
  ASSERT_EQ(view[2], 3);

  view = s.subspan(1);
  ASSERT_EQ(view.size(), size_t(2));
  ASSERT_EQ(view.data(), static_cast<const int *>(&a[1]));
  ASSERT_EQ(view[0], 2);
  ASSERT_EQ(view[1], 3);

  const span<int> CONST_S(a);
  view = CONST_S;
  ASSERT_EQ(view.size(), size_t(3));
  ASSERT_EQ(view.data(), static_cast<const int *>(&a[0]));

  view = span<int>();
  ASSERT_EQ(view.size(), size_t(0));
  ASSERT_TRUE(view.empty());
  ASSERT_EQ(view.data(), nullptr);
}

TEST(LlvmLibcSpanTest, Modify) {
  int a[] = {1, 2, 3};
  span<int> s(a);
  for (int &value : s)
    ++value;
  ASSERT_EQ(s.size(), size_t(3));
  ASSERT_EQ(s[0], 2);
  ASSERT_EQ(s[1], 3);
  ASSERT_EQ(s[2], 4);
}

TEST(LlvmLibcSpanTest, IteratorsAndReverseIterators) {
  Pair pairs[] = {{1, 10}, {2, 20}, {3, 30}};
  span<Pair> s(pairs);

  static_assert(is_same_v<decltype(s.begin()), Pair *>);
  static_assert(is_same_v<decltype(s.cbegin()), const Pair *>);

  ASSERT_EQ(s.cbegin(), static_cast<const Pair *>(&pairs[0]));
  ASSERT_EQ(s.cend(), static_cast<const Pair *>(&pairs[0] + 3));

  auto rit = s.rbegin();
  ASSERT_EQ(rit->first, 3);
  ASSERT_EQ(rit->second, 30);
  ASSERT_EQ(rit[2].first, 1);

  rit->second = 300;
  ASSERT_EQ(pairs[2].second, 300);

  auto prev = rit++;
  ASSERT_EQ(prev->first, 3);
  ASSERT_EQ(rit->first, 2);

  auto next = rit--;
  ASSERT_EQ(next->first, 2);
  ASSERT_EQ(rit->first, 3);

  ++rit;
  --rit;
  ASSERT_EQ(rit.base(), s.rbegin().base());
  ASSERT_TRUE(rit == s.rbegin());
  ASSERT_EQ(s.rend() - s.rbegin(), ptrdiff_t(3));

  // Implicit conversion from reverse_iterator to const_reverse_iterator.
  span<Pair>::const_reverse_iterator crit = s.rbegin();
  ASSERT_EQ(crit.base(), s.crbegin().base());
  ASSERT_TRUE(crit == s.crbegin());
  ASSERT_TRUE(crit != s.crend());
  ASSERT_EQ((crit + 2)->first, 1);
}

TEST(LlvmLibcSpanTest, SubSpan) {
  int a[] = {1, 2, 3};
  span<const int> s(a);
  {
    const auto SUB = s.subspan(0);
    ASSERT_EQ(SUB.size(), size_t(3));
    ASSERT_EQ(SUB[0], 1);
    ASSERT_EQ(SUB[1], 2);
    ASSERT_EQ(SUB[2], 3);
  }
  {
    const auto SUB = s.subspan(2);
    ASSERT_EQ(SUB.size(), size_t(1));
    ASSERT_EQ(SUB[0], 3);
  }
  {
    const auto SUB = s.subspan(3);
    ASSERT_EQ(SUB.size(), size_t(0));
  }
  {
    const auto SUB = s.subspan(0, 1);
    ASSERT_EQ(SUB.size(), size_t(1));
    ASSERT_EQ(SUB[0], 1);
  }
}

TEST(LlvmLibcSpanTest, FirstAndLastSubSpan) {
  int a[] = {1, 2, 3};
  span<const int> s(a);

  const auto FIRST = s.first(2);
  ASSERT_EQ(FIRST.size(), size_t(2));
  ASSERT_EQ(FIRST[0], 1);
  ASSERT_EQ(FIRST[1], 2);

  const auto LAST = s.last(2);
  ASSERT_EQ(LAST.size(), size_t(2));
  ASSERT_EQ(LAST[0], 2);
  ASSERT_EQ(LAST[1], 3);

  ASSERT_TRUE(s.first(0).empty());
  ASSERT_TRUE(s.last(0).empty());
}

TEST(LlvmLibcSpanTest, CompileTimeSubviews) {
  int a[] = {10, 20, 30, 40};
  span<int> s(a);

  const auto FIRST_ZERO = s.first<0>();
  ASSERT_TRUE(FIRST_ZERO.empty());

  const auto FIRST_TWO = s.first<2>();
  ASSERT_EQ(FIRST_TWO.size(), size_t(2));
  ASSERT_EQ(FIRST_TWO[0], 10);
  ASSERT_EQ(FIRST_TWO[1], 20);

  const auto LAST_ZERO = s.last<0>();
  ASSERT_TRUE(LAST_ZERO.empty());

  const auto LAST_TWO = s.last<2>();
  ASSERT_EQ(LAST_TWO.size(), size_t(2));
  ASSERT_EQ(LAST_TWO[0], 30);
  ASSERT_EQ(LAST_TWO[1], 40);

  const auto SUB_DYNAMIC = s.subspan<1>();
  ASSERT_EQ(SUB_DYNAMIC.size(), size_t(3));
  ASSERT_EQ(SUB_DYNAMIC[0], 20);
  ASSERT_EQ(SUB_DYNAMIC[2], 40);

  const auto SUB_FIXED = s.subspan<1, 2>();
  ASSERT_EQ(SUB_FIXED.size(), size_t(2));
  ASSERT_EQ(SUB_FIXED[0], 20);
  ASSERT_EQ(SUB_FIXED[1], 30);

  const auto SUB_END = s.subspan<4, 0>();
  ASSERT_TRUE(SUB_END.empty());
}

TEST(LlvmLibcSpanTest, AsBytesAndWritableBytes) {
  uint32_t values[] = {0x01020304U, 0x05060708U};
  span<uint32_t> s(values);

  span<const byte> ro_bytes = as_bytes(s);
  ASSERT_EQ(ro_bytes.size(), sizeof(values));
  ASSERT_EQ(ro_bytes.data(), reinterpret_cast<const byte *>(values));

  const auto *raw = reinterpret_cast<const unsigned char *>(values);
  for (size_t i = 0; i < ro_bytes.size(); ++i)
    EXPECT_EQ(to_integer<unsigned char>(ro_bytes[i]), raw[i]);

  span<byte> rw_bytes = as_writable_bytes(s);
  ASSERT_EQ(rw_bytes.size(), sizeof(values));
  for (byte &b : rw_bytes)
    b = byte{0};
  EXPECT_EQ(values[0], uint32_t(0));
  EXPECT_EQ(values[1], uint32_t(0));

  span<const uint32_t> cs(values);
  span<const byte> cs_bytes = as_bytes(cs);
  ASSERT_EQ(cs_bytes.size(), sizeof(values));
}

#ifndef NDEBUG
TEST(LlvmLibcSpanTest, PreconditionDeath) {
  int a[] = {1, 2, 3};
  span<int> empty_s;
  span<int> s(a);

  // Inverted [first, end) range.
  ASSERT_DEATH(([] {
                 int arr[] = {1, 2, 3};
                 span<int> bad(arr + 2, arr + 1);
                 (void)bad;
               }),
               WITH_SIGNAL(-1));
  ASSERT_DEATH(([&] {
                 span<int> bad(a, nullptr);
                 (void)bad;
               }),
               WITH_SIGNAL(-1));

  // Out-of-bounds operator[].
  ASSERT_DEATH(([&] { (void)empty_s[0]; }), WITH_SIGNAL(-1));
  ASSERT_DEATH(([&] { (void)s[3]; }), WITH_SIGNAL(-1));

  // Empty front() and back().
  ASSERT_DEATH(([&] { (void)empty_s.front(); }), WITH_SIGNAL(-1));
  ASSERT_DEATH(([&] { (void)empty_s.back(); }), WITH_SIGNAL(-1));

  // Out-of-bounds first() and last().
  ASSERT_DEATH(([&] { (void)s.first(4); }), WITH_SIGNAL(-1));
  ASSERT_DEATH(([&] { (void)s.first<4>(); }), WITH_SIGNAL(-1));
  ASSERT_DEATH(([&] { (void)s.last(4); }), WITH_SIGNAL(-1));
  ASSERT_DEATH(([&] { (void)s.last<4>(); }), WITH_SIGNAL(-1));

  // Out-of-bounds and wrapping subspan().
  ASSERT_DEATH(([&] { (void)s.subspan(4); }), WITH_SIGNAL(-1));
  ASSERT_DEATH(([&] { (void)s.subspan<4>(); }), WITH_SIGNAL(-1));
  ASSERT_DEATH(([&] { (void)s.subspan(2, 2); }), WITH_SIGNAL(-1));
  ASSERT_DEATH(([&] { (void)s.subspan<2, 2>(); }), WITH_SIGNAL(-1));
  ASSERT_DEATH(([&] {
                 (void)s.subspan(
                     2, LIBC_NAMESPACE::cpp::numeric_limits<size_t>::max() - 1);
               }),
               WITH_SIGNAL(-1));
}
#endif // NDEBUG
