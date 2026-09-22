//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class In>
// concept std::weakly_incrementable;

#include <iterator>

#include <concepts>
#include <memory>
#include <optional>

#include "../incrementable.h"
#include "test_macros.h"

static_assert(std::weakly_incrementable<int>);
static_assert(std::weakly_incrementable<int*>);
static_assert(std::weakly_incrementable<int**>);
static_assert(!std::weakly_incrementable<int[]>);
static_assert(!std::weakly_incrementable<int[10]>);
static_assert(!std::weakly_incrementable<double>);
static_assert(!std::weakly_incrementable<int&>);
static_assert(!std::weakly_incrementable<int()>);
static_assert(!std::weakly_incrementable<int (*)()>);
static_assert(!std::weakly_incrementable<int (&)()>);
static_assert(!std::weakly_incrementable<bool>);

struct S {};
static_assert(!std::weakly_incrementable<int S::*>);

#define CHECK_POINTER_TO_MEMBER_FUNCTIONS(qualifier)                                                                   \
  static_assert(!std::weakly_incrementable<int (S::*)() qualifier>);                                                   \
  static_assert(!std::weakly_incrementable<int (S::*)() qualifier noexcept>);                                          \
  static_assert(!std::weakly_incrementable<int (S::*)() qualifier&>);                                                  \
  static_assert(!std::weakly_incrementable<int (S::*)() qualifier & noexcept>);                                        \
  static_assert(!std::weakly_incrementable<int (S::*)() qualifier&&>);                                                 \
  static_assert(!std::weakly_incrementable < int (S::*)() qualifier&& noexcept >);

#define NO_QUALIFIER
CHECK_POINTER_TO_MEMBER_FUNCTIONS(NO_QUALIFIER);
CHECK_POINTER_TO_MEMBER_FUNCTIONS(const);
CHECK_POINTER_TO_MEMBER_FUNCTIONS(volatile);
CHECK_POINTER_TO_MEMBER_FUNCTIONS(const volatile);

static_assert(std::weakly_incrementable<postfix_increment_returns_void>);
static_assert(std::weakly_incrementable<postfix_increment_returns_copy>);
static_assert(std::weakly_incrementable<has_integral_minus>);
static_assert(std::weakly_incrementable<has_distinct_difference_type_and_minus>);
static_assert(!std::weakly_incrementable<missing_difference_type>);
static_assert(!std::weakly_incrementable<floating_difference_type>);
static_assert(!std::weakly_incrementable<non_const_minus>);
static_assert(!std::weakly_incrementable<non_integral_minus>);
static_assert(!std::weakly_incrementable<bad_difference_type_good_minus>);
static_assert(!std::weakly_incrementable<not_movable>);
static_assert(!std::weakly_incrementable<preinc_not_declared>);
static_assert(!std::weakly_incrementable<postinc_not_declared>);
static_assert(std::weakly_incrementable<not_default_initializable>);
static_assert(std::weakly_incrementable<incrementable_with_difference_type>);
static_assert(std::weakly_incrementable<incrementable_without_difference_type>);
static_assert(std::weakly_incrementable<difference_type_and_void_minus>);
static_assert(std::weakly_incrementable<noncopyable_with_difference_type>);
static_assert(std::weakly_incrementable<noncopyable_without_difference_type>);
static_assert(std::weakly_incrementable<noncopyable_with_difference_type_and_minus>);

#if !defined(TEST_HAS_NO_INT128)
static_assert(std::weakly_incrementable<extended_integral_difference_type<__int128>>);
#endif

#if defined(__BIT_TYPES_DEFINED__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wbit-int-extension"
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(8)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(9)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(10)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(11)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(12)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(13)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(14)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(15)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(16)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(17)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(18)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(19)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(20)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(21)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(22)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(23)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(24)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(25)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(26)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(27)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(28)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(29)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(30)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(31)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(32)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(33)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(34)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(35)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(36)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(37)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(38)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(39)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(40)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(41)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(42)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(43)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(44)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(45)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(46)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(47)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(48)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(49)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(50)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(51)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(52)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(53)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(54)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(55)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(56)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(57)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(58)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(59)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(60)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(61)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(62)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(63)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(64)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(65)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(66)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(67)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(68)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(69)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(70)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(71)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(72)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(73)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(74)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(75)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(76)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(77)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(78)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(79)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(80)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(81)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(82)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(83)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(84)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(85)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(86)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(87)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(88)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(89)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(90)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(91)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(92)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(93)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(94)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(95)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(96)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(97)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(98)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(99)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(100)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(101)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(102)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(103)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(104)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(105)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(106)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(107)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(108)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(109)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(110)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(111)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(112)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(113)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(114)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(115)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(116)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(117)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(118)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(119)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(120)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(121)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(122)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(123)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(124)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(125)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(126)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(127)>>);
static_assert(std::weakly_incrementable<extended_integral_difference_type<_BitInt(128)>>);
#  pragma GCC diagnostic pop
#endif
