//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// type_traits

// template<class... B> struct conjunction;                           // C++17
// template<class... B>
//   constexpr bool conjunction_v = conjunction<B...>::value;         // C++17

#include <cassert>
#include <type_traits>

#include "test_macros.h"

struct True  { static constexpr bool value = true; };
struct False { static constexpr bool value = false; };

struct MySpecialTrueType { static constexpr auto value = true; static constexpr auto MySpecial = 23; };
struct MyOtherSpecialTrueType { static constexpr auto value = -1; static constexpr auto MySpecial = 46; };
struct MySpecialFalseType { static constexpr auto value = false; static constexpr auto MySpecial = 37; };
struct HasNoValue {};
struct ExplicitlyConvertibleToBool { explicit constexpr operator bool() const { return true; } };
struct ValueExplicitlyConvertible { static constexpr ExplicitlyConvertibleToBool value {}; };

static_assert( std::conjunction<>::value);
static_assert( std::conjunction<std::true_type >::value);
static_assert(!std::conjunction<std::false_type>::value);

static_assert( std::conjunction_v<>);
static_assert( std::conjunction_v<std::true_type >);
static_assert(!std::conjunction_v<std::false_type>);

static_assert( std::conjunction<std::true_type,  std::true_type >::value);
static_assert(!std::conjunction<std::true_type,  std::false_type>::value);
static_assert(!std::conjunction<std::false_type, std::true_type >::value);
static_assert(!std::conjunction<std::false_type, std::false_type>::value);

static_assert( std::conjunction_v<std::true_type,  std::true_type >);
static_assert(!std::conjunction_v<std::true_type,  std::false_type>);
static_assert(!std::conjunction_v<std::false_type, std::true_type >);
static_assert(!std::conjunction_v<std::false_type, std::false_type>);

static_assert( std::conjunction<std::true_type,  std::true_type,  std::true_type >::value);
static_assert(!std::conjunction<std::true_type,  std::false_type, std::true_type >::value);
static_assert(!std::conjunction<std::false_type, std::true_type,  std::true_type >::value);
static_assert(!std::conjunction<std::false_type, std::false_type, std::true_type >::value);
static_assert(!std::conjunction<std::true_type,  std::true_type,  std::false_type>::value);
static_assert(!std::conjunction<std::true_type,  std::false_type, std::false_type>::value);
static_assert(!std::conjunction<std::false_type, std::true_type,  std::false_type>::value);
static_assert(!std::conjunction<std::false_type, std::false_type, std::false_type>::value);

static_assert( std::conjunction_v<std::true_type,  std::true_type,  std::true_type >);
static_assert(!std::conjunction_v<std::true_type,  std::false_type, std::true_type >);
static_assert(!std::conjunction_v<std::false_type, std::true_type,  std::true_type >);
static_assert(!std::conjunction_v<std::false_type, std::false_type, std::true_type >);
static_assert(!std::conjunction_v<std::true_type,  std::true_type,  std::false_type>);
static_assert(!std::conjunction_v<std::true_type,  std::false_type, std::false_type>);
static_assert(!std::conjunction_v<std::false_type, std::true_type,  std::false_type>);
static_assert(!std::conjunction_v<std::false_type, std::false_type, std::false_type>);

static_assert( std::conjunction<True >::value);
static_assert(!std::conjunction<False>::value);

static_assert( std::conjunction_v<True >);
static_assert(!std::conjunction_v<False>);

static_assert(std::is_base_of_v<MySpecialTrueType, std::conjunction<MyOtherSpecialTrueType, MySpecialTrueType>>);
static_assert(std::is_base_of_v<MyOtherSpecialTrueType, std::conjunction<MySpecialTrueType, MyOtherSpecialTrueType>>);
static_assert(std::is_base_of_v<MySpecialFalseType, std::conjunction<MySpecialFalseType, MyOtherSpecialTrueType>>);
static_assert(std::is_base_of_v<MySpecialFalseType, std::conjunction<MyOtherSpecialTrueType, MySpecialFalseType>>);

static_assert(std::is_base_of_v<std::false_type, std::conjunction<std::false_type, HasNoValue>>);

static_assert(!std::conjunction<std::false_type, HasNoValue>::value);
static_assert(!std::conjunction_v<std::false_type, HasNoValue>);

// Also check the case where HasNoValue is not the last in the list (https://llvm.org/PR584900).
static_assert(!std::conjunction<std::false_type, HasNoValue, std::true_type>::value);
static_assert(!std::conjunction_v<std::false_type, HasNoValue, std::true_type>);

static_assert(!std::conjunction<std::false_type, HasNoValue, std::false_type>::value);
static_assert(!std::conjunction_v<std::false_type, HasNoValue, std::false_type>);

static_assert(std::conjunction<MyOtherSpecialTrueType>::value == -1);
static_assert(std::conjunction_v<MyOtherSpecialTrueType>);

static_assert(std::is_base_of_v<ValueExplicitlyConvertible, std::conjunction<ValueExplicitlyConvertible>>);
static_assert(std::conjunction_v<ValueExplicitlyConvertible, std::true_type>);
