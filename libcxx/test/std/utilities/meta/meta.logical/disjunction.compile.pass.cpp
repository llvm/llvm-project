//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// type_traits

// template<class... B> struct disjunction;                           // C++17
// template<class... B>
//   constexpr bool disjunction_v = disjunction<B...>::value;         // C++17

#include <type_traits>
#include <cassert>

#include "test_macros.h"

struct True  { static constexpr bool value = true; };
struct False { static constexpr bool value = false; };

struct MySpecialTrueType { static constexpr auto value = -1; static constexpr auto MySpecial = 37; };
struct MySpecialFalseType { static constexpr auto value = false; static constexpr auto MySpecial = 23; };
struct MyOtherSpecialFalseType { static constexpr auto value = false; static constexpr auto MySpecial = 46; };
struct HasNoValue {};
struct ExplicitlyConvertibleToBool { explicit constexpr operator bool() const { return false; } };
struct ValueExplicitlyConvertible { static constexpr ExplicitlyConvertibleToBool value {}; };

static_assert(!std::disjunction<>::value);
static_assert( std::disjunction<std::true_type >::value);
static_assert(!std::disjunction<std::false_type>::value);

static_assert(!std::disjunction_v<>);
static_assert( std::disjunction_v<std::true_type >);
static_assert(!std::disjunction_v<std::false_type>);

static_assert( std::disjunction<std::true_type,  std::true_type >::value);
static_assert( std::disjunction<std::true_type,  std::false_type>::value);
static_assert( std::disjunction<std::false_type, std::true_type >::value);
static_assert(!std::disjunction<std::false_type, std::false_type>::value);

static_assert( std::disjunction_v<std::true_type,  std::true_type >);
static_assert( std::disjunction_v<std::true_type,  std::false_type>);
static_assert( std::disjunction_v<std::false_type, std::true_type >);
static_assert(!std::disjunction_v<std::false_type, std::false_type>);

static_assert( std::disjunction<std::true_type,  std::true_type,  std::true_type >::value);
static_assert( std::disjunction<std::true_type,  std::false_type, std::true_type >::value);
static_assert( std::disjunction<std::false_type, std::true_type,  std::true_type >::value);
static_assert( std::disjunction<std::false_type, std::false_type, std::true_type >::value);
static_assert( std::disjunction<std::true_type,  std::true_type,  std::false_type>::value);
static_assert( std::disjunction<std::true_type,  std::false_type, std::false_type>::value);
static_assert( std::disjunction<std::false_type, std::true_type,  std::false_type>::value);
static_assert(!std::disjunction<std::false_type, std::false_type, std::false_type>::value);

static_assert( std::disjunction_v<std::true_type,  std::true_type,  std::true_type >);
static_assert( std::disjunction_v<std::true_type,  std::false_type, std::true_type >);
static_assert( std::disjunction_v<std::false_type, std::true_type,  std::true_type >);
static_assert( std::disjunction_v<std::false_type, std::false_type, std::true_type >);
static_assert( std::disjunction_v<std::true_type,  std::true_type,  std::false_type>);
static_assert( std::disjunction_v<std::true_type,  std::false_type, std::false_type>);
static_assert( std::disjunction_v<std::false_type, std::true_type,  std::false_type>);
static_assert(!std::disjunction_v<std::false_type, std::false_type, std::false_type>);

static_assert ( std::disjunction<True >::value, "" );
static_assert (!std::disjunction<False>::value, "" );

static_assert ( std::disjunction_v<True >, "" );
static_assert (!std::disjunction_v<False>, "" );

static_assert(std::is_base_of_v<MySpecialFalseType, std::disjunction<MyOtherSpecialFalseType, MySpecialFalseType>>);
static_assert(std::is_base_of_v<MyOtherSpecialFalseType, std::disjunction<MySpecialFalseType, MyOtherSpecialFalseType>>);
static_assert(std::is_base_of_v<MySpecialTrueType, std::disjunction<MySpecialTrueType, MyOtherSpecialFalseType>>);
static_assert(std::is_base_of_v<MySpecialTrueType, std::disjunction<MyOtherSpecialFalseType, MySpecialTrueType>>);

static_assert(std::is_base_of_v<std::true_type, std::disjunction<std::true_type, HasNoValue>>);

static_assert(std::disjunction<std::true_type, HasNoValue>::value);
static_assert(std::disjunction_v<std::true_type, HasNoValue>);

// Also check the case where HasNoValue is not the last in the list (https://llvm.org/PR584900).
static_assert(std::disjunction<std::true_type, HasNoValue, std::true_type>::value);
static_assert(std::disjunction_v<std::true_type, HasNoValue, std::true_type>);

static_assert(std::disjunction<std::true_type, HasNoValue, std::false_type>::value);
static_assert(std::disjunction_v<std::true_type, HasNoValue, std::false_type>);

static_assert(std::disjunction<MySpecialTrueType>::value == -1);
static_assert(std::disjunction_v<MySpecialTrueType>);

static_assert(std::is_base_of_v<ValueExplicitlyConvertible, std::disjunction<ValueExplicitlyConvertible>>);
static_assert(std::disjunction_v<ValueExplicitlyConvertible, std::true_type>);
