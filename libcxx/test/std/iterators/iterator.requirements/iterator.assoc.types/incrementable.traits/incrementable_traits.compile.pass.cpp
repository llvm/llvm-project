//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class T>
// struct incrementable_traits;

#include <iterator>

#include <concepts>

#include "test_macros.h"

template <class T>
concept check_has_difference_type = requires {
  typename std::incrementable_traits<T>::difference_type;
};

template <class T, class Expected>
concept check_difference_type_matches =
  check_has_difference_type<T> &&
  std::same_as<typename std::incrementable_traits<T>::difference_type, Expected>;

template <class T, class Expected>
constexpr bool check_incrementable_traits() {
  constexpr bool result = check_difference_type_matches<T, Expected>;
  static_assert(check_difference_type_matches<T const, Expected> == result);
  return result;
}

static_assert(check_incrementable_traits<float*, std::ptrdiff_t>());
static_assert(check_incrementable_traits<float const*, std::ptrdiff_t>());
static_assert(check_incrementable_traits<float volatile*, std::ptrdiff_t>());
static_assert(check_incrementable_traits<float const volatile*, std::ptrdiff_t>());
static_assert(check_incrementable_traits<float**, std::ptrdiff_t>());

static_assert(check_incrementable_traits<int[], std::ptrdiff_t>());
static_assert(check_incrementable_traits<int[10], std::ptrdiff_t>());

static_assert(check_incrementable_traits<char, int>());
static_assert(check_incrementable_traits<signed char, int>());
static_assert(check_incrementable_traits<unsigned char, int>());
static_assert(check_incrementable_traits<short, int>());
static_assert(check_incrementable_traits<unsigned short, int>());
static_assert(check_incrementable_traits<int, int>());
static_assert(check_incrementable_traits<unsigned int, int>());
static_assert(check_incrementable_traits<long, long>());
static_assert(check_incrementable_traits<unsigned long, long>());
static_assert(check_incrementable_traits<long long, long long>());
static_assert(check_incrementable_traits<unsigned long long, long long>());

static_assert(check_incrementable_traits<int&, int>());
static_assert(check_incrementable_traits<int const&, int>());
static_assert(check_incrementable_traits<int volatile&, int>());
static_assert(check_incrementable_traits<int const volatile&, int>());
static_assert(check_incrementable_traits<int&&, int>());
static_assert(check_incrementable_traits<int const&&, int>());
static_assert(check_incrementable_traits<int volatile&&, int>());
static_assert(check_incrementable_traits<int const volatile&&, int>());

static_assert(check_incrementable_traits<int volatile, int>());
static_assert(check_incrementable_traits<int* volatile, std::ptrdiff_t>());

struct integral_difference_type {
  using difference_type = int;
};
static_assert(check_incrementable_traits<integral_difference_type, int>());

struct non_integral_difference_type {
  using difference_type = void;
};
static_assert(check_incrementable_traits<non_integral_difference_type, void>());

struct int_subtraction {
  friend int operator-(int_subtraction, int_subtraction);
};
static_assert(check_incrementable_traits<int_subtraction, int>());
static_assert(!check_incrementable_traits<int_subtraction volatile&, int>());
static_assert(!check_incrementable_traits<int_subtraction const volatile&, int>());

struct char_subtraction {
  friend char operator-(char_subtraction, char_subtraction);
};
static_assert(check_incrementable_traits<char_subtraction, signed char>());

struct unsigned_int_subtraction_with_cv {
  friend unsigned int operator-(unsigned_int_subtraction_with_cv const&, unsigned_int_subtraction_with_cv const&);
  friend unsigned int operator-(unsigned_int_subtraction_with_cv const volatile&, unsigned_int_subtraction_with_cv const volatile&);
};
static_assert(check_incrementable_traits<unsigned_int_subtraction_with_cv, int>());
static_assert(check_incrementable_traits<unsigned_int_subtraction_with_cv volatile&, int>());
static_assert(check_incrementable_traits<unsigned_int_subtraction_with_cv const volatile&, int>());

struct specialised_incrementable_traits {};
template <>
struct std::incrementable_traits<specialised_incrementable_traits> {
  using difference_type = int;
};
static_assert(check_incrementable_traits<specialised_incrementable_traits, int>());

static_assert(!check_has_difference_type<void>);
static_assert(!check_has_difference_type<float>);
static_assert(!check_has_difference_type<double>);
static_assert(!check_has_difference_type<long double>);
static_assert(!check_has_difference_type<float&>);
static_assert(!check_has_difference_type<float const&>);

static_assert(!check_has_difference_type<void*>);
static_assert(!check_has_difference_type<std::nullptr_t>);
static_assert(!check_has_difference_type<int()>);
static_assert(!check_has_difference_type<int() noexcept>);
static_assert(!check_has_difference_type<int (*)()>);
static_assert(!check_has_difference_type<int (*)() noexcept>);
static_assert(!check_has_difference_type<int (&)()>);
static_assert(!check_has_difference_type<int (&)() noexcept>);

#define TEST_POINTER_TO_MEMBER_FUNCTION(type, cv)                          \
  static_assert(!check_has_difference_type<int (type::*)() cv>);           \
  static_assert(!check_has_difference_type<int (type::*)() cv noexcept>);  \
  static_assert(!check_has_difference_type<int (type::*)() cv&>);          \
  static_assert(!check_has_difference_type<int (type::*)() cv& noexcept>); \
  static_assert(!check_has_difference_type<int (type::*)() cv&&>);         \
  static_assert(!check_has_difference_type<int (type::*)() cv&& noexcept>);

struct empty {};

#define NO_QUALIFIER
TEST_POINTER_TO_MEMBER_FUNCTION(empty, NO_QUALIFIER);
TEST_POINTER_TO_MEMBER_FUNCTION(empty, const);
TEST_POINTER_TO_MEMBER_FUNCTION(empty, volatile);
TEST_POINTER_TO_MEMBER_FUNCTION(empty, const volatile);

struct void_subtraction {
  friend void operator-(void_subtraction, void_subtraction);
};
static_assert(!check_has_difference_type<void_subtraction>);

#define TEST_NOT_DIFFERENCE_TYPE(S, qual1, qual2) \
  struct S {                                      \
    friend int operator-(S qual1, S qual2);       \
  };                                              \
  static_assert(!check_has_difference_type<S>)

TEST_NOT_DIFFERENCE_TYPE(A01, &, &);
TEST_NOT_DIFFERENCE_TYPE(A02, &, const&);
TEST_NOT_DIFFERENCE_TYPE(A03, &, volatile&);
TEST_NOT_DIFFERENCE_TYPE(A04, &, const volatile&);
TEST_NOT_DIFFERENCE_TYPE(A05, &, &&);
TEST_NOT_DIFFERENCE_TYPE(A06, &, const&&);
TEST_NOT_DIFFERENCE_TYPE(A07, &, volatile&&);
TEST_NOT_DIFFERENCE_TYPE(A08, &, const volatile&&);

TEST_NOT_DIFFERENCE_TYPE(A11, const&, &);
// TEST_NOT_DIFFERENCE_TYPE(A12, const&, const&); // == true
TEST_NOT_DIFFERENCE_TYPE(A13, const&, volatile&);
// TEST_NOT_DIFFERENCE_TYPE(A14, const&, const volatile&); // invalid
TEST_NOT_DIFFERENCE_TYPE(A15, const&, &&);
TEST_NOT_DIFFERENCE_TYPE(A16, const&, const&&);
TEST_NOT_DIFFERENCE_TYPE(A17, const&, volatile&&);
TEST_NOT_DIFFERENCE_TYPE(A18, const&, const volatile&&);

TEST_NOT_DIFFERENCE_TYPE(A21, volatile&, &);
TEST_NOT_DIFFERENCE_TYPE(A22, volatile&, const&);
TEST_NOT_DIFFERENCE_TYPE(A23, volatile&, volatile&);
TEST_NOT_DIFFERENCE_TYPE(A24, volatile&, const volatile&);
TEST_NOT_DIFFERENCE_TYPE(A25, volatile&, &&);
TEST_NOT_DIFFERENCE_TYPE(A26, volatile&, const&&);
TEST_NOT_DIFFERENCE_TYPE(A27, volatile&, volatile&&);
TEST_NOT_DIFFERENCE_TYPE(A28, volatile&, const volatile&&);

TEST_NOT_DIFFERENCE_TYPE(A31, const volatile&, &);
// TEST_NOT_DIFFERENCE_TYPE(A32, const volatile&, const&); //  invalid
TEST_NOT_DIFFERENCE_TYPE(A33, const volatile&, volatile&);
// TEST_NOT_DIFFERENCE_TYPE(A34, const volatile&, const volatile&); // invalid
TEST_NOT_DIFFERENCE_TYPE(A35, const volatile&, &&);
TEST_NOT_DIFFERENCE_TYPE(A36, const volatile&, const&&);
TEST_NOT_DIFFERENCE_TYPE(A37, const volatile&, volatile&&);
TEST_NOT_DIFFERENCE_TYPE(A38, const volatile&, const volatile&&);

TEST_NOT_DIFFERENCE_TYPE(A41, &&, &);
TEST_NOT_DIFFERENCE_TYPE(A42, &&, const&);
TEST_NOT_DIFFERENCE_TYPE(A43, &&, volatile&);
TEST_NOT_DIFFERENCE_TYPE(A44, &&, const volatile&);
TEST_NOT_DIFFERENCE_TYPE(A45, &&, &&);
TEST_NOT_DIFFERENCE_TYPE(A46, &&, const&&);
TEST_NOT_DIFFERENCE_TYPE(A47, &&, volatile&&);
TEST_NOT_DIFFERENCE_TYPE(A48, &&, const volatile&&);

TEST_NOT_DIFFERENCE_TYPE(A51, const&&, &);
TEST_NOT_DIFFERENCE_TYPE(A52, const&&, const&);
TEST_NOT_DIFFERENCE_TYPE(A53, const&&, volatile&);
TEST_NOT_DIFFERENCE_TYPE(A54, const&&, const volatile&);
TEST_NOT_DIFFERENCE_TYPE(A55, const&&, &&);
TEST_NOT_DIFFERENCE_TYPE(A56, const&&, const&&);
TEST_NOT_DIFFERENCE_TYPE(A57, const&&, volatile&&);
TEST_NOT_DIFFERENCE_TYPE(A58, const&&, const volatile&&);

TEST_NOT_DIFFERENCE_TYPE(A61, volatile&&, &);
TEST_NOT_DIFFERENCE_TYPE(A62, volatile&&, const&);
TEST_NOT_DIFFERENCE_TYPE(A63, volatile&&, volatile&);
TEST_NOT_DIFFERENCE_TYPE(A64, volatile&&, const volatile&);
TEST_NOT_DIFFERENCE_TYPE(A65, volatile&&, &&);
TEST_NOT_DIFFERENCE_TYPE(A66, volatile&&, const&&);
TEST_NOT_DIFFERENCE_TYPE(A67, volatile&&, volatile&&);
TEST_NOT_DIFFERENCE_TYPE(A68, volatile&&, const volatile&&);

TEST_NOT_DIFFERENCE_TYPE(A71, const volatile&&, &);
TEST_NOT_DIFFERENCE_TYPE(A72, const volatile&&, const&);
TEST_NOT_DIFFERENCE_TYPE(A73, const volatile&&, volatile&);
TEST_NOT_DIFFERENCE_TYPE(A74, const volatile&&, const volatile&);
TEST_NOT_DIFFERENCE_TYPE(A75, const volatile&&, &&);
TEST_NOT_DIFFERENCE_TYPE(A76, const volatile&&, const&&);
TEST_NOT_DIFFERENCE_TYPE(A77, const volatile&&, volatile&&);
TEST_NOT_DIFFERENCE_TYPE(A78, const volatile&&, const volatile&&);

TEST_NOT_DIFFERENCE_TYPE(A81, &, NO_QUALIFIER);
// TEST_NOT_DIFFERENCE_TYPE(A82, const&, NO_QUALIFIER); // == true
TEST_NOT_DIFFERENCE_TYPE(A83, volatile&, NO_QUALIFIER);
// TEST_NOT_DIFFERENCE_TYPE(A84, const volatile&, NO_QUALIFIER); // invalid
TEST_NOT_DIFFERENCE_TYPE(A85, &&, NO_QUALIFIER);
TEST_NOT_DIFFERENCE_TYPE(A86, const&&, NO_QUALIFIER);
TEST_NOT_DIFFERENCE_TYPE(A87, volatile&&, NO_QUALIFIER);
TEST_NOT_DIFFERENCE_TYPE(A88, const volatile&&, NO_QUALIFIER);

TEST_NOT_DIFFERENCE_TYPE(A91, NO_QUALIFIER, &);
// TEST_NOT_DIFFERENCE_TYPE(A92, NO_QUALIFIER, const&); // == true
TEST_NOT_DIFFERENCE_TYPE(A93, NO_QUALIFIER, volatile&);
// TEST_NOT_DIFFERENCE_TYPE(A94, NO_QUALIFIER, const volatile&); // invalid
TEST_NOT_DIFFERENCE_TYPE(A95, NO_QUALIFIER, &&);
TEST_NOT_DIFFERENCE_TYPE(A96, NO_QUALIFIER, const&&);
TEST_NOT_DIFFERENCE_TYPE(A97, NO_QUALIFIER, volatile&&);
TEST_NOT_DIFFERENCE_TYPE(A98, NO_QUALIFIER, const volatile&&);
