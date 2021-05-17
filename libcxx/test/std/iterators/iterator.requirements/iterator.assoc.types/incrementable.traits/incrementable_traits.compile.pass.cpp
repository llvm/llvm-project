//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10

// template<class T>
// struct incrementable_traits;

#include <iterator>

#include <concepts>

#include "test_macros.h"

// clang-format off
template <class T>
concept check_has_difference_type = requires {
  typename std::incrementable_traits<T>::difference_type;
};

template <class T, class Expected>
concept check_difference_type_matches =
  check_has_difference_type<T> &&
  std::same_as<typename std::incrementable_traits<T>::difference_type, Expected>;
// clang-format on

template <class T, class Expected>
[[nodiscard]] constexpr bool check_incrementable_traits() noexcept {
  constexpr bool result = check_difference_type_matches<T, Expected>;
  static_assert(check_difference_type_matches<T const, Expected> == result);
  return result;
}

static_assert(check_incrementable_traits<float*, std::ptrdiff_t>());
static_assert(check_incrementable_traits<float const*, std::ptrdiff_t>());
static_assert(check_incrementable_traits<float volatile*, std::ptrdiff_t>());
static_assert(
    check_incrementable_traits<float const volatile*, std::ptrdiff_t>());
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
  friend int operator-(int_subtraction, int_subtraction) noexcept;
};
static_assert(check_incrementable_traits<int_subtraction, int>());
static_assert(!check_incrementable_traits<int_subtraction volatile&, int>());
static_assert(
    !check_incrementable_traits<int_subtraction const volatile&, int>());

struct char_subtraction {
  friend char operator-(char_subtraction, char_subtraction) noexcept;
};
static_assert(check_incrementable_traits<char_subtraction, signed char>());

struct unsigned_int_subtraction_with_cv {
  friend unsigned int
  operator-(unsigned_int_subtraction_with_cv const&,
            unsigned_int_subtraction_with_cv const&) noexcept;
  friend unsigned int
  operator-(unsigned_int_subtraction_with_cv const volatile&,
            unsigned_int_subtraction_with_cv const volatile&) noexcept;
};
static_assert(
    check_incrementable_traits<unsigned_int_subtraction_with_cv, int>());
static_assert(check_incrementable_traits<
              unsigned_int_subtraction_with_cv volatile&, int>());
static_assert(check_incrementable_traits<
              unsigned_int_subtraction_with_cv const volatile&, int>());

struct specialised_incrementable_traits {};
namespace std {
template <>
struct incrementable_traits<specialised_incrementable_traits> {
  using difference_type = int;
};
} // namespace std
static_assert(
    check_incrementable_traits<specialised_incrementable_traits, int>());

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

#define TEST_POINTER_TO_MEMBER_FUNCTION(type, cv_qualifier)                    \
  static_assert(!check_has_difference_type<int (type::*)() cv_qualifier>);     \
  static_assert(                                                               \
      !check_has_difference_type<int (type::*)() cv_qualifier noexcept>);      \
  static_assert(!check_has_difference_type<int (type::*)() cv_qualifier&>);    \
  static_assert(                                                               \
      !check_has_difference_type<int (type::*)() cv_qualifier & noexcept>);    \
  static_assert(!check_has_difference_type<int (type::*)() cv_qualifier&&>);   \
  static_assert(!check_has_difference_type < int (type::*)()                   \
                                                 cv_qualifier&& noexcept >);   \
  /**/

struct empty {};

#define NO_QUALIFIER
TEST_POINTER_TO_MEMBER_FUNCTION(empty, NO_QUALIFIER);
TEST_POINTER_TO_MEMBER_FUNCTION(empty, const);
TEST_POINTER_TO_MEMBER_FUNCTION(empty, volatile);
TEST_POINTER_TO_MEMBER_FUNCTION(empty, const volatile);

struct void_subtraction {
  friend void operator-(void_subtraction, void_subtraction) noexcept;
};
static_assert(!check_has_difference_type<void_subtraction>);

#define TEST_NOT_DIFFERENCE_TYPE(qual1, qual2)                                 \
  struct TEST_CONCAT(test_subtraction_, __LINE__) {                            \
    friend int operator-(TEST_CONCAT(test_subtraction_, __LINE__) qual1,       \
                         TEST_CONCAT(test_subtraction_, __LINE__) qual2);      \
  };                                                                           \
  static_assert(!check_has_difference_type<TEST_CONCAT(test_subtraction_,      \
                                                       __LINE__)>) /**/

TEST_NOT_DIFFERENCE_TYPE(&, &);
TEST_NOT_DIFFERENCE_TYPE(&, const&);
TEST_NOT_DIFFERENCE_TYPE(&, volatile&);
TEST_NOT_DIFFERENCE_TYPE(&, const volatile&);
TEST_NOT_DIFFERENCE_TYPE(&, &&);
TEST_NOT_DIFFERENCE_TYPE(&, const&&);
TEST_NOT_DIFFERENCE_TYPE(&, volatile&&);
TEST_NOT_DIFFERENCE_TYPE(&, const volatile&&);

TEST_NOT_DIFFERENCE_TYPE(const&, &);
// TEST_NOT_DIFFERENCE_TYPE(const&, const&); // == true
TEST_NOT_DIFFERENCE_TYPE(const&, volatile&);
// TEST_NOT_DIFFERENCE_TYPE(const&, const volatile&); // invalid
TEST_NOT_DIFFERENCE_TYPE(const&, &&);
TEST_NOT_DIFFERENCE_TYPE(const&, const&&);
TEST_NOT_DIFFERENCE_TYPE(const&, volatile&&);
TEST_NOT_DIFFERENCE_TYPE(const&, const volatile&&);

TEST_NOT_DIFFERENCE_TYPE(volatile&, &);
TEST_NOT_DIFFERENCE_TYPE(volatile&, const&);
TEST_NOT_DIFFERENCE_TYPE(volatile&, volatile&);
TEST_NOT_DIFFERENCE_TYPE(volatile&, const volatile&);
TEST_NOT_DIFFERENCE_TYPE(volatile&, &&);
TEST_NOT_DIFFERENCE_TYPE(volatile&, const&&);
TEST_NOT_DIFFERENCE_TYPE(volatile&, volatile&&);
TEST_NOT_DIFFERENCE_TYPE(volatile&, const volatile&&);

TEST_NOT_DIFFERENCE_TYPE(const volatile&, &);
// TEST_NOT_DIFFERENCE_TYPE(const volatile&, const&); //  invalid
TEST_NOT_DIFFERENCE_TYPE(const volatile&, volatile&);
// TEST_NOT_DIFFERENCE_TYPE(const volatile&, const volatile&); // invalid
TEST_NOT_DIFFERENCE_TYPE(const volatile&, &&);
TEST_NOT_DIFFERENCE_TYPE(const volatile&, const&&);
TEST_NOT_DIFFERENCE_TYPE(const volatile&, volatile&&);
TEST_NOT_DIFFERENCE_TYPE(const volatile&, const volatile&&);

TEST_NOT_DIFFERENCE_TYPE(&&, &);
TEST_NOT_DIFFERENCE_TYPE(&&, const&);
TEST_NOT_DIFFERENCE_TYPE(&&, volatile&);
TEST_NOT_DIFFERENCE_TYPE(&&, const volatile&);
TEST_NOT_DIFFERENCE_TYPE(&&, &&);
TEST_NOT_DIFFERENCE_TYPE(&&, const&&);
TEST_NOT_DIFFERENCE_TYPE(&&, volatile&&);
TEST_NOT_DIFFERENCE_TYPE(&&, const volatile&&);

TEST_NOT_DIFFERENCE_TYPE(const&&, &);
TEST_NOT_DIFFERENCE_TYPE(const&&, const&);
TEST_NOT_DIFFERENCE_TYPE(const&&, volatile&);
TEST_NOT_DIFFERENCE_TYPE(const&&, const volatile&);
TEST_NOT_DIFFERENCE_TYPE(const&&, &&);
TEST_NOT_DIFFERENCE_TYPE(const&&, const&&);
TEST_NOT_DIFFERENCE_TYPE(const&&, volatile&&);
TEST_NOT_DIFFERENCE_TYPE(const&&, const volatile&&);

TEST_NOT_DIFFERENCE_TYPE(volatile&&, &);
TEST_NOT_DIFFERENCE_TYPE(volatile&&, const&);
TEST_NOT_DIFFERENCE_TYPE(volatile&&, volatile&);
TEST_NOT_DIFFERENCE_TYPE(volatile&&, const volatile&);
TEST_NOT_DIFFERENCE_TYPE(volatile&&, &&);
TEST_NOT_DIFFERENCE_TYPE(volatile&&, const&&);
TEST_NOT_DIFFERENCE_TYPE(volatile&&, volatile&&);
TEST_NOT_DIFFERENCE_TYPE(volatile&&, const volatile&&);

TEST_NOT_DIFFERENCE_TYPE(const volatile&&, &);
TEST_NOT_DIFFERENCE_TYPE(const volatile&&, const&);
TEST_NOT_DIFFERENCE_TYPE(const volatile&&, volatile&);
TEST_NOT_DIFFERENCE_TYPE(const volatile&&, const volatile&);
TEST_NOT_DIFFERENCE_TYPE(const volatile&&, &&);
TEST_NOT_DIFFERENCE_TYPE(const volatile&&, const&&);
TEST_NOT_DIFFERENCE_TYPE(const volatile&&, volatile&&);
TEST_NOT_DIFFERENCE_TYPE(const volatile&&, const volatile&&);

TEST_NOT_DIFFERENCE_TYPE(&, NO_QUALIFIER);
// TEST_NOT_DIFFERENCE_TYPE(const&, NO_QUALIFIER); // == true
TEST_NOT_DIFFERENCE_TYPE(volatile&, NO_QUALIFIER);
// TEST_NOT_DIFFERENCE_TYPE(const volatile&, NO_QUALIFIER); // invalid
TEST_NOT_DIFFERENCE_TYPE(&&, NO_QUALIFIER);
TEST_NOT_DIFFERENCE_TYPE(const&&, NO_QUALIFIER);
TEST_NOT_DIFFERENCE_TYPE(volatile&&, NO_QUALIFIER);
TEST_NOT_DIFFERENCE_TYPE(const volatile&&, NO_QUALIFIER);

TEST_NOT_DIFFERENCE_TYPE(NO_QUALIFIER, &);
// TEST_NOT_DIFFERENCE_TYPE(NO_QUALIFIER, const&); // == true
TEST_NOT_DIFFERENCE_TYPE(NO_QUALIFIER, volatile&);
// TEST_NOT_DIFFERENCE_TYPE(NO_QUALIFIER, const volatile&); // invalid
TEST_NOT_DIFFERENCE_TYPE(NO_QUALIFIER, &&);
TEST_NOT_DIFFERENCE_TYPE(NO_QUALIFIER, const&&);
TEST_NOT_DIFFERENCE_TYPE(NO_QUALIFIER, volatile&&);
TEST_NOT_DIFFERENCE_TYPE(NO_QUALIFIER, const volatile&&);
