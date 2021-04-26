//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// XFAIL: msvc && clang

// template<class T>
// struct indirectly_readable_traits;

#include <iterator>

#include <concepts>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// `value_type` and `element_type` member aliases aren't actually used to declare anytihng, so GCC
// thinks they're completely unused.
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"

// clang-format off
template <class T>
concept check_has_value_type = requires {
  typename std::indirectly_readable_traits<T>::value_type;
};

template <class T, class Expected>
concept check_value_type_matches =
  check_has_value_type<T> &&
  std::same_as<typename std::indirectly_readable_traits<T>::value_type, Expected>;
// clang-format on

template <class T>
constexpr bool check_pointer() {
  constexpr bool result = check_value_type_matches<T*, T>;
  static_assert(check_value_type_matches<T const*, T> == result);
  static_assert(check_value_type_matches<T volatile*, T> == result);
  static_assert(check_value_type_matches<T const volatile*, T> == result);

  static_assert(check_value_type_matches<T* const, T> == result);
  static_assert(check_value_type_matches<T const* const, T> == result);
  static_assert(check_value_type_matches<T volatile* const, T> == result);
  static_assert(check_value_type_matches<T const volatile* const, T> == result);

  return result;
}

static_assert(!check_pointer<void>());
static_assert(check_pointer<int>());
static_assert(check_pointer<long>());
static_assert(check_pointer<double>());
static_assert(check_pointer<double*>());

struct S {};
static_assert(check_pointer<S>());

template <class T>
constexpr bool check_array() {
  constexpr bool result = check_value_type_matches<T[], T>;
  static_assert(check_value_type_matches<T const[], T> == result);
  static_assert(check_value_type_matches<T volatile[], T> == result);
  static_assert(check_value_type_matches<T const volatile[], T> == result);
  static_assert(check_value_type_matches<T[10], T> == result);
  static_assert(check_value_type_matches<T const[10], T> == result);
  static_assert(check_value_type_matches<T volatile[10], T> == result);
  static_assert(check_value_type_matches<T const volatile[10], T> == result);
  return result;
}

static_assert(check_array<int>());
static_assert(check_array<long>());
static_assert(check_array<double>());
static_assert(check_array<double*>());
static_assert(check_array<S>());

template <class T, class Expected>
constexpr bool check_explicit_member() {
  constexpr bool result = check_value_type_matches<T, Expected>;
  static_assert(check_value_type_matches<T const, Expected> == result);
  return result;
}

struct has_value_type {
  using value_type = int;
};
static_assert(check_explicit_member<has_value_type, int>());
static_assert(check_explicit_member<std::vector<int>::iterator, int>());

struct has_element_type {
  using element_type = S;
};
static_assert(check_explicit_member<has_element_type, S>());

struct has_same_value_and_element_type {
  using value_type = int;
  using element_type = int;
};
static_assert(check_explicit_member<has_same_value_and_element_type, int>());
static_assert(check_explicit_member<std::shared_ptr<long>, long>());
static_assert(check_explicit_member<std::shared_ptr<long const>, long>());

// clang-format off
template<class T, class U>
requires std::same_as<std::remove_cv_t<T>, std::remove_cv_t<U> >
struct possibly_different_cv_qualifiers {
  using value_type = T;
  using element_type = U;
};
// clang-format on

static_assert(check_explicit_member<possibly_different_cv_qualifiers<int, int>, int>());
static_assert(check_explicit_member<possibly_different_cv_qualifiers<int, int const>, int>());
static_assert(check_explicit_member<possibly_different_cv_qualifiers<int, int volatile>, int>());
static_assert(check_explicit_member<possibly_different_cv_qualifiers<int, int const volatile>, int>());
static_assert(check_explicit_member<possibly_different_cv_qualifiers<int const, int>, int>());
static_assert(check_explicit_member<possibly_different_cv_qualifiers<int const, int const>, int>());
static_assert(check_explicit_member<possibly_different_cv_qualifiers<int const, int volatile>, int>());
static_assert(check_explicit_member<possibly_different_cv_qualifiers<int const, int const volatile>, int>());
static_assert(check_explicit_member<possibly_different_cv_qualifiers<int volatile, int>, int>());
static_assert(check_explicit_member<possibly_different_cv_qualifiers<int volatile, int const>, int>());
static_assert(check_explicit_member<possibly_different_cv_qualifiers<int volatile, int volatile>, int>());
static_assert(check_explicit_member<possibly_different_cv_qualifiers<int volatile, int const volatile>, int>());
static_assert(check_explicit_member<possibly_different_cv_qualifiers<int const volatile, int>, int>());
static_assert(check_explicit_member<possibly_different_cv_qualifiers<int const volatile, int const>, int>());
static_assert(check_explicit_member<possibly_different_cv_qualifiers<int const volatile, int volatile>, int>());
static_assert(check_explicit_member<possibly_different_cv_qualifiers<int const volatile, int const volatile>, int>());

struct S2 {};
namespace std {
template <>
struct indirectly_readable_traits<S2> {
  using value_type = int;
};
} // namespace std
static_assert(check_value_type_matches<S2, int>);
static_assert(check_value_type_matches<std::vector<int>, int>);
static_assert(check_value_type_matches<std::vector<int>::iterator, int>);
static_assert(check_value_type_matches<std::vector<int>::const_iterator, int>);
static_assert(check_value_type_matches<std::istream_iterator<int>, int>);
static_assert(check_value_type_matches<std::istreambuf_iterator<char>, char>);
static_assert(check_value_type_matches<std::optional<int>, int>);

template <class T>
constexpr bool check_ref() {
  struct ref_value {
    using value_type = T&;
  };
  constexpr bool result = check_has_value_type<ref_value>;

  struct ref_element {
    using element_type = T&;
  };
  static_assert(check_has_value_type<ref_element> == result);

  return result;
}

static_assert(!check_ref<int>());
static_assert(!check_ref<S>());
static_assert(!check_ref<std::shared_ptr<long> >());

static_assert(!check_has_value_type<void>);
static_assert(!check_has_value_type<std::nullptr_t>);
static_assert(!check_has_value_type<int>);
static_assert(!check_has_value_type<S>);

struct has_different_value_and_element_type {
  using value_type = int;
  using element_type = long;
};
static_assert(!check_has_value_type<has_different_value_and_element_type>);

struct void_value {
  using value_type = void;
};
static_assert(!check_has_value_type<void_value>);

struct void_element {
  using element_type = void;
};
static_assert(!check_has_value_type<void_element>);
