//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that __cpp17_*_iterator catch bad iterators

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <__iterator/cpp17_iterator_concepts.h>
#include <__iterator/iterator_traits.h>
#include <compare>
#include <cstddef>

struct missing_deref {
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::input_iterator_tag;
  using value_type = int;
  using reference = int&;

  missing_deref& operator++();
};

struct missing_preincrement {
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::input_iterator_tag;
  using value_type = int;
  using reference = int&;

  int& operator*();
};

template <class Derived>
struct valid_iterator {
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::input_iterator_tag;
  using value_type = int;
  using reference = int&;

  int& operator*() const;
  Derived& operator++();

  struct Proxy {
    int operator*();
  };

  Proxy operator++(int);
};

struct not_move_constructible : valid_iterator<not_move_constructible> {
  not_move_constructible(const not_move_constructible&) = default;
  not_move_constructible(not_move_constructible&&) = delete;
  not_move_constructible& operator=(not_move_constructible&&) = default;
  not_move_constructible& operator=(const not_move_constructible&) = default;
};

struct not_copy_constructible : valid_iterator<not_copy_constructible> {
  not_copy_constructible(const not_copy_constructible&) = delete;
  not_copy_constructible(not_copy_constructible&&) = default;
  not_copy_constructible& operator=(not_copy_constructible&&) = default;
  not_copy_constructible& operator=(const not_copy_constructible&) = default;
};

struct not_move_assignable : valid_iterator<not_move_assignable> {
  not_move_assignable(const not_move_assignable&) = default;
  not_move_assignable(not_move_assignable&&) = default;
  not_move_assignable& operator=(not_move_assignable&&) = delete;
  not_move_assignable& operator=(const not_move_assignable&) = default;
};

struct not_copy_assignable : valid_iterator<not_copy_assignable> {
  not_copy_assignable(const not_copy_assignable&) = default;
  not_copy_assignable(not_copy_assignable&&) = default;
  not_copy_assignable& operator=(not_copy_assignable&&) = default;
  not_copy_assignable& operator=(const not_copy_assignable&) = delete;
};

struct diff_t_not_signed : valid_iterator<diff_t_not_signed> {
  using difference_type = unsigned;
};

void check_iterator_requirements() {
  static_assert(std::__cpp17_iterator<missing_deref>); // expected-error {{static assertion failed}}
  // expected-note@*:* {{indirection requires pointer operand}}

  static_assert(std::__cpp17_iterator<missing_preincrement>); // expected-error {{static assertion failed}}
  // expected-note@*:* {{cannot increment value of type 'missing_preincrement'}}


  static_assert(std::__cpp17_iterator<not_move_constructible>); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because 'not_move_constructible' does not satisfy '__cpp17_move_constructible'}}

  static_assert(std::__cpp17_iterator<not_copy_constructible>); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because 'not_copy_constructible' does not satisfy '__cpp17_copy_constructible'}}

  static_assert(std::__cpp17_iterator<not_move_assignable>); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because 'not_move_assignable' does not satisfy '__cpp17_copy_assignable'}}

  static_assert(std::__cpp17_iterator<not_copy_assignable>); // expected-error {{static assertion failed}}
  // expectted-note@*:* {{because 'not_copy_assignable' does not satisfy '__cpp17_copy_assignable'}}

  static_assert(std::__cpp17_iterator<diff_t_not_signed>); // expected-error {{static assertion failed}}
  // expectted-note@*:* {{'is_signed_v<__iter_diff_t<diff_t_not_signed> >' evaluated to false}}
}

struct not_equality_comparable : valid_iterator<not_equality_comparable> {};
bool operator==(not_equality_comparable, not_equality_comparable) = delete;
bool operator!=(not_equality_comparable, not_equality_comparable);

struct not_unequality_comparable : valid_iterator<not_unequality_comparable> {};
bool operator==(not_unequality_comparable, not_unequality_comparable);
bool operator!=(not_unequality_comparable, not_unequality_comparable) = delete;

void check_input_iterator_requirements() {
  _LIBCPP_REQUIRE_CPP17_INPUT_ITERATOR(not_equality_comparable); // expected-error {{static assertion failed}}
  // expected-note@*:* {{'__lhs == __rhs' would be invalid: overload resolution selected deleted operator '=='}}

  _LIBCPP_REQUIRE_CPP17_INPUT_ITERATOR(not_unequality_comparable); // expected-error {{static assertion failed}}
  // expected-note@*:* {{'__lhs != __rhs' would be invalid: overload resolution selected deleted operator '!='}}
}

template <class Derived>
struct valid_forward_iterator : valid_iterator<Derived> {
  Derived& operator++();
  Derived operator++(int);

  friend bool operator==(Derived, Derived);
};

struct not_default_constructible : valid_forward_iterator<not_default_constructible> {
  not_default_constructible() = delete;
};

struct postincrement_not_ref : valid_iterator<postincrement_not_ref> {};
bool operator==(postincrement_not_ref, postincrement_not_ref);

void check_forward_iterator_requirements() {
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(not_default_constructible); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because 'not_default_constructible' does not satisfy '__cpp17_default_constructible'}}
  _LIBCPP_REQUIRE_CPP17_FORWARD_ITERATOR(postincrement_not_ref); // expected-error {{static assertion failed}}
#ifndef _AIX
  // expected-note@*:* {{because type constraint 'convertible_to<valid_iterator<postincrement_not_ref>::Proxy, const postincrement_not_ref &>' was not satisfied}}
#endif
}

struct missing_predecrement : valid_forward_iterator<missing_predecrement> {
  missing_deref operator--(int);
};

struct missing_postdecrement : valid_forward_iterator<missing_postdecrement> {
  missing_postdecrement& operator--();
};

struct not_returning_iter_reference : valid_forward_iterator<not_returning_iter_reference> {

  struct Proxy {
    operator const not_returning_iter_reference&();

    int operator*();
  };

  not_returning_iter_reference& operator--();
  Proxy operator--(int);
};

void check_bidirectional_iterator_requirements() {
  _LIBCPP_REQUIRE_CPP17_BIDIRECTIONAL_ITERATOR(missing_predecrement); // expected-error {{static assertion failed}}
  // expected-note@*:* {{cannot decrement value of type 'missing_predecrement'}}
  _LIBCPP_REQUIRE_CPP17_BIDIRECTIONAL_ITERATOR(missing_postdecrement); // expected-error {{static assertion failed}}
  // expected-note@*:* {{cannot decrement value of type 'missing_postdecrement'}}
  _LIBCPP_REQUIRE_CPP17_BIDIRECTIONAL_ITERATOR(not_returning_iter_reference); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because type constraint 'same_as<int, __iter_reference<not_returning_iter_reference> >' was not satisfied}}
}

template <class Derived>
struct valid_random_access_iterator : valid_forward_iterator<Derived> {
  using difference_type = typename valid_forward_iterator<Derived>::difference_type;

  Derived& operator--();
  Derived operator--(int);

  Derived& operator+=(difference_type);
  Derived& operator-=(difference_type);

  friend Derived operator+(valid_random_access_iterator, difference_type);
  friend Derived operator+(difference_type, valid_random_access_iterator);
  friend Derived operator-(valid_random_access_iterator, difference_type);
  friend Derived operator-(difference_type, valid_random_access_iterator);
  friend difference_type operator-(valid_random_access_iterator, valid_random_access_iterator);

  int& operator[](difference_type) const;

  friend std::strong_ordering operator<=>(Derived, Derived);
};

struct missing_plus_equals : valid_random_access_iterator<missing_plus_equals> {
  void operator+=(difference_type) = delete;
};

struct missing_minus_equals : valid_random_access_iterator<missing_minus_equals> {
  void operator-=(difference_type) = delete;
};

struct missing_plus_iter_diff : valid_random_access_iterator<missing_plus_iter_diff> {
  void operator+(difference_type) = delete;
};

struct missing_plus_diff_iter : valid_random_access_iterator<missing_plus_diff_iter> {
  friend missing_plus_diff_iter operator+(difference_type, missing_plus_diff_iter) = delete;
};

struct missing_plus_iter_diff_const_mut : valid_random_access_iterator<missing_plus_iter_diff_const_mut> {
  friend missing_plus_iter_diff_const_mut operator+(missing_plus_iter_diff_const_mut&, difference_type);
  friend missing_plus_iter_diff_const_mut operator+(const missing_plus_iter_diff_const_mut&, difference_type) = delete;
};

struct missing_plus_iter_diff_mut_const : valid_random_access_iterator<missing_plus_iter_diff_mut_const> {
  friend missing_plus_iter_diff_mut_const operator+(missing_plus_iter_diff_mut_const, difference_type);
  friend missing_plus_iter_diff_mut_const operator+(difference_type, missing_plus_iter_diff_mut_const&);
  friend missing_plus_iter_diff_mut_const operator+(difference_type, const missing_plus_iter_diff_mut_const&) = delete;
};

struct missing_minus_iter_diff_const : valid_random_access_iterator<missing_minus_iter_diff_const> {
  friend missing_minus_iter_diff_const operator-(missing_minus_iter_diff_const&, difference_type);
  friend missing_minus_iter_diff_const operator-(const missing_minus_iter_diff_const&, difference_type) = delete;
};

struct missing_minus_iter_iter : valid_random_access_iterator<missing_minus_iter_iter> {
  friend missing_minus_iter_iter operator-(missing_minus_iter_iter, missing_minus_iter_iter) = delete;
};

struct missing_minus_const_iter_iter : valid_random_access_iterator<missing_minus_const_iter_iter> {
  friend difference_type operator-(missing_minus_const_iter_iter&, missing_minus_const_iter_iter);
  friend difference_type operator-(const missing_minus_const_iter_iter&, missing_minus_const_iter_iter) = delete;
};

struct missing_minus_iter_const_iter : valid_random_access_iterator<missing_minus_iter_const_iter> {
  friend difference_type operator-(missing_minus_iter_const_iter, missing_minus_iter_const_iter&);
  friend difference_type operator-(missing_minus_iter_const_iter, const missing_minus_iter_const_iter&) = delete;
};

struct missing_minus_const_iter_const_iter : valid_random_access_iterator<missing_minus_const_iter_const_iter> {
  friend difference_type operator-(missing_minus_const_iter_const_iter&, missing_minus_const_iter_const_iter&);
  friend difference_type operator-(const missing_minus_const_iter_const_iter&, missing_minus_const_iter_const_iter&);
  friend difference_type operator-(missing_minus_const_iter_const_iter&, const missing_minus_const_iter_const_iter&);
  friend difference_type operator-(const missing_minus_const_iter_const_iter&, const missing_minus_const_iter_const_iter&) = delete;
};

struct missing_subscript_operator : valid_random_access_iterator<missing_subscript_operator> {
  int& operator[](difference_type) = delete;
};

struct missing_const_subscript_operator : valid_random_access_iterator<missing_const_subscript_operator> {
  int& operator[](difference_type);
  int& operator[](difference_type) const = delete;
};

struct missing_less : valid_random_access_iterator<missing_less> {
  friend bool operator<(missing_less, missing_less) = delete;
};

struct missing_const_mut_less : valid_random_access_iterator<missing_const_mut_less> {
  friend bool operator<(missing_const_mut_less&, missing_const_mut_less&);
  friend bool operator<(missing_const_mut_less&, const missing_const_mut_less&);
  friend bool operator<(const missing_const_mut_less&, missing_const_mut_less&) = delete;
  friend bool operator<(const missing_const_mut_less&, const missing_const_mut_less&);
};

struct missing_mut_const_less : valid_random_access_iterator<missing_mut_const_less> {
  friend bool operator<(missing_mut_const_less&, missing_mut_const_less&);
  friend bool operator<(missing_mut_const_less&, const missing_mut_const_less&) = delete;
  friend bool operator<(const missing_mut_const_less&, missing_mut_const_less&);
  friend bool operator<(const missing_mut_const_less&, const missing_mut_const_less&);
};

struct missing_const_const_less : valid_random_access_iterator<missing_const_const_less> {
  friend bool operator<(missing_const_const_less&, missing_const_const_less&);
  friend bool operator<(missing_const_const_less&, const missing_const_const_less&);
  friend bool operator<(const missing_const_const_less&, missing_const_const_less&);
  friend bool operator<(const missing_const_const_less&, const missing_const_const_less&) = delete;
};

struct missing_greater : valid_random_access_iterator<missing_greater> {
  friend bool operator>(missing_greater, missing_greater) = delete;
};

struct missing_const_mut_greater : valid_random_access_iterator<missing_const_mut_greater> {
  friend bool operator>(missing_const_mut_greater&, missing_const_mut_greater&);
  friend bool operator>(missing_const_mut_greater&, const missing_const_mut_greater&);
  friend bool operator>(const missing_const_mut_greater&, missing_const_mut_greater&) = delete;
  friend bool operator>(const missing_const_mut_greater&, const missing_const_mut_greater&);
};

struct missing_mut_const_greater : valid_random_access_iterator<missing_mut_const_greater> {
  friend bool operator>(missing_mut_const_greater&, missing_mut_const_greater&);
  friend bool operator>(missing_mut_const_greater&, const missing_mut_const_greater&) = delete;
  friend bool operator>(const missing_mut_const_greater&, missing_mut_const_greater&);
  friend bool operator>(const missing_mut_const_greater&, const missing_mut_const_greater&);
};

struct missing_const_const_greater : valid_random_access_iterator<missing_const_const_greater> {
  friend bool operator>(missing_const_const_greater&, missing_const_const_greater&);
  friend bool operator>(missing_const_const_greater&, const missing_const_const_greater&);
  friend bool operator>(const missing_const_const_greater&, missing_const_const_greater&);
  friend bool operator>(const missing_const_const_greater&, const missing_const_const_greater&) = delete;
};

struct missing_less_eq : valid_random_access_iterator<missing_less_eq> {
  friend bool operator<=(missing_less_eq, missing_less_eq) = delete;
};

struct missing_const_mut_less_eq : valid_random_access_iterator<missing_const_mut_less_eq> {
  friend bool operator<=(missing_const_mut_less_eq&, missing_const_mut_less_eq&);
  friend bool operator<=(missing_const_mut_less_eq&, const missing_const_mut_less_eq&);
  friend bool operator<=(const missing_const_mut_less_eq&, missing_const_mut_less_eq&) = delete;
  friend bool operator<=(const missing_const_mut_less_eq&, const missing_const_mut_less_eq&);
};

struct missing_mut_const_less_eq : valid_random_access_iterator<missing_mut_const_less_eq> {
  friend bool operator<=(missing_mut_const_less_eq&, missing_mut_const_less_eq&);
  friend bool operator<=(missing_mut_const_less_eq&, const missing_mut_const_less_eq&) = delete;
  friend bool operator<=(const missing_mut_const_less_eq&, missing_mut_const_less_eq&);
  friend bool operator<=(const missing_mut_const_less_eq&, const missing_mut_const_less_eq&);
};

struct missing_const_const_less_eq : valid_random_access_iterator<missing_const_const_less_eq> {
  friend bool operator<=(missing_const_const_less_eq&, missing_const_const_less_eq&);
  friend bool operator<=(missing_const_const_less_eq&, const missing_const_const_less_eq&);
  friend bool operator<=(const missing_const_const_less_eq&, missing_const_const_less_eq&);
  friend bool operator<=(const missing_const_const_less_eq&, const missing_const_const_less_eq&) = delete;
};

struct missing_greater_eq : valid_random_access_iterator<missing_greater_eq> {
  friend bool operator>=(missing_greater_eq, missing_greater_eq) = delete;
};

struct missing_const_mut_greater_eq : valid_random_access_iterator<missing_const_mut_greater_eq> {
  friend bool operator>=(missing_const_mut_greater_eq&, missing_const_mut_greater_eq&);
  friend bool operator>=(missing_const_mut_greater_eq&, const missing_const_mut_greater_eq&);
  friend bool operator>=(const missing_const_mut_greater_eq&, missing_const_mut_greater_eq&) = delete;
  friend bool operator>=(const missing_const_mut_greater_eq&, const missing_const_mut_greater_eq&);
};

struct missing_mut_const_greater_eq : valid_random_access_iterator<missing_mut_const_greater_eq> {
  friend bool operator>=(missing_mut_const_greater_eq&, missing_mut_const_greater_eq&);
  friend bool operator>=(missing_mut_const_greater_eq&, const missing_mut_const_greater_eq&) = delete;
  friend bool operator>=(const missing_mut_const_greater_eq&, missing_mut_const_greater_eq&);
  friend bool operator>=(const missing_mut_const_greater_eq&, const missing_mut_const_greater_eq&);
};

struct missing_const_const_greater_eq : valid_random_access_iterator<missing_const_const_greater_eq> {
  friend bool operator>=(missing_const_const_greater_eq&, missing_const_const_greater_eq&);
  friend bool operator>=(missing_const_const_greater_eq&, const missing_const_const_greater_eq&);
  friend bool operator>=(const missing_const_const_greater_eq&, missing_const_const_greater_eq&);
  friend bool operator>=(const missing_const_const_greater_eq&, const missing_const_const_greater_eq&) = delete;
};

void check_random_access_iterator() {
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_plus_equals); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because '__iter += __n' would be invalid: overload resolution selected deleted operator '+='}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_minus_equals); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because '__iter -= __n' would be invalid: overload resolution selected deleted operator '-='}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_plus_iter_diff); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because '__iter + __n' would be invalid: overload resolution selected deleted operator '+'}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_plus_diff_iter); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because '__n + __iter' would be invalid: overload resolution selected deleted operator '+'}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_plus_iter_diff_const_mut); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because 'std::as_const(__iter) + __n' would be invalid: overload resolution selected deleted operator '+'}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_plus_iter_diff_mut_const); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because '__n + std::as_const(__iter)' would be invalid: overload resolution selected deleted operator '+'}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_minus_iter_diff_const); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because 'std::as_const(__iter) - __n' would be invalid: overload resolution selected deleted operator '-'}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_minus_iter_iter); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because '__iter - __iter' would be invalid: overload resolution selected deleted operator '-'}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_minus_const_iter_iter); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because 'std::as_const(__iter) - __iter' would be invalid: overload resolution selected deleted operator '-'}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_minus_iter_const_iter); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because '__iter - std::as_const(__iter)' would be invalid: overload resolution selected deleted operator '-'}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_minus_const_iter_const_iter); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because 'std::as_const(__iter) - std::as_const(__iter)' would be invalid: overload resolution selected deleted operator '-'}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_subscript_operator); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because '__iter[__n]' would be invalid: overload resolution selected deleted operator '[]'}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_const_subscript_operator); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because 'std::as_const(__iter)[__n]' would be invalid: overload resolution selected deleted operator '[]'}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_less); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because '__iter < __iter' would be invalid: overload resolution selected deleted operator '<'}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_const_mut_less); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because 'std::as_const(__iter) < __iter' would be invalid: overload resolution selected deleted operator '<'}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_mut_const_less); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because '__iter < std::as_const(__iter)' would be invalid: overload resolution selected deleted operator '<'}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_const_const_less); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because 'std::as_const(__iter) < std::as_const(__iter)' would be invalid: overload resolution selected deleted operator '<'}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_greater); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because '__iter > __iter' would be invalid: overload resolution selected deleted operator '>'}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_const_mut_greater); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because 'std::as_const(__iter) > __iter' would be invalid: overload resolution selected deleted operator '>'}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_mut_const_greater); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because '__iter > std::as_const(__iter)' would be invalid: overload resolution selected deleted operator '>'}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_const_const_greater); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because 'std::as_const(__iter) > std::as_const(__iter)' would be invalid: overload resolution selected deleted operator '>'}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_less_eq); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because '__iter <= __iter' would be invalid: overload resolution selected deleted operator '<='}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_const_mut_less_eq); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because 'std::as_const(__iter) <= __iter' would be invalid: overload resolution selected deleted operator '<='}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_mut_const_less_eq); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because '__iter <= std::as_const(__iter)' would be invalid: overload resolution selected deleted operator '<='}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_const_const_less_eq); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because 'std::as_const(__iter) <= std::as_const(__iter)' would be invalid: overload resolution selected deleted operator '<='}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_greater_eq); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because '__iter >= __iter' would be invalid: overload resolution selected deleted operator '>='}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_const_mut_greater_eq); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because 'std::as_const(__iter) >= __iter' would be invalid: overload resolution selected deleted operator '>='}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_mut_const_greater_eq); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because '__iter >= std::as_const(__iter)' would be invalid: overload resolution selected deleted operator '>='}}
  _LIBCPP_REQUIRE_CPP17_RANDOM_ACCESS_ITERATOR(missing_const_const_greater_eq); // expected-error {{static assertion failed}}
  // expected-note@*:* {{because 'std::as_const(__iter) >= std::as_const(__iter)' would be invalid: overload resolution selected deleted operator '>='}}
}
