// RUN: %check_clang_tidy -std=c++17-or-later %s bugprone-unchecked-expected-access %t -- -- -I %S/Inputs/unchecked-expected-access

#include "tl/expected.hpp"
#include <utility>

struct Foo {
  void foo() const {}
};

constexpr auto add_one = [](int v) { return v + 1; };

// Unchecked accesses.

void unchecked_value_access(tl::expected<int, int> e) {
  e.value();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to 'tl::expected' value [bugprone-unchecked-expected-access]
}

void unchecked_deref_operator_access(tl::expected<int, int> e) {
  *e;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: unchecked access to 'tl::expected' value [bugprone-unchecked-expected-access]
}

void unchecked_arrow_operator_access(tl::expected<Foo, int> e) {
  e->foo();
  // CHECK-MESSAGES: :[[@LINE-1]]:4: warning: unchecked access to 'tl::expected' value [bugprone-unchecked-expected-access]
}

void access_in_negated_check(tl::expected<int, int> e) {
  if (!e) {
    *e;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to 'tl::expected' value [bugprone-unchecked-expected-access]
  }
}

void unchecked_error_access(tl::expected<int, int> e) {
  e.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to 'tl::expected' error [bugprone-unchecked-expected-access]
}

void error_access_in_has_value_check(tl::expected<int, int> e) {
  if (e.has_value()) {
    e.error();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: unchecked access to 'tl::expected' error [bugprone-unchecked-expected-access]
  }
}

void void_unchecked_error_access(tl::expected<void, int> e) {
  e.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to 'tl::expected' error [bugprone-unchecked-expected-access]
}

// Checked accesses.

void checked_access_with_has_value(tl::expected<Foo, int> e) {
  if (e.has_value()) {
    e.value();
    *e;
    e->foo();
  }
}

int checked_access_in_ternary(tl::expected<int, int> e) {
  return e ? *e : 0;
}

int checked_error_access_after_early_return(tl::expected<int, int> e) {
  if (e)
    return 0;
  return e.error();
}

void void_checked_error_access(tl::expected<void, int> e) {
  if (!e)
    e.error();
}

void value_or_is_not_an_access(tl::expected<int, int> e) {
  e.value_or(0);
}

void unexpected_value_is_not_diagnosed() {
  tl::unexpected<int> u(1);
  u.value();
}

// emplace.

int emplace_if_empty(tl::expected<int, int> e) {
  if (!e)
    e.emplace(0);
  return *e;
}

void error_access_after_emplace(tl::expected<int, int> e) {
  e.emplace(1);
  e.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to 'tl::expected' error [bugprone-unchecked-expected-access]
}

// Construction.

int default_construct() {
  tl::expected<int, int> e;
  return *e;
}

int construct_from_value() {
  tl::expected<int, int> e(5);
  return *e;
}

int copy_initialize_from_value() {
  tl::expected<int, int> e = 5;
  return *e;
}

void construct_in_place() {
  tl::expected<Foo, int> e(tl::in_place);
  e->foo();
}

int construct_from_unexpected() {
  tl::expected<int, int> e(tl::unexpected(1));
  return e.error();
}

int copy_initialize_from_make_unexpected() {
  tl::expected<int, int> e = tl::make_unexpected(1);
  return e.error();
}

int construct_unexpect() {
  tl::expected<int, int> e(tl::unexpect, 1);
  return e.error();
}

void void_construct_unexpect() {
  tl::expected<void, int> e(tl::unexpect, 1);
  e.error();
}

int convert_from_checked(tl::expected<long, int> b) {
  if (!b)
    return 0;
  tl::expected<int, int> a(b);
  return *a;
}

int copy_construct_from_checked(tl::expected<int, int> b) {
  if (!b)
    return 0;
  auto a = b;
  return *a;
}

int error_access_after_default_construct() {
  tl::expected<int, int> e;
  return e.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: unchecked access to 'tl::expected' error [bugprone-unchecked-expected-access]
}

void void_error_access_after_default_construct() {
  tl::expected<void, int> e;
  e.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to 'tl::expected' error [bugprone-unchecked-expected-access]
}

int value_access_after_construct_from_make_unexpected() {
  tl::expected<int, int> e(tl::make_unexpected(1));
  return *e;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unchecked access to 'tl::expected' value [bugprone-unchecked-expected-access]
}

int convert_from_unchecked(tl::expected<long, int> b) {
  tl::expected<int, int> a(b);
  return *a;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unchecked access to 'tl::expected' value [bugprone-unchecked-expected-access]
}

// Assignment.

int assign_value(tl::expected<int, int> e) {
  e = 5;
  return *e;
}

int assign_make_unexpected(tl::expected<int, int> e) {
  e = tl::make_unexpected(1);
  return e.error();
}

int copy_assign_from_checked(tl::expected<int, int> a,
                             tl::expected<int, int> b) {
  if (!b)
    return 0;
  a = b;
  return *a;
}

int value_access_after_assign_make_unexpected(tl::expected<int, int> e) {
  e = tl::make_unexpected(1);
  return *e;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unchecked access to 'tl::expected' value [bugprone-unchecked-expected-access]
}

int error_access_after_assign_value(tl::expected<int, int> e) {
  e = 5;
  return e.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: unchecked access to 'tl::expected' error [bugprone-unchecked-expected-access]
}

// Monadic operations.

int map_value_access(tl::expected<int, int> e) {
  if (!e)
    return 0;
  return *e.map(add_one);
}

int transform_value_access(tl::expected<int, int> e) {
  if (!e)
    return 0;
  return *e.transform(add_one);
}

int map_error_error_access(tl::expected<int, int> e) {
  if (e)
    return 0;
  return e.map_error(add_one).error();
}

int void_map(tl::expected<void, int> e) {
  if (!e)
    return 0;
  return *e.map([] { return 1; });
}

int map_unchecked(tl::expected<int, int> e) {
  return *e.map(add_one);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unchecked access to 'tl::expected' value [bugprone-unchecked-expected-access]
}

int map_error_unchecked(tl::expected<int, int> e) {
  return e.map_error(add_one).error();
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: unchecked access to 'tl::expected' error [bugprone-unchecked-expected-access]
}

// Comparisons. Unlike for `std::expected`, these call operators that are
// declared for both operand orders, and `operator!=` is declared explicitly.

int equal_to_value(tl::expected<int, int> e) {
  if (e == 5)
    return *e;
  return 0;
}

int value_equal_to_expected(tl::expected<int, int> e) {
  if (5 == e)
    return *e;
  return 0;
}

int not_equal_to_value_early_return(tl::expected<int, int> e) {
  if (e != 5)
    return 0;
  return *e;
}

int value_not_equal_to_expected_early_return(tl::expected<int, int> e) {
  if (5 != e)
    return 0;
  return *e;
}

int equal_to_expected_with_value(tl::expected<int, int> a,
                                 tl::expected<int, int> b) {
  if (a == b && b)
    return *a;
  return 0;
}

int not_equal_to_expected_early_return(tl::expected<int, int> a,
                                       tl::expected<int, int> b) {
  if (a != b)
    return 0;
  if (b)
    return *a;
  return 0;
}

void void_not_equal_to_expected_with_value(tl::expected<void, int> a,
                                           tl::expected<void, int> b) {
  if (a != b && a)
    b.error();
}

int equal_to_unexpected(tl::expected<int, int> e) {
  if (e == tl::make_unexpected(1))
    return e.error();
  return 0;
}

int unexpected_equal_to_expected(tl::expected<int, int> e) {
  if (tl::make_unexpected(1) == e)
    return e.error();
  return 0;
}

int not_equal_to_unexpected_early_return(tl::expected<int, int> e) {
  if (e != tl::make_unexpected(1))
    return 0;
  return e.error();
}

int unexpected_not_equal_to_expected_early_return(tl::expected<int, int> e) {
  if (tl::make_unexpected(1) != e)
    return 0;
  return e.error();
}

int not_equal_to_value(tl::expected<int, int> e) {
  if (e != 5)
    return *e;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: unchecked access to 'tl::expected' value [bugprone-unchecked-expected-access]
  return 0;
}

int value_not_equal_to_expected(tl::expected<int, int> e) {
  if (5 != e)
    return *e;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: unchecked access to 'tl::expected' value [bugprone-unchecked-expected-access]
  return 0;
}

int value_access_after_value_equal_to_expected(tl::expected<int, int> e) {
  if (5 == e)
    return 0;
  return *e;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unchecked access to 'tl::expected' value [bugprone-unchecked-expected-access]
}

int not_equal_to_expected_with_value(tl::expected<int, int> a,
                                     tl::expected<int, int> b) {
  if (a != b && b)
    return *a;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: unchecked access to 'tl::expected' value [bugprone-unchecked-expected-access]
  return 0;
}

int value_access_when_unexpected_equal_to_expected(tl::expected<int, int> e) {
  if (tl::make_unexpected(1) == e)
    return *e;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: unchecked access to 'tl::expected' value [bugprone-unchecked-expected-access]
  return 0;
}

int error_access_when_not_equal_to_unexpected(tl::expected<int, int> e) {
  if (e != tl::make_unexpected(1))
    return e.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: unchecked access to 'tl::expected' error [bugprone-unchecked-expected-access]
  return 0;
}

// Forwarding. This is the pattern that `tl::expected` uses to implement `map`,
// `transform` and `and_then`.

template <class Exp> int map_impl_like(Exp &&exp) {
  return exp.has_value() ? *std::forward<Exp>(exp)
                         : std::forward<Exp>(exp).error();
}

int instantiate_map_impl_like(tl::expected<int, int> e) {
  return map_impl_like(e);
}
