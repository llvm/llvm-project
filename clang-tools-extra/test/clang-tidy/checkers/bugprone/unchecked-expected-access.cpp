// RUN: %check_clang_tidy -std=c++23-or-later %s bugprone-unchecked-expected-access %t -- -- -I %S/Inputs/unchecked-expected-access

#include "std/types/expected.h"
#include <utility>

struct Foo {
  void foo() const {}
};

std::expected<int, int> make();

// Unchecked accesses.

void unchecked_value_access(std::expected<int, int> e) {
  e.value();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

void unchecked_deref_operator_access(std::expected<int, int> e) {
  *e;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

void unchecked_arrow_operator_access(std::expected<Foo, int> e) {
  e->foo();
  // CHECK-MESSAGES: :[[@LINE-1]]:4: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

void const_ref_unchecked_value_access(const std::expected<int, int> &e) {
  e.value();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

void const_ref_unchecked_deref_operator_access(
    const std::expected<int, int> &e) {
  *e;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

void const_ref_unchecked_arrow_operator_access(
    const std::expected<Foo, int> &e) {
  e->foo();
  // CHECK-MESSAGES: :[[@LINE-1]]:4: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

void unchecked_access_to_function_result() {
  auto e = make();
  e.value();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

void access_in_else_branch(std::expected<int, int> e) {
  if (e.has_value()) {
  } else {
    e.value();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
  }
}

void access_in_negated_check(std::expected<int, int> e) {
  if (!e) {
    *e;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
  }
}

int access_in_ternary_with_negated_check(std::expected<int, int> e) {
  return !e ? *e : 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

void void_unchecked_value_access(std::expected<void, int> e) {
  e.value();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

// Checked accesses.

void checked_access_with_has_value(std::expected<Foo, int> e) {
  if (e.has_value()) {
    e.value();
    *e;
    e->foo();
  }
}

void checked_access_with_operator_bool(std::expected<int, int> e) {
  if (e) {
    e.value();
    *e;
  }
}

void checked_access_after_early_return(std::expected<int, int> e) {
  if (!e)
    return;
  e.value();
}

int checked_access_after_early_return_with_has_value(
    std::expected<int, int> e) {
  if (!e.has_value())
    return 0;
  return *e;
}

int checked_access_in_ternary(std::expected<int, int> e) {
  return e ? *e : 0;
}

int checked_access_in_ternary_with_has_value(std::expected<int, int> e) {
  return e.has_value() ? e.value() : 0;
}

void value_or_is_not_an_access(std::expected<int, int> e) {
  e.value_or(0);
}

// Unchecked error accesses.

void unchecked_error_access(std::expected<int, int> e) {
  e.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to 'std::expected' error [bugprone-unchecked-expected-access]
}

void const_ref_unchecked_error_access(const std::expected<int, int> &e) {
  e.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to 'std::expected' error [bugprone-unchecked-expected-access]
}

void unchecked_error_access_to_function_result() {
  auto e = make();
  e.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to 'std::expected' error [bugprone-unchecked-expected-access]
}

void error_access_in_operator_bool_check(std::expected<int, int> e) {
  if (e) {
    e.error();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: unchecked access to 'std::expected' error [bugprone-unchecked-expected-access]
  }
}

void error_access_in_has_value_check(std::expected<int, int> e) {
  if (e.has_value()) {
    e.error();
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: unchecked access to 'std::expected' error [bugprone-unchecked-expected-access]
  }
}

void error_access_after_early_return(std::expected<int, int> e) {
  if (!e)
    return;
  e.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to 'std::expected' error [bugprone-unchecked-expected-access]
}

int error_access_in_ternary(std::expected<int, int> e) {
  return e ? e.error() : 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: unchecked access to 'std::expected' error [bugprone-unchecked-expected-access]
}

void void_unchecked_error_access(std::expected<void, int> e) {
  e.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to 'std::expected' error [bugprone-unchecked-expected-access]
}

// Checked error accesses.

void checked_error_access_with_operator_bool(std::expected<int, int> e) {
  if (!e) {
    e.error();
  }
}

void checked_error_access_with_has_value(std::expected<int, int> e) {
  if (!e.has_value()) {
    e.error();
  }
}

void checked_error_access_in_else_branch(std::expected<int, int> e) {
  if (e) {
  } else {
    e.error();
  }
}

int checked_error_access_after_early_return(std::expected<int, int> e) {
  if (e)
    return 0;
  return e.error();
}

int checked_error_access_in_ternary(std::expected<int, int> e) {
  return e ? 0 : e.error();
}

void checked_value_and_error_access(std::expected<int, int> e) {
  if (e) {
    *e;
  } else {
    e.error();
  }
}

void void_checked_error_access(std::expected<void, int> e) {
  if (!e)
    e.error();
}

void error_or_is_not_an_access(std::expected<int, int> e) {
  e.error_or(0);
}

void unexpected_error_is_not_diagnosed() {
  std::unexpected<int> u(1);
  u.error();
}

// emplace.

void emplace_then_access(std::expected<int, int> e) {
  e.emplace(1);
  e.value();
  *e;
}

void emplace_then_arrow_access(std::expected<Foo, int> e) {
  e.emplace();
  e->foo();
}

int emplace_if_empty(std::expected<int, int> e) {
  if (!e)
    e.emplace(0);
  return *e;
}

int emplace_through_pointer(std::expected<int, int> *p) {
  p->emplace(1);
  return **p;
}

void void_emplace_then_access(std::expected<void, int> e) {
  e.emplace();
  e.value();
}

void error_access_after_emplace(std::expected<int, int> e) {
  e.emplace(1);
  e.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to 'std::expected' error [bugprone-unchecked-expected-access]
}

int emplace_on_other_object(std::expected<int, int> a,
                            std::expected<int, int> b) {
  a.emplace(1);
  return *b;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

int emplace_on_one_path(std::expected<int, int> e, bool cond) {
  if (cond)
    e.emplace(1);
  return *e;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

void void_error_access_after_emplace(std::expected<void, int> e) {
  e.emplace();
  e.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to 'std::expected' error [bugprone-unchecked-expected-access]
}

// Comparison of two expected objects.

int equal_to_expected_with_value(std::expected<int, int> a,
                                 std::expected<int, int> b) {
  if (a == b && b)
    return *a;
  return 0;
}

int equal_to_expected_with_error(std::expected<int, int> a,
                                 std::expected<int, int> b) {
  if (a == b && !b)
    return a.error();
  return 0;
}

int equal_then_check_other(std::expected<int, int> a,
                           std::expected<int, int> b) {
  if (a == b) {
    if (b)
      return *a;
  }
  return 0;
}

int not_not_equal_to_expected_with_value(std::expected<int, int> a,
                                         std::expected<int, int> b) {
  if (a != b)
    return 0;
  if (b)
    return *a;
  return 0;
}

int equal_to_expected_with_other_value_type(std::expected<int, int> a,
                                            std::expected<long, int> b) {
  if (a == b && b)
    return *a;
  return 0;
}

void void_not_equal_to_expected_with_value(std::expected<void, int> a,
                                           std::expected<void, int> b) {
  if (a != b && a)
    b.error();
}

int equal_to_unchecked_expected(std::expected<int, int> a,
                                std::expected<int, int> b) {
  if (a == b)
    return *a;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
  return 0;
}

int not_equal_to_expected_with_value(std::expected<int, int> a,
                                     std::expected<int, int> b) {
  if (a != b && b)
    return *a;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
  return 0;
}

void not_equal_values_may_differ(std::expected<int, int> a,
                                 std::expected<int, int> b) {
  if (a != b && a)
    b.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: unchecked access to 'std::expected' error [bugprone-unchecked-expected-access]
}

int equal_or_other_has_value(std::expected<int, int> a,
                             std::expected<int, int> b) {
  if (a == b || b)
    return *a;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
  return 0;
}

// Comparison of expected with a value.

int equal_to_value(std::expected<int, int> e) {
  if (e == 5)
    return *e;
  return 0;
}

int value_equal_to_expected(std::expected<int, int> e) {
  if (5 == e)
    return e.value();
  return 0;
}

int not_equal_to_value_early_return(std::expected<int, int> e) {
  if (e != 5)
    return 0;
  return *e;
}

long equal_to_value_of_other_type(std::expected<long, int> e, int v) {
  if (e == v)
    return *e;
  return 0;
}

int not_equal_to_value(std::expected<int, int> e) {
  if (e != 5)
    return *e;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
  return 0;
}

int not_equal_to_value_after_equal(std::expected<int, int> e) {
  if (e == 5)
    return 0;
  return *e;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

int error_access_when_equal_to_value(std::expected<int, int> e) {
  if (e == 5)
    return e.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: unchecked access to 'std::expected' error [bugprone-unchecked-expected-access]
  return 0;
}

int error_access_when_not_equal_to_value(std::expected<int, int> e) {
  if (e != 5)
    return e.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: unchecked access to 'std::expected' error [bugprone-unchecked-expected-access]
  return 0;
}

int equal_to_value_or_other(std::expected<int, int> e, bool cond) {
  if (e == 5 || cond)
    return *e;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
  return 0;
}

// Comparison of expected with an unexpected.

int equal_to_unexpected(std::expected<int, int> e) {
  if (e == std::unexpected(1))
    return e.error();
  return 0;
}

int unexpected_equal_to_expected(std::expected<int, int> e) {
  if (std::unexpected(1) == e)
    return e.error();
  return 0;
}

int not_equal_to_unexpected_early_return(std::expected<int, int> e) {
  if (e != std::unexpected(1))
    return 0;
  return e.error();
}

void void_equal_to_unexpected(std::expected<void, int> e) {
  if (e == std::unexpected(1))
    e.error();
}

int value_access_when_equal_to_unexpected(std::expected<int, int> e) {
  if (e == std::unexpected(1))
    return *e;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
  return 0;
}

int error_access_when_not_equal_to_unexpected(std::expected<int, int> e) {
  if (e != std::unexpected(1))
    return e.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: unchecked access to 'std::expected' error [bugprone-unchecked-expected-access]
  return 0;
}

int error_access_after_equal_to_unexpected(std::expected<int, int> e) {
  if (e == std::unexpected(1))
    return 0;
  return e.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: unchecked access to 'std::expected' error [bugprone-unchecked-expected-access]
}

void void_value_access_when_equal_to_unexpected(std::expected<void, int> e) {
  if (e == std::unexpected(1))
    e.value();
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

// Construction.

int default_construct() {
  std::expected<int, int> e;
  return *e;
}

int value_initialize() {
  std::expected<int, int> e{};
  return e.value();
}

int construct_from_value() {
  std::expected<int, int> e(5);
  return *e;
}

int copy_initialize_from_value() {
  std::expected<int, int> e = 5;
  return *e;
}

void construct_in_place() {
  std::expected<Foo, int> e(std::in_place);
  e->foo();
}

void void_default_construct() {
  std::expected<void, int> e;
  e.value();
}

void void_construct_in_place() {
  std::expected<void, int> e(std::in_place);
  e.value();
}

int construct_temporary_from_value() {
  return std::expected<int, int>(5).value();
}

int construct_from_unexpected_rvalue() {
  std::expected<int, int> e(std::unexpected(1));
  return e.error();
}

int construct_from_unexpected_lvalue() {
  const std::unexpected<int> u(1);
  std::expected<int, int> e(u);
  return e.error();
}

int copy_initialize_from_unexpected() {
  std::expected<int, int> e = std::unexpected(1);
  return e.error();
}

int construct_unexpect() {
  std::expected<int, int> e(std::unexpect, 1);
  return e.error();
}

void void_construct_unexpect() {
  std::expected<void, int> e(std::unexpect, 1);
  e.error();
}

int convert_from_checked(std::expected<long, int> b) {
  if (!b)
    return 0;
  std::expected<int, int> a(b);
  return *a;
}

int copy_construct_from_checked(std::expected<int, int> b) {
  if (!b)
    return 0;
  auto a = b;
  return *a;
}

int move_construct_from_checked(std::expected<int, int> b) {
  if (!b)
    return 0;
  auto a = std::move(b);
  return *a;
}

void construct_nested_from_inner_with_error() {
  std::expected<int, int> inner(std::unexpect, 1);
  std::expected<std::expected<int, int>, int> outer(inner);
  outer.value();
}

void construct_nested_from_inner_rvalue_with_error() {
  std::expected<std::expected<int, int>, int> outer(
      std::expected<int, int>(std::unexpect, 1));
  outer.value();
}

void construct_nested_from_unchecked_inner(std::expected<int, int> inner) {
  std::expected<std::expected<int, int>, int> outer(inner);
  outer.value();
}

void convert_nested_from_checked(
    std::expected<std::expected<long, int>, int> b) {
  if (!b)
    return;
  std::expected<std::expected<int, int>, int> a(b);
  a.value();
}

struct FromExpected {
  FromExpected(std::expected<int, int>);
};

void construct_from_expected_as_value(std::expected<int, int> b) {
  std::expected<FromExpected, int> a(b);
  a.value();
}

int error_access_after_default_construct() {
  std::expected<int, int> e;
  return e.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: unchecked access to 'std::expected' error [bugprone-unchecked-expected-access]
}

int error_access_after_construct_from_value() {
  std::expected<int, int> e(5);
  return e.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: unchecked access to 'std::expected' error [bugprone-unchecked-expected-access]
}

int value_access_after_construct_from_unexpected() {
  std::expected<int, int> e(std::unexpected(1));
  return *e;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

int value_access_after_construct_unexpect() {
  std::expected<int, int> e(std::unexpect, 1);
  return *e;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

void void_value_access_after_construct_unexpect() {
  std::expected<void, int> e(std::unexpect, 1);
  e.value();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

int convert_from_unchecked(std::expected<long, int> b) {
  std::expected<int, int> a(b);
  return *a;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

int copy_construct_from_unchecked(std::expected<int, int> b) {
  auto a = b;
  return *a;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

void error_access_after_construct_nested_from_inner_with_error() {
  std::expected<int, int> inner(std::unexpect, 1);
  std::expected<std::expected<int, int>, int> outer(inner);
  outer.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: unchecked access to 'std::expected' error [bugprone-unchecked-expected-access]
}

void convert_nested_from_unchecked(
    std::expected<std::expected<long, int>, int> b) {
  std::expected<std::expected<int, int>, int> a(b);
  a.value();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

int default_construct_then_assign_unexpected() {
  std::expected<int, int> e;
  e = std::unexpected(1);
  return *e;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

// Assignment.

int assign_value(std::expected<int, int> e) {
  e = 5;
  return *e;
}

int assign_braces(std::expected<int, int> e) {
  e = {};
  return *e;
}

int assign_value_after_error(std::expected<int, int> e) {
  if (e)
    return 0;
  e = 5;
  return *e;
}

int assign_unexpected_lvalue(std::expected<int, int> e) {
  const std::unexpected<int> u(1);
  e = u;
  return e.error();
}

int assign_unexpected_rvalue(std::expected<int, int> e) {
  e = std::unexpected(1);
  return e.error();
}

int chained_assign_value(std::expected<int, int> e) {
  return (e = 5).value();
}

void void_assign_unexpected(std::expected<void, int> e) {
  e = std::unexpected(1);
  e.error();
}

int copy_assign_from_checked(std::expected<int, int> a,
                             std::expected<int, int> b) {
  if (!b)
    return 0;
  a = b;
  return *a;
}

int move_assign_from_checked(std::expected<int, int> a,
                             std::expected<int, int> b) {
  if (!b)
    return 0;
  a = std::move(b);
  return *a;
}

int assign_from_converted_checked(std::expected<int, int> a,
                                  std::expected<long, int> b) {
  if (!b)
    return 0;
  a = b;
  return *a;
}

void assign_nested_from_inner_with_error(
    std::expected<std::expected<int, int>, int> outer) {
  std::expected<int, int> inner(std::unexpect, 1);
  outer = inner;
  outer.value();
}

int value_access_after_assign_unexpected(std::expected<int, int> e) {
  e = std::unexpected(1);
  return *e;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

int error_access_after_assign_value(std::expected<int, int> e) {
  e = 5;
  return e.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: unchecked access to 'std::expected' error [bugprone-unchecked-expected-access]
}

int assign_value_on_one_path(std::expected<int, int> e, bool cond) {
  if (cond)
    e = 5;
  return *e;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

int chained_assign_unexpected(std::expected<int, int> e) {
  return (e = std::unexpected(1)).value();
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

void void_value_access_after_assign_unexpected(std::expected<void, int> e) {
  e = std::unexpected(1);
  e.value();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

int copy_assign_from_unchecked(std::expected<int, int> a,
                               std::expected<int, int> b) {
  a = b;
  return *a;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

int assign_from_converted_unchecked(std::expected<int, int> a,
                                    std::expected<long, int> b) {
  a = b;
  return *a;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

void error_access_after_assign_nested_from_inner_with_error(
    std::expected<std::expected<int, int>, int> outer) {
  std::expected<int, int> inner(std::unexpect, 1);
  outer = inner;
  outer.error();
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: unchecked access to 'std::expected' error [bugprone-unchecked-expected-access]
}

// Monadic operations.

constexpr auto add_one = [](int v) { return v + 1; };

int transform_value_access(std::expected<int, int> e) {
  if (!e)
    return 0;
  auto r = e.transform(add_one);
  return *r;
}

int transform_temporary_value_access(std::expected<int, int> e) {
  if (!e)
    return 0;
  return e.transform(add_one).value();
}

int transform_rvalue(std::expected<int, int> e) {
  if (!e)
    return 0;
  auto r = std::move(e).transform(add_one);
  return *r;
}

int transform_const(const std::expected<int, int> &e) {
  if (!e)
    return 0;
  return *e.transform(add_one);
}

int transform_error_access_when_source_has_error(std::expected<int, int> e) {
  if (e)
    return 0;
  return e.transform(add_one).error();
}

int transform_error_error_access(std::expected<int, int> e) {
  if (e)
    return 0;
  auto r = e.transform_error(add_one);
  return r.error();
}

int transform_error_value_access(std::expected<int, int> e) {
  if (!e)
    return 0;
  return *e.transform_error(add_one);
}

int chained_transform(std::expected<int, int> e) {
  if (!e)
    return 0;
  return *e.transform(add_one).transform(add_one);
}

void transform_to_void(std::expected<int, int> e) {
  if (!e)
    return;
  e.transform([](int) {}).value();
}

int transform_result_keeps_state(std::expected<int, int> e) {
  if (!e)
    return 0;
  auto r = e.transform(add_one);
  e = std::unexpected(1);
  return *r;
}

int void_transform(std::expected<void, int> e) {
  if (!e)
    return 0;
  return *e.transform([] { return 1; });
}

int void_transform_error(std::expected<void, int> e) {
  if (e)
    return 0;
  return e.transform_error(add_one).error();
}

int transform_unchecked(std::expected<int, int> e) {
  return *e.transform(add_one);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

int value_access_after_transform_of_error(std::expected<int, int> e) {
  if (e)
    return 0;
  return *e.transform(add_one);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

int transform_error_unchecked(std::expected<int, int> e) {
  return e.transform_error(add_one).error();
  // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: unchecked access to 'std::expected' error [bugprone-unchecked-expected-access]
}

int error_access_after_transform_error_of_value(std::expected<int, int> e) {
  if (!e)
    return 0;
  return e.transform_error(add_one).error();
  // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: unchecked access to 'std::expected' error [bugprone-unchecked-expected-access]
}

int void_transform_unchecked(std::expected<void, int> e) {
  return *e.transform([] { return 1; });
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

// Forwarding.

int forward_lvalue_after_check(std::expected<int, int> &e) {
  if (!e)
    return 0;
  return *std::forward<std::expected<int, int> &>(e);
}

int forward_rvalue_after_check(std::expected<int, int> e) {
  if (!e)
    return 0;
  return std::forward<std::expected<int, int>>(e).value();
}

int forward_error_after_check(std::expected<int, int> &e) {
  if (e)
    return 0;
  return std::forward<std::expected<int, int> &>(e).error();
}

template <class Exp> int forward_after_check(Exp &&e) {
  if (!e)
    return 0;
  return *std::forward<Exp>(e);
}

int instantiate_forward_after_check(std::expected<int, int> e) {
  return forward_after_check(e);
}

int forward_unchecked(std::expected<int, int> e) {
  return *std::forward<std::expected<int, int>>(e);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

int forward_value_access_when_error(std::expected<int, int> &e) {
  if (e)
    return 0;
  return *std::forward<std::expected<int, int> &>(e);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

int forward_error_access_when_value(std::expected<int, int> &e) {
  if (!e)
    return 0;
  return std::forward<std::expected<int, int> &>(e).error();
  // CHECK-MESSAGES: :[[@LINE-1]]:53: warning: unchecked access to 'std::expected' error [bugprone-unchecked-expected-access]
}

template <class Exp> int forward_without_check(Exp &&e) {
  return *std::forward<Exp>(e);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: unchecked access to 'std::expected' value [bugprone-unchecked-expected-access]
}

int instantiate_forward_without_check(std::expected<int, int> e) {
  return forward_without_check(e);
}
