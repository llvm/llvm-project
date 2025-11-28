// RUN: %clang_cc1 -fsyntax-only -verify -std=c++14 -triple x86_64-unknown-unknown %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++14 -triple x86_64-unknown-unknown %s -fexperimental-new-constant-interpreter

constexpr int arr[10] = {};

constexpr bool test_constexpr_valid() {
  __builtin_assume_dereferenceable(arr, 40);
  return true;
}
static_assert(test_constexpr_valid(), "");

constexpr bool test_constexpr_partial() {
  __builtin_assume_dereferenceable(&arr[5], 20);
  return true;
}
static_assert(test_constexpr_partial(), "");

constexpr bool test_constexpr_nullptr() {
  __builtin_assume_dereferenceable(nullptr, 4);
  return true;
}
static_assert(test_constexpr_nullptr(), ""); // expected-error {{not an integral constant expression}}

constexpr bool test_constexpr_too_large() {
  __builtin_assume_dereferenceable(arr, 100);
  return true;
}
static_assert(test_constexpr_too_large(), ""); // expected-error {{not an integral constant expression}}

constexpr int single_var = 42;
constexpr bool test_single_var() {
  __builtin_assume_dereferenceable(&single_var, 4);
  return true;
}
static_assert(test_single_var(), "");

constexpr bool test_exact_boundary() {
  __builtin_assume_dereferenceable(&arr[9], 4);
  return true;
}
static_assert(test_exact_boundary(), "");

constexpr bool test_one_over() {
  __builtin_assume_dereferenceable(&arr[9], 5);
  return true;
}
static_assert(test_one_over(), ""); // expected-error {{not an integral constant expression}}

constexpr bool test_zero_size() {
  __builtin_assume_dereferenceable(arr, 0);
  return true;
}
static_assert(test_zero_size(), ""); // expected-error {{not an integral constant expression}}

struct S {
  int x;
  int y;
};
constexpr S s = {1, 2};
constexpr bool test_struct_member() {
  __builtin_assume_dereferenceable(&s.x, 4);
  return true;
}
static_assert(test_struct_member(), "");
