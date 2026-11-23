// RUN: %clang_analyze_cc1 -triple aarch64-unknown-linux-gnu \
// RUN:   -analyzer-checker=core,security.VAList -verify %s
// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:   -analyzer-checker=core,security.VAList -verify %s

// Tests for recovering the state of a va_list that is passed to a function
// by value (on AArch64 va_list is a struct, and the analyzer core does not
// propagate the checker's per-region state to the callee's parameter copy).

#include <stdarg.h>

//===--------------------------------------------------------------------===//
// Test an initialized va_list is passed by value to function that copies it.
// No warnings expected, neither on the inlined path nor in the standalone
// analysis of the helper function.
//===--------------------------------------------------------------------===//

void consume(int);

void test_va_copy(va_list v) { // No warnings on the standalone path.
  va_list v_copy;
  // The expected warning comes from test_copy_released below.
  va_copy(v_copy, v); // expected-warning{{Already released va_list is copied}}
  consume(va_arg(v_copy, int));
  va_end(v_copy);
}

void test_va_copy_initialized(int a, ...) {
  va_list v;
  va_start(v, a);
  test_va_copy(v); // no warning
  va_end(v);
}

//===--------------------------------------------------------------------===//
// Test va_arg used directly on a by-value parameter. Again no warnings when
// the caller initialized the list, and no warnings in the standalone
// analysis of the helper function (unknown parameter state).
//===--------------------------------------------------------------------===//

void test_va_arg(va_list v) {
  // The expected warnings comes from test_va_arg_released and
  // test_va_arg_uninitialized_caller, respectively.
  // expected-warning@+2{{va_arg() is called on an already released va_list}}
  // expected-warning@+1{{va_arg() is called on an uninitialized va_list}}
  consume(va_arg(v, int));
}

void test_va_arg_initialized(int a, ...) {
  va_list v;
  va_start(v, a);
  test_va_arg(v); // no warning
  va_end(v);
}

//===--------------------------------------------------------------------===//
// Passing an already released va_list must still be diagnosed when the
// caller is visible.
//===--------------------------------------------------------------------===//

void test_va_copy_released(int a, ...) {
  va_list v;
  va_start(v, a);
  va_end(v);
  test_va_copy(v); // warns at va_copy inside test_va_copy
}

void test_va_arg_released(int a, ...) {
  va_list v;
  va_start(v, a);
  va_end(v);
  test_va_arg(v); // warns at va_arg inside test_va_arg;
}

//===--------------------------------------------------------------------===//
// Genuinely uninitialized locals must be diagnosed.
//===--------------------------------------------------------------------===//

void test_local_copy_uninitialized(void) {
  va_list v, c;
  va_copy(c, v); // expected-warning{{Uninitialized va_list is copied}}
}

void test_local_va_arg_uninitialized(void) {
  va_list v;
  int x = va_arg(v, int); // expected-warning{{va_arg() is called on an uninitialized va_list}}
  (void)x;
}

void test_local_va_end_uninitialized(void) {
  va_list v;
  va_end(v); // expected-warning{{va_end() is called on an uninitialized va_list}}
}

//===--------------------------------------------------------------------===//
// Missing va_end of the copy is detected on the inlined path. In the
// standalone analysis the copy has Unknown state and produces no leak warning.
//===--------------------------------------------------------------------===//

void test_va_copy_leak(va_list v) {
  va_list c;
  va_copy(c, v);
  consume(va_arg(c, int)); // expected-warning{{is leaked}}
}

void test_leak(int a, ...) {
  va_list v;
  va_start(v, a);
  test_va_copy_leak(v);
  va_end(v);
}

//===--------------------------------------------------------------------===//
// Passing uninitialized list by value must still be diagnosed on the inlined
// path.
//===--------------------------------------------------------------------===//

void test_va_copy_uninitialized(va_list v) {
  va_list v_copy;
  va_copy(v_copy, v); // expected-warning{{Uninitialized va_list is copied}}
  consume(va_arg(v_copy, int));
  va_end(v_copy);
}

void test_va_copy_uninitialized_caller(int a, ...) {
  va_list v;
  test_va_copy_uninitialized(v);
}

void test_va_arg_uninitialized_caller(int a, ...) {
  va_list v;
  test_va_arg(v); // warns at va_arg inside test_va_arg
}

//===--------------------------------------------------------------------===//
// The caller's local is dead after its last direct use, but its checker
// state is retained around the call and remains available to the inline
// frame.
//===--------------------------------------------------------------------===//

void test_va_arg_after_va_end(va_list v) {
  va_end(v); // no warning: initialized on the inlined path, unknown standalone
  // expected-warning@+1{{va_arg() is called on an already released va_list}}
  consume(va_arg(v, int));
}

void test_va_arg_after_va_end_caller(int a, ...) {
  va_list v;
  va_start(v, a);
  test_va_arg_after_va_end(v);
}

//===--------------------------------------------------------------------===//
// A caller that does not va_end its own list must still get the usual leak
// diagnostic after a successful helper function call.
//===--------------------------------------------------------------------===//

void test_va_copy_leaked_caller(int a, ...) {
  va_list v;
  va_start(v, a);
  test_va_copy(v);
} // expected-warning{{is leaked}}
