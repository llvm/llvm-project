// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -analyzer-checker=core,debug.ExprInspection

#include "Inputs/errno_var.h"
#include "Inputs/some_system_globals.h"
#include "Inputs/some_user_globals.h"

extern void abort() __attribute__((__noreturn__));
#define assert(expr) ((expr) ? (void)(0) : abort())

void invalidate_globals(void);
void clang_analyzer_eval(int x);

/// Test the special system 'errno'
void test_errno_constraint() {
  assert(errno > 2);
  clang_analyzer_eval(errno > 2); // expected-warning {{TRUE}}
  invalidate_globals();
  clang_analyzer_eval(errno > 2); // expected-warning {{UNKNOWN}}
}
void test_errno_assign(int x) {
  errno = x;
  clang_analyzer_eval(errno == x); // expected-warning {{TRUE}}
  invalidate_globals();
  clang_analyzer_eval(errno == x); // expected-warning {{UNKNOWN}}
}

/// Test user global variables
void test_my_const_user_global_constraint() {
  assert(my_const_user_global > 2);
  clang_analyzer_eval(my_const_user_global > 2); // expected-warning {{TRUE}}
  invalidate_globals();
  clang_analyzer_eval(my_const_user_global > 2); // expected-warning {{TRUE}}
}
void test_my_const_user_global_assign(int); // One cannot assign value to a const lvalue.

void test_my_mutable_user_global_constraint() {
  assert(my_mutable_user_global > 2);
  clang_analyzer_eval(my_mutable_user_global > 2); // expected-warning {{TRUE}}
  invalidate_globals();
  clang_analyzer_eval(my_mutable_user_global > 2); // expected-warning {{UNKNOWN}}
}
void test_my_mutable_user_global_assign(int x) {
  my_mutable_user_global = x;
  clang_analyzer_eval(my_mutable_user_global == x); // expected-warning {{TRUE}}
  invalidate_globals();
  clang_analyzer_eval(my_mutable_user_global == x); // expected-warning {{UNKNOWN}}
}

/// Test system global variables
void test_my_const_system_global_constraint() {
  assert(my_const_system_global > 2);
  clang_analyzer_eval(my_const_system_global > 2); // expected-warning {{TRUE}}
  invalidate_globals();
  clang_analyzer_eval(my_const_system_global > 2); // expected-warning {{TRUE}}
}
void test_my_const_system_global_assign(int);// One cannot assign value to a const lvalue.

void test_my_mutable_system_global_constraint() {
  assert(my_mutable_system_global > 2);
  clang_analyzer_eval(my_mutable_system_global > 2); // expected-warning {{TRUE}}
  invalidate_globals();
  clang_analyzer_eval(my_mutable_system_global > 2); // expected-warning {{UNKNOWN}}
}
void test_my_mutable_system_global_assign(int x) {
  my_mutable_system_global = x;
  clang_analyzer_eval(my_mutable_system_global == x); // expected-warning {{TRUE}}
  invalidate_globals();
  clang_analyzer_eval(my_mutable_system_global == x); // expected-warning {{UNKNOWN}}
}
