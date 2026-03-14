// RUN: %clang_cc1 -fsyntax-only -Wuninitialized -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wuninitialized -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

_Bool test_bool_no_false(void) {
  _Bool var; // expected-note {{initialize}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:12-[[@LINE-1]]:12}:" = 0"
  return var; // expected-warning {{uninitialized}}
}

#define bool _Bool
#define false (bool)0
#define true (bool)1
bool test_bool_with_false(void) {
  bool var; // expected-note {{initialize}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:11}:" = false"
  return var; // expected-warning {{uninitialized}}
}

bool test_bool_with_false_undefined(void) {
  bool
#undef false
      var; // expected-note {{initialize}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:10-[[@LINE-1]]:10}:" = 0"
  return var; // expected-warning {{uninitialized}}
}
