// RUN: %clang_cc1 -x c -fsyntax-only -verify -Wbool-operation %s
// RUN: %clang_cc1 -x c -fsyntax-only -verify -Wall %s
// RUN: %clang_cc1 -x c -fsyntax-only -Wbool-operation -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -Wbool-operation %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -verify -Wall %s
// RUN: %clang_cc1 -x c++ -fsyntax-only -Wbool-operation -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

#ifdef __cplusplus
typedef bool boolean;
#else
typedef _Bool boolean;
#endif

void test(boolean b, int i) {
  b = ~b; // expected-warning {{bitwise negation of a boolean expression always evaluates to 'true'; did you mean logical negation?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:8}:"!"
  b = ~(b); // expected-warning {{bitwise negation of a boolean expression always evaluates to 'true'; did you mean logical negation?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:8}:"!"
  b = ~i;
  i = ~b; // expected-warning {{bitwise negation of a boolean expression; did you mean logical negation?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:8}:"!"
  b = ~(i > 4); // expected-warning {{bitwise negation of a boolean expression always evaluates to 'true'; did you mean logical negation?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:7-[[@LINE-1]]:8}:"!"
}
