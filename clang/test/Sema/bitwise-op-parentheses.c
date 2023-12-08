// RUN: %clang_cc1 -fsyntax-only -verify %s -DSILENCE
// RUN: %clang_cc1 -fsyntax-only -verify %s -Wbitwise-op-parentheses
// RUN: %clang_cc1 -fsyntax-only -verify %s -Wparentheses
// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits %s -Wbitwise-op-parentheses 2>&1 | FileCheck %s

#ifdef SILENCE
// expected-no-diagnostics
#endif

void bitwise_op_parentheses(unsigned i) {
  (void)(i & i | i);
#ifndef SILENCE
  // expected-warning@-2 {{'&' within '|'}}
  // expected-note@-3 {{place parentheses around the '&' expression to silence this warning}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:10-[[@LINE-5]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:15-[[@LINE-6]]:15}:")"

  (void)(i | i & i);
#ifndef SILENCE
  // expected-warning@-2 {{'&' within '|'}}
  // expected-note@-3 {{place parentheses around the '&' expression to silence this warning}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:14-[[@LINE-5]]:14}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:19-[[@LINE-6]]:19}:")"

  (void)(i ^ i | i);
#ifndef SILENCE
  // expected-warning@-2 {{'^' within '|'}}
  // expected-note@-3 {{place parentheses around the '^' expression to silence this warning}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:10-[[@LINE-5]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:15-[[@LINE-6]]:15}:")"

  (void)(i | i ^ i);
#ifndef SILENCE
  // expected-warning@-2 {{'^' within '|'}}
  // expected-note@-3 {{place parentheses around the '^' expression to silence this warning}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:14-[[@LINE-5]]:14}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:19-[[@LINE-6]]:19}:")"

  (void)(i & i ^ i);
#ifndef SILENCE
  // expected-warning@-2 {{'&' within '^'}}
  // expected-note@-3 {{place parentheses around the '&' expression to silence this warning}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:10-[[@LINE-5]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:15-[[@LINE-6]]:15}:")"

  (void)(i ^ i & i);
#ifndef SILENCE
  // expected-warning@-2 {{'&' within '^'}}
  // expected-note@-3 {{place parentheses around the '&' expression to silence this warning}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:14-[[@LINE-5]]:14}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:19-[[@LINE-6]]:19}:")"
}
