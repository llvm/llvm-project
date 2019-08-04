// RUN: %clang_cc1 -fsyntax-only -verify %s -DSILENCE
// RUN: %clang_cc1 -fsyntax-only -verify %s -Wlogical-op-parentheses
// RUN: %clang_cc1 -fsyntax-only -verify %s -Wparentheses
// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits %s -Wlogical-op-parentheses 2>&1 | FileCheck %s

#ifdef SILENCE
// expected-no-diagnostics
#endif

void logical_op_parentheses(unsigned i) {
  (void)(i ||
             i && i);
#ifndef SILENCE
  // expected-warning@-2 {{'&&' within '||'}}
  // expected-note@-3 {{place parentheses around the '&&' expression to silence this warning}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:14-[[@LINE-5]]:14}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:20-[[@LINE-6]]:20}:")"

  (void)(i || i && "w00t");
  (void)("w00t" && i || i);

  (void)(i || i && "w00t" || i);
#ifndef SILENCE
  // expected-warning@-2 {{'&&' within '||'}}
  // expected-note@-3 {{place parentheses around the '&&' expression to silence this warning}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:15-[[@LINE-5]]:15}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:26-[[@LINE-6]]:26}:")"

  (void)(i || "w00t" && i || i);
#ifndef SILENCE
  // expected-warning@-2 {{'&&' within '||'}}
  // expected-note@-3 {{place parentheses around the '&&' expression to silence this warning}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:15-[[@LINE-5]]:15}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:26-[[@LINE-6]]:26}:")"

  (void)(i && i || 0);
  (void)(0 || i && i);
}
