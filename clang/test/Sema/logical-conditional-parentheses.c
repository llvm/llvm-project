// RUN: %clang_cc1 -std=c11 -fsyntax-only -verify -DSILENCE %s
// RUN: %clang_cc1 -std=c11 -fsyntax-only -verify -Wparentheses %s
// RUN: %clang_cc1 -std=c11 -fsyntax-only -verify -Wlogical-conditional-parentheses %s
// RUN: %clang_cc1 -std=c11 -fsyntax-only -verify -Wparentheses -Wno-logical-conditional-parentheses -DSILENCE %s
// RUN: %clang_cc1 -std=c11 -fsyntax-only -fdiagnostics-parseable-fixits -Wlogical-conditional-parentheses %s 2>&1 | FileCheck %s

#ifdef SILENCE
// expected-no-diagnostics
#endif

// Regression modeled on CVE-2026-130875 (C11 comparisons have type int).
int packet_looks_bogus(int i, int arg_cnt, unsigned flags, int len, int datalength) {
  return (i != arg_cnt) || (flags & 1u) ? (len > datalength) : (len != datalength);
#ifndef SILENCE
  // expected-warning@-2 {{operator '?:' has lower precedence than '||'; '||' will be evaluated first}}
  // expected-note@-3 {{place parentheses around the '||' expression to silence this warning}}
  // expected-note@-4 {{place parentheses around the '?:' expression to evaluate it first}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:10-[[@LINE-6]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-7]]:40-[[@LINE-7]]:40}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-8]]:28-[[@LINE-8]]:28}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-9]]:83-[[@LINE-9]]:83}:")"
}

int logical_and_with_bool_arms(_Bool a, _Bool b, _Bool x, _Bool y) {
  return a && b ? x : y;
#ifndef SILENCE
  // expected-warning@-2 {{operator '?:' has lower precedence than '&&'; '&&' will be evaluated first}}
  // expected-note@-3 {{place parentheses around the '&&' expression to silence this warning}}
  // expected-note@-4 {{place parentheses around the '?:' expression to evaluate it first}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:10-[[@LINE-6]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-7]]:16-[[@LINE-7]]:16}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-8]]:15-[[@LINE-8]]:15}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-9]]:24-[[@LINE-9]]:24}:")"
}

int logical_not_arms(int a, int b, int x, int y) {
  return a || b ? !x : !y;
#ifndef SILENCE
  // expected-warning@-2 {{operator '?:' has lower precedence than '||'; '||' will be evaluated first}}
  // expected-note@-3 {{place parentheses around the '||' expression to silence this warning}}
  // expected-note@-4 {{place parentheses around the '?:' expression to evaluate it first}}
#endif
}

// Negative controls: explicitly parenthesized groupings.
int explicit_parens_controls(int i, int arg_cnt, unsigned flags, int len, int datalength) {
  int r1 = ((i != arg_cnt) || (flags & 1u)) ? (len > datalength) : (len != datalength);
  int r2 = (i != arg_cnt) || ((flags & 1u) ? (len > datalength) : (len != datalength));
  return r1 && r2;
}

// Negative controls: non-boolean conditional arms (idiomatic C ternaries).
int non_boolean_arms(int a, int b, int x, int y, const char *p, const char *q) {
  int r1 = a || b ? x : y;
  int r2 = a && b ? 1 : 0;
  int r3 = a || b ? (x > y) : 0;
  const char *s = a && b ? p : q;
  return r1 + r2 + r3 + (s != 0);
}

// Negative control: GNU binary conditional extension (?:).
int gnu_binary_conditional(_Bool a, _Bool b, _Bool y) {
  return a || b ?: y;
}

// Negative controls: macros containing '?' or the logical operator.
#define UNPAREN_COND(A, B) A || B
#define MACRO_TERNARY(A, B, X, Y) A || B ? X : Y
int macro_controls(int a, int b, int x, int y) {
  int r1 = UNPAREN_COND(a, b) ? (x > y) : (x != y);
  int r2 = MACRO_TERNARY(a, b, x > y, x != y);
  return r1 && r2;
}
