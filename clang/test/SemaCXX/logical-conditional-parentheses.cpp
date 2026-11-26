// RUN: %clang_cc1 -fsyntax-only -verify -DSILENCE %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wparentheses %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wlogical-conditional-parentheses %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wparentheses -Wno-logical-conditional-parentheses -DSILENCE %s
// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits -Wlogical-conditional-parentheses %s 2>&1 | FileCheck %s

#ifdef SILENCE
// expected-no-diagnostics
#endif

bool predA(int x);
bool predB(int x);

bool or_with_comparisons(int i, int cnt, int flags, int len, int max_len) {
  return (i != cnt) || (flags & 1) ? (len > max_len) : (len != max_len);
#ifndef SILENCE
  // expected-warning@-2 {{operator '?:' has lower precedence than '||'; '||' will be evaluated first}}
  // expected-note@-3 {{place parentheses around the '||' expression to silence this warning}}
  // expected-note@-4 {{place parentheses around the '?:' expression to evaluate it first}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:10-[[@LINE-6]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-7]]:35-[[@LINE-7]]:35}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-8]]:24-[[@LINE-8]]:24}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-9]]:72-[[@LINE-9]]:72}:")"
}

bool and_with_bool_calls(bool a, bool b, int x) {
  return a && b ? predA(x) : predB(x);
#ifndef SILENCE
  // expected-warning@-2 {{operator '?:' has lower precedence than '&&'; '&&' will be evaluated first}}
  // expected-note@-3 {{place parentheses around the '&&' expression to silence this warning}}
  // expected-note@-4 {{place parentheses around the '?:' expression to evaluate it first}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:10-[[@LINE-6]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-7]]:16-[[@LINE-7]]:16}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-8]]:15-[[@LINE-8]]:15}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-9]]:38-[[@LINE-9]]:38}:")"
}

bool chained_or_with_bools(bool a, bool b, bool c, bool x, bool y) {
  return a || b || c ? x : y;
#ifndef SILENCE
  // expected-warning@-2 {{operator '?:' has lower precedence than '||'; '||' will be evaluated first}}
  // expected-note@-3 {{place parentheses around the '||' expression to silence this warning}}
  // expected-note@-4 {{place parentheses around the '?:' expression to evaluate it first}}
#endif
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:10-[[@LINE-6]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-7]]:21-[[@LINE-7]]:21}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-8]]:20-[[@LINE-8]]:20}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-9]]:29-[[@LINE-9]]:29}:")"
}

bool inverted_bool_literals(bool a, bool b) {
  // (a || b) ? false : true != a || (b ? false : true)
  bool r1 = a || b ? false : true;
#ifndef SILENCE
  // expected-warning@-2 {{operator '?:' has lower precedence than '||'; '||' will be evaluated first}}
  // expected-note@-3 {{place parentheses around the '||' expression to silence this warning}}
  // expected-note@-4 {{place parentheses around the '?:' expression to evaluate it first}}
#endif
  // (a && b) ? false : true != a && (b ? false : true)
  bool r2 = a && b ? false : true;
#ifndef SILENCE
  // expected-warning@-2 {{operator '?:' has lower precedence than '&&'; '&&' will be evaluated first}}
  // expected-note@-3 {{place parentheses around the '&&' expression to silence this warning}}
  // expected-note@-4 {{place parentheses around the '?:' expression to evaluate it first}}
#endif
  return r1 && r2;
}

template <typename T>
bool templated_nondependent(bool a, bool b, int x, int y, T) {
  return a || b ? (x > y) : (x != y);
#ifndef SILENCE
  // expected-warning@-2 {{operator '?:' has lower precedence than '||'; '||' will be evaluated first}}
  // expected-note@-3 {{place parentheses around the '||' expression to silence this warning}}
  // expected-note@-4 {{place parentheses around the '?:' expression to evaluate it first}}
#endif
}

template <typename T>
T generic_select(bool a, bool b, T x, T y) {
  // Do not warn merely because a generic template is instantiated with T=bool.
  return a || b ? x : y;
}

bool instantiate_templates(bool a, bool b, int x, int y) {
  return templated_nondependent(a, b, x, y, 0) &&
         generic_select<bool>(a, b, x > y, x != y) &&
         generic_select<int>(a, b, x, y) != 0;
}

// Negative controls: explicitly parenthesized groupings.
bool explicit_parens_controls(int i, int cnt, int flags, int len, int max_len) {
  bool r1 = ((i != cnt) || (flags & 1)) ? (len > max_len) : (len != max_len);
  bool r2 = (i != cnt) || ((flags & 1) ? (len > max_len) : (len != max_len));
  return r1 && r2;
}

// Negative controls: non-boolean branches (including pointers).
int non_boolean_arms(bool a, bool b, int x, int y, const char *p, const char *q) {
  int v1 = a || b ? x : y;
  int v2 = a && b ? x : y;
  const char *s = a && b ? p : q;
  return v1 + v2 + (s != nullptr ? 1 : 0);
}

// Negative controls: semantically equivalent groupings.
bool equivalent_groupings(bool a, bool b, bool x, bool y) {
  bool r1 = a || b ? true : y;
  bool r2 = a && b ? x : false;
  bool r3 = a || b ? true : false;
  bool r4 = a && b ? true : false;
  return r1 && r2 && r3 && r4;
}

// Negative control: overloaded logical operator (expression templates, etc.).
struct ExprVal {
  explicit operator bool() const;
};
ExprVal operator||(ExprVal, ExprVal);
bool overloaded_logical(ExprVal a, ExprVal b, bool x, bool y) {
  return a || b ? x : y;
}
