// RUN: %clang_cc1 -Wparentheses -fsyntax-only -verify %s
// RUN: %clang_cc1 -Wparentheses -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

// Test the various warnings under -Wparentheses
void if_assign(void) {
  int i;
  if (i = 4) {} // expected-warning {{assignment as a condition}} \
                // expected-note{{place parentheses around the assignment to silence this warning}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:7-[[@LINE-3]]:7}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:12-[[@LINE-4]]:12}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:9-[[@LINE-5]]:10}:"=="

  if ((i = 4)) {}
}

void bitwise_rel(unsigned i) {
  (void)(i & 0x2 == 0); // expected-warning {{& has lower precedence than ==}} \
                        // expected-note{{place parentheses around the '==' expression to silence this warning}} \
                        // expected-note{{place parentheses around the & expression to evaluate it first}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:14-[[@LINE-3]]:14}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:22-[[@LINE-4]]:22}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:10-[[@LINE-5]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:17-[[@LINE-6]]:17}:")"

  (void)(0 == i & 0x2); // expected-warning {{& has lower precedence than ==}} \
                        // expected-note{{place parentheses around the '==' expression to silence this warning}} \
                        // expected-note{{place parentheses around the & expression to evaluate it first}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:16-[[@LINE-4]]:16}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:15-[[@LINE-5]]:15}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:22-[[@LINE-6]]:22}:")"

  (void)(i & 0xff < 30); // expected-warning {{& has lower precedence than <}} \
                         // expected-note{{place parentheses around the '<' expression to silence this warning}} \
                         // expected-note{{place parentheses around the & expression to evaluate it first}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:14-[[@LINE-3]]:14}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:23-[[@LINE-4]]:23}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:10-[[@LINE-5]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:18-[[@LINE-6]]:18}:")"

  (void)((i & 0x2) == 0);
  (void)(i & (0x2 == 0));
  // Eager logical op
  (void)(i == 1 | i == 2 | i == 3);
  (void)(i != 1 & i != 2 & i != 3);

  (void)(i & i | i); // expected-warning {{'&' within '|'}} \
                     // expected-note {{place parentheses around the '&' expression to silence this warning}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:10-[[@LINE-2]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:15-[[@LINE-3]]:15}:")"

  (void)(i | i & i); // expected-warning {{'&' within '|'}} \
                     // expected-note {{place parentheses around the '&' expression to silence this warning}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:14-[[@LINE-2]]:14}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:19-[[@LINE-3]]:19}:")"

  (void)(i ||
             i && i); // expected-warning {{'&&' within '||'}} \
                      // expected-note {{place parentheses around the '&&' expression to silence this warning}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:14-[[@LINE-2]]:14}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:20-[[@LINE-3]]:20}:")"

  (void)(i || i && "w00t"); // no warning.
  (void)("w00t" && i || i); // no warning.

  (void)(i || i && "w00t" || i); // expected-warning {{'&&' within '||'}} \
                                 // expected-note {{place parentheses around the '&&' expression to silence this warning}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:15-[[@LINE-2]]:15}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:26-[[@LINE-3]]:26}:")"

  (void)(i || "w00t" && i || i); // expected-warning {{'&&' within '||'}} \
                                 // expected-note {{place parentheses around the '&&' expression to silence this warning}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:15-[[@LINE-2]]:15}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:26-[[@LINE-3]]:26}:")"

  (void)(i && i || 0); // no warning.
  (void)(0 || i && i); // no warning.
}

_Bool someConditionFunc();

void conditional_op(int x, int y, _Bool b, void* p) {
  (void)(x + someConditionFunc() ? 1 : 2); // expected-warning {{operator '?:' has lower precedence than '+'}} \
                                           // expected-note {{place parentheses around the '+' expression to silence this warning}} \
                                           // expected-note {{place parentheses around the '?:' expression to evaluate it first}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:33-[[@LINE-4]]:33}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:14-[[@LINE-5]]:14}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:41-[[@LINE-6]]:41}:")"

  (void)((x + someConditionFunc()) ? 1 : 2); // no warning

  (void)(x - b ? 1 : 2); // expected-warning {{operator '?:' has lower precedence than '-'}} \
                         // expected-note {{place parentheses around the '-' expression to silence this warning}} \
                         // expected-note {{place parentheses around the '?:' expression to evaluate it first}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:15-[[@LINE-4]]:15}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:14-[[@LINE-5]]:14}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:23-[[@LINE-6]]:23}:")"

  (void)(x * (x == y) ? 1 : 2); // expected-warning {{operator '?:' has lower precedence than '*'}} \
                                // expected-note {{place parentheses around the '*' expression to silence this warning}} \
                                // expected-note {{place parentheses around the '?:' expression to evaluate it first}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:22-[[@LINE-4]]:22}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:14-[[@LINE-5]]:14}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:30-[[@LINE-6]]:30}:")"

  (void)(x / !x ? 1 : 2); // expected-warning {{operator '?:' has lower precedence than '/'}} \
                          // expected-note {{place parentheses around the '/' expression to silence this warning}} \
                          // expected-note {{place parentheses around the '?:' expression to evaluate it first}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:10-[[@LINE-3]]:10}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:16-[[@LINE-4]]:16}:")"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:14-[[@LINE-5]]:14}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-6]]:24-[[@LINE-6]]:24}:")"

  (void)(x % 2 ? 1 : 2); // no warning

  (void)(x + p ? 1 : 2); // expected-warning {{operator '?:' has lower precedence than '+'}} expected-note 2{{place parentheses}}
  (void)(p + x ? 1 : 2); // no warning

  (void)(p + b ? 1 : 2); // expected-warning {{operator '?:' has lower precedence than '+'}} expected-note 2{{place parentheses}}

  (void)(x + y > 0 ? 1 : 2); // no warning
  (void)(x + (y > 0) ? 1 : 2); // expected-warning {{operator '?:' has lower precedence than '+'}} expected-note 2{{place parentheses}}
}

// RUN: not %clang_cc1 -fsyntax-only -Wparentheses -Werror -fdiagnostics-show-option %s 2>&1 | FileCheck %s -check-prefix=CHECK-FLAG
// CHECK-FLAG: error: using the result of an assignment as a condition without parentheses [-Werror,-Wparentheses]
