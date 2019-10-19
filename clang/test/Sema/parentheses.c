// RUN: %clang_cc1 -fsyntax-only -verify %s
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

  (void)(b ? 0xf0 : 0x10 | b ? 0x5 : 0x2); // expected-warning {{operator '?:' has lower precedence than '|'}} expected-note 2{{place parentheses}}

  (void)((b ? 0xf0 : 0x10) | (b ? 0x5 : 0x2));  // no warning, has parentheses
  (void)(b ? 0xf0 : (0x10 | b) ? 0x5 : 0x2);  // no warning, has parentheses

  (void)(x | b ? 1 : 2); // expected-warning {{operator '?:' has lower precedence than '|'}} expected-note 2{{place parentheses}}
  (void)(x & b ? 1 : 2); // expected-warning {{operator '?:' has lower precedence than '&'}} expected-note 2{{place parentheses}}

  (void)((x | b) ? 1 : 2);  // no warning, has parentheses
  (void)(x | (b ? 1 : 2));  // no warning, has parentheses
  (void)((x & b) ? 1 : 2);  // no warning, has parentheses
  (void)(x & (b ? 1 : 2));  // no warning, has parentheses

  // Only warn on uses of the bitwise operators, and not the logical operators.
  // The bitwise operators are more likely to be bugs while the logical
  // operators are more likely to be used correctly.  Since there is no
  // explicit logical-xor operator, the bitwise-xor is commonly used instead.
  // For this warning, treat the bitwise-xor as if it were a logical operator.
  (void)(x ^ b ? 1 : 2);  // no warning, ^ is often used as logical xor
  (void)(x || b ? 1 : 2);  // no warning, logical operator
  (void)(x && b ? 1 : 2);  // no warning, logical operator
}

// RUN: not %clang_cc1 -fsyntax-only -Wparentheses -Werror -fdiagnostics-show-option %s 2>&1 | FileCheck %s -check-prefix=CHECK-FLAG
// CHECK-FLAG: error: using the result of an assignment as a condition without parentheses [-Werror,-Wparentheses]
