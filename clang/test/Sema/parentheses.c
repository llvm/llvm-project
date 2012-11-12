// RUN: %clang_cc1 -Wparentheses -fsyntax-only -verify %s
// RUN: %clang_cc1 -Wparentheses -fixit %s -o - | %clang_cc1 -Wparentheses -Werror -

// Test the various warnings under -Wparentheses
void if_assign(void) {
  int i;
  if (i = 4) {} // expected-warning {{assignment as a condition}} \
                // expected-note{{use '==' to turn this assignment into an equality comparison}} \
  // expected-note{{place parentheses around the assignment to silence this warning}}
  if ((i = 4)) {}
}

void bitwise_rel(unsigned i) {
  (void)(i & 0x2 == 0); // expected-warning {{& has lower precedence than ==}} \
                        // expected-note{{place parentheses around the & expression to evaluate it first}} \
  // expected-note{{place parentheses around the '==' expression to silence this warning}}
  (void)(0 == i & 0x2); // expected-warning {{& has lower precedence than ==}} \
                        // expected-note{{place parentheses around the & expression to evaluate it first}} \
  // expected-note{{place parentheses around the '==' expression to silence this warning}}
  (void)(i & 0xff < 30); // expected-warning {{& has lower precedence than <}} \
                        // expected-note{{place parentheses around the & expression to evaluate it first}} \
  // expected-note{{place parentheses around the '<' expression to silence this warning}}
  (void)((i & 0x2) == 0);
  (void)(i & (0x2 == 0));
  // Eager logical op
  (void)(i == 1 | i == 2 | i == 3);
  (void)(i != 1 & i != 2 & i != 3);

  (void)(i & i | i); // expected-warning {{'&' within '|'}} \
                     // expected-note {{place parentheses around the '&' expression to silence this warning}}

  (void)(i | i & i); // expected-warning {{'&' within '|'}} \
                     // expected-note {{place parentheses around the '&' expression to silence this warning}}

  (void)(i ||
             i && i); // expected-warning {{'&&' within '||'}} \
                       // expected-note {{place parentheses around the '&&' expression to silence this warning}}
  (void)(i || i && "w00t"); // no warning.
  (void)("w00t" && i || i); // no warning.
  (void)(i || i && "w00t" || i); // expected-warning {{'&&' within '||'}} \
                                 // expected-note {{place parentheses around the '&&' expression to silence this warning}}
  (void)(i || "w00t" && i || i); // expected-warning {{'&&' within '||'}} \
                                 // expected-note {{place parentheses around the '&&' expression to silence this warning}}
  (void)(i && i || 0); // no warning.
  (void)(0 || i && i); // no warning.
}

_Bool someConditionFunc();

void conditional_op(int x, int y, _Bool b) {
  (void)(x + someConditionFunc() ? 1 : 2); // expected-warning {{operator '?:' has lower precedence than '+'}} \
                                           // expected-note {{place parentheses around the '?:' expression to evaluate it first}} \
                                           // expected-note {{place parentheses around the '+' expression to silence this warning}}

  (void)((x + someConditionFunc()) ? 1 : 2); // no warning

  (void)(x - b ? 1 : 2); // expected-warning {{operator '?:' has lower precedence than '-'}} \
                         // expected-note {{place parentheses around the '?:' expression to evaluate it first}} \
                         // expected-note {{place parentheses around the '-' expression to silence this warning}}

  (void)(x * (x == y) ? 1 : 2); // expected-warning {{operator '?:' has lower precedence than '*'}} \
                                // expected-note {{place parentheses around the '?:' expression to evaluate it first}} \
                                // expected-note {{place parentheses around the '*' expression to silence this warning}}

  (void)(x / !x ? 1 : 2); // expected-warning {{operator '?:' has lower precedence than '/'}} \
                          // expected-note {{place parentheses around the '?:' expression to evaluate it first}} \
                          // expected-note {{place parentheses around the '/' expression to silence this warning}}

  (void)(x % 2 ? 1 : 2); // no warning
}

// RUN: %clang_cc1 -fsyntax-only -Wparentheses -Werror -fdiagnostics-show-option %s 2>&1 | FileCheck %s
// CHECK: error: using the result of an assignment as a condition without parentheses [-Werror,-Wparentheses]
