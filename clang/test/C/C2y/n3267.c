// RUN: %clang_cc1 -std=c2y -verify %s

bool test_if() {
  if (true) {}
  if (bool x = true; x) {}
  if (bool x = false) return x;
  if ([[maybe_unused]] bool x = true) {}
  if (bool x [[maybe_unused]] = true) {}
  if ([[maybe_unused]] int x = 3; x > 0) {}
  if (struct A { int x;} a = {.x = 1}; a.x) {}
  if (int arr[] = {1,2,3}; arr[1]) {}
  if (auto x = 1; x) {}
  if (static_assert(true); true) {}
  if ([[clang::assume(1 > 0)]]; true) {}
  if ([[]]; true) {}
  if (auto x = 3) {}
  if (auto x = 3; x == 3) {}
  int y = 1;
  if (auto x = &y) {}
  if (auto x = &y; *x == 1) {}
  return false;
}

int test_switch() {
  int y = 1;
  switch (y) {}

  switch (int x = 1; x) {
  default:
    y += x;
  }

  switch (int x [[maybe_unused]] = 1) {}
  switch ([[maybe_unused]] int x = 1) {}

  switch (struct A { int x;} a = {.x = 1}; a.x) {}
  switch (int arr[] = {1,2,3}; arr[1]) {}
  switch (auto x = 1; x) {}
  switch (static_assert(true); 1) {
  default:
  }
  switch ([[clang::assume(1 > 0)]]; 1) {
  default:
  }

  switch ([[]]; 1) {
  default:
  }

  switch (auto x = 3) {default:}
  switch (auto x = 3; x) {default:}
  switch (auto x = &y; *x) {default:}

  switch (int x = 1) {
  default:
    return y + x;
  }
}

bool negative_test_if() {
  if (true; true) {} /* expected-error {{first clause in condition must be a declaration}}
                        expected-warning {{expression result unused}}*/
  if (true; ) {} /* expected-error {{first clause in condition must be a declaration}}
                    expected-error {{expected expression}}
                    expected-warning {{expression result unused}} */
  if (bool x = true; bool y = x) return y; // expected-error {{expected expression}}
  if (bool x = true; bool y = x; y) return y; /*expected-error {{expected expression}}
                                                expected-error {{expected ')'}}
                                                expected-note {{to match this '('}}
                                                expected-error {{use of undeclared identifier 'y'}} */

  if (true; bool y = true) return y; /* expected-error {{first clause in condition must be a declaration}}
                                        expected-error {{expected expression}}
                                        expected-warning {{expression result unused}}*/
  if (int x) {} // expected-error {{variable declaration in condition must have an initializer}}
  if (; true) {} // expected-error {{first clause in condition must be a declaration}}
  if (static_assert(1); static_assert(1); static_assert(1); static_assert(1); 1) {} /* expected-error {{expected expression}}
                                                                                       expected-error {{expected expression}}
                                                                                       expected-error {{expected expression}} */
  return false;
}

int negative_test_switch() {
  switch (true; 1) { /* expected-error {{first clause in condition must be a declaration}}
                        expected-warning {{expression result unused}} */
  default:
    break;
  }

  switch (int x) {} // expected-error {{variable declaration in condition must have an initializer}}

  switch (true; ) {} /* expected-error {{first clause in condition must be a declaration}}
                        expected-error {{expected expression}}
                        expected-warning {{expression result unused}} */

  switch (; 1) { // expected-error {{first clause in condition must be a declaration}}
  default:
  }

  switch (static_assert(1); static_assert(1); static_assert(1); static_assert(1); 1) { /* expected-error {{expected expression}}
                                                                                           expected-error {{expected expression}}
                                                                                           expected-error {{expected expression}} */
  default:
  }

  int y = 1;
  switch (auto x = &y) {default:} // expected-error {{statement requires expression of integer type ('int *' invalid)}}

  switch (int x = 1; int y = x) { // expected-error {{expected expression}}
  default:
    return y;
  }
}
