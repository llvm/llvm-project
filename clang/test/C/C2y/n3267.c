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
  if (__attribute__((assume(1 > 0))); true) {}
  if (__attribute__(()); true) {}
  if (__attribute__((deprecated)) auto x = 3) {}
  if (auto x __attribute__((deprecated)) = 3) {}
  if (__attribute__((deprecated)) auto x = 3) {x += 1;} /* expected-warning {{'x' is deprecated}}
                                                           expected-note {{'x' has been explicitly marked deprecated here}} */
  if (int x __attribute__((deprecated)) = 3; x) {} /* expected-warning {{'x' is deprecated}}
                                                      expected-note {{'x' has been explicitly marked deprecated here}} */
  if (__extension__ int x = 3; x) {}
  if (__extension__ int x = 3) {}
  if (__extension__ __extension__ auto x = 1; x) {}
  if (__extension__ [[maybe_unused]] int x = 3; x > 0) {}
  if (__extension__ static_assert(true); true) {}
  if (__extension__ struct ExtTag { int x; } a = {.x = 1}; a.x) {}
  // __extension__ does not silence ordinary (non-extension) diagnostics.
  if (__extension__ int dx __attribute__((deprecated)) = 3; dx) {} /* expected-warning {{'dx' is deprecated}}
                                                                      expected-note {{'dx' has been explicitly marked deprecated here}} */
  if (__extension__ [[]]; true) {}
  if (__extension__ enum ExtEnum { EA } e = EA; e) {}
  // The second selection-header form is 'declaration expression', whose first
  // clause is the full C 'declaration' grammar -- so these declaration
  // varieties are all valid in it. (Only the third form, the simple-declaration
  // that is itself the controlling expression, is limited to a single
  // declarator with a mandatory initializer.)
  if (int a = 1, b = 2; a + b) {}      // multiple declarators
  if (static int s = 0; s) {}          // storage-class specifier
  if (int fn(void); true) {}           // function declaration
  if (typedef int LocalT; true) {}     // typedef declaration
  if (_Static_assert(1, ""); true) {}  // _Static_assert declaration
  if (auto x = 3) {}
  if (auto x = 3; x == 3) {}
  int y = 1;
  if (auto x = &y) {}
  if (auto x = &y; *x == 1) {}
  // The classic 'T * x' ambiguity resolves to a declaration when T names a type.
  typedef int MyInt;
  if (MyInt * p = &y; p) {}            // declaration of a pointer
  if (MyInt (q) = 3; q) {}             // parenthesized declarator
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
  switch (__attribute__((assume(1 > 0))); 1) {default:}
  switch (__attribute__(()); 1) {default:}
  switch (__attribute__((deprecated)) auto x = 3) {default:}
  switch (auto x __attribute__((deprecated)) = 3) {default:}
  switch (__attribute__((deprecated)) auto x = 3) {default: x += 1;} /* expected-warning {{'x' is deprecated}}
                                                                        expected-note {{'x' has been explicitly marked deprecated here}} */
  switch (int x __attribute__((deprecated)) = 3; x) {default:} /* expected-warning {{'x' is deprecated}}
                                                                  expected-note {{'x' has been explicitly marked deprecated here}} */
  switch (__extension__ int x = 1; x) {default:}
  switch (__extension__ static_assert(true); 1) {default:}
  switch (__extension__ enum SwEnum { SA, SB } e = SA; e) {default:}
  switch (int a = 1, b = 2; a + b) {default:}

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
  // The third form (the declaration that is itself the controlling expression)
  // is limited to a single declarator, so a declarator-list is rejected.
  if (int x = 1, y = 2) {} /* expected-error {{expected ')'}}
                              expected-note {{to match this '('}} */
  if (; true) {} // expected-error {{first clause in condition must be a declaration}}
  if (__extension__; true) {} // expected-error {{first clause in condition must be a declaration}}
  if (__extension__ true; true) {} /* expected-error {{first clause in condition must be a declaration}}
                                      expected-warning {{expression result unused}} */
  if (struct Incomplete s; true) {} /* expected-error {{variable has incomplete type 'struct Incomplete'}}
                                       expected-note {{forward declaration of 'struct Incomplete'}} */
  int a = 2;
  if (a * 1; true) {} /* expected-error {{first clause in condition must be a declaration}}
                         expected-warning {{expression result unused}} */
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

  switch (__extension__; 1) { // expected-error {{first clause in condition must be a declaration}}
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
