// RUN: %clang_cc1 -triple i686-pc-linux-gnu -fsyntax-only -verify %s -Wabsolute-value
// RUN: %clang_cc1 -triple i686-pc-linux-gnu -fsyntax-only %s -Wabsolute-value -fdiagnostics-parseable-fixits 2>&1 | FileCheck %s

extern "C" {
  int abs(int);
  float fabsf(float);
}

namespace std {
  int abs(int);
  float abs(float);
}

void test(long long ll, double d, int i, float f) {
  // Suggest including cmath
  (void)abs(d);
  // expected-warning@-1{{using integer absolute value function 'abs' when argument is of floating point type}}
  // expected-note@-2{{use function 'std::abs' instead}}
  // expected-note@-3{{include the header <cmath> or explicitly provide a declaration for 'std::abs'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:9-[[@LINE-4]]:12}:"std::abs"

  (void)fabsf(d);
  // expected-warning@-1{{absolute value function 'fabsf' given an argument of type 'double' but has parameter of type 'float' which may cause truncation of value}}
  // expected-note@-2{{use function 'std::abs' instead}}
  // expected-note@-3{{include the header <cmath> or explicitly provide a declaration for 'std::abs'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:9-[[@LINE-4]]:14}:"std::abs"

  // Suggest including cstdlib
  (void)abs(ll);
  // expected-warning@-1{{absolute value function 'abs' given an argument of type 'long long' but has parameter of type 'int' which may cause truncation of value}}
  // expected-note@-2{{use function 'std::abs' instead}}
  // expected-note@-3{{include the header <cstdlib> or explicitly provide a declaration for 'std::abs'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:9-[[@LINE-4]]:12}:"std::abs"
  (void)fabsf(ll);
  // expected-warning@-1{{using floating point absolute value function 'fabsf' when argument is of integer type}}
  // expected-note@-2{{use function 'std::abs' instead}}
  // expected-note@-3{{include the header <cstdlib> or explicitly provide a declaration for 'std::abs'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:9-[[@LINE-4]]:14}:"std::abs"

  // Proper function already called, no warnings.
  (void)abs(i);
  (void)fabsf(f);

  // Declarations found, suggest name change.
  (void)fabsf(i);
  // expected-warning@-1{{using floating point absolute value function 'fabsf' when argument is of integer type}}
  // expected-note@-2{{use function 'std::abs' instead}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:9-[[@LINE-3]]:14}:"std::abs"
  (void)abs(f);
  // expected-warning@-1{{using integer absolute value function 'abs' when argument is of floating point type}}
  // expected-note@-2{{use function 'std::abs' instead}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:9-[[@LINE-3]]:12}:"std::abs"
}
