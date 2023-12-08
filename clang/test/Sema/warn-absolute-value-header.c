// RUN: %clang_cc1 -triple i686-pc-linux-gnu -fsyntax-only -verify %s -Wabsolute-value
// RUN: %clang_cc1 -triple i686-pc-linux-gnu -fsyntax-only %s -Wabsolute-value -fdiagnostics-parseable-fixits 2>&1 | FileCheck %s

int abs(int);

// Wrong signature
int fabsf(int);
// expected-warning@-1{{incompatible redeclaration of library function 'fabsf'}}
// expected-note@-2{{'fabsf' is a builtin with type 'float (float)'}}

void test_int(int i, unsigned u, long long ll, float f, double d) {
  (void)abs(i);

  // Remove abs call
  (void)abs(u);
  // expected-warning@-1{{taking the absolute value of unsigned type 'unsigned int' has no effect}}
  // expected-note@-2{{remove the call to 'abs' since unsigned values cannot be negative}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:9-[[@LINE-3]]:12}:""

  int llabs;
  (void)llabs;
  // Conflict in names, no notes
  (void)abs(ll);
  // expected-warning@-1{{absolute value function 'abs' given an argument of type 'long long' but has parameter of type 'int' which may cause truncation of value}}

  // Conflict in names, no notes
  (void)abs(f);
  // expected-warning@-1{{using integer absolute value function 'abs' when argument is of floating point type}}

  // Suggest header.
  (void)abs(d);
  // expected-warning@-1{{using integer absolute value function 'abs' when argument is of floating point type}}
  // expected-note@-2{{use function 'fabs' instead}}
  // expected-note@-3{{include the header <math.h> or explicitly provide a declaration for 'fabs'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:9-[[@LINE-4]]:12}:"fabs"
}
