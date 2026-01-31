// RUN: not %clang_cc1 -verify -verify-directives %s 2>&1 | FileCheck %s

void f1(A, A);
// expected-error@-1 {{unknown type name 'A'}}
// expected-error@-2 1 {{unknown type name 'A'}}

void f2(A, A);
// expected-error@-1 2 {{unknown type name 'A'}}

// CHECK:      error: 'expected-error' diagnostics seen but not expected:
// CHECK-NEXT:   Line 8: exactly one diagnostic can be matched

void f3(A, A);
// expected-error@-1 0-1 {{unknown type name 'A'}}
// expected-error@-2 0-1 {{unknown type name 'A'}}

// CHECK-NEXT:   Line 14: exactly one diagnostic can be matched
// CHECK-NEXT:   Line 15: exactly one diagnostic can be matched

void f4(A, A);
// expected-error@-1 1-2 {{unknown type name 'A'}}

// CHECK-NEXT:   Line 21: exactly one diagnostic can be matched

void f5(A);
// expected-error@-1 0-2 {{unknown type name 'A'}}

// CHECK-NEXT:   Line 26: exactly one diagnostic can be matched

void f6(A, A);
// expected-error@-1 + {{unknown type name 'A'}}

// CHECK-NEXT:   Line 31: exactly one diagnostic can be matched

void f7(A, A);
// expected-error@-1 0+ {{unknown type name 'A'}}

// CHECK-NEXT:   Line 36: exactly one diagnostic can be matched

void f8(A, A);
// expected-error@-1 1+ {{unknown type name 'A'}}

// CHECK-NEXT:   Line 41: exactly one diagnostic can be matched

// CHECK-NEXT: 8 errors generated.
