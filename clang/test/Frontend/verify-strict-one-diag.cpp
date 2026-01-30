// RUN: not %clang_cc1 -verify -verify-strict %s 2>&1 | FileCheck %s

void f1(A, A);
// expected-error@-1 {{unknown type name 'A'}}
// expected-error@-2 1 {{unknown type name 'A'}}

void f2(A, A);
// expected-error@-1 2 {{unknown type name 'A'}}

// CHECK:      error: 'expected-error' diagnostics seen but not expected:
// CHECK-NEXT:   Line 7: unknown type name 'A'
// CHECK-NEXT:   Line 7: unknown type name 'A'
// CHECK-NEXT:   Line 8: exactly one diagnostic can be matched

void f3(A, A);
// expected-error@-1 0-1 {{unknown type name 'A'}}
// expected-error@-2 0-1 {{unknown type name 'A'}}

// CHECK-NEXT:   Line 15: unknown type name 'A'
// CHECK-NEXT:   Line 15: unknown type name 'A'
// CHECK-NEXT:   Line 16: exactly one diagnostic can be matched
// CHECK-NEXT:   Line 17: exactly one diagnostic can be matched

void f4(A, A);
// expected-error@-1 1-2 {{unknown type name 'A'}}

// CHECK-NEXT:   Line 24: unknown type name 'A'
// CHECK-NEXT:   Line 24: unknown type name 'A'
// CHECK-NEXT:   Line 25: exactly one diagnostic can be matched

void f5(A);
// expected-error@-1 0-2 {{unknown type name 'A'}}

// CHECK-NEXT:   Line 31: unknown type name 'A'
// CHECK-NEXT:   Line 32: exactly one diagnostic can be matched

void f6(A, A);
// expected-error@-1 + {{unknown type name 'A'}}

// CHECK-NEXT:   Line 37: unknown type name 'A'
// CHECK-NEXT:   Line 37: unknown type name 'A'
// CHECK-NEXT:   Line 38: exactly one diagnostic can be matched

void f7(A, A);
// expected-error@-1 0+ {{unknown type name 'A'}}

// CHECK-NEXT:   Line 44: unknown type name 'A'
// CHECK-NEXT:   Line 44: unknown type name 'A'
// CHECK-NEXT:   Line 45: exactly one diagnostic can be matched

void f8(A, A);
// expected-error@-1 1+ {{unknown type name 'A'}}

// CHECK-NEXT:   Line 51: unknown type name 'A'
// CHECK-NEXT:   Line 51: unknown type name 'A'
// CHECK-NEXT:   Line 52: exactly one diagnostic can be matched

// CHECK-NEXT: 21 errors generated.
