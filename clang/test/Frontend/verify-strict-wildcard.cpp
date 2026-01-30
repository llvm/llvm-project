// RUN: not %clang_cc1 -verify -verify-strict %s 2>&1 | FileCheck %s

void f1(A);
// expected-error@* {{unknown type name 'A'}}

// CHECK:      error: 'expected-error' diagnostics seen but not expected:
// CHECK-NEXT:   Line 3: unknown type name 'A'
// CHECK-NEXT:   Line 4: cannot use wildcard for diagnostic location

void f2(A);
// expected-error@*:6 {{unknown type name 'A'}}

// CHECK-NEXT:   Line 10: unknown type name 'A'
// CHECK-NEXT:   Line 11: cannot use wildcard for diagnostic location

void f3(A);
// expected-error@*:* {{unknown type name 'A'}}

// CHECK-NEXT:   Line 16: unknown type name 'A'
// CHECK-NEXT:   Line 17: cannot use wildcard for diagnostic location

// CHECK-NEXT: 6 errors generated.
