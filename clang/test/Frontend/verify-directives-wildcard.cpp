// RUN: not %clang_cc1 -verify -verify-directives %s 2>&1 | FileCheck %s

void f1(A);
// expected-error@* {{unknown type name 'A'}}

// CHECK:      error: 'expected-error' diagnostics seen but not expected:
// CHECK-NEXT:   Line 4: diagnostic verification mode disallows use of a wildcard for diagnostic location

void f2(A);
// expected-error@*:* {{unknown type name 'A'}}

// CHECK-NEXT:   Line 10: diagnostic verification mode disallows use of a wildcard for diagnostic location

// CHECK-NEXT: 2 errors generated.
