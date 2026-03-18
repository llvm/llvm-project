// RUN: not %clang_cc1 -verify -verify-directives %s 2>&1 | FileCheck %s

void f1(A);
// expected-error@-1 {{unknown type}}

// CHECK:      error: diagnostic messages of 'error' severity not fully matched:
// CHECK-NEXT:   'expected-error' at line 4 in {{.*}}: unknown type
// CHECK-NEXT:      does not fully match diagnostic at line 3: unknown type name 'A'

// CHECK-NEXT: 1 error generated.
