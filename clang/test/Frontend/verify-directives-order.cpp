// RUN: not %clang_cc1 -verify -verify-directives %s 2>&1 | FileCheck %s

void f1(A); // #f1
void f2(B); // #f2
// expected-error@#f2 {{unknown type name 'B'}}
// expected-error@#f1 {{unknown type name 'A'}}

// CHECK:      error: all diagnostics were successfully matched, but out-of-order directives were found:
// CHECK-NEXT:   'expected-error' at line 5 in {{.*}}: unknown type name 'B'
// CHECK-NEXT:     matches diagnostic at line 4, but diagnostic at line 3 was emitted first:
// CHECK-NEXT:       unknown type name 'A'
// CHECK-NEXT:   'expected-error' at line 6 in {{.*}}: unknown type name 'A'
// CHECK-NEXT:     matches diagnostic at line 3, but diagnostic at line 4 was emitted first:
// CHECK-NEXT:       unknown type name 'B'
