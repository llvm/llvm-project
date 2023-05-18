// RUN: not %clang_cc1 -verify %s 2>&1 | FileCheck %s

#include "verify-any-file.h"
// expected-error@*:* {{unknown type name 'unexpected'}}

// expected-error@*:* {{missing error}}

// expected-error@*:123 {{invalid line : "*" required}}
//
//      CHECK: error: 'expected-error' diagnostics expected but not seen:
// CHECK-NEXT:   File * Line * (directive at {{.*}}verify-any-file.c:6): missing error
// CHECK-NEXT: error: 'expected-error' diagnostics seen but not expected:
// CHECK-NEXT:   File {{.*}}verify-any-file.c Line 8: missing or invalid line number following '@' in expected '*'
// CHECK-NEXT: 2 errors generated.
