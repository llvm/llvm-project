// RUN: not %clang_cc1 -Wfatal-errors -verify %s 2>&1 | FileCheck %s

#error first fatal
// expected-error@-1 {{first fatal}}

#error second fatal
// expected-error@-1 {{second fatal}}


//      CHECK: error: 'expected-error' diagnostics expected but not seen:
// CHECK-NEXT:   Line 6 (directive at {{.*}}verify-fatal.c:7): second fatal
// CHECK-NEXT: 1 error generated.
