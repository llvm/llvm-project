// RUN: %clang_cc1 -arcmt-action=check -verify %s
// RUN: not %clang_cc1 -arcmt-action=check -verify %t.invalid 2>&1 | FileCheck %s

#if 0
// expected-error {{should be ignored}}
#endif

#error should not be ignored
// expected-error@-1 {{should not be ignored}}

#error
// expected-error@-1 {{}}

//      CHECK: error: no expected directives found: consider use of 'expected-no-diagnostics'
// CHECK-NEXT: error: 'expected-error' diagnostics seen but not expected:
// CHECK-NEXT:   (frontend): error reading '{{.*}}verify.m.tmp.invalid'
// CHECK-NEXT: 2 errors generated.
