// This test-case runs several sub-tests on -verify to ensure that correct
// diagnostics are generated in relation to the mis-use and non-use of the
// 'expected-no-diagnostics' directive.

// RUN: %clang_cc1 -DTEST1 -verify %s 2>&1 | FileCheck -check-prefix=CHECK1 %s
#ifdef TEST1
// expected-no-diagnostics
// expected-note {{}}

//      CHECK1: error: 'error' diagnostics seen but not expected:
// CHECK1-NEXT:   Line 8: expected directive cannot follow 'expected-no-diagnostics' directive
// CHECK1-NEXT: 1 error generated.
#endif

// RUN: %clang_cc1 -DTEST2 -verify %s 2>&1 | FileCheck -check-prefix=CHECK2 %s
#ifdef TEST2
#warning X
// expected-warning@-1 {{X}}
// expected-no-diagnostics

//      CHECK2: error: 'error' diagnostics seen but not expected:
// CHECK2-NEXT:   Line 19: 'expected-no-diagnostics' directive cannot follow other expected directives
// CHECK2-NEXT: 1 error generated.
#endif

// RUN: %clang_cc1 -DTEST3 -verify %s 2>&1 | FileCheck -check-prefix=CHECK3 %s
// RUN: %clang_cc1 -verify 2>&1 | FileCheck -check-prefix=CHECK3 %s
#ifdef TEST3
// no directives

//      CHECK3: error: no expected directives found: consider use of 'expected-no-diagnostics'
// CHECK3-NEXT: 1 error generated.
#endif

// RUN: %clang_cc1 -E -DTEST4 -verify %s 2>&1 | FileCheck -check-prefix=CHECK4 %s
#ifdef TEST4
#warning X
// expected-warning@-1 {{X}}

// CHECK4-NOT: error: no expected directives found: consider use of 'expected-no-diagnostics'
#endif
