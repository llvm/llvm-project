// RUN: not %clang_cc1 -DTEST_SWITCH -verify-ignore-unexpected=remark,aoeu,note -verify %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-BAD-SWITCH %s
#ifdef TEST_SWITCH
// expected-no-diagnostics
#endif
// CHECK-BAD-SWITCH: error: 'expected-error' diagnostics seen but not expected:
// CHECK-BAD-SWITCH-NEXT: (frontend): invalid value 'aoeu' in '-verify-ignore-unexpected='

// RUN: %clang_cc1 -DTEST1 -verify %s
// RUN: %clang_cc1 -DTEST1 -verify -verify-ignore-unexpected %s
#ifdef TEST1
#warning MyWarning1
    // expected-warning@-1 {{MyWarning1}}
int x; // expected-note {{previous definition is here}}
float x; // expected-error {{redefinition of 'x'}}
#endif

// RUN: not %clang_cc1 -DTEST2 -verify %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-UNEXP %s
// RUN: not %clang_cc1 -DTEST2 -verify -verify-ignore-unexpected= %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-UNEXP %s
// RUN: not %clang_cc1 -DTEST2 -verify -verify-ignore-unexpected=note %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-NOTE %s
// RUN: not %clang_cc1 -DTEST2 -verify -verify-ignore-unexpected=warning %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-WARN %s
// RUN: not %clang_cc1 -DTEST2 -verify -verify-ignore-unexpected=error %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-ERR %s
#ifdef TEST2
#warning MyWarning2
int x;
float x;
#endif
// CHECK-UNEXP: no expected directives found
// CHECK-UNEXP-NEXT: 'expected-error' diagnostics seen but not expected
// CHECK-UNEXP-NEXT: Line {{[0-9]+}}: redefinition of 'x'
// CHECK-UNEXP-NEXT: 'expected-warning' diagnostics seen but not expected
// CHECK-UNEXP-NEXT: Line {{[0-9]+}}: MyWarning2
// CHECK-UNEXP-NEXT: 'expected-note' diagnostics seen but not expected
// CHECK-UNEXP-NEXT: Line {{[0-9]+}}: previous definition is here
// CHECK-UNEXP-NEXT: 4 errors generated.

// CHECK-NOTE: no expected directives found
// CHECK-NOTE-NEXT: 'expected-error' diagnostics seen but not expected
// CHECK-NOTE-NEXT: Line {{[0-9]+}}: redefinition of 'x'
// CHECK-NOTE-NEXT: 'expected-warning' diagnostics seen but not expected
// CHECK-NOTE-NEXT: Line {{[0-9]+}}: MyWarning2
// CHECK-NOTE-NEXT: 3 errors generated.

// CHECK-WARN: no expected directives found
// CHECK-WARN-NEXT: 'expected-error' diagnostics seen but not expected
// CHECK-WARN-NEXT: Line {{[0-9]+}}: redefinition of 'x'
// CHECK-WARN-NEXT: 'expected-note' diagnostics seen but not expected
// CHECK-WARN-NEXT: Line {{[0-9]+}}: previous definition is here
// CHECK-WARN-NEXT: 3 errors generated.

// CHECK-ERR: no expected directives found
// CHECK-ERR-NEXT: 'expected-warning' diagnostics seen but not expected
// CHECK-ERR-NEXT: Line {{[0-9]+}}: MyWarning2
// CHECK-ERR-NEXT: 'expected-note' diagnostics seen but not expected
// CHECK-ERR-NEXT: Line {{[0-9]+}}: previous definition is here
// CHECK-ERR-NEXT: 3 errors generated.

// RUN: not %clang_cc1 -DTEST3 -verify -verify-ignore-unexpected %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-EXP %s
#ifdef TEST3
// expected-error {{test3}}
#endif
// CHECK-EXP: 'expected-error' diagnostics expected but not seen
// CHECK-EXP-NEXT: Line {{[0-9]+}}: test3

// RUN: not %clang_cc1 -DTEST4 -verify -verify-ignore-unexpected %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-NOEXP %s
// RUN: not %clang_cc1 -DTEST4 -verify -verify-ignore-unexpected=warning,error,note %s 2>&1 \
// RUN:     | FileCheck -check-prefix=CHECK-NOEXP %s
#ifdef TEST4
#warning MyWarning4
int x;
float x;
#endif
// CHECK-NOEXP: error: no expected directives found
// CHECK-NOEXP-NEXT: 1 error generated
