#if 0
// RUN: not %clang_cc1 -verify %s 2>&1 | FileCheck %s

// Please note that all comments are inside "#if 0" blocks so that
// VerifyDiagnosticConsumer sees no comments while processing this
// test-case (and hence no expected-* directives).
#endif

#include "verify2.h"
#error source

#if 0
// expected-error {{should be ignored}}

//      CHECK: error: no expected directives found: consider use of 'expected-no-diagnostics'
// CHECK-NEXT: error: 'expected-error' diagnostics seen but not expected:
// CHECK-NEXT:   Line 5: header
// CHECK-NEXT:   Line 10: source
// CHECK-NEXT: 3 errors generated.
#endif

#ifdef CHECK2
// RUN: not %clang_cc1 -DCHECK2 -verify %s 2>&1 | FileCheck -check-prefix=CHECK2 %s

// The following checks that -verify can match "any line" in an included file.
// The location of the diagnostic need therefore only match in the file, not to
// a specific line number.  This is useful where -verify is used as a testing
// tool for 3rd-party libraries where headers may change and the specific line
// number of a diagnostic in a header is not important.

// expected-error@verify2.h:* {{header}}
// expected-error@verify2.h:* {{unknown}}

//      CHECK2: error: 'expected-error' diagnostics expected but not seen:
// CHECK2-NEXT:   File {{.*}}verify2.h Line * (directive at {{.*}}verify2.c:32): unknown
// CHECK2-NEXT: error: 'expected-error' diagnostics seen but not expected:
// CHECK2-NEXT:   File {{.*}}verify2.c Line 10: source
// CHECK2-NEXT: 2 errors generated.
#endif
