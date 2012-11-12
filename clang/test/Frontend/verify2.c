#if 0
// RUN: %clang_cc1 -verify %s 2>&1 | FileCheck %s

// Please note that all comments are inside "#if 0" blocks so that
// VerifyDiagnosticConsumer sees no comments while processing this
// test-case (and hence no expected-* directives).
#endif

#include "verify2.h"
#error source

#if 0
// expected-error {{should be ignored}}

//      CHECK: error: no expected directives found: consider use of 'expected-no-diagnostics'
// CHECK-NEXT: error: 'error' diagnostics seen but not expected:
// CHECK-NEXT:   Line 1: header
// CHECK-NEXT:   Line 10: source
// CHECK-NEXT: 3 errors generated.
#endif
