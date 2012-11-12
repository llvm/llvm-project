void foo();

// RUN: %clang -Wall -fsyntax-only %s --serialize-diagnostics %t
// RUN: c-index-test -read-diagnostics %t 2>&1 | FileCheck %s
// RUN: rm -f  %t

// NOTE: it is important that this test case contains no issues.  It tests
// that serialize diagnostics work in the absence of any issues.

// CHECK: Number of diagnostics: 0
