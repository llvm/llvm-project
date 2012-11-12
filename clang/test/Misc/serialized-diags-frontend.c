// RUN: rm -f %t
// RUN: %clang -fsyntax-only %s -Wblahblah --serialize-diagnostics %t > /dev/null 2>&1 || true
// RUN: c-index-test -read-diagnostics %t 2>&1 | FileCheck %s

// This test case tests that we can handle frontend diagnostics.

// CHECK: warning: unknown warning option '-Wblahblah'
// CHECK: Number of diagnostics: 1
