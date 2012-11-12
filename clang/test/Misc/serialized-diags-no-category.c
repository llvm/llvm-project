#error foo
#error bar

// RUN: rm -f %t
// RUN: %clang -ferror-limit=1 -fsyntax-only %s --serialize-diagnostics %t > /dev/null 2>&1 || true
// RUN: c-index-test -read-diagnostics %t 2>&1 | FileCheck %s

// This test case tests that we can handle both fatal errors and errors without categories.

// CHECK: {{.*[/\\]}}serialized-diags-no-category.c:1:2: error: foo []
// CHECK: Number of diagnostics: 2

