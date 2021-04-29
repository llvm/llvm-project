// RUN: rm -f %t.diag
// RUN: not %clang -c %s --serialize-diagnostics %t.diag -o /dev/null
// RUN: c-index-test -read-diagnostics %t.diag 2>&1 | FileCheck %s

# 1 "" 1
void 1();

// CHECK: :1:6: error:
