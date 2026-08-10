// RUN: rm -rf %t && mkdir %t
// RUN: not %clang_cc1 %s -unknown-argument -serialize-diagnostic-file %t/diag -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: unknown argument: '-unknown-argument'
