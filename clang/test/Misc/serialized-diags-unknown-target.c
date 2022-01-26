// RUN: rm -rf %t && mkdir %t
// RUN: not %clang_cc1 %s -triple blah-unknown-unknown -serialize-diagnostic-file %t/diag -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: unknown target triple 'blah-unknown-unknown', please use -triple or -arch
