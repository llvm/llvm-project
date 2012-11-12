// RUN: not %clang_cc1 %s -cake-is-lie 2> %t.log
// RUN: FileCheck %s -input-file=%t.log

// CHECK: unknown argument
