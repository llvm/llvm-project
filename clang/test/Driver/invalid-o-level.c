// RUN: not %clang_cc1 %s -O900 2> %t.log
// RUN: FileCheck %s -input-file=%t.log

// CHECK: invalid value '900' in '-O900'
