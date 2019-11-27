// RUN: %clang -### -g -fdebug-prefix-map=old=new %s 2>&1 | FileCheck %s
// RUN: %clang -### -g -ffile-prefix-map=old=new %s 2>&1 | FileCheck %s

// CHECK: cc1as
// CHECK-SAME: -fdebug-prefix-map=old=new

// More tests for this flag in debug-prefix-map.c.
