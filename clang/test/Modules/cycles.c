// RUN: rm -rf %t
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -fmodules-cache-path=%t -F %S/Inputs %s 2>&1 | FileCheck %s
// FIXME: When we have a syntax for modules in C, use that.
@import MutuallyRecursive1;

// CHECK: While building module 'MutuallyRecursive1' imported from
// CHECK: While building module 'MutuallyRecursive2' imported from
// CHECK: MutuallyRecursive2.h:3:9: fatal error: cyclic dependency in module 'MutuallyRecursive1': MutuallyRecursive1 -> MutuallyRecursive2 -> MutuallyRecursive1
// CHECK: MutuallyRecursive1.h:2:9: fatal error: could not build module 'MutuallyRecursive2'
// CHECK: cycles.c:4:9: fatal error: could not build module 'MutuallyRecursive1'
// CHECK: 3 errors generated


