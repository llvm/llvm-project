// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -x objective-c -fmodule-cache-path %t -F %S/Inputs %s 2>&1 | FileCheck %s
// FIXME: When we have a syntax for modules in C, use that.
@__experimental_modules_import MutuallyRecursive1;

// FIXME: Lots of redundant diagnostics here, because the preprocessor
// can't currently tell the parser not to try to load the module again.

// CHECK: MutuallyRecursive2.h:3:32: fatal error: cyclic dependency in module 'MutuallyRecursive1': MutuallyRecursive1 -> MutuallyRecursive2 -> MutuallyRecursive1
// CHECK: MutuallyRecursive1.h:2:32: fatal error: could not build module 'MutuallyRecursive2'
// CHECK: cycles.c:4:32: fatal error: could not build module 'MutuallyRecursive1'

