// Test that implicit module builds diagnose redefinition of a module when the
// same modulemap file contains duplicate module declarations.

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: not %clang_cc1 -x objective-c -fmodules -fimplicit-module-maps \
// RUN:   -I %t/include \
// RUN:   -fmodules-cache-path=%t/cache \
// RUN:   %t/test.m 2>&1 | FileCheck %s

// CHECK: module.modulemap:9:8: error: redefinition of module 'A'
// CHECK: module.modulemap:1:8: note: previously defined here
// CHECK: fatal error: could not build module 'A'

//--- include/module.modulemap
module A {
    header "A.h"
}

module A1 {
    header "A1.h"
}

module A {
    header "A.h"
}

//--- include/A.h
// empty

//--- include/A1.h
// empty

//--- test.m
@import A;
