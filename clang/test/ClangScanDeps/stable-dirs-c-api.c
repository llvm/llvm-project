// REQUIRES: shell
// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: c-index-test core -scan-deps -working-dir %S -- %clang \
// RUN:   -c %t/client.c -fmodules -fmodules-cache-path=%t/module-cache \
// RUN:   -isysroot %t/Sysroot -I %t/Sysroot/usr/include 2>&1 | FileCheck %s \
// RUN:   -implicit-check-not error: -implicit-check-not=warning:

//--- Sysroot/usr/include/A/module.modulemap
module A {
  umbrella "."
}

//--- Sysroot/usr/include/A/A.h
typedef int A_t;

//--- client.c
#include <A/A.h>


// CHECK:       module: 
// CHECK-NEXT:    name: A
// CHECK:         is-in-stable-directories: 1 

