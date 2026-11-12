// RUN: not %clang_cc1 -fmodules -fmodule-map-file=does-not-exist.modulemap -verify %s 2>&1 | FileCheck %s

// CHECK: module map file 'does-not-exist.modulemap' not found
