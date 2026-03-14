// RUN: %clang_cc1 -triple x86_64-linux-android -emit-pch -o %t %s
// RUN: %clang_cc1 -x ast -ast-print %t | FileCheck %s
// REQUIRES: x86-registered-target
extern int a1_0[sizeof(long double) == 16 ? 1 : -1];
extern int a1_i[__alignof(long double) == 16 ? 1 : -1];

// Verify that long double is 128 bit IEEEquad

long double foo = 1.0E4000L;
// CHECK: long double foo = 1.00000000000000000000000000000000004E+4000L;
