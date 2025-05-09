// Linkage types of global variables
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o -  | FileCheck %s

int aaaa;
// CHECK: cir.global external @aaaa
static int bbbb;
// CHECK: cir.global internal @_ZL4bbbb
inline int cccc;
// CHECK: cir.global linkonce_odr @cccc
[[gnu::selectany]] int dddd;
// CHECK: cir.global weak_odr @dddd
