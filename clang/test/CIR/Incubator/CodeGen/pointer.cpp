// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Global pointer should be zero initialized by default.
int *ptr;
// CHECK: cir.global external @ptr = #cir.ptr<null> : !cir.ptr<!s32i>
