// RUN: cir-clang %s -O0 -S -emit-cir | FileCheck %s

int normal_var = 10;

__attribute__((visibility("hidden")))
int hidden_var = 10;

__attribute__((visibility("hidden")))
static int hidden_static_var = 10;

// CHECK-LABEL: cir.global external dso_local @normal_var
// CHECK: #cir.int<10> : !s32i

// CHECK-LABEL: cir.global hidden external dso_local @hidden_var
// CHECK: #cir.int<10> : !s32i

// CHECK-LABEL: cir.global hidden internal dso_local @hidden_static_var
// CHECK: #cir.int<10> : !s32i