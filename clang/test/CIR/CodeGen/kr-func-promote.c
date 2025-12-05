// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s

// CHECK: cir.func {{.*}}@foo(%arg0: !s32i
// CHECK:   %0 = cir.alloca !s16i, !cir.ptr<!s16i>, ["x", init]
// CHECK:   %1 = cir.cast integral %arg0 : !s32i -> !s16i
// CHECK:   cir.store %1, %0 : !s16i, !cir.ptr<!s16i>
void foo(x) short x; {}

// CHECK: cir.func no_proto dso_local @bar(%arg0: !cir.double
// CHECK:   %0 = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["f", init]
// CHECK:   %1 = cir.cast floating %arg0 : !cir.double -> !cir.float
// CHECK:   cir.store %1, %0 : !cir.float, !cir.ptr<!cir.float>
void bar(f) float f; {}
