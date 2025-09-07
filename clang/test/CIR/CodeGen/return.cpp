// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s

int &ret0(int &x) {
  return x;
}

// CHECK: cir.func dso_local @_Z4ret0Ri
// CHECK:   %0 = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["x", init, const] {alignment = 8 : i64}
// CHECK:   %1 = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["__retval"] {alignment = 8 : i64}
// CHECK:   cir.store{{.*}} %arg0, %0 : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK:   %2 = cir.load %0 : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK:   cir.store{{.*}} %2, %1 : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK:   %3 = cir.load %1 : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK:   cir.return %3 : !cir.ptr<!s32i>

int unreachable_after_return() {
  return 0;
  return 1;
}

// CHECK: cir.func dso_local @_Z24unreachable_after_returnv
// CHECK-NEXT:   %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK-NEXT:   %1 = cir.const #cir.int<0> : !s32i
// CHECK-NEXT:   cir.store{{.*}} %1, %0 : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:   cir.br ^bb1
// CHECK-NEXT: ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:   %2 = cir.load %0 : !cir.ptr<!s32i>, !s32i
// CHECK-NEXT:   cir.return %2 : !s32i
// CHECK-NEXT: ^bb2:  // no predecessors
// CHECK-NEXT:   %3 = cir.const #cir.int<1> : !s32i
// CHECK-NEXT:   cir.store{{.*}} %3, %0 : !s32i, !cir.ptr<!s32i>
// CHECK-NEXT:   cir.br ^bb1
// CHECK-NEXT: }
