// RUN: %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s

void use(int x, unsigned int y, float f) {
  // CHECK: cir.func{{.*}}(%[[X_ARG:.*]]: !s32i{{.*}}, %[[Y_ARG:.*]]: !u32i{{.*}}, %[[F_ARG:.*]]: !cir.float{{.*}}){{.*}}{
  // CHECK-NEXT: %[[X_ALLOC:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
  // CHECK-NEXT: %[[Y_ALLOC:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["y", init]
  // CHECK-NEXT: %[[F_ALLOC:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["f", init]
  // CHECK-NEXT: cir.store %[[X_ARG]], %[[X_ALLOC]] : !s32i, !cir.ptr<!s32i>
  // CHECK-NEXT: cir.store %[[Y_ARG]], %[[Y_ALLOC]] : !u32i, !cir.ptr<!u32i>
  // CHECK-NEXT: cir.store %[[F_ARG]], %[[F_ALLOC]] : !cir.float, !cir.ptr<!cir.float>

  // CHECK-NEXT: acc.atomic.read %[[X_ALLOC]] = %[[Y_ALLOC]] : !cir.ptr<!s32i>, !cir.ptr<!u32i>, !s32i
#pragma acc atomic read
  x = y;

  // CHECK-NEXT: %[[X_LOAD:.*]] = cir.load{{.*}} %[[X_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[X_CAST:.*]] = cir.cast integral %[[X_LOAD]] : !s32i -> !u32i
  // CHECK-NEXT: %[[Y_LOAD:.*]] = cir.load{{.*}} %[[Y_ALLOC]] : !cir.ptr<!u32i>, !u32i
  // CHECK-NEXT: %[[CMP:.*]] = cir.cmp(eq, %[[X_CAST]], %[[Y_LOAD]]) : !u32i, !cir.bool
  // CHECK-NEXT: %[[CMP_CAST:.*]] = builtin.unrealized_conversion_cast %[[CMP]] : !cir.bool to i1
  // CHECK-NEXT: acc.atomic.read if(%[[CMP_CAST]]) %[[F_ALLOC]] = %[[Y_ALLOC]] : !cir.ptr<!cir.float>, !cir.ptr<!u32i>, !cir.float
#pragma acc atomic read if (x == y)
  f = y;
}
