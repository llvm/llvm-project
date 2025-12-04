// RUN: %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s

extern "C" bool condition(int x, unsigned int y, float f);
extern "C" double do_thing(float f);

struct ConvertsToScalar {
  operator float();
};

void use(int x, unsigned int y, float f, ConvertsToScalar cts) {
  // CHECK: cir.func{{.*}}(%[[X_ARG:.*]]: !s32i{{.*}}, %[[Y_ARG:.*]]: !u32i{{.*}}, %[[F_ARG:.*]]: !cir.float{{.*}}){{.*}}, %[[CTS_ARG:.*]]: !rec_ConvertsToScalar{{.*}}) {
  // CHECK-NEXT: %[[X_ALLOC:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
  // CHECK-NEXT: %[[Y_ALLOC:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["y", init]
  // CHECK-NEXT: %[[F_ALLOC:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["f", init]
  // CHECK-NEXT: %[[CTS_ALLOC:.*]] = cir.alloca !rec_ConvertsToScalar, !cir.ptr<!rec_ConvertsToScalar>, ["cts", init]
  //
  // CHECK-NEXT: cir.store %[[X_ARG]], %[[X_ALLOC]] : !s32i, !cir.ptr<!s32i>
  // CHECK-NEXT: cir.store %[[Y_ARG]], %[[Y_ALLOC]] : !u32i, !cir.ptr<!u32i>
  // CHECK-NEXT: cir.store %[[F_ARG]], %[[F_ALLOC]] : !cir.float, !cir.ptr<!cir.float>
  // CHECK-NEXT: cir.store %[[CTS_ARG]], %[[CTS_ALLOC]] : !rec_ConvertsToScalar, !cir.ptr<!rec_ConvertsToScalar>

  // CHECK-NEXT: %[[Y_LOAD:.*]] = cir.load {{.*}}%[[Y_ALLOC]] : !cir.ptr<!u32i>, !u32i
  // CHECK-NEXT: %[[Y_TO_FLOAT:.*]] = cir.cast int_to_float %[[Y_LOAD]] : !u32i -> !cir.float
  // CHECK-NEXT: %[[F_LOAD:.*]] = cir.load {{.*}}%[[F_ALLOC]] : !cir.ptr<!cir.float>, !cir.float
  // CHECK-NEXT: %[[MUL:.*]] = cir.binop(mul, %[[Y_TO_FLOAT]], %[[F_LOAD]]) : !cir.float
  // CHECK-NEXT: %[[RHS_CAST:.*]] = cir.cast float_to_int %[[MUL]] : !cir.float -> !s32i
  // CHECK-NEXT: acc.atomic.write %[[X_ALLOC]] = %[[RHS_CAST]] : !cir.ptr<!s32i>, !s32i
#pragma acc atomic write
  x = y * f;

  // CHECK-NEXT: %[[F_LOAD:.*]] = cir.load {{.*}}%[[F_ALLOC]] : !cir.ptr<!cir.float>, !cir.float
  // CHECK-NEXT: %[[CALL:.*]] = cir.call @do_thing(%[[F_LOAD]]) : (!cir.float) -> !cir.double
  // CHECK-NEXT: %[[CALL_CAST:.*]] = cir.cast float_to_int %[[CALL]] : !cir.double -> !u32i
  // CHECK-NEXT: acc.atomic.write %[[Y_ALLOC]] = %[[CALL_CAST]] : !cir.ptr<!u32i>, !u32i
#pragma acc atomic write
  y = do_thing(f);

  // CHECK-NEXT: %[[X_LOAD:.*]] = cir.load {{.*}}%[[X_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[X_CAST:.*]] = cir.cast int_to_float %[[X_LOAD]] : !s32i -> !cir.float
  // CHECK-NEXT: %[[THING_CALL:.*]] = cir.call @do_thing(%[[X_CAST]]) : (!cir.float) -> !cir.double
  // CHECK-NEXT: %[[THING_CAST:.*]] = cir.cast floating %[[THING_CALL]] : !cir.double -> !cir.float
  // CHECK-NEXT: %[[X_LOAD:.*]] = cir.load {{.*}}%[[X_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[Y_LOAD:.*]] = cir.load {{.*}}%[[Y_ALLOC]] : !cir.ptr<!u32i>, !u32i
  // CHECK-NEXT: %[[F_LOAD:.*]] = cir.load {{.*}}%[[F_ALLOC]] : !cir.ptr<!cir.float>, !cir.float
  // CHECK-NEXT: %[[COND_CALL:.*]] = cir.call @condition(%[[X_LOAD]], %[[Y_LOAD]], %[[F_LOAD]]) : (!s32i, !u32i, !cir.float) -> !cir.bool
  // CHECK-NEXT: %[[COND_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_CALL]] : !cir.bool to i1
  // CHECK-NEXT: acc.atomic.write if(%[[COND_CAST]]) %[[F_ALLOC]] = %[[THING_CAST]] : !cir.ptr<!cir.float>, !cir.float
#pragma acc atomic write if (condition(x, y, f))
  f = do_thing(x);

  // CHECK-NEXT: %[[CTS_CONV_CALL:.*]] = cir.call @{{.*}}(%[[CTS_ALLOC]]) : (!cir.ptr<!rec_ConvertsToScalar>) -> !cir.float
  // CHECK-NEXT: acc.atomic.write %[[F_ALLOC]] = %[[CTS_CONV_CALL]] : !cir.ptr<!cir.float>, !cir.float
#pragma acc atomic write
  f = cts;
}
