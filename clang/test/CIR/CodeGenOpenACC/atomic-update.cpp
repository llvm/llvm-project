// RUN: %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s

struct HasOps {
  operator float();
  int thing();
};

void use(int x, unsigned int y, float f, HasOps ops) {
  // CHECK: cir.func{{.*}}(%[[X_ARG:.*]]: !s32i{{.*}}, %[[Y_ARG:.*]]: !u32i{{.*}}, %[[F_ARG:.*]]: !cir.float{{.*}}){{.*}}, %[[OPS_ARG:.*]]: !rec_HasOps{{.*}}) {
  // CHECK-NEXT: %[[X_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
  // CHECK-NEXT: %[[Y_ALLOCA:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["y", init]
  // CHECK-NEXT: %[[F_ALLOCA:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["f", init]
  // CHECK-NEXT: %[[OPS_ALLOCA:.*]] = cir.alloca !rec_HasOps, !cir.ptr<!rec_HasOps>, ["ops", init]
  // CHECK-NEXT: cir.store %[[X_ARG]], %[[X_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // CHECK-NEXT: cir.store %[[Y_ARG]], %[[Y_ALLOCA]] : !u32i, !cir.ptr<!u32i>
  // CHECK-NEXT: cir.store %[[F_ARG]], %[[F_ALLOCA]] : !cir.float, !cir.ptr<!cir.float>
  // CHECK-NEXT: cir.store %[[OPS_ARG]], %[[OPS_ALLOCA]] : !rec_HasOps, !cir.ptr<!rec_HasOps>

  // CHECK-NEXT: acc.atomic.update %[[X_ALLOCA]] : !cir.ptr<!s32i> {
  // CHECK-NEXT: ^bb0(%[[RECIPE_ARG:.*]]: !s32i{{.*}}):
  // CHECK-NEXT: %[[TEMP_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[RECIPE_ARG]], %[[TEMP_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load{{.*}} %[[TEMP_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[TEMP_LOAD]]) nsw : !s32i, !s32i
  // CHECK-NEXT: cir.store {{.*}}%[[INC]], %[[TEMP_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load{{.*}} %[[TEMP_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.yield %[[TEMP_LOAD]] : !s32i
  // CHECK-NEXT: }
#pragma acc atomic update
  ++x;

  // CHECK-NEXT: acc.atomic.update %[[Y_ALLOCA]] : !cir.ptr<!u32i> {
  // CHECK-NEXT: ^bb0(%[[RECIPE_ARG:.*]]: !u32i{{.*}}):
  // CHECK-NEXT: %[[TEMP_ALLOCA:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[RECIPE_ARG]], %[[TEMP_ALLOCA]] : !u32i, !cir.ptr<!u32i>
  //
  // CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load{{.*}} %[[TEMP_ALLOCA]] : !cir.ptr<!u32i>, !u32i
  // CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[TEMP_LOAD]]) : !u32i, !u32i
  // CHECK-NEXT: cir.store {{.*}}%[[INC]], %[[TEMP_ALLOCA]] : !u32i, !cir.ptr<!u32i>
  // 
  // CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load{{.*}} %[[TEMP_ALLOCA]] : !cir.ptr<!u32i>, !u32i
  // CHECK-NEXT: acc.yield %[[TEMP_LOAD]] : !u32i
  // CHECK-NEXT: }
#pragma acc atomic update
  y++;

  // CHECK-NEXT: acc.atomic.update %[[F_ALLOCA]] : !cir.ptr<!cir.float> {
  // CHECK-NEXT: ^bb0(%[[RECIPE_ARG:.*]]: !cir.float{{.*}}):
  // CHECK-NEXT: %[[TEMP_ALLOCA:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[RECIPE_ARG]], %[[TEMP_ALLOCA]] : !cir.float, !cir.ptr<!cir.float>
  //
  // CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load{{.*}} %[[TEMP_ALLOCA]] : !cir.ptr<!cir.float>, !cir.float
  // CHECK-NEXT: %[[INC:.*]] = cir.unary(dec, %[[TEMP_LOAD]]) : !cir.float, !cir.float
  // CHECK-NEXT: cir.store {{.*}}%[[INC]], %[[TEMP_ALLOCA]] : !cir.float, !cir.ptr<!cir.float>
  // 
  // CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load{{.*}} %[[TEMP_ALLOCA]] : !cir.ptr<!cir.float>, !cir.float
  // CHECK-NEXT: acc.yield %[[TEMP_LOAD]] : !cir.float
  // CHECK-NEXT: }
#pragma acc atomic update
  f--;

  // CHECK-NEXT: acc.atomic.update %[[X_ALLOCA]] : !cir.ptr<!s32i> {
  // CHECK-NEXT: ^bb0(%[[RECIPE_ARG:.*]]: !s32i{{.*}}):
  // CHECK-NEXT: %[[TEMP_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[RECIPE_ARG]], %[[TEMP_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[F_LOAD:.*]] = cir.load{{.*}} %[[F_ALLOCA]] : !cir.ptr<!cir.float>, !cir.float
  // CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load{{.*}} %[[TEMP_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[INT_TO_F:.*]] = cir.cast int_to_float %[[TEMP_LOAD]] : !s32i -> !cir.float
  // CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[INT_TO_F]], %[[F_LOAD]]) : !cir.float
  // CHECK-NEXT: %[[F_TO_INT:.*]] = cir.cast float_to_int %[[ADD]] : !cir.float -> !s32i
  // CHECK-NEXT: cir.store{{.*}} %[[F_TO_INT]], %[[TEMP_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load{{.*}} %[[TEMP_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.yield %[[TEMP_LOAD]] : !s32i
  // CHECK-NEXT: }
#pragma acc atomic update
  x += f;

  // CHECK-NEXT: acc.atomic.update %[[F_ALLOCA]] : !cir.ptr<!cir.float> {
  // CHECK-NEXT: ^bb0(%[[RECIPE_ARG:.*]]: !cir.float{{.*}}):
  // CHECK-NEXT: %[[TEMP_ALLOCA:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[RECIPE_ARG]], %[[TEMP_ALLOCA]] : !cir.float, !cir.ptr<!cir.float>
  //
  // CHECK-NEXT: %[[Y_LOAD:.*]] = cir.load{{.*}} %[[Y_ALLOCA]] : !cir.ptr<!u32i>, !u32i
  // CHECK-NEXT: %[[INT_TO_F:.*]] = cir.cast int_to_float %[[Y_LOAD]] : !u32i -> !cir.float
  // CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load{{.*}} %[[TEMP_ALLOCA]] : !cir.ptr<!cir.float>, !cir.float
  // CHECK-NEXT: %[[DIV:.*]] = cir.binop(div, %[[TEMP_LOAD]], %[[INT_TO_F]]) : !cir.float
  // CHECK-NEXT: cir.store{{.*}} %[[DIV]], %[[TEMP_ALLOCA]] : !cir.float, !cir.ptr<!cir.float>
  //
  // CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load{{.*}} %[[TEMP_ALLOCA]] : !cir.ptr<!cir.float>, !cir.float
  // CHECK-NEXT: acc.yield %[[TEMP_LOAD]] : !cir.float
  // CHECK-NEXT: }
#pragma acc atomic update
  f /= y;

  // CHECK-NEXT: acc.atomic.update %[[Y_ALLOCA]] : !cir.ptr<!u32i> {
  // CHECK-NEXT: ^bb0(%[[RECIPE_ARG:.*]]: !u32i{{.*}}):
  // CHECK-NEXT: %[[TEMP_ALLOCA:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[RECIPE_ARG]], %[[TEMP_ALLOCA]] : !u32i, !cir.ptr<!u32i>
  //
  // CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load{{.*}} %[[TEMP_ALLOCA]] : !cir.ptr<!u32i>, !u32i
  // CHECK-NEXT: %[[CALL:.*]] = cir.call {{.*}}(%[[OPS_ALLOCA]]) : (!cir.ptr<!rec_HasOps>) -> !s32i
  // CHECK-NEXT: %[[CALL_CAST:.*]] = cir.cast integral %[[CALL]] : !s32i -> !u32i
  // CHECK-NEXT: %[[MUL:.*]] = cir.binop(mul, %[[TEMP_LOAD]], %[[CALL_CAST]]) : !u32i
  // CHECK-NEXT: cir.store{{.*}} %[[MUL]], %[[TEMP_ALLOCA]] : !u32i, !cir.ptr<!u32i>
  //
  // CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load{{.*}} %[[TEMP_ALLOCA]] : !cir.ptr<!u32i>, !u32i
  // CHECK-NEXT: acc.yield %[[TEMP_LOAD]] : !u32i
  // CHECK-NEXT: }

#pragma acc atomic update
  y = y * ops.thing();

  // CHECK-NEXT: acc.atomic.update %[[X_ALLOCA]] : !cir.ptr<!s32i> {
  // CHECK-NEXT: ^bb0(%[[RECIPE_ARG:.*]]: !s32i{{.*}}):
  // CHECK-NEXT: %[[TEMP_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[RECIPE_ARG]], %[[TEMP_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[CALL:.*]] = cir.call {{.*}}(%[[OPS_ALLOCA]]) : (!cir.ptr<!rec_HasOps>) -> !s32i
  // CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load{{.*}} %[[TEMP_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[OR:.*]] = cir.binop(or, %[[CALL]], %[[INT_TO_F]]) : !s32i
  // CHECK-NEXT: cir.store{{.*}} %[[OR]], %[[TEMP_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load{{.*}} %[[TEMP_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.yield %[[TEMP_LOAD]] : !s32i
  // CHECK-NEXT: }
#pragma acc atomic update
  x = ops.thing() | x;

  // CHECK-NEXT: %[[X_LOAD:.*]] = cir.load{{.*}} %[[X_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[BOOL_CAST:.*]] = cir.cast int_to_bool %[[X_LOAD]] : !s32i -> !cir.bool
  // CHECK-NEXT: %[[X_CAST:.*]] = builtin.unrealized_conversion_cast %[[BOOL_CAST]] : !cir.bool to i1
  // CHECK-NEXT: acc.atomic.update if(%[[X_CAST]]) %[[F_ALLOCA]] : !cir.ptr<!cir.float> {
  // CHECK-NEXT: ^bb0(%[[RECIPE_ARG:.*]]: !cir.float{{.*}}):
  // CHECK-NEXT: %[[TEMP_ALLOCA:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[RECIPE_ARG]], %[[TEMP_ALLOCA]] : !cir.float, !cir.ptr<!cir.float>
  //
  // CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load{{.*}} %[[TEMP_ALLOCA]] : !cir.ptr<!cir.float>, !cir.float
  // CHECK-NEXT: %[[CALL:.*]] = cir.call {{.*}}(%[[OPS_ALLOCA]]) : (!cir.ptr<!rec_HasOps>) -> !cir.float
  // CHECK-NEXT: %[[SUB:.*]] = cir.binop(sub, %[[TEMP_LOAD]], %[[CALL]]) : !cir.float
  // CHECK-NEXT: cir.store{{.*}} %[[SUB]], %[[TEMP_ALLOCA]] : !cir.float, !cir.ptr<!cir.float>
  //
  // CHECK-NEXT: %[[TEMP_LOAD:.*]] = cir.load{{.*}} %[[TEMP_ALLOCA]] : !cir.ptr<!cir.float>, !cir.float
  // CHECK-NEXT: acc.yield %[[TEMP_LOAD]] : !cir.float
  // CHECK-NEXT: }
#pragma acc atomic update if (x)
  f = f - ops;
}
