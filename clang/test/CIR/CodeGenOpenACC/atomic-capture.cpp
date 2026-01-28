// RUN: %clang_cc1 -fopenacc -triple x86_64-linux-gnu -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir -triple x86_64-linux-pc %s -o - | FileCheck %s

struct HasOps {
  operator float();
  int thing();
  int operator++();
  int operator++(int);
};

void use(int x, int v, float f, HasOps ops) {
  // CHECK: cir.func{{.*}}(%[[X_ARG:.*]]: !s32i{{.*}}, %[[V_ARG:.*]]: !s32i{{.*}}, %[[F_ARG:.*]]: !cir.float{{.*}}){{.*}}, %[[OPS_ARG:.*]]: !rec_HasOps{{.*}}) {
  // CHECK-NEXT: %[[X_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
  // CHECK-NEXT: %[[V_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["v", init]
  // CHECK-NEXT: %[[F_ALLOCA:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["f", init]
  // CHECK-NEXT: %[[OPS_ALLOCA:.*]] = cir.alloca !rec_HasOps, !cir.ptr<!rec_HasOps>, ["ops", init]
  // CHECK-NEXT: cir.store %[[X_ARG]], %[[X_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // CHECK-NEXT: cir.store %[[V_ARG]], %[[V_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // CHECK-NEXT: cir.store %[[F_ARG]], %[[F_ALLOCA]] : !cir.float, !cir.ptr<!cir.float>
  // CHECK-NEXT: cir.store %[[OPS_ARG]], %[[OPS_ALLOCA]] : !rec_HasOps, !cir.ptr<!rec_HasOps>

  // CHECK-NEXT: %[[X_LOAD:.*]] = cir.load{{.*}} %[[X_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[V_LOAD:.*]] = cir.load{{.*}} %[[V_ALLOCA]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CMP:.*]] = cir.cmp(ne, %[[X_LOAD]], %[[V_LOAD]]) : !s32i, !cir.bool
  // CHECK-NEXT: %[[IF_COND_CAST:.*]] = builtin.unrealized_conversion_cast %[[CMP:.*]] : !cir.bool to i1
  // CHECK-NEXT: acc.atomic.capture if(%[[IF_COND_CAST]]) {
  // CHECK-NEXT: acc.atomic.read %[[V_ALLOCA]] = %[[X_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.atomic.update %[[X_ALLOCA]] : !cir.ptr<!s32i> {
  // CHECK-NEXT: ^bb0(%[[X_VAR:.*]]: !s32i{{.*}}):
  // CHECK-NEXT: %[[X_VAR_ALLOC:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[X_VAR]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[X_VAR_LOAD]]) nsw : !s32i, !s32i
  // CHECK-NEXT: cir.store{{.*}} %[[INC]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.yield %[[X_VAR_LOAD]] : !s32i
  // CHECK-NEXT: }
  // CHECK-NEXT: }
#pragma acc atomic capture if (x != v)
  v = x++;

  // CHECK-NEXT: acc.atomic.capture {
  // CHECK-NEXT: acc.atomic.update %[[X_ALLOCA]] : !cir.ptr<!s32i> {
  // CHECK-NEXT: ^bb0(%[[X_VAR:.*]]: !s32i{{.*}}):
  // CHECK-NEXT: %[[X_VAR_ALLOC:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[X_VAR]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[X_VAR_LOAD]]) nsw : !s32i, !s32i
  // CHECK-NEXT: cir.store{{.*}} %[[INC]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.yield %[[X_VAR_LOAD]] : !s32i
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.atomic.read %[[V_ALLOCA]] = %[[X_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: }
#pragma acc atomic capture
  v = ++x;

  // CHECK-NEXT: acc.atomic.capture {
  // CHECK-NEXT: acc.atomic.read %[[V_ALLOCA]] = %[[X_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.atomic.update %[[X_ALLOCA]] : !cir.ptr<!s32i> {
  // CHECK-NEXT: ^bb0(%[[X_VAR:.*]]: !s32i{{.*}}):
  // CHECK-NEXT: %[[X_VAR_ALLOC:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[X_VAR]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[DEC:.*]] = cir.unary(dec, %[[X_VAR_LOAD]]) nsw : !s32i, !s32i
  // CHECK-NEXT: cir.store{{.*}} %[[DEC]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.yield %[[X_VAR_LOAD]] : !s32i
  // CHECK-NEXT: }
  // CHECK-NEXT: }
#pragma acc atomic capture
  v = x--;

  // CHECK-NEXT: acc.atomic.capture {
  // CHECK-NEXT: acc.atomic.update %[[X_ALLOCA]] : !cir.ptr<!s32i> {
  // CHECK-NEXT: ^bb0(%[[X_VAR:.*]]: !s32i{{.*}}):
  // CHECK-NEXT: %[[X_VAR_ALLOC:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[X_VAR]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  // 
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[DEC:.*]] = cir.unary(dec, %[[X_VAR_LOAD]]) nsw : !s32i, !s32i
  // CHECK-NEXT: cir.store{{.*}} %[[DEC]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.yield %[[X_VAR_LOAD]] : !s32i
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.atomic.read %[[V_ALLOCA]] = %[[X_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: }
#pragma acc atomic capture
  v = --x;

  // CHECK-NEXT: acc.atomic.capture {
  // CHECK-NEXT: acc.atomic.update %[[X_ALLOCA]] : !cir.ptr<!s32i> {
  // CHECK-NEXT: ^bb0(%[[X_VAR:.*]]: !s32i{{.*}}):
  // CHECK-NEXT: %[[X_VAR_ALLOC:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[X_VAR]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[F_LOAD:.*]] = cir.load{{.*}} %[[F_ALLOCA]] : !cir.ptr<!cir.float>, !cir.float
  // CHECK-NEXT: %[[ONE_INT:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = cir.cast int_to_float %[[ONE_INT]] : !s32i -> !cir.float
  // CHECK-NEXT: %[[MUL:.*]] = cir.binop(mul, %[[F_LOAD]], %[[ONE_CAST]]) : !cir.float
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[X_CAST:.*]] = cir.cast int_to_float %[[X_VAR_LOAD]] : !s32i -> !cir.float
  // CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[X_CAST]], %[[MUL]]) : !cir.float
  // CHECK-NEXT: %[[INT_CAST:.*]] = cir.cast float_to_int %[[ADD]] : !cir.float -> !s32i
  // CHECK-NEXT: cir.store{{.*}} %[[INT_CAST]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.yield %[[X_VAR_LOAD]] : !s32i
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.atomic.read %[[V_ALLOCA]] = %[[X_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: }
#pragma acc atomic capture
  v = x += f * 1;

  // CHECK-NEXT: acc.atomic.capture {
  // CHECK-NEXT: acc.atomic.update %[[X_ALLOCA]] : !cir.ptr<!s32i> {
  // CHECK-NEXT: ^bb0(%[[X_VAR:.*]]: !s32i{{.*}}):
  // CHECK-NEXT: %[[X_VAR_ALLOC:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[X_VAR]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[X_CAST:.*]] = cir.cast int_to_float %[[X_VAR_LOAD]] : !s32i -> !cir.float
  // CHECK-NEXT: %[[F_LOAD:.*]] = cir.load{{.*}} %[[F_ALLOCA]] : !cir.ptr<!cir.float>, !cir.float
  // CHECK-NEXT: %[[ONE_INT:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = cir.cast int_to_float %[[ONE_INT]] : !s32i -> !cir.float
  // CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[F_LOAD]], %[[ONE_CAST]]) : !cir.float
  // CHECK-NEXT: %[[MUL:.*]] = cir.binop(mul, %[[X_CAST]], %[[ADD]]) : !cir.float
  // CHECK-NEXT: %[[INT_CAST:.*]] = cir.cast float_to_int %[[MUL]] : !cir.float -> !s32i
  // CHECK-NEXT: cir.store{{.*}} %[[INT_CAST]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.yield %[[X_VAR_LOAD]] : !s32i
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.atomic.read %[[V_ALLOCA]] = %[[X_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: }
#pragma acc atomic capture
  v = x = x * (f + 1);

  // CHECK-NEXT: acc.atomic.capture {
  // CHECK-NEXT: acc.atomic.update %[[X_ALLOCA]] : !cir.ptr<!s32i> {
  // CHECK-NEXT: ^bb0(%[[X_VAR:.*]]: !s32i{{.*}}):
  // CHECK-NEXT: %[[X_VAR_ALLOC:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[X_VAR]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[F_LOAD:.*]] = cir.load{{.*}} %[[F_ALLOCA]] : !cir.ptr<!cir.float>, !cir.float
  // CHECK-NEXT: %[[ONE_INT:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = cir.cast int_to_float %[[ONE_INT]] : !s32i -> !cir.float
  // CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[F_LOAD]], %[[ONE_CAST]]) : !cir.float
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[X_CAST:.*]] = cir.cast int_to_float %[[X_VAR_LOAD]] : !s32i -> !cir.float
  // CHECK-NEXT: %[[MUL:.*]] = cir.binop(mul, %[[ADD]], %[[X_CAST]]) : !cir.float
  // CHECK-NEXT: %[[INT_CAST:.*]] = cir.cast float_to_int %[[MUL]] : !cir.float -> !s32i
  // CHECK-NEXT: cir.store{{.*}} %[[INT_CAST]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.yield %[[X_VAR_LOAD]] : !s32i
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.atomic.read %[[V_ALLOCA]] = %[[X_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: }
#pragma acc atomic capture
  v = x = (f + 1) * x;

  // CHECK-NEXT: acc.atomic.capture {
  // CHECK-NEXT: acc.atomic.read %[[V_ALLOCA]] = %[[X_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.atomic.update %[[X_ALLOCA]] : !cir.ptr<!s32i> {
  // CHECK-NEXT: ^bb0(%[[X_VAR:.*]]: !s32i{{.*}}):
  // CHECK-NEXT: %[[X_VAR_ALLOC:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[X_VAR]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[F_LOAD:.*]] = cir.load{{.*}} %[[F_ALLOCA]] : !cir.ptr<!cir.float>, !cir.float
  // CHECK-NEXT: %[[ONE_INT:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = cir.cast int_to_float %[[ONE_INT]] : !s32i -> !cir.float
  // CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[F_LOAD]], %[[ONE_CAST]]) : !cir.float
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[X_CAST:.*]] = cir.cast int_to_float %[[X_VAR_LOAD]] : !s32i -> !cir.float
  // CHECK-NEXT: %[[MUL:.*]] = cir.binop(mul, %[[X_CAST]], %[[ADD]]) : !cir.float
  // CHECK-NEXT: %[[INT_CAST:.*]] = cir.cast float_to_int %[[MUL]] : !cir.float -> !s32i
  // CHECK-NEXT: cir.store{{.*}} %[[INT_CAST]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.yield %[[X_VAR_LOAD]] : !s32i
  // CHECK-NEXT: }
  // CHECK-NEXT: }
#pragma acc atomic capture
  {
    v = x; x *= f + 1;
  }

  // CHECK-NEXT: acc.atomic.capture {
  // CHECK-NEXT: acc.atomic.update %[[X_ALLOCA]] : !cir.ptr<!s32i> {
  // CHECK-NEXT: ^bb0(%[[X_VAR:.*]]: !s32i{{.*}}):
  // CHECK-NEXT: %[[X_VAR_ALLOC:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[X_VAR]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[F_LOAD:.*]] = cir.load{{.*}} %[[F_ALLOCA]] : !cir.ptr<!cir.float>, !cir.float
  // CHECK-NEXT: %[[ONE_INT:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = cir.cast int_to_float %[[ONE_INT]] : !s32i -> !cir.float
  // CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[F_LOAD]], %[[ONE_CAST]]) : !cir.float
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[X_CAST:.*]] = cir.cast int_to_float %[[X_VAR_LOAD]] : !s32i -> !cir.float
  // CHECK-NEXT: %[[SUB:.*]] = cir.binop(sub, %[[X_CAST]], %[[ADD]]) : !cir.float
  // CHECK-NEXT: %[[INT_CAST:.*]] = cir.cast float_to_int %[[SUB]] : !cir.float -> !s32i
  // CHECK-NEXT: cir.store{{.*}} %[[INT_CAST]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.yield %[[X_VAR_LOAD]] : !s32i
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.atomic.read %[[V_ALLOCA]] = %[[X_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: }
#pragma acc atomic capture
  {
    x -= f + 1;
    v = x;
  }

  // CHECK-NEXT: acc.atomic.capture {
  // CHECK-NEXT: acc.atomic.read %[[V_ALLOCA]] = %[[X_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.atomic.update %[[X_ALLOCA]] : !cir.ptr<!s32i> {
  // CHECK-NEXT: ^bb0(%[[X_VAR:.*]]: !s32i{{.*}}):
  // CHECK-NEXT: %[[X_VAR_ALLOC:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[X_VAR]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[X_CAST:.*]] = cir.cast int_to_float %[[X_VAR_LOAD]] : !s32i -> !cir.float
  // CHECK-NEXT: %[[F_LOAD:.*]] = cir.load{{.*}} %[[F_ALLOCA]] : !cir.ptr<!cir.float>, !cir.float
  // CHECK-NEXT: %[[ONE_INT:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = cir.cast int_to_float %[[ONE_INT]] : !s32i -> !cir.float
  // CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[F_LOAD]], %[[ONE_CAST]]) : !cir.float
  // CHECK-NEXT: %[[DIV:.*]] = cir.binop(div, %[[X_CAST]], %[[ADD]]) : !cir.float
  // CHECK-NEXT: %[[INT_CAST:.*]] = cir.cast float_to_int %[[DIV]] : !cir.float -> !s32i
  // CHECK-NEXT: cir.store{{.*}} %[[INT_CAST]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.yield %[[X_VAR_LOAD]] : !s32i
  // CHECK-NEXT: }
  // CHECK-NEXT: }
#pragma acc atomic capture
  {
    v = x;
    x = x / (f + 1);
  }

  // CHECK-NEXT: acc.atomic.capture {
  // CHECK-NEXT: acc.atomic.read %[[V_ALLOCA]] = %[[X_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.atomic.update %[[X_ALLOCA]] : !cir.ptr<!s32i> {
  // CHECK-NEXT: ^bb0(%[[X_VAR:.*]]: !s32i{{.*}}):
  // CHECK-NEXT: %[[X_VAR_ALLOC:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[X_VAR]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[F_LOAD:.*]] = cir.load{{.*}} %[[F_ALLOCA]] : !cir.ptr<!cir.float>, !cir.float
  // CHECK-NEXT: %[[OPS_CONV:.*]] = cir.call @{{.*}}(%[[OPS_ALLOCA]]) : (!cir.ptr<!rec_HasOps>) -> !cir.float
  // CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[F_LOAD]], %[[OPS_CONV]]) : !cir.float
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[X_CAST:.*]] = cir.cast int_to_float %[[X_VAR_LOAD]] : !s32i -> !cir.float
  // CHECK-NEXT: %[[DIV:.*]] = cir.binop(div, %[[ADD]], %[[X_CAST]]) : !cir.float
  // CHECK-NEXT: %[[INT_CAST:.*]] = cir.cast float_to_int %[[DIV]] : !cir.float -> !s32i
  // CHECK-NEXT: cir.store{{.*}} %[[INT_CAST]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.yield %[[X_VAR_LOAD]] : !s32i
  // CHECK-NEXT: }
  // CHECK-NEXT: }
#pragma acc atomic capture
  {
    v = x;
    x = (f + ops) / x;
  }

  // CHECK-NEXT: acc.atomic.capture {
  // CHECK-NEXT: acc.atomic.update %[[X_ALLOCA]] : !cir.ptr<!s32i> {
  // CHECK-NEXT: ^bb0(%[[X_VAR:.*]]: !s32i{{.*}}):
  // CHECK-NEXT: %[[X_VAR_ALLOC:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[X_VAR]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[X_CAST:.*]] = cir.cast int_to_float %[[X_VAR_LOAD]] : !s32i -> !cir.float
  // CHECK-NEXT: %[[F_LOAD:.*]] = cir.load{{.*}} %[[F_ALLOCA]] : !cir.ptr<!cir.float>, !cir.float
  // CHECK-NEXT: %[[ONE_INT:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = cir.cast int_to_float %[[ONE_INT]] : !s32i -> !cir.float
  // CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[F_LOAD]], %[[ONE_CAST]]) : !cir.float
  // CHECK-NEXT: %[[DIV:.*]] = cir.binop(div, %[[X_CAST]], %[[ADD]]) : !cir.float
  // CHECK-NEXT: %[[INT_CAST:.*]] = cir.cast float_to_int %[[DIV]] : !cir.float -> !s32i
  // CHECK-NEXT: cir.store{{.*}} %[[INT_CAST]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.yield %[[X_VAR_LOAD]] : !s32i
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.atomic.read %[[V_ALLOCA]] = %[[X_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: }
#pragma acc atomic capture
  {
    x = x / (f + 1);
    v = x;
  }

  // CHECK-NEXT: acc.atomic.capture {
  // CHECK-NEXT: acc.atomic.update %[[X_ALLOCA]] : !cir.ptr<!s32i> {
  // CHECK-NEXT: ^bb0(%[[X_VAR:.*]]: !s32i{{.*}}):
  // CHECK-NEXT: %[[X_VAR_ALLOC:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[X_VAR]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[F_LOAD:.*]] = cir.load{{.*}} %[[F_ALLOCA]] : !cir.ptr<!cir.float>, !cir.float
  // CHECK-NEXT: %[[OPS_CONV:.*]] = cir.call @{{.*}}(%[[OPS_ALLOCA]]) : (!cir.ptr<!rec_HasOps>) -> !cir.float
  // CHECK-NEXT: %[[ADD:.*]] = cir.binop(add, %[[F_LOAD]], %[[OPS_CONV]]) : !cir.float
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[X_CAST:.*]] = cir.cast int_to_float %[[X_VAR_LOAD]] : !s32i -> !cir.float
  // CHECK-NEXT: %[[DIV:.*]] = cir.binop(div, %[[ADD]], %[[X_CAST]]) : !cir.float
  // CHECK-NEXT: %[[INT_CAST:.*]] = cir.cast float_to_int %[[DIV]] : !cir.float -> !s32i
  // CHECK-NEXT: cir.store{{.*}} %[[INT_CAST]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.yield %[[X_VAR_LOAD]] : !s32i
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.atomic.read %[[V_ALLOCA]] = %[[X_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: }
#pragma acc atomic capture
  {
    x = (f + ops) / x;
    v = x;
  }

  // CHECK-NEXT: %[[OPS_CONV:.*]] = cir.call @{{.*}}(%[[OPS_ALLOCA]]) : (!cir.ptr<!rec_HasOps>) -> !cir.float
  // CHECK-NEXT: %[[OPS_CONV_TO_INT:.*]] = cir.cast float_to_int %[[OPS_CONV]] : !cir.float -> !s32i
  //
  // CHECK-NEXT: acc.atomic.capture {
  // CHECK-NEXT: acc.atomic.read %[[V_ALLOCA]] = %[[X_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.atomic.write %[[X_ALLOCA]] = %[[OPS_CONV_TO_INT]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: }
#pragma acc atomic capture
  {
    v = x;
    x = ops;
  }

  // CHECK-NEXT: acc.atomic.capture {
  // CHECK-NEXT: acc.atomic.read %[[V_ALLOCA]] = %[[X_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.atomic.update %[[X_ALLOCA]] : !cir.ptr<!s32i> {
  // CHECK-NEXT: ^bb0(%[[X_VAR:.*]]: !s32i{{.*}}):
  // CHECK-NEXT: %[[X_VAR_ALLOC:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[X_VAR]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[X_VAR_LOAD]]) nsw : !s32i, !s32i
  // CHECK-NEXT: cir.store{{.*}} %[[INC]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.yield %[[X_VAR_LOAD]] : !s32i
  // CHECK-NEXT: }
  // CHECK-NEXT: }
#pragma acc atomic capture
  {
    v = x;
    x++;
  }

  // CHECK-NEXT: acc.atomic.capture {
  // CHECK-NEXT: acc.atomic.read %[[V_ALLOCA]] = %[[X_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.atomic.update %[[X_ALLOCA]] : !cir.ptr<!s32i> {
  // CHECK-NEXT: ^bb0(%[[X_VAR:.*]]: !s32i{{.*}}):
  // CHECK-NEXT: %[[X_VAR_ALLOC:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[X_VAR]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[X_VAR_LOAD]]) nsw : !s32i, !s32i
  // CHECK-NEXT: cir.store{{.*}} %[[INC]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.yield %[[X_VAR_LOAD]] : !s32i
  // CHECK-NEXT: }
  // CHECK-NEXT: }
#pragma acc atomic capture
  {
    v = x;
    ++x;
  }

  // CHECK-NEXT: acc.atomic.capture {
  // CHECK-NEXT: acc.atomic.update %[[X_ALLOCA]] : !cir.ptr<!s32i> {
  // CHECK-NEXT: ^bb0(%[[X_VAR:.*]]: !s32i{{.*}}):
  // CHECK-NEXT: %[[X_VAR_ALLOC:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[X_VAR]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[X_VAR_LOAD]]) nsw : !s32i, !s32i
  // CHECK-NEXT: cir.store{{.*}} %[[INC]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.yield %[[X_VAR_LOAD]] : !s32i
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.atomic.read %[[V_ALLOCA]] = %[[X_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: }
#pragma acc atomic capture
  {
    x++;
    v = x;
  }

  // CHECK-NEXT: acc.atomic.capture {
  // CHECK-NEXT: acc.atomic.update %[[X_ALLOCA]] : !cir.ptr<!s32i> {
  // CHECK-NEXT: ^bb0(%[[X_VAR:.*]]: !s32i{{.*}}):
  // CHECK-NEXT: %[[X_VAR_ALLOC:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[X_VAR]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[INC:.*]] = cir.unary(inc, %[[X_VAR_LOAD]]) nsw : !s32i, !s32i
  // CHECK-NEXT: cir.store{{.*}} %[[INC]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.yield %[[X_VAR_LOAD]] : !s32i
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.atomic.read %[[V_ALLOCA]] = %[[X_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: }
#pragma acc atomic capture
  {
    ++x;
    v = x;
  }

  // CHECK-NEXT: acc.atomic.capture {
  // CHECK-NEXT: acc.atomic.read %[[V_ALLOCA]] = %[[X_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.atomic.update %[[X_ALLOCA]] : !cir.ptr<!s32i> {
  // CHECK-NEXT: ^bb0(%[[X_VAR:.*]]: !s32i{{.*}}):
  // CHECK-NEXT: %[[X_VAR_ALLOC:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[X_VAR]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[DEC:.*]] = cir.unary(dec, %[[X_VAR_LOAD]]) nsw : !s32i, !s32i
  // CHECK-NEXT: cir.store{{.*}} %[[DEC]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.yield %[[X_VAR_LOAD]] : !s32i
  // CHECK-NEXT: }
  // CHECK-NEXT: }
#pragma acc atomic capture
  {
    v = x;
    x--;
  }

  // CHECK-NEXT: acc.atomic.capture {
  // CHECK-NEXT: acc.atomic.read %[[V_ALLOCA]] = %[[X_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.atomic.update %[[X_ALLOCA]] : !cir.ptr<!s32i> {
  // CHECK-NEXT: ^bb0(%[[X_VAR:.*]]: !s32i{{.*}}):
  // CHECK-NEXT: %[[X_VAR_ALLOC:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[X_VAR]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[DEC:.*]] = cir.unary(dec, %[[X_VAR_LOAD]]) nsw : !s32i, !s32i
  // CHECK-NEXT: cir.store{{.*}} %[[DEC]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.yield %[[X_VAR_LOAD]] : !s32i
  // CHECK-NEXT: }
  // CHECK-NEXT: }
#pragma acc atomic capture
  {
    v = x;
    --x;
  }

  // CHECK-NEXT: acc.atomic.capture {
  // CHECK-NEXT: acc.atomic.update %[[X_ALLOCA]] : !cir.ptr<!s32i> {
  // CHECK-NEXT: ^bb0(%[[X_VAR:.*]]: !s32i{{.*}}):
  // CHECK-NEXT: %[[X_VAR_ALLOC:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[X_VAR]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[DEC:.*]] = cir.unary(dec, %[[X_VAR_LOAD]]) nsw : !s32i, !s32i
  // CHECK-NEXT: cir.store{{.*}} %[[DEC]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.yield %[[X_VAR_LOAD]] : !s32i
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.atomic.read %[[V_ALLOCA]] = %[[X_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: }
#pragma acc atomic capture
  {
    x--;
    v = x;
  }

  // CHECK-NEXT: acc.atomic.capture {
  // CHECK-NEXT: acc.atomic.update %[[X_ALLOCA]] : !cir.ptr<!s32i> {
  // CHECK-NEXT: ^bb0(%[[X_VAR:.*]]: !s32i{{.*}}):
  // CHECK-NEXT: %[[X_VAR_ALLOC:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x_var", init]
  // CHECK-NEXT: cir.store %[[X_VAR]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[DEC:.*]] = cir.unary(dec, %[[X_VAR_LOAD]]) nsw : !s32i, !s32i
  // CHECK-NEXT: cir.store{{.*}} %[[DEC]], %[[X_VAR_ALLOC]] : !s32i, !cir.ptr<!s32i>
  //
  // CHECK-NEXT: %[[X_VAR_LOAD:.*]] = cir.load{{.*}} %[[X_VAR_ALLOC]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: acc.yield %[[X_VAR_LOAD]] : !s32i
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.atomic.read %[[V_ALLOCA]] = %[[X_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: }
#pragma acc atomic capture
  {
    --x;
    v = x;
  }
}
