// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

void acc_host_data(int cond, int var1, int var2, int *arr) {
  // CHECK: cir.func{{.*}} @acc_host_data(%[[ARG_COND:.*]]: !s32i {{.*}}, %[[ARG_V1:.*]]: !s32i {{.*}}, %[[ARG_V2:.*]]: !s32i {{.*}}, %[[ARG_ARR:.*]]: !cir.ptr<!s32i> {{.*}}) {
  // CHECK-NEXT: %[[COND:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["cond", init]
  // CHECK-NEXT: %[[V1:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["var1", init]
  // CHECK-NEXT: %[[V2:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["var2", init]
  // CHECK-NEXT: %[[ARR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arr", init]
  // CHECK-NEXT: cir.store %[[ARG_COND]], %[[COND]] : !s32i, !cir.ptr<!s32i>
  // CHECK-NEXT: cir.store %[[ARG_V1]], %[[V1]] : !s32i, !cir.ptr<!s32i>
  // CHECK-NEXT: cir.store %[[ARG_V2]], %[[V2]] : !s32i, !cir.ptr<!s32i>
  // CHECK-NEXT: cir.store %[[ARG_ARR]], %[[ARR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>

#pragma acc host_data use_device(var1)
  {}
  // CHECK-NEXT: %[[USE_DEV1:.*]] = acc.use_device varPtr(%[[V1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "var1"}
  // CHECK-NEXT: acc.host_data dataOperands(%[[USE_DEV1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
#pragma acc host_data use_device(var1, var2)
  {}
  // CHECK-NEXT: %[[USE_DEV1:.*]] = acc.use_device varPtr(%[[V1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "var1"}
  // CHECK-NEXT: %[[USE_DEV2:.*]] = acc.use_device varPtr(%[[V2]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "var2"}
  // CHECK-NEXT: acc.host_data dataOperands(%[[USE_DEV1]], %[[USE_DEV2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc host_data use_device(var1, var2) if_present
  {}
  // CHECK-NEXT: %[[USE_DEV1:.*]] = acc.use_device varPtr(%[[V1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "var1"}
  // CHECK-NEXT: %[[USE_DEV2:.*]] = acc.use_device varPtr(%[[V2]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "var2"}
  // CHECK-NEXT: acc.host_data dataOperands(%[[USE_DEV1]], %[[USE_DEV2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {ifPresent}

#pragma acc host_data use_device(var1, var2) if(cond)
  {}
  // CHECK-NEXT: %[[USE_DEV1:.*]] = acc.use_device varPtr(%[[V1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "var1"}
  // CHECK-NEXT: %[[USE_DEV2:.*]] = acc.use_device varPtr(%[[V2]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "var2"}
  // CHECK-NEXT: %[[LOAD_COND:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[COND_BOOL:.*]] = cir.cast int_to_bool %[[LOAD_COND]] : !s32i -> !cir.bool
  // CHECK-NEXT: %[[COND_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_BOOL]] : !cir.bool to i1
  // CHECK-NEXT: acc.host_data if(%[[COND_CAST]]) dataOperands(%[[USE_DEV1]], %[[USE_DEV2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc host_data use_device(var1, var2) if(cond) if_present
  {}
  // CHECK-NEXT: %[[USE_DEV1:.*]] = acc.use_device varPtr(%[[V1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "var1"}
  // CHECK-NEXT: %[[USE_DEV2:.*]] = acc.use_device varPtr(%[[V2]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "var2"}
  // CHECK-NEXT: %[[LOAD_COND:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[COND_BOOL:.*]] = cir.cast int_to_bool %[[LOAD_COND]] : !s32i -> !cir.bool
  // CHECK-NEXT: %[[COND_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_BOOL]] : !cir.bool to i1
  // CHECK-NEXT: acc.host_data if(%[[COND_CAST]]) dataOperands(%[[USE_DEV1]], %[[USE_DEV2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {ifPresent}

#pragma acc host_data use_device(arr[0:var1])
  {}
  // CHECK-NEXT: %[[ZERO:.*]] = cir.const #cir.int<0>
  // CHECK-NEXT: %[[ZERO_CAST:.*]] = builtin.unrealized_conversion_cast %[[ZERO]] : !s32i to si32
  // CHECK-NEXT: %[[VAR1_LOAD:.*]] = cir.load{{.*}} %[[V1]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[VAR1_CAST:.*]] = builtin.unrealized_conversion_cast %[[VAR1_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[CONST_ZERO:.*]] = arith.constant 0
  // CHECK-NEXT: %[[CONST_ONE:.*]] = arith.constant 1
  // CHECK-NEXT: %[[BOUNDS:.*]] = acc.bounds lowerbound(%[[ZERO_CAST]] : si32) extent(%[[VAR1_CAST]] : si32) stride(%[[CONST_ONE]] : i64) startIdx(%[[CONST_ZERO]] : i64)
  // CHECK-NEXT: %[[USE_DEV1:.*]] = acc.use_device varPtr(%[[ARR]] : !cir.ptr<!cir.ptr<!s32i>>) bounds(%[[BOUNDS]]) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arr[0:var1]"}
  // CHECK-NEXT: acc.host_data dataOperands(%[[USE_DEV1]] : !cir.ptr<!cir.ptr<!s32i>>)
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
}
