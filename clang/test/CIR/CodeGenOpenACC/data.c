// RUN: %clang_cc1 -fopenacc -emit-cir -fclangir %s -o - | FileCheck %s

void acc_data(int cond) {
  // CHECK: cir.func @acc_data(%[[ARG:.*]]: !s32i{{.*}}) {
  // CHECK-NEXT: %[[COND:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["cond", init]
  // CHECK-NEXT: cir.store %[[ARG]], %[[COND]] : !s32i, !cir.ptr<!s32i>

#pragma acc data default(none)
  {
    int i = 0;
    ++i;
  }
  // CHECK-NEXT: acc.data {
  // CHECK-NEXT: cir.alloca
  // CHECK-NEXT: cir.const
  // CHECK-NEXT: cir.store
  // CHECK-NEXT: cir.load
  // CHECK-NEXT: cir.unary
  // CHECK-NEXT: cir.store
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(present)
  {
    int i = 0;
    ++i;
  }
  // CHECK-NEXT: acc.data {
  // CHECK-NEXT: cir.alloca
  // CHECK-NEXT: cir.const
  // CHECK-NEXT: cir.store
  // CHECK-NEXT: cir.load
  // CHECK-NEXT: cir.unary
  // CHECK-NEXT: cir.store
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue present>}

#pragma acc data default(none) async
  {}
  // CHECK-NEXT: acc.data {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {asyncOnly = [#acc.device_type<none>], defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) async(cond)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.data async(%[[CONV_CAST]] : si32) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) async device_type(nvidia, radeon) async
  {}
  // CHECK-NEXT: acc.data {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {asyncOnly = [#acc.device_type<none>, #acc.device_type<nvidia>, #acc.device_type<radeon>], defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) async(3) device_type(nvidia, radeon) async(cond)
  {}
  // CHECK-NEXT: %[[THREE_LITERAL:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.data async(%[[THREE_CAST]] : si32, %[[CONV_CAST]] : si32 [#acc.device_type<nvidia>], %[[CONV_CAST]] : si32 [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) async device_type(nvidia, radeon) async(cond)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.data async(%[[CONV_CAST]] : si32 [#acc.device_type<nvidia>], %[[CONV_CAST]] : si32 [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {asyncOnly = [#acc.device_type<none>], defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) async(3) device_type(nvidia, radeon) async
  {}
  // CHECK-NEXT: %[[THREE_LITERAL:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.data async(%[[THREE_CAST]] : si32) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {asyncOnly = [#acc.device_type<nvidia>, #acc.device_type<radeon>], defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) if(cond)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[BOOL_CAST:.*]] = cir.cast(int_to_bool, %[[COND_LOAD]] : !s32i), !cir.bool
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[BOOL_CAST]] : !cir.bool to i1
  // CHECK-NEXT: acc.data if(%[[CONV_CAST]]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) if(1)
  {}
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[BOOL_CAST:.*]] = cir.cast(int_to_bool, %[[ONE_LITERAL]] : !s32i), !cir.bool
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[BOOL_CAST]] : !cir.bool to i1
  // CHECK-NEXT: acc.data if(%[[CONV_CAST]]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) if(cond == 1)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[EQ_RES:.*]] = cir.cmp(eq, %[[COND_LOAD]], %[[ONE_LITERAL]]) : !s32i, !cir.bool
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[EQ_RES]] : !cir.bool to i1
  // CHECK-NEXT: acc.data if(%[[CONV_CAST]]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}

  // CHECK-NEXT: cir.return
}
