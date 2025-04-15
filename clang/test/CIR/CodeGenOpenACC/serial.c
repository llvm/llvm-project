// RUN: %clang_cc1 -fopenacc -emit-cir -fclangir %s -o - | FileCheck %s

void acc_serial(int cond) {
  // CHECK: cir.func @acc_serial(%[[ARG:.*]]: !s32i{{.*}}) {
  // CHECK-NEXT: %[[COND:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["cond", init]
  // CHECK-NEXT: cir.store %[[ARG]], %[[COND]] : !s32i, !cir.ptr<!s32i>
#pragma acc serial
  {}

  // CHECK-NEXT: acc.serial {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT:}

#pragma acc serial default(none)
  {}
  // CHECK-NEXT: acc.serial {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc serial default(present)
  {}
  // CHECK-NEXT: acc.serial {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue present>}

#pragma acc serial
  while(1){}
  // CHECK-NEXT: acc.serial {
  // CHECK-NEXT: cir.scope {
  // CHECK-NEXT: cir.while {
  // CHECK-NEXT: %[[INT:.*]] = cir.const #cir.int<1>
  // CHECK-NEXT: %[[CAST:.*]] = cir.cast(int_to_bool, %[[INT]] :
  // CHECK-NEXT: cir.condition(%[[CAST]])
  // CHECK-NEXT: } do {
  // CHECK-NEXT: cir.yield
  // cir.while do end:
  // CHECK-NEXT: }
  // cir.scope end:
  // CHECK-NEXT: }
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT:}

#pragma acc serial self
  {}
  // CHECK-NEXT: acc.serial {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } attributes {selfAttr}

#pragma acc serial self(cond)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[BOOL_CAST:.*]] = cir.cast(int_to_bool, %[[COND_LOAD]] : !s32i), !cir.bool
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[BOOL_CAST]] : !cir.bool to i1
  // CHECK-NEXT: acc.serial self(%[[CONV_CAST]]) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial self(0)
  {}
  // CHECK-NEXT: %[[ZERO_LITERAL:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK-NEXT: %[[BOOL_CAST:.*]] = cir.cast(int_to_bool, %[[ZERO_LITERAL]] : !s32i), !cir.bool
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[BOOL_CAST]] : !cir.bool to i1
  // CHECK-NEXT: acc.serial self(%[[CONV_CAST]]) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

  // CHECK-NEXT: cir.return
}
