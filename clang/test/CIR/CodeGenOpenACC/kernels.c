// RUN: %clang_cc1 -fopenacc -emit-cir -fclangir %s -o - | FileCheck %s

void acc_kernels(int cond) {
  // CHECK: cir.func @acc_kernels(%[[ARG:.*]]: !s32i{{.*}}) {
  // CHECK-NEXT: %[[COND:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["cond", init]
  // CHECK-NEXT: cir.store %[[ARG]], %[[COND]] : !s32i, !cir.ptr<!s32i>
#pragma acc kernels
  {}

  // CHECK-NEXT: acc.kernels {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT:}

#pragma acc kernels default(none)
  {}
  // CHECK-NEXT: acc.kernels {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc kernels default(present)
  {}
  // CHECK-NEXT: acc.kernels {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue present>}

#pragma acc kernels
  while(1){}
  // CHECK-NEXT: acc.kernels {
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
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT:}

#pragma acc kernels self
  {}
  // CHECK-NEXT: acc.kernels {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {selfAttr}

#pragma acc kernels self(cond)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[BOOL_CAST:.*]] = cir.cast(int_to_bool, %[[COND_LOAD]] : !s32i), !cir.bool
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[BOOL_CAST]] : !cir.bool to i1
  // CHECK-NEXT: acc.kernels self(%[[CONV_CAST]]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels self(0)
  {}
  // CHECK-NEXT: %[[ZERO_LITERAL:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK-NEXT: %[[BOOL_CAST:.*]] = cir.cast(int_to_bool, %[[ZERO_LITERAL]] : !s32i), !cir.bool
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[BOOL_CAST]] : !cir.bool to i1
  // CHECK-NEXT: acc.kernels self(%[[CONV_CAST]]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
 
  // CHECK-NEXT: cir.return
}
