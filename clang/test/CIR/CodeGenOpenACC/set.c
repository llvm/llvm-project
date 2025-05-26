// RUN: %clang_cc1 -fopenacc -emit-cir -fclangir %s -o - | FileCheck %s

void acc_set(int cond) {
  // CHECK: cir.func @acc_set(%[[ARG:.*]]: !s32i{{.*}}) {
  // CHECK-NEXT: %[[COND:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["cond", init]
  // CHECK-NEXT: cir.store %[[ARG]], %[[COND]] : !s32i, !cir.ptr<!s32i>

#pragma acc set device_type(*)
  // CHECK-NEXT: acc.set attributes {device_type = #acc.device_type<star>}

  // Set doesn't allow multiple device_type clauses, so no need to test them.
#pragma acc set device_type(radeon)
  // CHECK-NEXT: acc.set attributes {device_type = #acc.device_type<radeon>}

#pragma acc set default_async(cond)
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[COND_CONV:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.set default_async(%[[COND_CONV]] : si32)

#pragma acc set default_async(1)
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CONV:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.set default_async(%[[ONE_CONV]] : si32)

#pragma acc set device_num(cond) if (cond)
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[COND_CONV:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[COND_CAST:.*]] = cir.cast(int_to_bool, %[[COND_LOAD]] : !s32i), !cir.bool
  // CHECK-NEXT: %[[BOOL_CONV:.*]] = builtin.unrealized_conversion_cast %[[COND_CAST]] : !cir.bool to i1
  // CHECK-NEXT: acc.set device_num(%[[COND_CONV]] : si32) if(%[[BOOL_CONV]])

#pragma acc set device_type(radeon) default_async(1) device_num(cond) if (cond)
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CONV:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[COND_CONV:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[COND_CAST:.*]] = cir.cast(int_to_bool, %[[COND_LOAD]] : !s32i), !cir.bool
  // CHECK-NEXT: %[[BOOL_CONV:.*]] = builtin.unrealized_conversion_cast %[[COND_CAST]] : !cir.bool to i1
  // CHECK-NEXT: acc.set default_async(%[[ONE_CONV]] : si32) device_num(%[[COND_CONV]] : si32) if(%[[BOOL_CONV]]) attributes {device_type = #acc.device_type<radeon>}

  // CHECK-NEXT: cir.return
}
