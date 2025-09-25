// RUN: %clang_cc1 -fopenacc -emit-cir -fclangir %s -o - | FileCheck %s

void acc_shutdown(int cond) {
  // CHECK: cir.func{{.*}} @acc_shutdown(%[[ARG:.*]]: !s32i{{.*}}) {
  // CHECK-NEXT: %[[COND:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["cond", init]
  // CHECK-NEXT: cir.store %[[ARG]], %[[COND]] : !s32i, !cir.ptr<!s32i>
#pragma acc shutdown
// CHECK-NEXT: acc.shutdown loc(#{{[a-zA-Z0-9]+}}){{$}}

#pragma acc shutdown device_type(*)
  // CHECK-NEXT: acc.shutdown attributes {device_types = [#acc.device_type<star>]}
#pragma acc shutdown device_type(nvidia)
  // CHECK-NEXT: acc.shutdown attributes {device_types = [#acc.device_type<nvidia>]}
#pragma acc shutdown device_type(NVIDIA)
  // CHECK-NEXT: acc.shutdown attributes {device_types = [#acc.device_type<nvidia>]}
#pragma acc shutdown device_type(HoSt) device_type(MuLtIcORe)
  // CHECK-NEXT: acc.shutdown attributes {device_types = [#acc.device_type<host>, #acc.device_type<multicore>]}

#pragma acc shutdown if(cond)
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[COND_CAST:.*]] = cir.cast(int_to_bool, %[[COND_LOAD]] : !s32i), !cir.bool
  // CHECK-NEXT: %[[BOOL_CONV:.*]] = builtin.unrealized_conversion_cast %[[COND_CAST]] : !cir.bool to i1
  // CHECK-NEXT: acc.shutdown if(%[[BOOL_CONV]])

#pragma acc shutdown if(1)
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_TO_BOOL_CAST:.*]] = cir.cast(int_to_bool, %[[ONE_LITERAL]] : !s32i), !cir.bool
  // CHECK-NEXT: %[[BOOL_CONV:.*]] = builtin.unrealized_conversion_cast %[[ONE_TO_BOOL_CAST]] : !cir.bool to i1
  // CHECK-NEXT: acc.shutdown if(%[[BOOL_CONV]])

#pragma acc shutdown device_num(cond)
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[COND_CONV:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.shutdown device_num(%[[COND_CONV]] : si32)

#pragma acc shutdown device_num(1)
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CONV:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.shutdown device_num(%[[ONE_CONV]] : si32)

#pragma acc shutdown if(cond) device_num(cond) device_type(*)
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[COND_CAST:.*]] = cir.cast(int_to_bool, %[[COND_LOAD]] : !s32i), !cir.bool
  // CHECK-NEXT: %[[BOOL_CONV:.*]] = builtin.unrealized_conversion_cast %[[COND_CAST]] : !cir.bool to i1
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[COND_CONV:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.shutdown device_num(%[[COND_CONV]] : si32) if(%[[BOOL_CONV]]) attributes {device_types = [#acc.device_type<star>]}
}
