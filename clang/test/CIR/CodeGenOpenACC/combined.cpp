// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

extern "C" void acc_combined(int N, int cond) {
  // CHECK: cir.func @acc_combined(%[[ARG_N:.*]]: !s32i loc{{.*}}, %[[ARG_COND:.*]]: !s32i loc{{.*}}) {
  // CHECK-NEXT: %[[ALLOCA_N:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["N", init]
  // CHECK-NEXT: %[[COND:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["cond", init]
  // CHECK-NEXT: cir.store %[[ARG_N]], %[[ALLOCA_N]] : !s32i, !cir.ptr<!s32i>
  // CHECK-NEXT: cir.store %[[ARG_COND]], %[[COND]] : !s32i, !cir.ptr<!s32i>

#pragma acc parallel loop
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.parallel combined(loop) {
  // CHECK: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial loop
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.serial combined(loop) {
  // CHECK: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.yield
  // CHECK-NEXT: } loc

#pragma acc kernels loop
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.kernels combined(loop) {
  // CHECK: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc parallel loop default(none)
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.parallel combined(loop) {
  // CHECK: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>} loc

#pragma acc serial loop default(present)
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.serial combined(loop) {
  // CHECK: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue present>} loc

#pragma acc kernels loop default(none)
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.kernels combined(loop) {
  // CHECK: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>} loc

#pragma acc parallel loop seq
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.parallel combined(loop) {
  // CHECK: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {seq = [#acc.device_type<none>]} loc
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial loop device_type(nvidia, radeon) seq
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.serial combined(loop) {
  // CHECK: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {seq = [#acc.device_type<nvidia>, #acc.device_type<radeon>, #acc.device_type<none>]} loc
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
#pragma acc kernels loop seq device_type(nvidia, radeon)
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.kernels combined(loop) {
  // CHECK: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {seq = [#acc.device_type<none>]} loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc parallel loop auto
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.parallel combined(loop) {
  // CHECK: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {auto_ = [#acc.device_type<none>]} loc
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial loop device_type(nvidia, radeon) auto
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.serial combined(loop) {
  // CHECK: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {auto_ = [#acc.device_type<nvidia>, #acc.device_type<radeon>], seq = [#acc.device_type<none>]} loc
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
#pragma acc kernels loop auto device_type(nvidia, radeon)
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.kernels combined(loop) {
  // CHECK: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {auto_ = [#acc.device_type<none>]} loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc parallel loop independent
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.parallel combined(loop) {
  // CHECK: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {independent = [#acc.device_type<none>]} loc
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial loop device_type(nvidia, radeon) independent
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.serial combined(loop) {
  // CHECK: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {independent = [#acc.device_type<nvidia>, #acc.device_type<radeon>], seq = [#acc.device_type<none>]} loc
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
#pragma acc kernels loop independent device_type(nvidia, radeon)
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.kernels combined(loop) {
  // CHECK: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {independent = [#acc.device_type<none>]} loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc

  #pragma acc parallel loop collapse(1) device_type(radeon)
  for(unsigned I = 0; I < N; ++I)
    for(unsigned J = 0; J < N; ++J)
      for(unsigned K = 0; K < N; ++K);
  // CHECK: acc.parallel combined(loop) {
  // CHECK: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {collapse = [1], collapseDeviceType = [#acc.device_type<none>], independent = [#acc.device_type<none>]}
  // CHECK: acc.yield
  // CHECK-NEXT: } loc

  #pragma acc serial loop collapse(1) device_type(radeon) collapse (2)
  for(unsigned I = 0; I < N; ++I)
    for(unsigned J = 0; J < N; ++J)
      for(unsigned K = 0; K < N; ++K);
  // CHECK: acc.serial combined(loop) {
  // CHECK: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {collapse = [1, 2], collapseDeviceType = [#acc.device_type<none>, #acc.device_type<radeon>], seq = [#acc.device_type<none>]}
  // CHECK: acc.yield
  // CHECK-NEXT: } loc

  #pragma acc kernels loop collapse(1) device_type(radeon, nvidia) collapse (2)
  for(unsigned I = 0; I < N; ++I)
    for(unsigned J = 0; J < N; ++J)
      for(unsigned K = 0; K < N; ++K);
  // CHECK: acc.kernels combined(loop) {
  // CHECK: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {auto_ = [#acc.device_type<none>], collapse = [1, 2, 2], collapseDeviceType = [#acc.device_type<none>, #acc.device_type<radeon>, #acc.device_type<nvidia>]}
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc
  #pragma acc parallel loop collapse(1) device_type(radeon, nvidia) collapse(2) device_type(host) collapse(3)
  for(unsigned I = 0; I < N; ++I)
    for(unsigned J = 0; J < N; ++J)
      for(unsigned K = 0; K < N; ++K);
  // CHECK: acc.parallel combined(loop) {
  // CHECK: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {collapse = [1, 2, 2, 3], collapseDeviceType = [#acc.device_type<none>, #acc.device_type<radeon>, #acc.device_type<nvidia>, #acc.device_type<host>], independent = [#acc.device_type<none>]}
  // CHECK: acc.yield
  // CHECK-NEXT: } loc

#pragma acc kernels loop self
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.kernels combined(loop) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {selfAttr}

#pragma acc serial loop self(N)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[N_LOAD:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[BOOL_CAST:.*]] = cir.cast(int_to_bool, %[[N_LOAD]] : !s32i), !cir.bool
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[BOOL_CAST]] : !cir.bool to i1
  // CHECK-NEXT: acc.serial combined(loop) self(%[[CONV_CAST]]) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc parallel loop if(N)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[N_LOAD:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[BOOL_CAST:.*]] = cir.cast(int_to_bool, %[[N_LOAD]] : !s32i), !cir.bool
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[BOOL_CAST]] : !cir.bool to i1
  // CHECK-NEXT: acc.parallel combined(loop) if(%[[CONV_CAST]]) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial loop if(1)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[BOOL_CAST:.*]] = cir.cast(int_to_bool, %[[ONE_LITERAL]] : !s32i), !cir.bool
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[BOOL_CAST]] : !cir.bool to i1
  // CHECK-NEXT: acc.serial combined(loop) if(%[[CONV_CAST]]) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc kernels loop if(N == 1)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[N_LOAD:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[EQ_RES:.*]] = cir.cmp(eq, %[[N_LOAD]], %[[ONE_LITERAL]]) : !s32i, !cir.bool
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[EQ_RES]] : !cir.bool to i1
  // CHECK-NEXT: acc.kernels combined(loop) if(%[[CONV_CAST]]) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc parallel loop if(N == 1) self(N == 2)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[N_LOAD:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[EQ_RES_IF:.*]] = cir.cmp(eq, %[[N_LOAD]], %[[ONE_LITERAL]]) : !s32i, !cir.bool
  // CHECK-NEXT: %[[CONV_CAST_IF:.*]] = builtin.unrealized_conversion_cast %[[EQ_RES_IF]] : !cir.bool to i1
  // CHECK-NEXT: %[[N_LOAD:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[EQ_RES_SELF:.*]] = cir.cmp(eq, %[[N_LOAD]], %[[TWO_LITERAL]]) : !s32i, !cir.bool
  // CHECK-NEXT: %[[CONV_CAST_SELF:.*]] = builtin.unrealized_conversion_cast %[[EQ_RES_SELF]] : !cir.bool to i1
  // CHECK-NEXT: acc.parallel combined(loop) self(%[[CONV_CAST_SELF]]) if(%[[CONV_CAST_IF]]) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

  #pragma acc parallel loop tile(1, 2, 3)
  for(unsigned I = 0; I < N; ++I)
    for(unsigned J = 0; J < N; ++J)
      for(unsigned K = 0; K < N; ++K);
  // CHECK-NEXT: acc.parallel combined(loop) {
  // CHECK: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[TWO_CONST:.*]] = arith.constant 2 : i64
  // CHECK-NEXT: %[[THREE_CONST:.*]] = arith.constant 3 : i64
  // CHECK-NEXT: acc.loop combined(parallel) tile({%[[ONE_CONST]] : i64, %[[TWO_CONST]] : i64, %[[THREE_CONST]] : i64}) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  #pragma acc serial loop tile(2) device_type(radeon)
  for(unsigned I = 0; I < N; ++I)
    for(unsigned J = 0; J < N; ++J)
      for(unsigned K = 0; K < N; ++K);
  // CHECK-NEXT: acc.serial combined(loop) {
  // CHECK-NEXT: %[[TWO_CONST:.*]] = arith.constant 2 : i64
  // CHECK-NEXT: acc.loop combined(serial) tile({%[[TWO_CONST]] : i64}) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  #pragma acc kernels loop tile(2) device_type(radeon) tile (1, *)
  for(unsigned I = 0; I < N; ++I)
    for(unsigned J = 0; J < N; ++J)
      for(unsigned K = 0; K < N; ++K);
  // CHECK-NEXT: acc.kernels combined(loop) {
  // CHECK-NEXT: %[[TWO_CONST:.*]] = arith.constant 2 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[STAR_CONST:.*]] = arith.constant -1 : i64
  // CHECK-NEXT: acc.loop combined(kernels) tile({%[[TWO_CONST]] : i64}, {%[[ONE_CONST]] : i64, %[[STAR_CONST]] : i64} [#acc.device_type<radeon>]) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  #pragma acc parallel loop tile(*) device_type(radeon, nvidia) tile (1, 2)
  for(unsigned I = 0; I < N; ++I)
    for(unsigned J = 0; J < N; ++J)
      for(unsigned K = 0; K < N; ++K);
  // CHECK-NEXT: acc.parallel combined(loop) {
  // CHECK-NEXT: %[[STAR_CONST:.*]] = arith.constant -1 : i64
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[TWO_CONST:.*]] = arith.constant 2 : i64
  // CHECK-NEXT: acc.loop combined(parallel) tile({%[[STAR_CONST]] : i64}, {%[[ONE_CONST]] : i64, %[[TWO_CONST]] : i64} [#acc.device_type<radeon>], {%[[ONE_CONST]] : i64, %[[TWO_CONST]] : i64} [#acc.device_type<nvidia>]) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  #pragma acc serial loop tile(1) device_type(radeon, nvidia) tile(2, 3) device_type(host) tile(*, *, *)
  for(unsigned I = 0; I < N; ++I)
    for(unsigned J = 0; J < N; ++J)
      for(unsigned K = 0; K < N; ++K);
  // CHECK-NEXT: acc.serial combined(loop) {
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[TWO_CONST:.*]] = arith.constant 2 : i64
  // CHECK-NEXT: %[[THREE_CONST:.*]] = arith.constant 3 : i64
  // CHECK-NEXT: %[[STAR_CONST:.*]] = arith.constant -1 : i64
  // CHECK-NEXT: %[[STAR2_CONST:.*]] = arith.constant -1 : i64
  // CHECK-NEXT: %[[STAR3_CONST:.*]] = arith.constant -1 : i64
  // CHECK-NEXT: acc.loop combined(serial) tile({%[[ONE_CONST]] : i64}, {%[[TWO_CONST]] : i64, %[[THREE_CONST]] : i64} [#acc.device_type<radeon>], {%[[TWO_CONST]] : i64, %[[THREE_CONST]] : i64} [#acc.device_type<nvidia>], {%[[STAR_CONST]] : i64, %[[STAR2_CONST]] : i64, %[[STAR3_CONST]] : i64} [#acc.device_type<host>]) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc parallel loop gang
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.parallel combined(loop) {
  // CHECK-NEXT: acc.loop combined(parallel) gang {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc parallel loop gang device_type(nvidia) gang
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.parallel combined(loop) {
  // CHECK-NEXT: acc.loop combined(parallel) gang([#acc.device_type<none>, #acc.device_type<nvidia>]) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc parallel loop gang(dim:1) device_type(nvidia) gang(dim:2)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.parallel combined(loop) {
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[TWO_CONST:.*]] = arith.constant 2 : i64
  // CHECK-NEXT: acc.loop combined(parallel) gang({dim=%[[ONE_CONST]] : i64}, {dim=%[[TWO_CONST]] : i64} [#acc.device_type<nvidia>]) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc parallel loop gang(static:N, dim: 1) device_type(nvidia, radeon) gang(static:*, dim : 2)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.parallel combined(loop) {
  // CHECK-NEXT: %[[N_LOAD:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[N_CONV:.*]] = builtin.unrealized_conversion_cast %[[N_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_CONST:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[STAR_CONST:.*]] = arith.constant -1 : i64
  // CHECK-NEXT: %[[TWO_CONST:.*]] = arith.constant 2 : i64
  // CHECK-NEXT: acc.loop combined(parallel) gang({static=%[[N_CONV]] : si32, dim=%[[ONE_CONST]] : i64}, {static=%[[STAR_CONST]] : i64, dim=%[[TWO_CONST]] : i64} [#acc.device_type<nvidia>], {static=%[[STAR_CONST]] : i64, dim=%[[TWO_CONST]] : i64} [#acc.device_type<radeon>]) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc kernels loop gang(num:N) device_type(nvidia, radeon) gang(num:N)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.kernels combined(loop) {
  // CHECK-NEXT: %[[N_LOAD:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[N_CONV:.*]] = builtin.unrealized_conversion_cast %[[N_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[N_LOAD2:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[N_CONV2:.*]] = builtin.unrealized_conversion_cast %[[N_LOAD2]] : !s32i to si32
  // CHECK-NEXT: acc.loop combined(kernels) gang({num=%[[N_CONV]] : si32}, {num=%[[N_CONV2]] : si32} [#acc.device_type<nvidia>], {num=%[[N_CONV2]] : si32} [#acc.device_type<radeon>]) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
#pragma acc kernels loop gang(static:N) device_type(nvidia) gang(static:*)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.kernels combined(loop) {
  // CHECK-NEXT: %[[N_LOAD:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[N_CONV:.*]] = builtin.unrealized_conversion_cast %[[N_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[STAR_CONST:.*]] = arith.constant -1 : i64
  // CHECK-NEXT: acc.loop combined(kernels) gang({static=%[[N_CONV]] : si32}, {static=%[[STAR_CONST]] : i64} [#acc.device_type<nvidia>]) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
#pragma acc kernels loop gang(static:N, num: N + 1) device_type(nvidia) gang(static:*, num : N + 2)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.kernels combined(loop) {
  // CHECK-NEXT: %[[N_LOAD:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[N_CONV:.*]] = builtin.unrealized_conversion_cast %[[N_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[N_LOAD2:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CIR_ONE_CONST:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[N_PLUS_ONE:.*]] = cir.binop(add, %[[N_LOAD2]], %[[CIR_ONE_CONST]]) nsw : !s32i
  // CHECK-NEXT: %[[N_PLUS_ONE_CONV:.*]] = builtin.unrealized_conversion_cast %[[N_PLUS_ONE]] : !s32i to si32
  // CHECK-NEXT: %[[STAR_CONST:.*]] = arith.constant -1 : i64
  // CHECK-NEXT: %[[N_LOAD3:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CIR_TWO_CONST:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[N_PLUS_TWO:.*]] = cir.binop(add, %[[N_LOAD3]], %[[CIR_TWO_CONST]]) nsw : !s32i
  // CHECK-NEXT: %[[N_PLUS_TWO_CONV:.*]] = builtin.unrealized_conversion_cast %[[N_PLUS_TWO]] : !s32i to si32
  // CHECK-NEXT: acc.loop combined(kernels) gang({static=%[[N_CONV]] : si32, num=%[[N_PLUS_ONE_CONV]] : si32}, {static=%[[STAR_CONST]] : i64, num=%[[N_PLUS_TWO_CONV]] : si32} [#acc.device_type<nvidia>]) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels loop worker
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.kernels combined(loop) {
  // CHECK-NEXT: acc.loop combined(kernels) worker {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels loop worker(N)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.kernels combined(loop) {
  // CHECK-NEXT: %[[N_LOAD:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[N_CONV:.*]] = builtin.unrealized_conversion_cast %[[N_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.loop combined(kernels) worker(%[[N_CONV]] : si32) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels loop worker device_type(nvidia, radeon) worker
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.kernels combined(loop) {
  // CHECK-NEXT: acc.loop combined(kernels) worker([#acc.device_type<none>, #acc.device_type<nvidia>, #acc.device_type<radeon>]) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels loop worker(N) device_type(nvidia, radeon) worker
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.kernels combined(loop) {
  // CHECK-NEXT: %[[N_LOAD:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[N_CONV:.*]] = builtin.unrealized_conversion_cast %[[N_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.loop combined(kernels) worker([#acc.device_type<nvidia>, #acc.device_type<radeon>], %[[N_CONV]] : si32) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels loop worker device_type(nvidia, radeon) worker(N)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.kernels combined(loop) {
  // CHECK-NEXT: %[[N_LOAD:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[N_CONV:.*]] = builtin.unrealized_conversion_cast %[[N_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.loop combined(kernels) worker([#acc.device_type<none>], %[[N_CONV]] : si32 [#acc.device_type<nvidia>], %[[N_CONV]] : si32 [#acc.device_type<radeon>]) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels loop worker(N) device_type(nvidia, radeon) worker(N + 1)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.kernels combined(loop) {
  // CHECK-NEXT: %[[N_LOAD:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[N_CONV:.*]] = builtin.unrealized_conversion_cast %[[N_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[N_LOAD2:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[ONE_CONST:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[N_PLUS_ONE:.*]] = cir.binop(add, %[[N_LOAD2]], %[[ONE_CONST]]) nsw : !s32i
  // CHECK-NEXT: %[[N_PLUS_ONE_CONV:.*]] = builtin.unrealized_conversion_cast %[[N_PLUS_ONE]] : !s32i to si32
  // CHECK-NEXT: acc.loop combined(kernels) worker(%[[N_CONV]] : si32, %[[N_PLUS_ONE_CONV]] : si32 [#acc.device_type<nvidia>], %[[N_PLUS_ONE_CONV]] : si32 [#acc.device_type<radeon>]) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels loop device_type(nvidia, radeon) worker(num:N + 1)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.kernels combined(loop) {
  // CHECK-NEXT: %[[N_LOAD:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[ONE_CONST:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[N_PLUS_ONE:.*]] = cir.binop(add, %[[N_LOAD]], %[[ONE_CONST]]) nsw : !s32i
  // CHECK-NEXT: %[[N_PLUS_ONE_CONV:.*]] = builtin.unrealized_conversion_cast %[[N_PLUS_ONE]] : !s32i to si32
  // CHECK-NEXT: acc.loop combined(kernels) worker(%[[N_PLUS_ONE_CONV]] : si32 [#acc.device_type<nvidia>], %[[N_PLUS_ONE_CONV]] : si32 [#acc.device_type<radeon>]) {
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc


#pragma acc kernels loop worker vector device_type(nvidia) worker vector
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.kernels combined(loop) {
  // CHECK-NEXT: acc.loop combined(kernels) worker([#acc.device_type<none>, #acc.device_type<nvidia>]) vector([#acc.device_type<none>, #acc.device_type<nvidia>])
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels loop vector
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.kernels combined(loop) {
  // CHECK: acc.loop combined(kernels) vector {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels loop vector(N)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.kernels combined(loop) {
  // CHECK-NEXT: %[[N_LOAD:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[N_CONV:.*]] = builtin.unrealized_conversion_cast %[[N_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.loop combined(kernels) vector(%[[N_CONV]] : si32) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels loop vector device_type(nvidia, radeon) vector
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.kernels combined(loop) {
  // CHECK-NEXT: acc.loop combined(kernels) vector([#acc.device_type<none>, #acc.device_type<nvidia>, #acc.device_type<radeon>]) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels loop vector(N) device_type(nvidia, radeon) vector
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.kernels combined(loop) {
  // CHECK-NEXT: %[[N_LOAD:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[N_CONV:.*]] = builtin.unrealized_conversion_cast %[[N_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.loop combined(kernels) vector([#acc.device_type<nvidia>, #acc.device_type<radeon>], %[[N_CONV]] : si32) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels loop vector(N) device_type(nvidia, radeon) vector(N + 1)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.kernels combined(loop) {
  // CHECK-NEXT: %[[N_LOAD:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[N_CONV:.*]] = builtin.unrealized_conversion_cast %[[N_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[N_LOAD2:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[ONE_CONST:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[N_PLUS_ONE:.*]] = cir.binop(add, %[[N_LOAD2]], %[[ONE_CONST]]) nsw : !s32i
  // CHECK-NEXT: %[[N_PLUS_ONE_CONV:.*]] = builtin.unrealized_conversion_cast %[[N_PLUS_ONE]] : !s32i to si32
  // CHECK-NEXT: acc.loop combined(kernels) vector(%[[N_CONV]] : si32, %[[N_PLUS_ONE_CONV]] : si32 [#acc.device_type<nvidia>], %[[N_PLUS_ONE_CONV]] : si32 [#acc.device_type<radeon>]) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels loop device_type(nvidia, radeon) vector(length:N + 1)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.kernels combined(loop) {
  // CHECK-NEXT: %[[N_LOAD:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[ONE_CONST:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[N_PLUS_ONE:.*]] = cir.binop(add, %[[N_LOAD]], %[[ONE_CONST]]) nsw : !s32i
  // CHECK-NEXT: %[[N_PLUS_ONE_CONV:.*]] = builtin.unrealized_conversion_cast %[[N_PLUS_ONE]] : !s32i to si32
  // CHECK-NEXT: acc.loop combined(kernels) vector(%[[N_PLUS_ONE_CONV]] : si32 [#acc.device_type<nvidia>], %[[N_PLUS_ONE_CONV]] : si32 [#acc.device_type<radeon>]) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels loop worker(N) vector(N) device_type(nvidia) worker(N) vector(N)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.kernels combined(loop) {
  // CHECK-NEXT: %[[N_LOAD:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[N_CONV:.*]] = builtin.unrealized_conversion_cast %[[N_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[N_LOAD2:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[N_CONV2:.*]] = builtin.unrealized_conversion_cast %[[N_LOAD2]] : !s32i to si32
  // CHECK-NEXT: %[[N_LOAD3:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[N_CONV3:.*]] = builtin.unrealized_conversion_cast %[[N_LOAD3]] : !s32i to si32
  // CHECK-NEXT: %[[N_LOAD4:.*]] = cir.load{{.*}} %[[ALLOCA_N]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[N_CONV4:.*]] = builtin.unrealized_conversion_cast %[[N_LOAD4]] : !s32i to si32
  // CHECK-NEXT: acc.loop combined(kernels) worker(%[[N_CONV]] : si32, %[[N_CONV3]] : si32 [#acc.device_type<nvidia>]) vector(%[[N_CONV2]] : si32, %[[N_CONV4]] : si32 [#acc.device_type<nvidia>]) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc parallel loop wait
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.parallel combined(loop) wait {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial loop wait device_type(nvidia) wait
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.serial combined(loop) wait([#acc.device_type<none>, #acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc kernels loop wait(1) device_type(nvidia) wait
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.kernels combined(loop) wait([#acc.device_type<nvidia>], {%[[ONE_CAST]] : si32}) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc parallel loop wait device_type(nvidia) wait(1)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.parallel combined(loop) wait([#acc.device_type<none>], {%[[ONE_CAST]] : si32} [#acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial loop wait(1) device_type(nvidia) wait(1)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL2:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL2]] : !s32i to si32
  // CHECK-NEXT: acc.serial combined(loop) wait({%[[ONE_CAST]] : si32}, {%[[ONE_CAST2]] : si32} [#acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc kernels loop wait(devnum: cond : 1)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.kernels combined(loop) wait({devnum: %[[CONV_CAST]] : si32, %[[ONE_CAST]] : si32}) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc parallel loop wait(devnum: cond : 1) device_type(nvidia) wait(devnum: cond : 1)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST2:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.parallel combined(loop) wait({devnum: %[[CONV_CAST]] : si32, %[[ONE_CAST]] : si32}, {devnum: %[[CONV_CAST2]] : si32, %[[ONE_CAST2]] : si32} [#acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial loop wait(devnum: cond : 1, 2)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.serial combined(loop) wait({devnum: %[[CONV_CAST]] : si32, %[[ONE_CAST]] : si32, %[[TWO_CAST]] : si32}) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc kernels loop wait(devnum: cond : 1, 2) device_type(nvidia, radeon) wait(devnum: cond : 1, 2)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST2:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST2:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.kernels combined(loop) wait({devnum: %[[CONV_CAST]] : si32, %[[ONE_CAST]] : si32, %[[TWO_CAST]] : si32}, {devnum: %[[CONV_CAST2]] : si32, %[[ONE_CAST2]] : si32, %[[TWO_CAST2]] : si32} [#acc.device_type<nvidia>], {devnum: %[[CONV_CAST2]] : si32, %[[ONE_CAST2]] : si32, %[[TWO_CAST2]] : si32} [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc parallel loop wait(cond,  1)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.parallel combined(loop) wait({%[[CONV_CAST]] : si32, %[[ONE_CAST]] : si32}) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial loop wait(queues: cond,  1) device_type(radeon)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.serial combined(loop) wait({%[[CONV_CAST]] : si32, %[[ONE_CAST]] : si32}) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc parallel loop num_gangs(1)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.parallel combined(loop) num_gangs({%[[ONE_CAST]] : si32}) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc kernels loop num_gangs(cond)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.kernels combined(loop) num_gangs({%[[CONV_CAST]] : si32}) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc parallel loop num_gangs(1, cond, 2)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.parallel combined(loop) num_gangs({%[[ONE_CAST]] : si32, %[[CONV_CAST]] : si32, %[[TWO_CAST]] : si32}) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc kernels loop num_gangs(1) device_type(radeon) num_gangs(cond)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.kernels combined(loop) num_gangs({%[[ONE_CAST]] : si32}, {%[[CONV_CAST]] : si32} [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc parallel loop num_gangs(1, cond, 2) device_type(radeon) num_gangs(4, 5, 6)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[FOUR_LITERAL:.*]] = cir.const #cir.int<4> : !s32i
  // CHECK-NEXT: %[[FOUR_CAST:.*]] = builtin.unrealized_conversion_cast %[[FOUR_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[FIVE_LITERAL:.*]] = cir.const #cir.int<5> : !s32i
  // CHECK-NEXT: %[[FIVE_CAST:.*]] = builtin.unrealized_conversion_cast %[[FIVE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[SIX_LITERAL:.*]] = cir.const #cir.int<6> : !s32i
  // CHECK-NEXT: %[[SIX_CAST:.*]] = builtin.unrealized_conversion_cast %[[SIX_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.parallel combined(loop) num_gangs({%[[ONE_CAST]] : si32, %[[CONV_CAST]] : si32, %[[TWO_CAST]] : si32}, {%[[FOUR_CAST]] : si32, %[[FIVE_CAST]] : si32, %[[SIX_CAST]] : si32} [#acc.device_type<radeon>])
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc parallel loop num_gangs(1, cond, 2) device_type(radeon, nvidia) num_gangs(4, 5, 6)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[FOUR_LITERAL:.*]] = cir.const #cir.int<4> : !s32i
  // CHECK-NEXT: %[[FOUR_CAST:.*]] = builtin.unrealized_conversion_cast %[[FOUR_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[FIVE_LITERAL:.*]] = cir.const #cir.int<5> : !s32i
  // CHECK-NEXT: %[[FIVE_CAST:.*]] = builtin.unrealized_conversion_cast %[[FIVE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[SIX_LITERAL:.*]] = cir.const #cir.int<6> : !s32i
  // CHECK-NEXT: %[[SIX_CAST:.*]] = builtin.unrealized_conversion_cast %[[SIX_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.parallel combined(loop) num_gangs({%[[ONE_CAST]] : si32, %[[CONV_CAST]] : si32, %[[TWO_CAST]] : si32}, {%[[FOUR_CAST]] : si32, %[[FIVE_CAST]] : si32, %[[SIX_CAST]] : si32} [#acc.device_type<radeon>], {%[[FOUR_CAST]] : si32, %[[FIVE_CAST]] : si32, %[[SIX_CAST]] : si32} [#acc.device_type<nvidia>])
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc parallel loop num_workers(cond)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.parallel combined(loop) num_workers(%[[CONV_CAST]] : si32) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc kernels loop num_workers(cond) device_type(nvidia) num_workers(2u)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !u32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !u32i to ui32
  // CHECK-NEXT: acc.kernels combined(loop) num_workers(%[[CONV_CAST]] : si32, %[[TWO_CAST]] : ui32 [#acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc parallel loop num_workers(cond) device_type(nvidia, host) num_workers(2) device_type(radeon) num_workers(3)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[THREE_LITERAL:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.parallel combined(loop) num_workers(%[[CONV_CAST]] : si32, %[[TWO_CAST]] : si32 [#acc.device_type<nvidia>], %[[TWO_CAST]] : si32 [#acc.device_type<host>], %[[THREE_CAST]] : si32 [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc kernels loop num_workers(cond) device_type(nvidia) num_workers(2) device_type(radeon, multicore) num_workers(4)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[FOUR_LITERAL:.*]] = cir.const #cir.int<4> : !s32i
  // CHECK-NEXT: %[[FOUR_CAST:.*]] = builtin.unrealized_conversion_cast %[[FOUR_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.kernels combined(loop) num_workers(%[[CONV_CAST]] : si32, %[[TWO_CAST]] : si32 [#acc.device_type<nvidia>], %[[FOUR_CAST]] : si32 [#acc.device_type<radeon>], %[[FOUR_CAST]] : si32 [#acc.device_type<multicore>]) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc parallel loop device_type(nvidia) num_workers(2) device_type(radeon) num_workers(3)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[THREE_LITERAL:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.parallel combined(loop) num_workers(%[[TWO_CAST]] : si32 [#acc.device_type<nvidia>], %[[THREE_CAST]] : si32 [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  //
#pragma acc parallel loop vector_length(cond)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.parallel combined(loop) vector_length(%[[CONV_CAST]] : si32) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc kernels loop vector_length(cond) device_type(nvidia) vector_length(2u)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !u32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !u32i to ui32
  // CHECK-NEXT: acc.kernels combined(loop) vector_length(%[[CONV_CAST]] : si32, %[[TWO_CAST]] : ui32 [#acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc parallel loop vector_length(cond) device_type(nvidia, host) vector_length(2) device_type(radeon) vector_length(3)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[THREE_LITERAL:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.parallel combined(loop) vector_length(%[[CONV_CAST]] : si32, %[[TWO_CAST]] : si32 [#acc.device_type<nvidia>], %[[TWO_CAST]] : si32 [#acc.device_type<host>], %[[THREE_CAST]] : si32 [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc kernels loop vector_length(cond) device_type(nvidia) vector_length(2) device_type(radeon, multicore) vector_length(4)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[FOUR_LITERAL:.*]] = cir.const #cir.int<4> : !s32i
  // CHECK-NEXT: %[[FOUR_CAST:.*]] = builtin.unrealized_conversion_cast %[[FOUR_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.kernels combined(loop) vector_length(%[[CONV_CAST]] : si32, %[[TWO_CAST]] : si32 [#acc.device_type<nvidia>], %[[FOUR_CAST]] : si32 [#acc.device_type<radeon>], %[[FOUR_CAST]] : si32 [#acc.device_type<multicore>]) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc parallel loop device_type(nvidia) vector_length(2) device_type(radeon) vector_length(3)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[THREE_LITERAL:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.parallel combined(loop) vector_length(%[[TWO_CAST]] : si32 [#acc.device_type<nvidia>], %[[THREE_CAST]] : si32 [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc parallel loop async
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.parallel combined(loop) async {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial loop async(cond)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.serial combined(loop) async(%[[CONV_CAST]] : si32) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc kernels loop async device_type(nvidia, radeon) async
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: acc.kernels combined(loop) async([#acc.device_type<none>, #acc.device_type<nvidia>, #acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc parallel loop async(3) device_type(nvidia, radeon) async(cond)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[THREE_LITERAL:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.parallel combined(loop) async(%[[THREE_CAST]] : si32, %[[CONV_CAST]] : si32 [#acc.device_type<nvidia>], %[[CONV_CAST]] : si32 [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial loop async device_type(nvidia, radeon) async(cond)
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.serial combined(loop) async([#acc.device_type<none>], %[[CONV_CAST]] : si32 [#acc.device_type<nvidia>], %[[CONV_CAST]] : si32 [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc kernels loop async(3) device_type(nvidia, radeon) async
  for(unsigned I = 0; I < N; ++I);
  // CHECK-NEXT: %[[THREE_LITERAL:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.kernels combined(loop) async([#acc.device_type<nvidia>, #acc.device_type<radeon>], %[[THREE_CAST]] : si32) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
}
extern "C" void acc_combined_data_clauses(int *arg1, int *arg2) {
  // CHECK: cir.func @acc_combined_data_clauses(%[[ARG1_PARAM:.*]]: !cir.ptr<!s32i>{{.*}}, %[[ARG2_PARAM:.*]]: !cir.ptr<!s32i>{{.*}}) {
  // CHECK-NEXT: %[[ARG1:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arg1", init]
  // CHECK-NEXT: %[[ARG2:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arg2", init]
  // CHECK-NEXT: cir.store %[[ARG1_PARAM]], %[[ARG1]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CHECK-NEXT: cir.store %[[ARG2_PARAM]], %[[ARG2]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>

#pragma acc parallel loop deviceptr(arg1)
  for(unsigned I = 0; I < 5; ++I);
  // CHECK-NEXT: %[[DEVPTR1:.*]] = acc.deviceptr varPtr(%[[ARG1]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg1"}
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[DEVPTR1]] : !cir.ptr<!cir.ptr<!s32i>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial loop deviceptr(arg2)
  for(unsigned I = 0; I < 5; ++I);
  // CHECK-NEXT: %[[DEVPTR2:.*]] = acc.deviceptr varPtr(%[[ARG2]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg2"}
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[DEVPTR2]] : !cir.ptr<!cir.ptr<!s32i>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc kernels loop deviceptr(arg1, arg2)
  for(unsigned I = 0; I < 5; ++I);
  // CHECK-NEXT: %[[DEVPTR1:.*]] = acc.deviceptr varPtr(%[[ARG1]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg1"}
  // CHECK-NEXT: %[[DEVPTR2:.*]] = acc.deviceptr varPtr(%[[ARG2]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg2"}
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[DEVPTR1]], %[[DEVPTR2]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!s32i>>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc parallel loop deviceptr(arg1) async
  for(unsigned I = 0; I < 5; ++I);
  // CHECK-NEXT: %[[DEVPTR1:.*]] = acc.deviceptr varPtr(%[[ARG1]] : !cir.ptr<!cir.ptr<!s32i>>) async -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg1"}
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[DEVPTR1]] : !cir.ptr<!cir.ptr<!s32i>>) async {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial loop deviceptr(arg2) async device_type(nvidia)
  for(unsigned I = 0; I < 5; ++I);
  // CHECK-NEXT: %[[DEVPTR2:.*]] = acc.deviceptr varPtr(%[[ARG2]] : !cir.ptr<!cir.ptr<!s32i>>) async -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg2"}
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[DEVPTR2]] : !cir.ptr<!cir.ptr<!s32i>>) async {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc kernels loop deviceptr(arg1, arg2) device_type(nvidia) async
  for(unsigned I = 0; I < 5; ++I);
  // CHECK-NEXT: %[[DEVPTR1:.*]] = acc.deviceptr varPtr(%[[ARG1]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<nvidia>]) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg1"}
  // CHECK-NEXT: %[[DEVPTR2:.*]] = acc.deviceptr varPtr(%[[ARG2]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<nvidia>]) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg2"}
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[DEVPTR1]], %[[DEVPTR2]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc parallel loop no_create(arg1)
  for(unsigned I = 0; I < 5; ++I);
  // CHECK-NEXT: %[[NOCREATE1:.*]] = acc.nocreate varPtr(%[[ARG1]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg1"}
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[NOCREATE1]] : !cir.ptr<!cir.ptr<!s32i>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[NOCREATE1]] : !cir.ptr<!cir.ptr<!s32i>>) {dataClause = #acc<data_clause acc_no_create>, name = "arg1"}

#pragma acc serial loop no_create(arg2)
  for(unsigned I = 0; I < 5; ++I);
  // CHECK-NEXT: %[[NOCREATE2:.*]] = acc.nocreate varPtr(%[[ARG2]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg2"}
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[NOCREATE2]] : !cir.ptr<!cir.ptr<!s32i>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[NOCREATE2]] : !cir.ptr<!cir.ptr<!s32i>>) {dataClause = #acc<data_clause acc_no_create>, name = "arg2"}

#pragma acc kernels loop no_create(arg1, arg2) device_type(host) async
  for(unsigned I = 0; I < 5; ++I);
  // CHECK-NEXT: %[[NOCREATE1:.*]] = acc.nocreate varPtr(%[[ARG1]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<host>]) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg1"}
  // CHECK-NEXT: %[[NOCREATE2:.*]] = acc.nocreate varPtr(%[[ARG2]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<host>]) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg2"}
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[NOCREATE1]], %[[NOCREATE2]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<host>]) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[NOCREATE2]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<host>]) {dataClause = #acc<data_clause acc_no_create>, name = "arg2"}
  // CHECK-NEXT: acc.delete accPtr(%[[NOCREATE1]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<host>]) {dataClause = #acc<data_clause acc_no_create>, name = "arg1"}

#pragma acc parallel loop present(arg1)
  for(unsigned I = 0; I < 5; ++I);
  // CHECK-NEXT: %[[PRESENT1:.*]] = acc.present varPtr(%[[ARG1]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg1"}
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[PRESENT1]] : !cir.ptr<!cir.ptr<!s32i>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[PRESENT1]] : !cir.ptr<!cir.ptr<!s32i>>) {dataClause = #acc<data_clause acc_present>, name = "arg1"}

#pragma acc serial loop present(arg2)
  for(unsigned I = 0; I < 5; ++I);
  // CHECK-NEXT: %[[PRESENT2:.*]] = acc.present varPtr(%[[ARG2]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg2"}
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[PRESENT2]] : !cir.ptr<!cir.ptr<!s32i>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[PRESENT2]] : !cir.ptr<!cir.ptr<!s32i>>) {dataClause = #acc<data_clause acc_present>, name = "arg2"}

#pragma acc kernels loop present(arg1, arg2) device_type(host) async
  for(unsigned I = 0; I < 5; ++I);
  // CHECK-NEXT: %[[PRESENT1:.*]] = acc.present varPtr(%[[ARG1]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<host>]) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg1"}
  // CHECK-NEXT: %[[PRESENT2:.*]] = acc.present varPtr(%[[ARG2]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<host>]) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg2"}
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[PRESENT1]], %[[PRESENT2]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<host>]) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[PRESENT2]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<host>]) {dataClause = #acc<data_clause acc_present>, name = "arg2"}
  // CHECK-NEXT: acc.delete accPtr(%[[PRESENT1]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<host>]) {dataClause = #acc<data_clause acc_present>, name = "arg1"}

#pragma acc parallel loop attach(arg1)
  for(unsigned I = 0; I < 5; ++I);
  // CHECK-NEXT: %[[ATTACH1:.*]] = acc.attach varPtr(%[[ARG1]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg1"}
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[ATTACH1]] : !cir.ptr<!cir.ptr<!s32i>>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.detach accPtr(%[[ATTACH1]] : !cir.ptr<!cir.ptr<!s32i>>) {dataClause = #acc<data_clause acc_attach>, name = "arg1"}

#pragma acc serial loop attach(arg2)
  for(unsigned I = 0; I < 5; ++I);
  // CHECK-NEXT: %[[ATTACH2:.*]] = acc.attach varPtr(%[[ARG2]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg2"}
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[ATTACH2]] : !cir.ptr<!cir.ptr<!s32i>>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.detach accPtr(%[[ATTACH2]] : !cir.ptr<!cir.ptr<!s32i>>) {dataClause = #acc<data_clause acc_attach>, name = "arg2"}

#pragma acc kernels loop attach(arg1, arg2) device_type(host) async
  for(unsigned I = 0; I < 5; ++I);
  // CHECK-NEXT: %[[ATTACH1:.*]] = acc.attach varPtr(%[[ARG1]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<host>]) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg1"}
  // CHECK-NEXT: %[[ATTACH2:.*]] = acc.attach varPtr(%[[ARG2]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<host>]) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg2"}
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[ATTACH1]], %[[ATTACH2]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<host>]) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.detach accPtr(%[[ATTACH2]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<host>]) {dataClause = #acc<data_clause acc_attach>, name = "arg2"}
  // CHECK-NEXT: acc.detach accPtr(%[[ATTACH1]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<host>]) {dataClause = #acc<data_clause acc_attach>, name = "arg1"}

  // Checking the automatic-addition of parallelism clauses.
#pragma acc parallel loop
    for(unsigned I = 0; I < 5; ++I);
  // CHECK-NEXT: acc.parallel combined(loop) {
  // CHECK-NEXT:  acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {independent = [#acc.device_type<none>]} loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc kernels loop
    for(unsigned I = 0; I < 5; ++I);
  // CHECK-NEXT: acc.kernels combined(loop) {
  // CHECK-NEXT:  acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {auto_ = [#acc.device_type<none>]} loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc serial loop
    for(unsigned I = 0; I < 5; ++I);
  // CHECK-NEXT: acc.serial combined(loop) {
  // CHECK-NEXT:  acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {seq = [#acc.device_type<none>]} loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial loop worker
    for(unsigned I = 0; I < 5; ++I);
  // CHECK-NEXT: acc.serial combined(loop) {
  // CHECK-NEXT:  acc.loop combined(serial) worker {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {auto_ = [#acc.device_type<none>]} loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial loop vector
    for(unsigned I = 0; I < 5; ++I);
  // CHECK-NEXT: acc.serial combined(loop) {
  // CHECK-NEXT:  acc.loop combined(serial) vector {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {auto_ = [#acc.device_type<none>]} loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial loop gang
    for(unsigned I = 0; I < 5; ++I);
  // CHECK-NEXT: acc.serial combined(loop) {
  // CHECK-NEXT:  acc.loop combined(serial) gang {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {auto_ = [#acc.device_type<none>]} loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
}
