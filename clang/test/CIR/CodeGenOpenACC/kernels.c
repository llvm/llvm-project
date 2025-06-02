// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

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
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
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

#pragma acc kernels if(cond)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[BOOL_CAST:.*]] = cir.cast(int_to_bool, %[[COND_LOAD]] : !s32i), !cir.bool
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[BOOL_CAST]] : !cir.bool to i1
  // CHECK-NEXT: acc.kernels if(%[[CONV_CAST]]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels if(1)
  {}
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[BOOL_CAST:.*]] = cir.cast(int_to_bool, %[[ONE_LITERAL]] : !s32i), !cir.bool
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[BOOL_CAST]] : !cir.bool to i1
  // CHECK-NEXT: acc.kernels if(%[[CONV_CAST]]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels if(cond == 1)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[EQ_RES:.*]] = cir.cmp(eq, %[[COND_LOAD]], %[[ONE_LITERAL]]) : !s32i, !cir.bool
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[EQ_RES]] : !cir.bool to i1
  // CHECK-NEXT: acc.kernels if(%[[CONV_CAST]]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels if(cond == 1) self(cond == 2)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[EQ_RES_IF:.*]] = cir.cmp(eq, %[[COND_LOAD]], %[[ONE_LITERAL]]) : !s32i, !cir.bool
  // CHECK-NEXT: %[[CONV_CAST_IF:.*]] = builtin.unrealized_conversion_cast %[[EQ_RES_IF]] : !cir.bool to i1
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[EQ_RES_SELF:.*]] = cir.cmp(eq, %[[COND_LOAD]], %[[TWO_LITERAL]]) : !s32i, !cir.bool
  // CHECK-NEXT: %[[CONV_CAST_SELF:.*]] = builtin.unrealized_conversion_cast %[[EQ_RES_SELF]] : !cir.bool to i1
  // CHECK-NEXT: acc.kernels self(%[[CONV_CAST_SELF]]) if(%[[CONV_CAST_IF]]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels num_workers(cond)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.kernels num_workers(%[[CONV_CAST]] : si32) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels num_workers(cond) device_type(nvidia) num_workers(2u)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !u32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !u32i to ui32
  // CHECK-NEXT: acc.kernels num_workers(%[[CONV_CAST]] : si32, %[[TWO_CAST]] : ui32 [#acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels num_workers(cond) device_type(nvidia, host) num_workers(2) device_type(radeon) num_workers(3)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[THREE_LITERAL:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.kernels num_workers(%[[CONV_CAST]] : si32, %[[TWO_CAST]] : si32 [#acc.device_type<nvidia>], %[[TWO_CAST]] : si32 [#acc.device_type<host>], %[[THREE_CAST]] : si32 [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels num_workers(cond) device_type(nvidia) num_workers(2) device_type(radeon, multicore) num_workers(3)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[THREE_LITERAL:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.kernels num_workers(%[[CONV_CAST]] : si32, %[[TWO_CAST]] : si32 [#acc.device_type<nvidia>], %[[THREE_CAST]] : si32 [#acc.device_type<radeon>], %[[THREE_CAST]] : si32 [#acc.device_type<multicore>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels device_type(nvidia) num_workers(2) device_type(radeon) num_workers(3)
  {}
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[THREE_LITERAL:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.kernels num_workers(%[[TWO_CAST]] : si32 [#acc.device_type<nvidia>], %[[THREE_CAST]] : si32 [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels vector_length(cond)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.kernels vector_length(%[[CONV_CAST]] : si32) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels vector_length(cond) device_type(nvidia) vector_length(2u)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !u32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !u32i to ui32
  // CHECK-NEXT: acc.kernels vector_length(%[[CONV_CAST]] : si32, %[[TWO_CAST]] : ui32 [#acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels vector_length(cond) device_type(nvidia, host) vector_length(2) device_type(radeon) vector_length(3)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[THREE_LITERAL:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.kernels vector_length(%[[CONV_CAST]] : si32, %[[TWO_CAST]] : si32 [#acc.device_type<nvidia>], %[[TWO_CAST]] : si32 [#acc.device_type<host>], %[[THREE_CAST]] : si32 [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels vector_length(cond) device_type(nvidia) vector_length(2) device_type(radeon, multicore) vector_length(3)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[THREE_LITERAL:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.kernels vector_length(%[[CONV_CAST]] : si32, %[[TWO_CAST]] : si32 [#acc.device_type<nvidia>], %[[THREE_CAST]] : si32 [#acc.device_type<radeon>], %[[THREE_CAST]] : si32 [#acc.device_type<multicore>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels device_type(nvidia) vector_length(2) device_type(radeon) vector_length(3)
  {}
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[THREE_LITERAL:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.kernels vector_length(%[[TWO_CAST]] : si32 [#acc.device_type<nvidia>], %[[THREE_CAST]] : si32 [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels async
  {}
  // CHECK-NEXT: acc.kernels async {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels async(cond)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.kernels async(%[[CONV_CAST]] : si32) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels async device_type(nvidia, radeon) async
  {}
  // CHECK-NEXT: acc.kernels async([#acc.device_type<none>, #acc.device_type<nvidia>, #acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels async(3) device_type(nvidia, radeon) async(cond)
  {}
  // CHECK-NEXT: %[[THREE_LITERAL:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.kernels async(%[[THREE_CAST]] : si32, %[[CONV_CAST]] : si32 [#acc.device_type<nvidia>], %[[CONV_CAST]] : si32 [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels async device_type(nvidia, radeon) async(cond)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.kernels async([#acc.device_type<none>], %[[CONV_CAST]] : si32 [#acc.device_type<nvidia>], %[[CONV_CAST]] : si32 [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels async(3) device_type(nvidia, radeon) async
  {}
  // CHECK-NEXT: %[[THREE_LITERAL:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.kernels async([#acc.device_type<nvidia>, #acc.device_type<radeon>], %[[THREE_CAST]] : si32) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels num_gangs(1)
  {}
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.kernels num_gangs({%[[ONE_CAST]] : si32}) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels num_gangs(cond)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.kernels num_gangs({%[[CONV_CAST]] : si32}) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels num_gangs(1) device_type(radeon) num_gangs(cond)
  {}
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.kernels num_gangs({%[[ONE_CAST]] : si32}, {%[[CONV_CAST]] : si32} [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels num_gangs(1) device_type(radeon) num_gangs(6)
  {}
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[SIX_LITERAL:.*]] = cir.const #cir.int<6> : !s32i
  // CHECK-NEXT: %[[SIX_CAST:.*]] = builtin.unrealized_conversion_cast %[[SIX_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.kernels num_gangs({%[[ONE_CAST]] : si32}, {%[[SIX_CAST]] : si32} [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels num_gangs(cond) device_type(radeon, nvidia) num_gangs(4)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[FOUR_LITERAL:.*]] = cir.const #cir.int<4> : !s32i
  // CHECK-NEXT: %[[FOUR_CAST:.*]] = builtin.unrealized_conversion_cast %[[FOUR_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.kernels num_gangs({%[[CONV_CAST]] : si32}, {%[[FOUR_CAST]] : si32} [#acc.device_type<radeon>], {%[[FOUR_CAST]] : si32} [#acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels wait
  {}
  // CHECK-NEXT: acc.kernels wait {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels wait device_type(nvidia) wait
  {}
  // CHECK-NEXT: acc.kernels wait([#acc.device_type<none>, #acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels wait(1) device_type(nvidia) wait
  {}
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.kernels wait([#acc.device_type<nvidia>], {%[[ONE_CAST]] : si32}) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels wait device_type(nvidia) wait(1)
  {}
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.kernels wait([#acc.device_type<none>], {%[[ONE_CAST]] : si32} [#acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels wait(1) device_type(nvidia) wait(1)
  {}
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL2:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL2]] : !s32i to si32
  // CHECK-NEXT: acc.kernels wait({%[[ONE_CAST]] : si32}, {%[[ONE_CAST2]] : si32} [#acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels wait(devnum: cond : 1)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.kernels wait({devnum: %[[CONV_CAST]] : si32, %[[ONE_CAST]] : si32}) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels wait(devnum: cond : 1) device_type(nvidia) wait(devnum: cond : 1)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST2:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.kernels wait({devnum: %[[CONV_CAST]] : si32, %[[ONE_CAST]] : si32}, {devnum: %[[CONV_CAST2]] : si32, %[[ONE_CAST2]] : si32} [#acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels wait(devnum: cond : 1, 2)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.kernels wait({devnum: %[[CONV_CAST]] : si32, %[[ONE_CAST]] : si32, %[[TWO_CAST]] : si32}) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels wait(devnum: cond : 1, 2) device_type(nvidia, radeon) wait(devnum: cond : 1, 2)
  {}
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
  // CHECK-NEXT: acc.kernels wait({devnum: %[[CONV_CAST]] : si32, %[[ONE_CAST]] : si32, %[[TWO_CAST]] : si32}, {devnum: %[[CONV_CAST2]] : si32, %[[ONE_CAST2]] : si32, %[[TWO_CAST2]] : si32} [#acc.device_type<nvidia>], {devnum: %[[CONV_CAST2]] : si32, %[[ONE_CAST2]] : si32, %[[TWO_CAST2]] : si32} [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels wait(cond,  1)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.kernels wait({%[[CONV_CAST]] : si32, %[[ONE_CAST]] : si32}) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc kernels wait(queues: cond,  1) device_type(radeon)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.kernels wait({%[[CONV_CAST]] : si32, %[[ONE_CAST]] : si32}) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

  // CHECK-NEXT: cir.return
}
