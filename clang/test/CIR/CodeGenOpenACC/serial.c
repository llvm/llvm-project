// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

void acc_serial(int cond) {
  // CHECK: cir.func{{.*}} @acc_serial(%[[ARG:.*]]: !s32i{{.*}}) {
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
  // CHECK-NEXT: %[[CAST:.*]] = cir.cast int_to_bool %[[INT]]
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
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[BOOL_CAST:.*]] = cir.cast int_to_bool %[[COND_LOAD]] : !s32i -> !cir.bool
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[BOOL_CAST]] : !cir.bool to i1
  // CHECK-NEXT: acc.serial self(%[[CONV_CAST]]) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial self(0)
  {}
  // CHECK-NEXT: %[[ZERO_LITERAL:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK-NEXT: %[[BOOL_CAST:.*]] = cir.cast int_to_bool %[[ZERO_LITERAL]] : !s32i -> !cir.bool
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[BOOL_CAST]] : !cir.bool to i1
  // CHECK-NEXT: acc.serial self(%[[CONV_CAST]]) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial if(cond)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[BOOL_CAST:.*]] = cir.cast int_to_bool %[[COND_LOAD]] : !s32i -> !cir.bool
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[BOOL_CAST]] : !cir.bool to i1
  // CHECK-NEXT: acc.serial if(%[[CONV_CAST]]) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial if(1)
  {}
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[BOOL_CAST:.*]] = cir.cast int_to_bool %[[ONE_LITERAL]] : !s32i -> !cir.bool
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[BOOL_CAST]] : !cir.bool to i1
  // CHECK-NEXT: acc.serial if(%[[CONV_CAST]]) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial if(cond == 1)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[EQ_RES:.*]] = cir.cmp(eq, %[[COND_LOAD]], %[[ONE_LITERAL]]) : !s32i, !cir.bool
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[EQ_RES]] : !cir.bool to i1
  // CHECK-NEXT: acc.serial if(%[[CONV_CAST]]) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial if(cond == 1) self(cond == 2)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[EQ_RES_IF:.*]] = cir.cmp(eq, %[[COND_LOAD]], %[[ONE_LITERAL]]) : !s32i, !cir.bool
  // CHECK-NEXT: %[[CONV_CAST_IF:.*]] = builtin.unrealized_conversion_cast %[[EQ_RES_IF]] : !cir.bool to i1
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[EQ_RES_SELF:.*]] = cir.cmp(eq, %[[COND_LOAD]], %[[TWO_LITERAL]]) : !s32i, !cir.bool
  // CHECK-NEXT: %[[CONV_CAST_SELF:.*]] = builtin.unrealized_conversion_cast %[[EQ_RES_SELF]] : !cir.bool to i1
  // CHECK-NEXT: acc.serial self(%[[CONV_CAST_SELF]]) if(%[[CONV_CAST_IF]]) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial async
  {}
  // CHECK-NEXT: acc.serial async {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial async(cond)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.serial async(%[[CONV_CAST]] : si32) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial async device_type(nvidia, radeon) async
  {}
  // CHECK-NEXT: acc.serial async([#acc.device_type<none>, #acc.device_type<nvidia>, #acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: }  loc

#pragma acc serial async(3) device_type(nvidia, radeon) async(cond)
  {}
  // CHECK-NEXT: %[[THREE_LITERAL:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.serial async(%[[THREE_CAST]] : si32, %[[CONV_CAST]] : si32 [#acc.device_type<nvidia>], %[[CONV_CAST]] : si32 [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial async device_type(nvidia, radeon) async(cond)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.serial async([#acc.device_type<none>], %[[CONV_CAST]] : si32 [#acc.device_type<nvidia>], %[[CONV_CAST]] : si32 [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial async(3) device_type(nvidia, radeon) async
  {}
  // CHECK-NEXT: %[[THREE_LITERAL:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.serial async([#acc.device_type<nvidia>, #acc.device_type<radeon>], %[[THREE_CAST]] : si32) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial wait
  {}
  // CHECK-NEXT: acc.serial wait {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial wait device_type(nvidia) wait
  {}
  // CHECK-NEXT: acc.serial wait([#acc.device_type<none>, #acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial wait(1) device_type(nvidia) wait
  {}
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.serial wait([#acc.device_type<nvidia>], {%[[ONE_CAST]] : si32}) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial wait device_type(nvidia) wait(1)
  {}
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.serial wait([#acc.device_type<none>], {%[[ONE_CAST]] : si32} [#acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial wait(1) device_type(nvidia) wait(1)
  {}
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL2:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL2]] : !s32i to si32
  // CHECK-NEXT: acc.serial wait({%[[ONE_CAST]] : si32}, {%[[ONE_CAST2]] : si32} [#acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial wait(devnum: cond : 1)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.serial wait({devnum: %[[CONV_CAST]] : si32, %[[ONE_CAST]] : si32}) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial wait(devnum: cond : 1) device_type(nvidia) wait(devnum: cond : 1)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST2:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.serial wait({devnum: %[[CONV_CAST]] : si32, %[[ONE_CAST]] : si32}, {devnum: %[[CONV_CAST2]] : si32, %[[ONE_CAST2]] : si32} [#acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial wait(devnum: cond : 1, 2)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.serial wait({devnum: %[[CONV_CAST]] : si32, %[[ONE_CAST]] : si32, %[[TWO_CAST]] : si32}) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial wait(devnum: cond : 1, 2) device_type(nvidia, radeon) wait(devnum: cond : 1, 2)
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
  // CHECK-NEXT: acc.serial wait({devnum: %[[CONV_CAST]] : si32, %[[ONE_CAST]] : si32, %[[TWO_CAST]] : si32}, {devnum: %[[CONV_CAST2]] : si32, %[[ONE_CAST2]] : si32, %[[TWO_CAST2]] : si32} [#acc.device_type<nvidia>], {devnum: %[[CONV_CAST2]] : si32, %[[ONE_CAST2]] : si32, %[[TWO_CAST2]] : si32} [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial wait(cond,  1)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.serial wait({%[[CONV_CAST]] : si32, %[[ONE_CAST]] : si32}) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial wait(queues: cond,  1) device_type(radeon)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.serial wait({%[[CONV_CAST]] : si32, %[[ONE_CAST]] : si32}) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

  // CHECK-NEXT: cir.return
}

void acc_serial_data_clauses(int *arg1, int *arg2) {
  // CHECK: cir.func{{.*}} @acc_serial_data_clauses(%[[ARG1_PARAM:.*]]: !cir.ptr<!s32i>{{.*}}, %[[ARG2_PARAM:.*]]: !cir.ptr<!s32i>{{.*}}) {
  // CHECK-NEXT: %[[ARG1:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arg1", init]
  // CHECK-NEXT: %[[ARG2:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arg2", init]
  // CHECK-NEXT: cir.store %[[ARG1_PARAM]], %[[ARG1]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CHECK-NEXT: cir.store %[[ARG2_PARAM]], %[[ARG2]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>

#pragma acc serial deviceptr(arg1)
  ;
  // CHECK-NEXT: %[[DEVPTR1:.*]] = acc.deviceptr varPtr(%[[ARG1]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg1"}
  // CHECK-NEXT: acc.serial dataOperands(%[[DEVPTR1]] : !cir.ptr<!cir.ptr<!s32i>>) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial deviceptr(arg1, arg2)
  ;
  // CHECK-NEXT: %[[DEVPTR1:.*]] = acc.deviceptr varPtr(%[[ARG1]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg1"}
  // CHECK-NEXT: %[[DEVPTR2:.*]] = acc.deviceptr varPtr(%[[ARG2]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg2"}
  // CHECK-NEXT: acc.serial dataOperands(%[[DEVPTR1]], %[[DEVPTR2]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!s32i>>) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial deviceptr(arg1) async
  ;
  // CHECK-NEXT: %[[DEVPTR1:.*]] = acc.deviceptr varPtr(%[[ARG1]] : !cir.ptr<!cir.ptr<!s32i>>) async -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg1"}
  // CHECK-NEXT: acc.serial dataOperands(%[[DEVPTR1]] : !cir.ptr<!cir.ptr<!s32i>>) async {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial deviceptr(arg1, arg2) device_type(radeon) async
  ;
  // CHECK-NEXT: %[[DEVPTR1:.*]] = acc.deviceptr varPtr(%[[ARG1]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<radeon>]) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg1"}
  // CHECK-NEXT: %[[DEVPTR2:.*]] = acc.deviceptr varPtr(%[[ARG2]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<radeon>]) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg2"}
  // CHECK-NEXT: acc.serial dataOperands(%[[DEVPTR1]], %[[DEVPTR2]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial no_create(arg1)
  ;
  // CHECK-NEXT: %[[NOCREATE1:.*]] = acc.nocreate varPtr(%[[ARG1]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg1"}
  // CHECK-NEXT: acc.serial dataOperands(%[[NOCREATE1]] : !cir.ptr<!cir.ptr<!s32i>>) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[NOCREATE1]] : !cir.ptr<!cir.ptr<!s32i>>) {dataClause = #acc<data_clause acc_no_create>, name = "arg1"}
#pragma acc serial no_create(arg1, arg2) device_type(nvidia) async
  ;
  // CHECK-NEXT: %[[NOCREATE1:.*]] = acc.nocreate varPtr(%[[ARG1]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<nvidia>]) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg1"}
  // CHECK-NEXT: %[[NOCREATE2:.*]] = acc.nocreate varPtr(%[[ARG2]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<nvidia>]) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg2"}
  // CHECK-NEXT: acc.serial dataOperands(%[[NOCREATE1]], %[[NOCREATE2]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[NOCREATE2]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<nvidia>]) {dataClause = #acc<data_clause acc_no_create>, name = "arg2"}
  // CHECK-NEXT: acc.delete accPtr(%[[NOCREATE1]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<nvidia>]) {dataClause = #acc<data_clause acc_no_create>, name = "arg1"}

#pragma acc serial present(arg1)
  ;
  // CHECK-NEXT: %[[PRESENT1:.*]] = acc.present varPtr(%[[ARG1]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg1"}
  // CHECK-NEXT: acc.serial dataOperands(%[[PRESENT1]] : !cir.ptr<!cir.ptr<!s32i>>) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[PRESENT1]] : !cir.ptr<!cir.ptr<!s32i>>) {dataClause = #acc<data_clause acc_present>, name = "arg1"}
#pragma acc serial present(arg1, arg2) device_type(nvidia) async
  ;
  // CHECK-NEXT: %[[PRESENT1:.*]] = acc.present varPtr(%[[ARG1]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<nvidia>]) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg1"}
  // CHECK-NEXT: %[[PRESENT2:.*]] = acc.present varPtr(%[[ARG2]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<nvidia>]) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg2"}
  // CHECK-NEXT: acc.serial dataOperands(%[[PRESENT1]], %[[PRESENT2]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[PRESENT2]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<nvidia>]) {dataClause = #acc<data_clause acc_present>, name = "arg2"}
  // CHECK-NEXT: acc.delete accPtr(%[[PRESENT1]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<nvidia>]) {dataClause = #acc<data_clause acc_present>, name = "arg1"}

#pragma acc serial attach(arg1)
  ;
  // CHECK-NEXT: %[[ATTACH1:.*]] = acc.attach varPtr(%[[ARG1]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg1"}
  // CHECK-NEXT: acc.serial dataOperands(%[[ATTACH1]] : !cir.ptr<!cir.ptr<!s32i>>) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.detach accPtr(%[[ATTACH1]] : !cir.ptr<!cir.ptr<!s32i>>) {dataClause = #acc<data_clause acc_attach>, name = "arg1"}
#pragma acc serial attach(arg1, arg2) device_type(nvidia) async
  ;
  // CHECK-NEXT: %[[ATTACH1:.*]] = acc.attach varPtr(%[[ARG1]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<nvidia>]) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg1"}
  // CHECK-NEXT: %[[ATTACH2:.*]] = acc.attach varPtr(%[[ARG2]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<nvidia>]) -> !cir.ptr<!cir.ptr<!s32i>> {name = "arg2"}
  // CHECK-NEXT: acc.serial dataOperands(%[[ATTACH1]], %[[ATTACH2]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.detach accPtr(%[[ATTACH2]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<nvidia>]) {dataClause = #acc<data_clause acc_attach>, name = "arg2"}
  // CHECK-NEXT: acc.detach accPtr(%[[ATTACH1]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<nvidia>]) {dataClause = #acc<data_clause acc_attach>, name = "arg1"}
}
