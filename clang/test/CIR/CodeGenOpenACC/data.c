// RUN: %clang_cc1 -fopenacc -emit-cir -fclangir %s -o - | FileCheck %s

void acc_data(int cond) {
  // CHECK: cir.func{{.*}} @acc_data(%[[ARG:.*]]: !s32i{{.*}}) {
  // CHECK-NEXT: %[[COND:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["cond", init]

  int *ptr;
  // CHECK-NEXT: %[[PTR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["ptr"]
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
  // CHECK-NEXT: acc.data async {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) async(cond)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.data async(%[[CONV_CAST]] : si32) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) async device_type(nvidia, radeon) async
  {}
  // CHECK-NEXT: acc.data async([#acc.device_type<none>, #acc.device_type<nvidia>, #acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) async(3) device_type(nvidia, radeon) async(cond)
  {}
  // CHECK-NEXT: %[[THREE_LITERAL:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.data async(%[[THREE_CAST]] : si32, %[[CONV_CAST]] : si32 [#acc.device_type<nvidia>], %[[CONV_CAST]] : si32 [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) async device_type(nvidia, radeon) async(cond)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: acc.data async([#acc.device_type<none>], %[[CONV_CAST]] : si32 [#acc.device_type<nvidia>], %[[CONV_CAST]] : si32 [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) async(3) device_type(nvidia, radeon) async
  {}
  // CHECK-NEXT: %[[THREE_LITERAL:.*]] = cir.const #cir.int<3> : !s32i
  // CHECK-NEXT: %[[THREE_CAST:.*]] = builtin.unrealized_conversion_cast %[[THREE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.data async([#acc.device_type<nvidia>, #acc.device_type<radeon>], %[[THREE_CAST]] : si32) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) if(cond)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[BOOL_CAST:.*]] = cir.cast int_to_bool %[[COND_LOAD]] : !s32i -> !cir.bool
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[BOOL_CAST]] : !cir.bool to i1
  // CHECK-NEXT: acc.data if(%[[CONV_CAST]]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) if(1)
  {}
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[BOOL_CAST:.*]] = cir.cast int_to_bool %[[ONE_LITERAL]] : !s32i -> !cir.bool
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[BOOL_CAST]] : !cir.bool to i1
  // CHECK-NEXT: acc.data if(%[[CONV_CAST]]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) if(cond == 1)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[EQ_RES:.*]] = cir.cmp(eq, %[[COND_LOAD]], %[[ONE_LITERAL]]) : !s32i, !cir.bool
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[EQ_RES]] : !cir.bool to i1
  // CHECK-NEXT: acc.data if(%[[CONV_CAST]]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) wait
  {}
  // CHECK-NEXT: acc.data wait {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) wait device_type(nvidia) wait
  {}
  // CHECK-NEXT: acc.data wait([#acc.device_type<none>, #acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) wait(1) device_type(nvidia) wait
  {}
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.data wait([#acc.device_type<nvidia>], {%[[ONE_CAST]] : si32}) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) wait device_type(nvidia) wait(1)
  {}
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.data wait([#acc.device_type<none>], {%[[ONE_CAST]] : si32} [#acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) wait(1) device_type(nvidia) wait(1)
  {}
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL2:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL2]] : !s32i to si32
  // CHECK-NEXT: acc.data wait({%[[ONE_CAST]] : si32}, {%[[ONE_CAST2]] : si32} [#acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) wait(devnum: cond : 1)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.data wait({devnum: %[[CONV_CAST]] : si32, %[[ONE_CAST]] : si32}) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) wait(devnum: cond : 1) device_type(nvidia) wait(devnum: cond : 1)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST2:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST2:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.data wait({devnum: %[[CONV_CAST]] : si32, %[[ONE_CAST]] : si32}, {devnum: %[[CONV_CAST2]] : si32, %[[ONE_CAST2]] : si32} [#acc.device_type<nvidia>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) wait(devnum: cond : 1, 2)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: %[[TWO_LITERAL:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.data wait({devnum: %[[CONV_CAST]] : si32, %[[ONE_CAST]] : si32, %[[TWO_CAST]] : si32}) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) wait(devnum: cond : 1, 2) device_type(nvidia, radeon) wait(devnum: cond : 1, 2)
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
  // CHECK-NEXT: acc.data wait({devnum: %[[CONV_CAST]] : si32, %[[ONE_CAST]] : si32, %[[TWO_CAST]] : si32}, {devnum: %[[CONV_CAST2]] : si32, %[[ONE_CAST2]] : si32, %[[TWO_CAST2]] : si32} [#acc.device_type<nvidia>], {devnum: %[[CONV_CAST2]] : si32, %[[ONE_CAST2]] : si32, %[[TWO_CAST2]] : si32} [#acc.device_type<radeon>]) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) wait(cond,  1)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.data wait({%[[CONV_CAST]] : si32, %[[ONE_CAST]] : si32}) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data default(none) wait(queues: cond,  1) device_type(radeon)
  {}
  // CHECK-NEXT: %[[COND_LOAD:.*]] = cir.load{{.*}} %[[COND]] : !cir.ptr<!s32i>, !s32i
  // CHECK-NEXT: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[COND_LOAD]] : !s32i to si32
  // CHECK-NEXT: %[[ONE_LITERAL:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_LITERAL]] : !s32i to si32
  // CHECK-NEXT: acc.data wait({%[[CONV_CAST]] : si32, %[[ONE_CAST]] : si32}) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: attributes {defaultAttr = #acc<defaultvalue none>}

#pragma acc data deviceptr(ptr)
  {}
  // CHECK-NEXT: %[[DEV_PTR:.*]] = acc.deviceptr varPtr(%[[PTR]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "ptr"}
  // CHECK-NEXT: acc.data dataOperands(%[[DEV_PTR]] : !cir.ptr<!cir.ptr<!s32i>>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
#pragma acc data deviceptr(ptr) device_type(radeon) async
  {}
  // CHECK-NEXT: %[[DEV_PTR:.*]] = acc.deviceptr varPtr(%[[PTR]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<radeon>]) -> !cir.ptr<!cir.ptr<!s32i>> {name = "ptr"}
  // CHECK-NEXT: acc.data async([#acc.device_type<radeon>]) dataOperands(%[[DEV_PTR]] : !cir.ptr<!cir.ptr<!s32i>>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc data present(cond)
  {}
  // CHECK-NEXT: %[[PRESENT:.*]] = acc.present varPtr(%[[COND]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "cond"}
  // CHECK-NEXT: acc.data dataOperands(%[[PRESENT]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[PRESENT]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_present>, name = "cond"}

#pragma acc data present(cond) device_type(radeon) async
  {}
  // CHECK-NEXT: %[[PRESENT:.*]] = acc.present varPtr(%[[COND]] : !cir.ptr<!s32i>) async([#acc.device_type<radeon>]) -> !cir.ptr<!s32i> {name = "cond"}
  // CHECK-NEXT: acc.data async([#acc.device_type<radeon>]) dataOperands(%[[PRESENT]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[PRESENT]] : !cir.ptr<!s32i>) async([#acc.device_type<radeon>]) {dataClause = #acc<data_clause acc_present>, name = "cond"}

#pragma acc data attach(ptr)
  {}
  // CHECK-NEXT: %[[ATTACH:.*]] = acc.attach varPtr(%[[PTR]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "ptr"}
  // CHECK-NEXT: acc.data dataOperands(%[[ATTACH]] : !cir.ptr<!cir.ptr<!s32i>>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.detach accPtr(%[[ATTACH]] : !cir.ptr<!cir.ptr<!s32i>>) {dataClause = #acc<data_clause acc_attach>, name = "ptr"}

#pragma acc data attach(ptr) device_type(radeon) async
  {}
  // CHECK-NEXT: %[[ATTACH:.*]] = acc.attach varPtr(%[[PTR]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<radeon>]) -> !cir.ptr<!cir.ptr<!s32i>> {name = "ptr"}
  // CHECK-NEXT: acc.data async([#acc.device_type<radeon>]) dataOperands(%[[ATTACH]] : !cir.ptr<!cir.ptr<!s32i>>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.detach accPtr(%[[ATTACH]] : !cir.ptr<!cir.ptr<!s32i>>) async([#acc.device_type<radeon>]) {dataClause = #acc<data_clause acc_attach>, name = "ptr"}

  // CHECK-NEXT: cir.return
}
