// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s
void acc_data(int parmVar, int *ptrParmVar) {
  // CHECK: cir.func{{.*}} @acc_data(%[[ARG:.*]]: !s32i{{.*}}, %[[PTRARG:.*]]: !cir.ptr<!s32i>{{.*}}) {
  // CHECK-NEXT: %[[PARM:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["parmVar", init]
  // CHECK-NEXT: %[[PTRPARM:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["ptrParmVar", init]
  // CHECK-NEXT: cir.store %[[ARG]], %[[PARM]] : !s32i, !cir.ptr<!s32i>
  // CHECK-NEXT: cir.store %[[PTRARG]], %[[PTRPARM]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>

#pragma acc enter data copyin(parmVar)
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "parmVar", structured = false}
  // CHECK-NEXT: acc.enter_data dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>)

#pragma acc enter data copyin(readonly, alwaysin: parmVar)
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier readonly,alwaysin>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.enter_data dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>)

#pragma acc enter data copyin(readonly, alwaysin: parmVar) async
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) async -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier readonly,alwaysin>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.enter_data async dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>)

#pragma acc enter data async copyin(readonly, alwaysin: parmVar)
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) async -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier readonly,alwaysin>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.enter_data async dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>)

#pragma acc enter data copyin(readonly, alwaysin: parmVar) async(parmVar)
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[PARM_CAST:.*]] = builtin.unrealized_conversion_cast %[[PARM_LOAD]]
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) async(%[[PARM_CAST]] : si32) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier readonly,alwaysin>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.enter_data async(%[[PARM_CAST]] : si32) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>)

#pragma acc enter data async(parmVar) copyin(readonly, alwaysin: parmVar)
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[PARM_CAST:.*]] = builtin.unrealized_conversion_cast %[[PARM_LOAD]]
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) async(%[[PARM_CAST]] : si32) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier readonly,alwaysin>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.enter_data async(%[[PARM_CAST]] : si32) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>)

#pragma acc enter data create(parmVar)
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "parmVar", structured = false}
  // CHECK-NEXT: acc.enter_data dataOperands(%[[CREATE1]] : !cir.ptr<!s32i>)

#pragma acc enter data create(zero: parmVar)
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier zero>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.enter_data dataOperands(%[[CREATE1]] : !cir.ptr<!s32i>)

#pragma acc enter data create(zero: parmVar) async
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) async -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier zero>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.enter_data async dataOperands(%[[CREATE1]] : !cir.ptr<!s32i>)

#pragma acc enter data create(zero: parmVar) async(parmVar)
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[PARM_CAST:.*]] = builtin.unrealized_conversion_cast %[[PARM_LOAD]]
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) async(%[[PARM_CAST]] : si32) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier zero>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.enter_data async(%[[PARM_CAST]] : si32) dataOperands(%[[CREATE1]] : !cir.ptr<!s32i>)

#pragma acc enter data attach(ptrParmVar)
  // CHECK-NEXT: %[[ATTACH1:.*]] = acc.attach varPtr(%[[PTRPARM]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "ptrParmVar", structured = false}
  // CHECK-NEXT: acc.enter_data dataOperands(%[[ATTACH1]] : !cir.ptr<!cir.ptr<!s32i>>)

#pragma acc enter data attach(ptrParmVar) async
  // CHECK-NEXT: %[[ATTACH1:.*]] = acc.attach varPtr(%[[PTRPARM]] : !cir.ptr<!cir.ptr<!s32i>>) async -> !cir.ptr<!cir.ptr<!s32i>> {name = "ptrParmVar", structured = false}
  // CHECK-NEXT: acc.enter_data async dataOperands(%[[ATTACH1]] : !cir.ptr<!cir.ptr<!s32i>>)

#pragma acc enter data attach(ptrParmVar) async(parmVar)
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[PARM_CAST:.*]] = builtin.unrealized_conversion_cast %[[PARM_LOAD]]
  // CHECK-NEXT: %[[ATTACH1:.*]] = acc.attach varPtr(%[[PTRPARM]] : !cir.ptr<!cir.ptr<!s32i>>) async(%[[PARM_CAST]] : si32) -> !cir.ptr<!cir.ptr<!s32i>> {name = "ptrParmVar", structured = false}
  // CHECK-NEXT: acc.enter_data async(%[[PARM_CAST]] : si32) dataOperands(%[[ATTACH1]] : !cir.ptr<!cir.ptr<!s32i>>)

#pragma acc enter data if (parmVar == 1) copyin(parmVar)
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[ONE_CONST:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[CMP:.*]] = cir.cmp(eq, %[[PARM_LOAD]], %[[ONE_CONST]])
  // CHECK-NEXT: %[[CMP_CAST:.*]] = builtin.unrealized_conversion_cast %[[CMP]]
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "parmVar", structured = false}
  // CHECK-NEXT: acc.enter_data if(%[[CMP_CAST]]) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>)

#pragma acc enter data async if (parmVar == 1) copyin(parmVar)
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[ONE_CONST:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[CMP:.*]] = cir.cmp(eq, %[[PARM_LOAD]], %[[ONE_CONST]])
  // CHECK-NEXT: %[[CMP_CAST:.*]] = builtin.unrealized_conversion_cast %[[CMP]]
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) async -> !cir.ptr<!s32i> {name = "parmVar", structured = false}
  // CHECK-NEXT: acc.enter_data if(%[[CMP_CAST]]) async dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>)

#pragma acc enter data if (parmVar == 1) async(parmVar) copyin(parmVar)
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[ONE_CONST:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[CMP:.*]] = cir.cmp(eq, %[[PARM_LOAD]], %[[ONE_CONST]])
  // CHECK-NEXT: %[[CMP_CAST:.*]] = builtin.unrealized_conversion_cast %[[CMP]]
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[PARM_CAST:.*]] = builtin.unrealized_conversion_cast %[[PARM_LOAD]]
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) async(%[[PARM_CAST]] : si32) -> !cir.ptr<!s32i> {name = "parmVar", structured = false}
  // CHECK-NEXT: acc.enter_data if(%[[CMP_CAST]]) async(%[[PARM_CAST]] : si32) dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>)

#pragma acc enter data wait create(parmVar)
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "parmVar", structured = false}
  // CHECK-NEXT: acc.enter_data wait dataOperands(%[[CREATE1]] : !cir.ptr<!s32i>)

#pragma acc enter data wait(1) create(parmVar)
  // CHECK-NEXT: %[[ONE_CONST:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_CONST]]
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "parmVar", structured = false}
  // CHECK-NEXT: acc.enter_data wait(%[[ONE_CAST]] : si32) dataOperands(%[[CREATE1]] : !cir.ptr<!s32i>)

#pragma acc enter data wait(parmVar, 1, 2) create(parmVar)
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[PARM_CAST:.*]] = builtin.unrealized_conversion_cast %[[PARM_LOAD]]
  // CHECK-NEXT: %[[ONE_CONST:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_CONST]]
  // CHECK-NEXT: %[[TWO_CONST:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_CONST]]
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "parmVar", structured = false}
  // CHECK-NEXT: acc.enter_data wait(%[[PARM_CAST]], %[[ONE_CAST]], %[[TWO_CAST]] : si32, si32, si32) dataOperands(%[[CREATE1]] : !cir.ptr<!s32i>)

#pragma acc enter data wait(devnum: parmVar: 1, 2) create(parmVar)
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[PARM_CAST:.*]] = builtin.unrealized_conversion_cast %[[PARM_LOAD]]
  // CHECK-NEXT: %[[ONE_CONST:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_CONST]]
  // CHECK-NEXT: %[[TWO_CONST:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_CONST]]
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "parmVar", structured = false}
  // CHECK-NEXT: acc.enter_data wait_devnum(%[[PARM_CAST]] : si32) wait(%[[ONE_CAST]], %[[TWO_CAST]] : si32, si32) dataOperands(%[[CREATE1]] : !cir.ptr<!s32i>)

}
