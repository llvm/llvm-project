// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s
void acc_data(int parmVar, int *ptrParmVar) {
  // CHECK: cir.func{{.*}} @acc_data(%[[ARG:.*]]: !s32i{{.*}}, %[[PTRARG:.*]]: !cir.ptr<!s32i>{{.*}}) {
  // CHECK-NEXT: %[[PARM:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["parmVar", init]
  // CHECK-NEXT: %[[PTRPARM:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["ptrParmVar", init]
  // CHECK-NEXT: cir.store %[[ARG]], %[[PARM]] : !s32i, !cir.ptr<!s32i>
  // CHECK-NEXT: cir.store %[[PTRARG]], %[[PTRPARM]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>

#pragma acc exit data copyout(parmVar)
  // CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyout>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.exit_data dataOperands(%[[GDP]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.copyout accPtr(%[[GDP]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {name = "parmVar", structured = false}

#pragma acc exit data copyout(zero, alwaysout: parmVar)
  // CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyout>, modifiers = #acc<data_clause_modifier zero,alwaysout>,  name = "parmVar", structured = false}
  // CHECK-NEXT: acc.exit_data dataOperands(%[[GDP]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.copyout accPtr(%[[GDP]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {modifiers = #acc<data_clause_modifier zero,alwaysout>, name = "parmVar", structured = false}

#pragma acc exit data copyout(zero, alwaysout: parmVar) async
  // CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) async -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyout>, modifiers = #acc<data_clause_modifier zero,alwaysout>,  name = "parmVar", structured = false}
  // CHECK-NEXT: acc.exit_data async dataOperands(%[[GDP]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.copyout accPtr(%[[GDP]] : !cir.ptr<!s32i>) async to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {modifiers = #acc<data_clause_modifier zero,alwaysout>, name = "parmVar", structured = false}

#pragma acc exit data async copyout(zero, alwaysout: parmVar)
  // CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) async -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyout>, modifiers = #acc<data_clause_modifier zero,alwaysout>,  name = "parmVar", structured = false}
  // CHECK-NEXT: acc.exit_data async dataOperands(%[[GDP]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.copyout accPtr(%[[GDP]] : !cir.ptr<!s32i>) async to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {modifiers = #acc<data_clause_modifier zero,alwaysout>, name = "parmVar", structured = false}

#pragma acc exit data finalize copyout(zero, alwaysout: parmVar) async(parmVar)
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[PARM_CAST:.*]] = builtin.unrealized_conversion_cast %[[PARM_LOAD]]
  // CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) async(%[[PARM_CAST]] : si32) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyout>, modifiers = #acc<data_clause_modifier zero,alwaysout>,  name = "parmVar", structured = false}
  // CHECK-NEXT: acc.exit_data async(%[[PARM_CAST]] : si32) dataOperands(%[[GDP]] : !cir.ptr<!s32i>) attributes {finalize}
  // CHECK-NEXT: acc.copyout accPtr(%[[GDP]] : !cir.ptr<!s32i>) async(%[[PARM_CAST]] : si32) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {modifiers = #acc<data_clause_modifier zero,alwaysout>, name = "parmVar", structured = false}

#pragma acc exit data async(parmVar) copyout(zero, alwaysout: parmVar)
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[PARM_CAST:.*]] = builtin.unrealized_conversion_cast %[[PARM_LOAD]]
  // CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) async(%[[PARM_CAST]] : si32) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyout>, modifiers = #acc<data_clause_modifier zero,alwaysout>,  name = "parmVar", structured = false}
  // CHECK-NEXT: acc.exit_data async(%[[PARM_CAST]] : si32) dataOperands(%[[GDP]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.copyout accPtr(%[[GDP]] : !cir.ptr<!s32i>) async(%[[PARM_CAST]] : si32) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {modifiers = #acc<data_clause_modifier zero,alwaysout>, name = "parmVar", structured = false}

#pragma acc exit data delete(parmVar) finalize
  // CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_delete>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.exit_data dataOperands(%[[GDP]] : !cir.ptr<!s32i>) attributes {finalize}
  // CHECK-NEXT: acc.delete accPtr(%[[GDP]] : !cir.ptr<!s32i>) {name = "parmVar", structured = false}

#pragma acc exit data delete(parmVar) async(parmVar)
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[PARM_CAST:.*]] = builtin.unrealized_conversion_cast %[[PARM_LOAD]]
  // CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>)  async(%[[PARM_CAST]] : si32) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_delete>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.exit_data async(%[[PARM_CAST]] : si32) dataOperands(%[[GDP]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.delete accPtr(%[[GDP]] : !cir.ptr<!s32i>) async(%[[PARM_CAST]] : si32) {name = "parmVar", structured = false}

#pragma acc exit data detach(ptrParmVar)
  // CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[PTRPARM]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {dataClause = #acc<data_clause acc_detach>, name = "ptrParmVar", structured = false}
  // CHECK-NEXT: acc.exit_data dataOperands(%[[GDP]] : !cir.ptr<!cir.ptr<!s32i>>)
  // CHECK-NEXT: acc.detach accPtr(%[[GDP]] : !cir.ptr<!cir.ptr<!s32i>>) {name = "ptrParmVar", structured = false}

#pragma acc exit data detach(ptrParmVar) async
  // CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[PTRPARM]] : !cir.ptr<!cir.ptr<!s32i>>) async -> !cir.ptr<!cir.ptr<!s32i>> {dataClause = #acc<data_clause acc_detach>, name = "ptrParmVar", structured = false}
  // CHECK-NEXT: acc.exit_data async dataOperands(%[[GDP]] : !cir.ptr<!cir.ptr<!s32i>>)
  // CHECK-NEXT: acc.detach accPtr(%[[GDP]] : !cir.ptr<!cir.ptr<!s32i>>) async {name = "ptrParmVar", structured = false}

#pragma acc exit data detach(ptrParmVar) async(parmVar) finalize
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[PARM_CAST:.*]] = builtin.unrealized_conversion_cast %[[PARM_LOAD]]
  // CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[PTRPARM]] : !cir.ptr<!cir.ptr<!s32i>>) async(%[[PARM_CAST]] : si32) -> !cir.ptr<!cir.ptr<!s32i>> {dataClause = #acc<data_clause acc_detach>, name = "ptrParmVar", structured = false}
  // CHECK-NEXT: acc.exit_data async(%[[PARM_CAST]] : si32) dataOperands(%[[GDP]] : !cir.ptr<!cir.ptr<!s32i>>) attributes {finalize}
  // CHECK-NEXT: acc.detach accPtr(%[[GDP]] : !cir.ptr<!cir.ptr<!s32i>>) async(%[[PARM_CAST]] : si32) {name = "ptrParmVar", structured = false}

#pragma acc exit data if (parmVar == 1) copyout(parmVar)
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[ONE_CONST:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[CMP:.*]] = cir.cmp(eq, %[[PARM_LOAD]], %[[ONE_CONST]])
  // CHECK-NEXT: %[[CMP_CAST:.*]] = builtin.unrealized_conversion_cast %[[CMP]]
  // CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyout>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.exit_data if(%[[CMP_CAST]]) dataOperands(%[[GDP]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.copyout accPtr(%[[GDP]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {name = "parmVar", structured = false}

#pragma acc exit data async if (parmVar == 1) copyout(parmVar)
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[ONE_CONST:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[CMP:.*]] = cir.cmp(eq, %[[PARM_LOAD]], %[[ONE_CONST]])
  // CHECK-NEXT: %[[CMP_CAST:.*]] = builtin.unrealized_conversion_cast %[[CMP]]
  // CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) async -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyout>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.exit_data if(%[[CMP_CAST]]) async dataOperands(%[[GDP]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.copyout accPtr(%[[GDP]] : !cir.ptr<!s32i>) async to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {name = "parmVar", structured = false}

#pragma acc exit data if (parmVar == 1) async(parmVar) copyout(parmVar)
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[ONE_CONST:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[CMP:.*]] = cir.cmp(eq, %[[PARM_LOAD]], %[[ONE_CONST]])
  // CHECK-NEXT: %[[CMP_CAST:.*]] = builtin.unrealized_conversion_cast %[[CMP]]
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[PARM_CAST:.*]] = builtin.unrealized_conversion_cast %[[PARM_LOAD]]
  // CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) async(%[[PARM_CAST]] : si32) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyout>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.exit_data if(%[[CMP_CAST]]) async(%[[PARM_CAST]] : si32) dataOperands(%[[GDP]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.copyout accPtr(%[[GDP]] : !cir.ptr<!s32i>) async(%[[PARM_CAST]] : si32) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {name = "parmVar", structured = false}

#pragma acc exit data wait delete(parmVar)
  // CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_delete>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.exit_data wait dataOperands(%[[GDP]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.delete accPtr(%[[GDP]] : !cir.ptr<!s32i>) {name = "parmVar", structured = false}

#pragma acc exit data wait(1) delete(parmVar)
  // CHECK-NEXT: %[[ONE_CONST:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_CONST]]
  // CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_delete>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.exit_data wait(%[[ONE_CAST]] : si32) dataOperands(%[[GDP]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.delete accPtr(%[[GDP]] : !cir.ptr<!s32i>) {name = "parmVar", structured = false}

#pragma acc exit data wait(parmVar, 1, 2) delete(parmVar) finalize
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[PARM_CAST:.*]] = builtin.unrealized_conversion_cast %[[PARM_LOAD]]
  // CHECK-NEXT: %[[ONE_CONST:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_CONST]]
  // CHECK-NEXT: %[[TWO_CONST:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_CONST]]
  // CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_delete>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.exit_data wait(%[[PARM_CAST]], %[[ONE_CAST]], %[[TWO_CAST]] : si32, si32, si32) dataOperands(%[[GDP]] : !cir.ptr<!s32i>) attributes {finalize}
  // CHECK-NEXT: acc.delete accPtr(%[[GDP]] : !cir.ptr<!s32i>) {name = "parmVar", structured = false}

#pragma acc exit data wait(devnum: parmVar: 1, 2) delete(parmVar)
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[PARM_CAST:.*]] = builtin.unrealized_conversion_cast %[[PARM_LOAD]]
  // CHECK-NEXT: %[[ONE_CONST:.*]] = cir.const #cir.int<1> : !s32i
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_CONST]]
  // CHECK-NEXT: %[[TWO_CONST:.*]] = cir.const #cir.int<2> : !s32i
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_CONST]]
  // CHECK-NEXT: %[[GDP:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_delete>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.exit_data wait_devnum(%[[PARM_CAST]] : si32) wait(%[[ONE_CAST]], %[[TWO_CAST]] : si32, si32) dataOperands(%[[GDP]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.delete accPtr(%[[GDP]] : !cir.ptr<!s32i>) {name = "parmVar", structured = false}
}
