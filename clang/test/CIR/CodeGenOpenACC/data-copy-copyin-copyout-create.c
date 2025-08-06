// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

void acc_data(int parmVar) {
  // CHECK: cir.func{{.*}} @acc_data(%[[ARG:.*]]: !s32i{{.*}}) {
  // CHECK-NEXT: %[[PARM:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["parmVar", init]
  int localVar1;
  // CHECK-NEXT: %[[LV1:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["localVar1"]
  // CHECK-NEXT: cir.store %[[ARG]], %[[PARM]]

#pragma acc data copy(parmVar)
  ;
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "parmVar"}
  // CHECK-NEXT: acc.data dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "parmVar"}
#pragma acc data copy(parmVar) copy(localVar1)
  ;
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "parmVar"}
  // CHECK-NEXT: %[[COPYIN2:.*]] = acc.copyin varPtr(%[[LV1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "localVar1"}
  // CHECK-NEXT: acc.data dataOperands(%[[COPYIN1]], %[[COPYIN2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN2]] : !cir.ptr<!s32i>) to varPtr(%[[LV1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "localVar1"}
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "parmVar"}
#pragma acc data copy(parmVar, localVar1)
  ;
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "parmVar"}
  // CHECK-NEXT: %[[COPYIN2:.*]] = acc.copyin varPtr(%[[LV1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, name = "localVar1"}
  // CHECK-NEXT: acc.data dataOperands(%[[COPYIN1]], %[[COPYIN2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN2]] : !cir.ptr<!s32i>) to varPtr(%[[LV1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "localVar1"}
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, name = "parmVar"}

#pragma acc data copy(capture, always, alwaysin, alwaysout:  parmVar)
  ;
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier always,capture>, name = "parmVar"}
  // CHECK-NEXT: acc.data dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier always,capture>, name = "parmVar"}

#pragma acc data copy(capture: parmVar) copy(always, alwaysin, alwaysout:  parmVar)
  ;
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier capture>, name = "parmVar"}
  // CHECK-NEXT: %[[COPYIN2:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier always>, name = "parmVar"}
  // CHECK-NEXT: acc.data dataOperands(%[[COPYIN1]], %[[COPYIN2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN2]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier always>, name = "parmVar"}
  // CHECK-NEXT: acc.copyout accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copy>, modifiers = #acc<data_clause_modifier capture>, name = "parmVar"}

#pragma acc data copyin(parmVar)
  ;
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "parmVar"}
  // CHECK-NEXT: acc.data dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, name = "parmVar"}
#pragma acc data copyin(parmVar) copyin(localVar1)
  ;
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "parmVar"}
  // CHECK-NEXT: %[[COPYIN2:.*]] = acc.copyin varPtr(%[[LV1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "localVar1"}
  // CHECK-NEXT: acc.data dataOperands(%[[COPYIN1]], %[[COPYIN2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN2]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, name = "localVar1"}
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, name = "parmVar"}
#pragma acc data copyin(parmVar, localVar1)
  ;
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "parmVar"}
  // CHECK-NEXT: %[[COPYIN2:.*]] = acc.copyin varPtr(%[[LV1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "localVar1"}
  // CHECK-NEXT: acc.data dataOperands(%[[COPYIN1]], %[[COPYIN2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN2]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, name = "localVar1"}
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, name = "parmVar"}

#pragma acc data copyin(capture, always, alwaysin, readonly:  parmVar)
  ;
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier always,readonly,capture>, name = "parmVar"}
  // CHECK-NEXT: acc.data dataOperands(%[[COPYIN1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier always,readonly,capture>, name = "parmVar"}

#pragma acc data copyin(capture: parmVar) copyin(readonly, always, alwaysin:  parmVar)
  ;
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier capture>, name = "parmVar"}
  // CHECK-NEXT: %[[COPYIN2:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier always,readonly>, name = "parmVar"}
  // CHECK-NEXT: acc.data dataOperands(%[[COPYIN1]], %[[COPYIN2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN2]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier always,readonly>, name = "parmVar"}
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier capture>, name = "parmVar"}

#pragma acc data copyout(parmVar)
  ;
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyout>, name = "parmVar"}
  // CHECK-NEXT: acc.data dataOperands(%[[CREATE1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[CREATE1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {name = "parmVar"}

#pragma acc data copyout(parmVar) copyout(localVar1)
  ;
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyout>, name = "parmVar"}
  // CHECK-NEXT: %[[CREATE2:.*]] = acc.create varPtr(%[[LV1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyout>, name = "localVar1"}
  // CHECK-NEXT: acc.data dataOperands(%[[CREATE1]], %[[CREATE2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[CREATE2]] : !cir.ptr<!s32i>) to varPtr(%[[LV1]] : !cir.ptr<!s32i>) {name = "localVar1"}
  // CHECK-NEXT: acc.copyout accPtr(%[[CREATE1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {name = "parmVar"}

#pragma acc data copyout(parmVar, localVar1)
  ;
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyout>, name = "parmVar"}
  // CHECK-NEXT: %[[CREATE2:.*]] = acc.create varPtr(%[[LV1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyout>, name = "localVar1"}
  // CHECK-NEXT: acc.data dataOperands(%[[CREATE1]], %[[CREATE2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[CREATE2]] : !cir.ptr<!s32i>) to varPtr(%[[LV1]] : !cir.ptr<!s32i>) {name = "localVar1"}
  // CHECK-NEXT: acc.copyout accPtr(%[[CREATE1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {name = "parmVar"}

#pragma acc data copyout(capture, zero, alwaysout:  parmVar)
  ;
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyout>, modifiers = #acc<data_clause_modifier zero,alwaysout,capture>, name = "parmVar"}
  // CHECK-NEXT: acc.data dataOperands(%[[CREATE1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[CREATE1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {modifiers = #acc<data_clause_modifier zero,alwaysout,capture>, name = "parmVar"}

#pragma acc data copyout(capture: parmVar) copyout(always, alwaysout:  parmVar)
  ;
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyout>, modifiers = #acc<data_clause_modifier capture>, name = "parmVar"}
  // CHECK-NEXT: %[[CREATE2:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyout>, modifiers = #acc<data_clause_modifier always>, name = "parmVar"}
  // CHECK-NEXT: acc.data dataOperands(%[[CREATE1]], %[[CREATE2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[CREATE2]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {modifiers = #acc<data_clause_modifier always>, name = "parmVar"}
  // CHECK-NEXT: acc.copyout accPtr(%[[CREATE1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {modifiers = #acc<data_clause_modifier capture>, name = "parmVar"}

#pragma acc data create(parmVar)
  ;
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "parmVar"}
  // CHECK-NEXT: acc.data dataOperands(%[[CREATE1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[CREATE1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_create>, name = "parmVar"}

#pragma acc data create(parmVar) create(localVar1)
  ;
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "parmVar"}
  // CHECK-NEXT: %[[CREATE2:.*]] = acc.create varPtr(%[[LV1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "localVar1"}
  // CHECK-NEXT: acc.data dataOperands(%[[CREATE1]], %[[CREATE2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[CREATE2]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_create>, name = "localVar1"}
  // CHECK-NEXT: acc.delete accPtr(%[[CREATE1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_create>, name = "parmVar"}

#pragma acc data create(parmVar, localVar1)
  ;
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "parmVar"}
  // CHECK-NEXT: %[[CREATE2:.*]] = acc.create varPtr(%[[LV1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "localVar1"}
  // CHECK-NEXT: acc.data dataOperands(%[[CREATE1]], %[[CREATE2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[CREATE2]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_create>, name = "localVar1"}
  // CHECK-NEXT: acc.delete accPtr(%[[CREATE1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_create>, name = "parmVar"}


#pragma acc data create(capture, zero:  parmVar)
  ;
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier zero,capture>, name = "parmVar"}
  // CHECK-NEXT: acc.data dataOperands(%[[CREATE1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[CREATE1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_create>, modifiers = #acc<data_clause_modifier zero,capture>, name = "parmVar"}

#pragma acc data create(capture: parmVar) create(zero:  parmVar)
  ;
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier capture>, name = "parmVar"}
  // CHECK-NEXT: %[[CREATE2:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier zero>, name = "parmVar"}
  // CHECK-NEXT: acc.data dataOperands(%[[CREATE1]], %[[CREATE2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[CREATE2]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_create>, modifiers = #acc<data_clause_modifier zero>, name = "parmVar"}
  // CHECK-NEXT: acc.delete accPtr(%[[CREATE1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_create>, modifiers = #acc<data_clause_modifier capture>, name = "parmVar"}

#pragma acc data no_create(parmVar)
  ;
  // CHECK-NEXT: %[[NOCREATE1:.*]] = acc.nocreate varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "parmVar"}
  // CHECK-NEXT: acc.data dataOperands(%[[NOCREATE1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[NOCREATE1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_no_create>, name = "parmVar"}

#pragma acc data no_create(parmVar) no_create(localVar1)
  ;
  // CHECK-NEXT: %[[NOCREATE1:.*]] = acc.nocreate varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "parmVar"}
  // CHECK-NEXT: %[[NOCREATE2:.*]] = acc.nocreate varPtr(%[[LV1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "localVar1"}
  // CHECK-NEXT: acc.data dataOperands(%[[NOCREATE1]], %[[NOCREATE2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[NOCREATE2]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_no_create>, name = "localVar1"}
  // CHECK-NEXT: acc.delete accPtr(%[[NOCREATE1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_no_create>, name = "parmVar"}

#pragma acc data no_create(parmVar, localVar1)
  ;
  // CHECK-NEXT: %[[NOCREATE1:.*]] = acc.nocreate varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "parmVar"}
  // CHECK-NEXT: %[[NOCREATE2:.*]] = acc.nocreate varPtr(%[[LV1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "localVar1"}
  // CHECK-NEXT: acc.data dataOperands(%[[NOCREATE1]], %[[NOCREATE2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[NOCREATE2]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_no_create>, name = "localVar1"}
  // CHECK-NEXT: acc.delete accPtr(%[[NOCREATE1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_no_create>, name = "parmVar"}
}
