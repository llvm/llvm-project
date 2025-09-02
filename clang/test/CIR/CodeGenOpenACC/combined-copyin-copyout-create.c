// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

void acc_combined(int parmVar) {
  // CHECK: cir.func{{.*}} @acc_combined(%[[ARG:.*]]: !s32i{{.*}}) {
  // CHECK-NEXT: %[[PARM:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["parmVar", init]

  int localVar1;
  // CHECK-NEXT: %[[LV1:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["localVar1"]
  float localVar2;
  // CHECK-NEXT: %[[LV2:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["localVar2"]
  // CHECK-NEXT: cir.store %[[ARG]], %[[PARM]]
#pragma acc parallel loop copyin(parmVar) copyout(localVar1) create(localVar2)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "parmVar"}
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[LV1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyout>, name = "localVar1"}
  // CHECK-NEXT: %[[CREATE2:.*]] = acc.create varPtr(%[[LV2]] : !cir.ptr<!cir.float>) -> !cir.ptr<!cir.float> {name = "localVar2"}
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[COPYIN1]], %[[CREATE1]], %[[CREATE2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !cir.ptr<!cir.float>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[CREATE2]] : !cir.ptr<!cir.float>) {dataClause = #acc<data_clause acc_create>, name = "localVar2"}
  // CHECK-NEXT: acc.copyout accPtr(%[[CREATE1]] : !cir.ptr<!s32i>) to varPtr(%[[LV1]] : !cir.ptr<!s32i>) {name = "localVar1"}
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, name = "parmVar"}

#pragma acc serial loop copyin(parmVar, localVar1)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "parmVar"}
  // CHECK-NEXT: %[[COPYIN2:.*]] = acc.copyin varPtr(%[[LV1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "localVar1"}
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]], %[[COPYIN2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN2]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, name = "localVar1"}
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, name = "parmVar"}

#pragma acc kernels loop copyout(parmVar, localVar1)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyout>, name = "parmVar"}
  // CHECK-NEXT: %[[CREATE2:.*]] = acc.create varPtr(%[[LV1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyout>, name = "localVar1"}
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[CREATE1]], %[[CREATE2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[CREATE2]] : !cir.ptr<!s32i>) to varPtr(%[[LV1]] : !cir.ptr<!s32i>) {name = "localVar1"}
  // CHECK-NEXT: acc.copyout accPtr(%[[CREATE1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {name = "parmVar"}

#pragma acc parallel loop create (parmVar, localVar2)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "parmVar"}
  // CHECK-NEXT: %[[CREATE2:.*]] = acc.create varPtr(%[[LV2]] : !cir.ptr<!cir.float>) -> !cir.ptr<!cir.float> {name = "localVar2"}
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[CREATE1]], %[[CREATE2]] : !cir.ptr<!s32i>, !cir.ptr<!cir.float>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[CREATE2]] : !cir.ptr<!cir.float>) {dataClause = #acc<data_clause acc_create>, name = "localVar2"}
  // CHECK-NEXT: acc.delete accPtr(%[[CREATE1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_create>, name = "parmVar"}

#pragma acc serial loop copyin(capture: parmVar) copyin(always: localVar1)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier capture>, name = "parmVar"}
  // CHECK-NEXT: %[[COPYIN2:.*]] = acc.copyin varPtr(%[[LV1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier always>, name = "localVar1"}
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]], %[[COPYIN2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN2]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier always>, name = "localVar1"}
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier capture>, name = "parmVar"}

#pragma acc kernels loop copyout(capture: parmVar) copyout(always: localVar1)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyout>, modifiers = #acc<data_clause_modifier capture>, name = "parmVar"}
  // CHECK-NEXT: %[[CREATE2:.*]] = acc.create varPtr(%[[LV1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyout>, modifiers = #acc<data_clause_modifier always>, name = "localVar1"}
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[CREATE1]], %[[CREATE2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[CREATE2]] : !cir.ptr<!s32i>) to varPtr(%[[LV1]] : !cir.ptr<!s32i>) {modifiers = #acc<data_clause_modifier always>, name = "localVar1"}
  // CHECK-NEXT: acc.copyout accPtr(%[[CREATE1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {modifiers = #acc<data_clause_modifier capture>, name = "parmVar"}

#pragma acc parallel loop create(capture: parmVar)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier capture>, name = "parmVar"}
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[CREATE1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[CREATE1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_create>, modifiers = #acc<data_clause_modifier capture>, name = "parmVar"}

#pragma acc serial loop copyin(capture, always: parmVar, localVar1)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier always,capture>, name = "parmVar"}
  // CHECK-NEXT: %[[COPYIN2:.*]] = acc.copyin varPtr(%[[LV1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier always,capture>, name = "localVar1"}
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[COPYIN1]], %[[COPYIN2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN2]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier always,capture>, name = "localVar1"}
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier always,capture>, name = "parmVar"}

#pragma acc kernels loop copyin(readonly, always, alwaysin, capture: parmVar, localVar1, localVar2)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier always,readonly,capture>, name = "parmVar"}
  // CHECK-NEXT: %[[COPYIN2:.*]] = acc.copyin varPtr(%[[LV1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier always,readonly,capture>, name = "localVar1"}
  // CHECK-NEXT: %[[COPYIN3:.*]] = acc.copyin varPtr(%[[LV2]] : !cir.ptr<!cir.float>) -> !cir.ptr<!cir.float> {modifiers = #acc<data_clause_modifier always,readonly,capture>, name = "localVar2"}
  // CHECK-NEXT: acc.kernels combined(loop) dataOperands(%[[COPYIN1]], %[[COPYIN2]], %[[COPYIN3]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !cir.ptr<!cir.float>) {
  // CHECK-NEXT: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN3]] : !cir.ptr<!cir.float>) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier always,readonly,capture>, name = "localVar2"}
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN2]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier always,readonly,capture>, name = "localVar1"}
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_copyin>, modifiers = #acc<data_clause_modifier always,readonly,capture>, name = "parmVar"}

#pragma acc parallel loop copyout(zero, always, alwaysout, capture: parmVar, localVar1, localVar2)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyout>, modifiers = #acc<data_clause_modifier always,zero,capture>, name = "parmVar"}
  // CHECK-NEXT: %[[CREATE2:.*]] = acc.create varPtr(%[[LV1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_copyout>, modifiers = #acc<data_clause_modifier always,zero,capture>, name = "localVar1"}
  // CHECK-NEXT: %[[CREATE3:.*]] = acc.create varPtr(%[[LV2]] : !cir.ptr<!cir.float>) -> !cir.ptr<!cir.float> {dataClause = #acc<data_clause acc_copyout>, modifiers = #acc<data_clause_modifier always,zero,capture>, name = "localVar2"}
  // CHECK-NEXT: acc.parallel combined(loop) dataOperands(%[[CREATE1]], %[[CREATE2]], %[[CREATE3]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !cir.ptr<!cir.float>) {
  // CHECK-NEXT: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[CREATE3]] : !cir.ptr<!cir.float>) to varPtr(%[[LV2]] : !cir.ptr<!cir.float>) {modifiers = #acc<data_clause_modifier always,zero,capture>, name = "localVar2"}
  // CHECK-NEXT: acc.copyout accPtr(%[[CREATE2]] : !cir.ptr<!s32i>) to varPtr(%[[LV1]] : !cir.ptr<!s32i>) {modifiers = #acc<data_clause_modifier always,zero,capture>, name = "localVar1"}
  // CHECK-NEXT: acc.copyout accPtr(%[[CREATE1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {modifiers = #acc<data_clause_modifier always,zero,capture>, name = "parmVar"}

#pragma acc serial loop create(zero, capture: parmVar, localVar1, localVar2)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier zero,capture>, name = "parmVar"}
  // CHECK-NEXT: %[[CREATE2:.*]] = acc.create varPtr(%[[LV1]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {modifiers = #acc<data_clause_modifier zero,capture>, name = "localVar1"}
  // CHECK-NEXT: %[[CREATE3:.*]] = acc.create varPtr(%[[LV2]] : !cir.ptr<!cir.float>) -> !cir.ptr<!cir.float> {modifiers = #acc<data_clause_modifier zero,capture>, name = "localVar2"}
  // CHECK-NEXT: acc.serial combined(loop) dataOperands(%[[CREATE1]], %[[CREATE2]], %[[CREATE3]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !cir.ptr<!cir.float>) {
  // CHECK-NEXT: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[CREATE3]] : !cir.ptr<!cir.float>) {dataClause = #acc<data_clause acc_create>, modifiers = #acc<data_clause_modifier zero,capture>, name = "localVar2"}
  // CHECK-NEXT: acc.delete accPtr(%[[CREATE2]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_create>, modifiers = #acc<data_clause_modifier zero,capture>, name = "localVar1"}
  // CHECK-NEXT: acc.delete accPtr(%[[CREATE1]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_create>, modifiers = #acc<data_clause_modifier zero,capture>, name = "parmVar"}
}
