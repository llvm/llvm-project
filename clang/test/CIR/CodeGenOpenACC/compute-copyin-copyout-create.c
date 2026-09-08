// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

void acc_compute(int parmVar) {
  // CHECK: cir.func{{.*}} @acc_compute(%[[ARG:.*]]: !s32i{{.*}}) {{.*}}{
  // CHECK-NEXT: %[[PARM:.*]] = cir.alloca "parmVar" {{.*}} init : !cir.ptr<!s32i>

  int localVar1;
  // CHECK-NEXT: %[[LV1:.*]] = cir.alloca "localVar1" {{.*}} : !cir.ptr<!s32i>
  float localVar2;
  // CHECK-NEXT: %[[LV2:.*]] = cir.alloca "localVar2" {{.*}} : !cir.ptr<!cir.float>
  // CHECK-NEXT: cir.store %[[ARG]], %[[PARM]]

#pragma acc parallel copyin(parmVar) copyout(localVar1) create(localVar2)
  ;
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) name("parmVar") -> !cir.ptr<!s32i>
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[LV1]] : !cir.ptr<!s32i>) dataClause(acc_copyout) name("localVar1") -> !cir.ptr<!s32i>
  // CHECK-NEXT: %[[CREATE2:.*]] = acc.create varPtr(%[[LV2]] : !cir.ptr<!cir.float>) name("localVar2") -> !cir.ptr<!cir.float>
  // CHECK-NEXT: acc.parallel dataOperands(%[[COPYIN1]], %[[CREATE1]], %[[CREATE2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !cir.ptr<!cir.float>) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[CREATE2]] : !cir.ptr<!cir.float>) dataClause(acc_create) name("localVar2")
  // CHECK-NEXT: acc.copyout accPtr(%[[CREATE1]] : !cir.ptr<!s32i>) to varPtr(%[[LV1]] : !cir.ptr<!s32i>) name("localVar1")
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) dataClause(acc_copyin) name("parmVar")

#pragma acc serial copyin(parmVar, localVar1)
  ;
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) name("parmVar") -> !cir.ptr<!s32i>
  // CHECK-NEXT: %[[COPYIN2:.*]] = acc.copyin varPtr(%[[LV1]] : !cir.ptr<!s32i>) name("localVar1") -> !cir.ptr<!s32i>
  // CHECK-NEXT: acc.serial dataOperands(%[[COPYIN1]], %[[COPYIN2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN2]] : !cir.ptr<!s32i>) dataClause(acc_copyin) name("localVar1")
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) dataClause(acc_copyin) name("parmVar")

#pragma acc kernels copyout(parmVar, localVar1)
  ;
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) dataClause(acc_copyout) name("parmVar") -> !cir.ptr<!s32i>
  // CHECK-NEXT: %[[CREATE2:.*]] = acc.create varPtr(%[[LV1]] : !cir.ptr<!s32i>) dataClause(acc_copyout) name("localVar1") -> !cir.ptr<!s32i>
  // CHECK-NEXT: acc.kernels dataOperands(%[[CREATE1]], %[[CREATE2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[CREATE2]] : !cir.ptr<!s32i>) to varPtr(%[[LV1]] : !cir.ptr<!s32i>) name("localVar1")
  // CHECK-NEXT: acc.copyout accPtr(%[[CREATE1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) name("parmVar")

#pragma acc parallel create (parmVar, localVar2)
  ;
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) name("parmVar") -> !cir.ptr<!s32i>
  // CHECK-NEXT: %[[CREATE2:.*]] = acc.create varPtr(%[[LV2]] : !cir.ptr<!cir.float>) name("localVar2") -> !cir.ptr<!cir.float>
  // CHECK-NEXT: acc.parallel dataOperands(%[[CREATE1]], %[[CREATE2]] : !cir.ptr<!s32i>, !cir.ptr<!cir.float>) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[CREATE2]] : !cir.ptr<!cir.float>) dataClause(acc_create) name("localVar2")
  // CHECK-NEXT: acc.delete accPtr(%[[CREATE1]] : !cir.ptr<!s32i>) dataClause(acc_create) name("parmVar")

#pragma acc serial copyin(capture: parmVar) copyin(always: localVar1)
  ;
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) name("parmVar") <modifiers = [capture]> -> !cir.ptr<!s32i> loc
  // CHECK-NEXT: %[[COPYIN2:.*]] = acc.copyin varPtr(%[[LV1]] : !cir.ptr<!s32i>) name("localVar1") <modifiers = [always]> -> !cir.ptr<!s32i> loc
  // CHECK-NEXT: acc.serial dataOperands(%[[COPYIN1]], %[[COPYIN2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN2]] : !cir.ptr<!s32i>) dataClause(acc_copyin) name("localVar1") <modifiers = [always]>
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) dataClause(acc_copyin) name("parmVar") <modifiers = [capture]>

#pragma acc kernels copyout(capture: parmVar) copyout(always: localVar1)
  ;
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) dataClause(acc_copyout) name("parmVar") <modifiers = [capture]> -> !cir.ptr<!s32i>
  // CHECK-NEXT: %[[CREATE2:.*]] = acc.create varPtr(%[[LV1]] : !cir.ptr<!s32i>) dataClause(acc_copyout) name("localVar1") <modifiers = [always]> -> !cir.ptr<!s32i>
  // CHECK-NEXT: acc.kernels dataOperands(%[[CREATE1]], %[[CREATE2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[CREATE2]] : !cir.ptr<!s32i>) to varPtr(%[[LV1]] : !cir.ptr<!s32i>) name("localVar1") <modifiers = [always]>
  // CHECK-NEXT: acc.copyout accPtr(%[[CREATE1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) name("parmVar") <modifiers = [capture]>

#pragma acc parallel create(capture: parmVar)
  ;
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) name("parmVar") <modifiers = [capture]> -> !cir.ptr<!s32i>
  // CHECK-NEXT: acc.parallel dataOperands(%[[CREATE1]] : !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[CREATE1]] : !cir.ptr<!s32i>) dataClause(acc_create) name("parmVar") <modifiers = [capture]>

#pragma acc serial copyin(capture, always: parmVar, localVar1)
  ;
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) name("parmVar") <modifiers = [always,capture]> -> !cir.ptr<!s32i> loc
  // CHECK-NEXT: %[[COPYIN2:.*]] = acc.copyin varPtr(%[[LV1]] : !cir.ptr<!s32i>) name("localVar1") <modifiers = [always,capture]> -> !cir.ptr<!s32i> loc
  // CHECK-NEXT: acc.serial dataOperands(%[[COPYIN1]], %[[COPYIN2]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN2]] : !cir.ptr<!s32i>) dataClause(acc_copyin) name("localVar1") <modifiers = [always,capture]>
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) dataClause(acc_copyin) name("parmVar") <modifiers = [always,capture]>

#pragma acc kernels copyin(readonly, always, alwaysin, capture: parmVar, localVar1, localVar2)
  ;
  // CHECK-NEXT: %[[COPYIN1:.*]] = acc.copyin varPtr(%[[PARM]] : !cir.ptr<!s32i>) name("parmVar") <modifiers = [always,readonly,capture]> -> !cir.ptr<!s32i> loc
  // CHECK-NEXT: %[[COPYIN2:.*]] = acc.copyin varPtr(%[[LV1]] : !cir.ptr<!s32i>) name("localVar1") <modifiers = [always,readonly,capture]> -> !cir.ptr<!s32i> loc
  // CHECK-NEXT: %[[COPYIN3:.*]] = acc.copyin varPtr(%[[LV2]] : !cir.ptr<!cir.float>) name("localVar2") <modifiers = [always,readonly,capture]> -> !cir.ptr<!cir.float> loc
  // CHECK-NEXT: acc.kernels dataOperands(%[[COPYIN1]], %[[COPYIN2]], %[[COPYIN3]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !cir.ptr<!cir.float>) {
  // CHECK-NEXT: acc.terminator
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN3]] : !cir.ptr<!cir.float>) dataClause(acc_copyin) name("localVar2") <modifiers = [always,readonly,capture]>
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN2]] : !cir.ptr<!s32i>) dataClause(acc_copyin) name("localVar1") <modifiers = [always,readonly,capture]>
  // CHECK-NEXT: acc.delete accPtr(%[[COPYIN1]] : !cir.ptr<!s32i>) dataClause(acc_copyin) name("parmVar") <modifiers = [always,readonly,capture]>

#pragma acc parallel copyout(zero, always, alwaysout, capture: parmVar, localVar1, localVar2)
  ;
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) dataClause(acc_copyout) name("parmVar") <modifiers = [always,zero,capture]> -> !cir.ptr<!s32i>
  // CHECK-NEXT: %[[CREATE2:.*]] = acc.create varPtr(%[[LV1]] : !cir.ptr<!s32i>) dataClause(acc_copyout) name("localVar1") <modifiers = [always,zero,capture]> -> !cir.ptr<!s32i>
  // CHECK-NEXT: %[[CREATE3:.*]] = acc.create varPtr(%[[LV2]] : !cir.ptr<!cir.float>) dataClause(acc_copyout) name("localVar2") <modifiers = [always,zero,capture]> -> !cir.ptr<!cir.float>
  // CHECK-NEXT: acc.parallel dataOperands(%[[CREATE1]], %[[CREATE2]], %[[CREATE3]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !cir.ptr<!cir.float>) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.copyout accPtr(%[[CREATE3]] : !cir.ptr<!cir.float>) to varPtr(%[[LV2]] : !cir.ptr<!cir.float>) name("localVar2") <modifiers = [always,zero,capture]>
  // CHECK-NEXT: acc.copyout accPtr(%[[CREATE2]] : !cir.ptr<!s32i>) to varPtr(%[[LV1]] : !cir.ptr<!s32i>) name("localVar1") <modifiers = [always,zero,capture]>
  // CHECK-NEXT: acc.copyout accPtr(%[[CREATE1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) name("parmVar") <modifiers = [always,zero,capture]>

#pragma acc serial create(zero, capture: parmVar, localVar1, localVar2)
  ;
  // CHECK-NEXT: %[[CREATE1:.*]] = acc.create varPtr(%[[PARM]] : !cir.ptr<!s32i>) name("parmVar") <modifiers = [zero,capture]> -> !cir.ptr<!s32i>
  // CHECK-NEXT: %[[CREATE2:.*]] = acc.create varPtr(%[[LV1]] : !cir.ptr<!s32i>) name("localVar1") <modifiers = [zero,capture]> -> !cir.ptr<!s32i>
  // CHECK-NEXT: %[[CREATE3:.*]] = acc.create varPtr(%[[LV2]] : !cir.ptr<!cir.float>) name("localVar2") <modifiers = [zero,capture]> -> !cir.ptr<!cir.float>
  // CHECK-NEXT: acc.serial dataOperands(%[[CREATE1]], %[[CREATE2]], %[[CREATE3]] : !cir.ptr<!s32i>, !cir.ptr<!s32i>, !cir.ptr<!cir.float>) {
  // CHECK-NEXT: acc.yield
  // CHECK-NEXT: } loc
  // CHECK-NEXT: acc.delete accPtr(%[[CREATE3]] : !cir.ptr<!cir.float>) dataClause(acc_create) name("localVar2") <modifiers = [zero,capture]>
  // CHECK-NEXT: acc.delete accPtr(%[[CREATE2]] : !cir.ptr<!s32i>) dataClause(acc_create) name("localVar1") <modifiers = [zero,capture]>
  // CHECK-NEXT: acc.delete accPtr(%[[CREATE1]] : !cir.ptr<!s32i>) dataClause(acc_create) name("parmVar") <modifiers = [zero,capture]>
}
