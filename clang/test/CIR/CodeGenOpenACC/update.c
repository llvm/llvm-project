// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

void acc_update(int parmVar, int *ptrParmVar) {
  // CHECK: cir.func{{.*}} @acc_update(%[[ARG:.*]]: !s32i{{.*}}, %[[PTRARG:.*]]: !cir.ptr<!s32i>{{.*}}) {
  // CHECK-NEXT: %[[PARM:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["parmVar", init]
  // CHECK-NEXT: %[[PTRPARM:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["ptrParmVar", init]
  // CHECK-NEXT: cir.store %[[ARG]], %[[PARM]] : !s32i, !cir.ptr<!s32i>
  // CHECK-NEXT: cir.store %[[PTRARG]], %[[PTRPARM]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>

#pragma acc update device(parmVar)
  // CHECK-NEXT: %[[UPD_DEV1:.*]] = acc.update_device varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "parmVar", structured = false}
  // CHECK-NEXT: acc.update dataOperands(%[[UPD_DEV1]] : !cir.ptr<!s32i>)

#pragma acc update device(parmVar, ptrParmVar)
  // CHECK-NEXT: %[[UPD_DEV1:.*]] = acc.update_device varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "parmVar", structured = false}
  // CHECK-NEXT: %[[UPD_DEV2:.*]] = acc.update_device varPtr(%[[PTRPARM]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "ptrParmVar", structured = false}
  // CHECK-NEXT: acc.update dataOperands(%[[UPD_DEV1]], %[[UPD_DEV2]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>)

#pragma acc update device(parmVar) device(ptrParmVar)
  // CHECK-NEXT: %[[UPD_DEV1:.*]] = acc.update_device varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {name = "parmVar", structured = false}
  // CHECK-NEXT: %[[UPD_DEV2:.*]] = acc.update_device varPtr(%[[PTRPARM]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "ptrParmVar", structured = false}
  // CHECK-NEXT: acc.update dataOperands(%[[UPD_DEV1]], %[[UPD_DEV2]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>)

#pragma acc update host(parmVar)
  // CHECK-NEXT: %[[GDP1:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_update_host>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.update dataOperands(%[[GDP1]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {name = "parmVar", structured = false}

#pragma acc update host(parmVar, ptrParmVar)
  // CHECK-NEXT: %[[GDP1:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_update_host>, name = "parmVar", structured = false}
  // CHECK-NEXT: %[[GDP2:.*]] = acc.getdeviceptr varPtr(%[[PTRPARM]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {dataClause = #acc<data_clause acc_update_host>, name = "ptrParmVar", structured = false}
  // CHECK-NEXT: acc.update dataOperands(%[[GDP1]], %[[GDP2]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>)
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP2]] : !cir.ptr<!cir.ptr<!s32i>>) to varPtr(%[[PTRPARM]] : !cir.ptr<!cir.ptr<!s32i>>) {name = "ptrParmVar", structured = false}
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {name = "parmVar", structured = false}

#pragma acc update host(parmVar) host(ptrParmVar)
  // CHECK-NEXT: %[[GDP1:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_update_host>, name = "parmVar", structured = false}
  // CHECK-NEXT: %[[GDP2:.*]] = acc.getdeviceptr varPtr(%[[PTRPARM]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {dataClause = #acc<data_clause acc_update_host>, name = "ptrParmVar", structured = false}
  // CHECK-NEXT: acc.update dataOperands(%[[GDP1]], %[[GDP2]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>)
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP2]] : !cir.ptr<!cir.ptr<!s32i>>) to varPtr(%[[PTRPARM]] : !cir.ptr<!cir.ptr<!s32i>>) {name = "ptrParmVar", structured = false}
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {name = "parmVar", structured = false}

#pragma acc update self(parmVar)
  // CHECK-NEXT: %[[GDP1:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.update dataOperands(%[[GDP1]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}

#pragma acc update self(parmVar, ptrParmVar)
  // CHECK-NEXT: %[[GDP1:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}
  // CHECK-NEXT: %[[GDP2:.*]] = acc.getdeviceptr varPtr(%[[PTRPARM]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {dataClause = #acc<data_clause acc_update_self>, name = "ptrParmVar", structured = false}
  // CHECK-NEXT: acc.update dataOperands(%[[GDP1]], %[[GDP2]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>)
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP2]] : !cir.ptr<!cir.ptr<!s32i>>) to varPtr(%[[PTRPARM]] : !cir.ptr<!cir.ptr<!s32i>>) {dataClause = #acc<data_clause acc_update_self>, name = "ptrParmVar", structured = false}
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}

#pragma acc update self(parmVar) self(ptrParmVar)
  // CHECK-NEXT: %[[GDP1:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}
  // CHECK-NEXT: %[[GDP2:.*]] = acc.getdeviceptr varPtr(%[[PTRPARM]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {dataClause = #acc<data_clause acc_update_self>, name = "ptrParmVar", structured = false}
  // CHECK-NEXT: acc.update dataOperands(%[[GDP1]], %[[GDP2]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>)
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP2]] : !cir.ptr<!cir.ptr<!s32i>>) to varPtr(%[[PTRPARM]] : !cir.ptr<!cir.ptr<!s32i>>) {dataClause = #acc<data_clause acc_update_self>, name = "ptrParmVar", structured = false}
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}

#pragma acc update self(parmVar) device(ptrParmVar)
  // CHECK-NEXT: %[[GDP1:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}
  // CHECK-NEXT: %[[UPD_DEV2:.*]] = acc.update_device varPtr(%[[PTRPARM]] : !cir.ptr<!cir.ptr<!s32i>>) -> !cir.ptr<!cir.ptr<!s32i>> {name = "ptrParmVar", structured = false}
  // CHECK-NEXT: acc.update dataOperands(%[[GDP1]], %[[UPD_DEV2]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>)
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}
}
