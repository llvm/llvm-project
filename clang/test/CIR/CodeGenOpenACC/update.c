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

#pragma acc update self(parmVar) if (parmVar == 1)
  // CHECK-NEXT: %[[GDP1:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[ONE_CONST:.*]] = cir.const #cir.int<1>
  // CHECK-NEXT: %[[CMP:.*]] = cir.cmp(eq, %[[PARM_LOAD]], %[[ONE_CONST]])
  // CHECK-NEXT: %[[CMP_CAST:.*]] = builtin.unrealized_conversion_cast %[[CMP]]
  // CHECK-NEXT: acc.update if(%[[CMP_CAST]]) dataOperands(%[[GDP1]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}
#pragma acc update self(parmVar) if (parmVar == 1) if_present
  // CHECK-NEXT: %[[GDP1:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[ONE_CONST:.*]] = cir.const #cir.int<1>
  // CHECK-NEXT: %[[CMP:.*]] = cir.cmp(eq, %[[PARM_LOAD]], %[[ONE_CONST]])
  // CHECK-NEXT: %[[CMP_CAST:.*]] = builtin.unrealized_conversion_cast %[[CMP]]
  // CHECK-NEXT: acc.update if(%[[CMP_CAST]]) dataOperands(%[[GDP1]] : !cir.ptr<!s32i>) attributes {ifPresent}
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}

#pragma acc update self(parmVar) wait
  // CHECK-NEXT: %[[GDP1:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.update wait dataOperands(%[[GDP1]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}

#pragma acc update self(parmVar) wait device_type(nvidia)
  // CHECK-NEXT: %[[GDP1:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.update wait dataOperands(%[[GDP1]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}

#pragma acc update self(parmVar) device_type(radeon) wait
  // CHECK-NEXT: %[[GDP1:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.update wait([#acc.device_type<radeon>]) dataOperands(%[[GDP1]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}

#pragma acc update self(parmVar) wait(parmVar)
  // CHECK-NEXT: %[[GDP1:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[PARM_CAST:.*]] = builtin.unrealized_conversion_cast %[[PARM_LOAD]]
  // CHECK-NEXT: acc.update wait({%[[PARM_CAST]] : si32}) dataOperands(%[[GDP1]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}

#pragma acc update self(parmVar) wait(parmVar) device_type(nvidia)
  // CHECK-NEXT: %[[GDP1:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[PARM_CAST:.*]] = builtin.unrealized_conversion_cast %[[PARM_LOAD]]
  // CHECK-NEXT: acc.update wait({%[[PARM_CAST]] : si32}) dataOperands(%[[GDP1]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}

#pragma acc update self(parmVar) device_type(radeon) wait(parmVar)
  // CHECK-NEXT: %[[GDP1:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[PARM_CAST:.*]] = builtin.unrealized_conversion_cast %[[PARM_LOAD]]
  // CHECK-NEXT: acc.update wait({%[[PARM_CAST]] : si32} [#acc.device_type<radeon>]) dataOperands(%[[GDP1]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}

#pragma acc update self(parmVar) device_type(radeon) wait(parmVar, 1, 2)
  // CHECK-NEXT: %[[GDP1:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[PARM_CAST:.*]] = builtin.unrealized_conversion_cast %[[PARM_LOAD]]
  // CHECK-NEXT: %[[ONE_CONST:.*]] = cir.const #cir.int<1>
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_CONST]]
  // CHECK-NEXT: %[[TWO_CONST:.*]] = cir.const #cir.int<2>
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_CONST]]
  // CHECK-NEXT: acc.update wait({%[[PARM_CAST]] : si32, %[[ONE_CAST]] : si32, %[[TWO_CAST]] : si32} [#acc.device_type<radeon>]) dataOperands(%[[GDP1]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}

#pragma acc update self(parmVar) device_type(radeon) wait(devnum:parmVar: 1, 2)
  // CHECK-NEXT: %[[GDP1:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[PARM_CAST:.*]] = builtin.unrealized_conversion_cast %[[PARM_LOAD]]
  // CHECK-NEXT: %[[ONE_CONST:.*]] = cir.const #cir.int<1>
  // CHECK-NEXT: %[[ONE_CAST:.*]] = builtin.unrealized_conversion_cast %[[ONE_CONST]]
  // CHECK-NEXT: %[[TWO_CONST:.*]] = cir.const #cir.int<2>
  // CHECK-NEXT: %[[TWO_CAST:.*]] = builtin.unrealized_conversion_cast %[[TWO_CONST]]
  // CHECK-NEXT: acc.update wait({devnum: %[[PARM_CAST]] : si32, %[[ONE_CAST]] : si32, %[[TWO_CAST]] : si32} [#acc.device_type<radeon>]) dataOperands(%[[GDP1]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP1]] : !cir.ptr<!s32i>) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}

#pragma acc update self(parmVar) async
  // CHECK-NEXT: %[[GDP1:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) async -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.update async dataOperands(%[[GDP1]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP1]] : !cir.ptr<!s32i>) async to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}

#pragma acc update self(parmVar) async device_type(nvidia)
  // CHECK-NEXT: %[[GDP1:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) async -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.update async dataOperands(%[[GDP1]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP1]] : !cir.ptr<!s32i>) async to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}

#pragma acc update self(parmVar) device_type(radeon) async
  // CHECK-NEXT: %[[GDP1:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) async([#acc.device_type<radeon>]) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.update async([#acc.device_type<radeon>]) dataOperands(%[[GDP1]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP1]] : !cir.ptr<!s32i>) async([#acc.device_type<radeon>]) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}

#pragma acc update self(parmVar) async(parmVar)
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[PARM_CAST:.*]] = builtin.unrealized_conversion_cast %[[PARM_LOAD]]
  // CHECK-NEXT: %[[GDP1:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) async(%[[PARM_CAST]] : si32) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.update async(%[[PARM_CAST]] : si32) dataOperands(%[[GDP1]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP1]] : !cir.ptr<!s32i>) async(%[[PARM_CAST]] : si32) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}

#pragma acc update self(parmVar) async(parmVar) device_type(nvidia)
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[PARM_CAST:.*]] = builtin.unrealized_conversion_cast %[[PARM_LOAD]]
  // CHECK-NEXT: %[[GDP1:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) async(%[[PARM_CAST]] : si32) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.update async(%[[PARM_CAST]] : si32) dataOperands(%[[GDP1]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP1]] : !cir.ptr<!s32i>) async(%[[PARM_CAST]] : si32) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}

#pragma acc update self(parmVar) device_type(radeon) async(parmVar)
  // CHECK-NEXT: %[[PARM_LOAD:.*]] = cir.load{{.*}} %[[PARM]]
  // CHECK-NEXT: %[[PARM_CAST:.*]] = builtin.unrealized_conversion_cast %[[PARM_LOAD]]
  // CHECK-NEXT: %[[GDP1:.*]] = acc.getdeviceptr varPtr(%[[PARM]] : !cir.ptr<!s32i>) async(%[[PARM_CAST]] : si32 [#acc.device_type<radeon>]) -> !cir.ptr<!s32i> {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}
  // CHECK-NEXT: acc.update async(%[[PARM_CAST]] : si32 [#acc.device_type<radeon>]) dataOperands(%[[GDP1]] : !cir.ptr<!s32i>)
  // CHECK-NEXT: acc.update_host accPtr(%[[GDP1]] : !cir.ptr<!s32i>) async(%[[PARM_CAST]] : si32 [#acc.device_type<radeon>]) to varPtr(%[[PARM]] : !cir.ptr<!s32i>) {dataClause = #acc<data_clause acc_update_self>, name = "parmVar", structured = false}
}
