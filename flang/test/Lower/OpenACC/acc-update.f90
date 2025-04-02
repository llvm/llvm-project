! This test checks lowering of OpenACC update directive.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine acc_update
  integer :: async = 1
  real, dimension(10, 10) :: a, b, c
  logical :: ifCondition = .TRUE.

! CHECK: %[[A:.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ea"}
! CHECK: %[[DECLA:.*]]:2 = hlfir.declare %[[A]]
! CHECK: %[[B:.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Eb"}
! CHECK: %[[DECLB:.*]]:2 = hlfir.declare %[[B]]
! CHECK: %[[C:.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ec"}
! CHECK: %[[DECLC:.*]]:2 = hlfir.declare %[[C]]

  !$acc update host(a)
! CHECK: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: acc.update dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}
! CHECK: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc update host(a) if_present
! CHECK: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: acc.update dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) attributes {ifPresent}{{$}}
! CHECK: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc update self(a)
! CHECK: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_self>, name = "a", structured = false}
! CHECK: acc.update dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}
! CHECK: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_update_self>, name = "a", structured = false}

  !$acc update host(a) if(.true.)
! CHECK: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: %[[IF1:.*]] = arith.constant true
! CHECK: acc.update if(%[[IF1]]) dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}
! CHECK: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc update host(a) if(ifCondition)
! CHECK: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: %[[IFCOND:.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
! CHECK: %[[IF2:.*]] = fir.convert %[[IFCOND]] : (!fir.logical<4>) -> i1
! CHECK: acc.update if(%[[IF2]]) dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}
! CHECK: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc update host(a) host(b) host(c)
! CHECK: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: %[[DEVPTR_B:.*]] = acc.getdeviceptr varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "b", structured = false}
! CHECK: %[[DEVPTR_C:.*]] = acc.getdeviceptr varPtr(%[[DECLC]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "c", structured = false}
! CHECK: acc.update dataOperands(%[[DEVPTR_A]], %[[DEVPTR_B]], %[[DEVPTR_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>){{$}}
! CHECK: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}
! CHECK: acc.update_host accPtr(%[[DEVPTR_B]] : !fir.ref<!fir.array<10x10xf32>>) to varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10x10xf32>>) {name = "b", structured = false}
! CHECK: acc.update_host accPtr(%[[DEVPTR_C]] : !fir.ref<!fir.array<10x10xf32>>) to varPtr(%[[DECLC]]#0 : !fir.ref<!fir.array<10x10xf32>>) {name = "c", structured = false}

  !$acc update host(a) host(b) device(c)
! CHECK: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: %[[DEVPTR_B:.*]] = acc.getdeviceptr varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "b", structured = false}
! CHECK: %[[DEVPTR_C:.*]] = acc.update_device varPtr(%[[DECLC]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {name = "c", structured = false}
! CHECK: acc.update dataOperands(%[[DEVPTR_C]], %[[DEVPTR_A]], %[[DEVPTR_B]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>){{$}}
! CHECK: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}
! CHECK: acc.update_host accPtr(%[[DEVPTR_B]] : !fir.ref<!fir.array<10x10xf32>>) to varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10x10xf32>>) {name = "b", structured = false}

  !$acc update host(a) async
! CHECK: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {asyncOnly = [#acc.device_type<none>], dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: acc.update async dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>)
! CHECK: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) {asyncOnly = [#acc.device_type<none>], name = "a", structured = false}

  !$acc update host(a) wait
! CHECK: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: acc.update wait dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>)
! CHECK: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc update host(a) async wait
! CHECK: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {asyncOnly = [#acc.device_type<none>], dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: acc.update async wait dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>)
! CHECK: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) {asyncOnly = [#acc.device_type<none>], name = "a", structured = false}

  !$acc update host(a) async(1)
! CHECK: [[ASYNC1:%.*]] = arith.constant 1 : i32
! CHECK: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) async([[ASYNC1]] : i32) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: acc.update async([[ASYNC1]] : i32) dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>)
! CHECK: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) async([[ASYNC1]] : i32) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc update host(a) async(async)
! CHECK: [[ASYNC2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) async([[ASYNC2]] : i32) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: acc.update async([[ASYNC2]] : i32) dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>)
! CHECK: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) async([[ASYNC2]] : i32) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc update host(a) wait(1)
! CHECK: [[WAIT1:%.*]] = arith.constant 1 : i32
! CHECK: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: acc.update wait({[[WAIT1]] : i32}) dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>)
! CHECK: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc update host(a) wait(queues: 1, 2)
! CHECK: [[WAIT2:%.*]] = arith.constant 1 : i32
! CHECK: [[WAIT3:%.*]] = arith.constant 2 : i32
! CHECK: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: acc.update wait({[[WAIT2]] : i32, [[WAIT3]] : i32}) dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>)
! CHECK: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc update host(a) wait(devnum: 1: queues: 1, 2)
! CHECK: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: acc.update wait({devnum: %c1{{.*}} : i32, %c1{{.*}} : i32, %c2{{.*}} : i32}) dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>)
! CHECK: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc update host(a) device_type(host, nvidia) async
! CHECK: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {asyncOnly = [#acc.device_type<host>, #acc.device_type<nvidia>], dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: acc.update async([#acc.device_type<host>, #acc.device_type<nvidia>]) dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>)
! CHECK: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) {asyncOnly = [#acc.device_type<host>, #acc.device_type<nvidia>], name = "a", structured = false}

end subroutine acc_update
