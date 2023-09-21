! This test checks lowering of OpenACC update directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s --check-prefixes=CHECK,FIR
! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK,HLFIR

subroutine acc_update
  integer :: async = 1
  real, dimension(10, 10) :: a, b, c
  logical :: ifCondition = .TRUE.

!CHECK: %[[A:.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ea"}
! HLFIR: %[[DECLA:.*]]:2 = hlfir.declare %[[A]]
!CHECK: %[[B:.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Eb"}
! HLFIR: %[[DECLB:.*]]:2 = hlfir.declare %[[B]]
!CHECK: %[[C:.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ec"}
! HLFIR: %[[DECLC:.*]]:2 = hlfir.declare %[[C]]

  !$acc update host(a)
! FIR:   %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! HLFIR: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: acc.update dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}
! FIR:   acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}
! HLFIR: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc update host(a) if_present
! FIR:   %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! HLFIR: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: acc.update dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) attributes {ifPresent}{{$}}
! FIR:   acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}
! HLFIR: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc update self(a)
! FIR:   %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_self>, name = "a", structured = false}
! HLFIR: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_self>, name = "a", structured = false}
! CHECK: acc.update dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}
! FIR:   acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_update_self>, name = "a", structured = false}
! HLFIR: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) {dataClause = #acc<data_clause acc_update_self>, name = "a", structured = false}

  !$acc update host(a) if(.true.)
! FIR:   %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! HLFIR: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: %[[IF1:.*]] = arith.constant true
! CHECK: acc.update if(%[[IF1]]) dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}
! FIR:   acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}
! HLFIR: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc update host(a) if(ifCondition)
! FIR:   %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! HLFIR: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: %[[IFCOND:.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
! CHECK: %[[IF2:.*]] = fir.convert %[[IFCOND]] : (!fir.logical<4>) -> i1
! CHECK: acc.update if(%[[IF2]]) dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}
! FIR:   acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}
! HLFIR: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc update host(a) host(b) host(c)
! FIR:   %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! HLFIR: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! FIR:   %[[DEVPTR_B:.*]] = acc.getdeviceptr varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "b", structured = false}
! HLFIR: %[[DEVPTR_B:.*]] = acc.getdeviceptr varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "b", structured = false}
! FIR:   %[[DEVPTR_C:.*]] = acc.getdeviceptr varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "c", structured = false}
! HLFIR: %[[DEVPTR_C:.*]] = acc.getdeviceptr varPtr(%[[DECLC]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "c", structured = false}
! CHECK: acc.update dataOperands(%[[DEVPTR_A]], %[[DEVPTR_B]], %[[DEVPTR_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>){{$}}
! FIR:   acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}
! HLFIR: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}
! FIR:   acc.update_host accPtr(%[[DEVPTR_B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) {name = "b", structured = false}
! HLFIR: acc.update_host accPtr(%[[DEVPTR_B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10x10xf32>>) {name = "b", structured = false}
! FIR:   acc.update_host accPtr(%[[DEVPTR_C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) {name = "c", structured = false}
! HLFIR: acc.update_host accPtr(%[[DEVPTR_C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLC]]#1 : !fir.ref<!fir.array<10x10xf32>>) {name = "c", structured = false}

  !$acc update host(a) host(b) device(c)
! FIR:   %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! HLFIR: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! FIR:   %[[DEVPTR_B:.*]] = acc.getdeviceptr varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "b", structured = false}
! HLFIR: %[[DEVPTR_B:.*]] = acc.getdeviceptr varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "b", structured = false}
! FIR:   %[[DEVPTR_C:.*]] = acc.update_device varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "c", structured = false}
! HLFIR: %[[DEVPTR_C:.*]] = acc.update_device varPtr(%[[DECLC]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {name = "c", structured = false}
! CHECK: acc.update dataOperands(%[[DEVPTR_C]], %[[DEVPTR_A]], %[[DEVPTR_B]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>){{$}}
! FIR:   acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}
! HLFIR: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}
! FIR:   acc.update_host accPtr(%[[DEVPTR_B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) {name = "b", structured = false}
! HLFIR: acc.update_host accPtr(%[[DEVPTR_B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLB]]#1 : !fir.ref<!fir.array<10x10xf32>>) {name = "b", structured = false}

  !$acc update host(a) async
! FIR:   %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! HLFIR: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: acc.update dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) attributes {async}
! FIR:   acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}
! HLFIR: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc update host(a) wait
! FIR:   %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! HLFIR: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: acc.update dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) attributes {wait}
! FIR:   acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}
! HLFIR: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc update host(a) async wait
! FIR:   %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! HLFIR: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: acc.update dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) attributes {async, wait}
! FIR:   acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}
! HLFIR: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc update host(a) async(1)
! FIR:   %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! HLFIR: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: [[ASYNC1:%.*]] = arith.constant 1 : i32
! CHECK: acc.update async([[ASYNC1]] : i32) dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>)
! FIR:   acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}
! HLFIR: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc update host(a) async(async)
! FIR:   %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! HLFIR: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: [[ASYNC2:%.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK: acc.update async([[ASYNC2]] : i32) dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>)
! FIR:   acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}
! HLFIR: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc update host(a) wait(1)
! FIR:   %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! HLFIR: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: [[WAIT1:%.*]] = arith.constant 1 : i32
! CHECK: acc.update wait([[WAIT1]] : i32) dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>)
! FIR:   acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}
! HLFIR: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc update host(a) wait(queues: 1, 2)
! FIR:   %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! HLFIR: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: [[WAIT2:%.*]] = arith.constant 1 : i32
! CHECK: [[WAIT3:%.*]] = arith.constant 2 : i32
! CHECK: acc.update wait([[WAIT2]], [[WAIT3]] : i32, i32) dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>)
! FIR:   acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}
! HLFIR: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc update host(a) wait(devnum: 1: queues: 1, 2)
! FIR:   %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! HLFIR: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: [[WAIT4:%.*]] = arith.constant 1 : i32
! CHECK: [[WAIT5:%.*]] = arith.constant 2 : i32
! CHECK: [[WAIT6:%.*]] = arith.constant 1 : i32
! CHECK: acc.update wait_devnum([[WAIT6]] : i32) wait([[WAIT4]], [[WAIT5]] : i32, i32) dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>)
! FIR:   acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}
! HLFIR: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc update host(a) device_type(1, 2)
! FIR:   %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! HLFIR: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: [[DEVTYPE1:%.*]] = arith.constant 1 : i32
! CHECK: [[DEVTYPE2:%.*]] = arith.constant 2 : i32
! CHECK: acc.update device_type([[DEVTYPE1]], [[DEVTYPE2]] : i32, i32) dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}
! FIR:   acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}
! HLFIR: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc update host(a) device_type(*)
! FIR:   %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! HLFIR: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_update_host>, name = "a", structured = false}
! CHECK: [[DEVTYPE3:%.*]] = arith.constant -1 : index
! CHECK: acc.update device_type([[DEVTYPE3]] : index) dataOperands(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}
! FIR:   acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}
! HLFIR: acc.update_host accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%{{.*}}, %{{.*}}) to varPtr(%[[DECLA]]#1 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

end subroutine acc_update
