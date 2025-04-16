! This test checks lowering of OpenACC exit data directive.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine acc_exit_data
  integer :: async = 1
  real, dimension(10, 10) :: a, b, c
  real, pointer :: d
  logical :: ifCondition = .TRUE.

!CHECK: %[[A:.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ea"}
!CHECK: %[[DECLA:.*]]:2 = hlfir.declare %[[A]]
!CHECK: %[[B:.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Eb"}
!CHECK: %[[DECLB:.*]]:2 = hlfir.declare %[[B]]
!CHECK: %[[C:.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ec"}
!CHECK: %[[DECLC:.*]]:2 = hlfir.declare %[[C]]
!CHECK: %[[D:.*]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "d", uniq_name = "{{.*}}Ed"}
!CHECK: %[[DECLD:.*]]:2 = hlfir.declare %[[D]]

  !$acc exit data delete(a)
!CHECK: %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_delete>, name = "a", structured = false}
!CHECK: acc.exit_data dataOperands(%[[DEVPTR]] : !fir.ref<!fir.array<10x10xf32>>)
!CHECK: acc.delete accPtr(%[[DEVPTR]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc exit data delete(a) if(.true.)
!CHECK: %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_delete>, name = "a", structured = false}
!CHECK: %[[IF1:.*]] = arith.constant true
!CHECK: acc.exit_data if(%[[IF1]]) dataOperands(%[[DEVPTR]] : !fir.ref<!fir.array<10x10xf32>>)
!CHECK: acc.delete accPtr(%[[DEVPTR]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc exit data delete(a) if(ifCondition)
!CHECK: %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_delete>, name = "a", structured = false}
!CHECK: %[[IFCOND:.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
!CHECK: %[[IF2:.*]] = fir.convert %[[IFCOND]] : (!fir.logical<4>) -> i1
!CHECK: acc.exit_data if(%[[IF2]]) dataOperands(%[[DEVPTR]] : !fir.ref<!fir.array<10x10xf32>>){{$}}
!CHECK: acc.delete accPtr(%[[DEVPTR]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc exit data delete(a) delete(b) delete(c)
!CHECK: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_delete>, name = "a", structured = false}
!CHECK: %[[DEVPTR_B:.*]] = acc.getdeviceptr varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_delete>, name = "b", structured = false}
!CHECK: %[[DEVPTR_C:.*]] = acc.getdeviceptr varPtr(%[[DECLC]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_delete>, name = "c", structured = false}
!CHECK: acc.exit_data dataOperands(%[[DEVPTR_A]], %[[DEVPTR_B]], %[[DEVPTR_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>){{$}}
!CHECK: acc.delete accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}
!CHECK: acc.delete accPtr(%[[DEVPTR_B]] : !fir.ref<!fir.array<10x10xf32>>) {name = "b", structured = false}
!CHECK: acc.delete accPtr(%[[DEVPTR_C]] : !fir.ref<!fir.array<10x10xf32>>) {name = "c", structured = false}

  !$acc exit data copyout(a) delete(b) detach(d)
!CHECK: %[[DEVPTR_A:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_copyout>, name = "a", structured = false}
!CHECK: %[[DEVPTR_B:.*]] = acc.getdeviceptr varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_delete>, name = "b", structured = false}
!CHECK: %[[DEVPTR_D:.*]] = acc.getdeviceptr varPtr(%[[DECLD]]#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>> {dataClause = #acc<data_clause acc_detach>, name = "d", structured = false}
!CHECK: acc.exit_data dataOperands(%[[DEVPTR_A]], %[[DEVPTR_B]], %[[DEVPTR_D]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.box<!fir.ptr<f32>>>)
!CHECK: acc.copyout accPtr(%[[DEVPTR_A]] : !fir.ref<!fir.array<10x10xf32>>) to varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}
!CHECK: acc.delete accPtr(%[[DEVPTR_B]] : !fir.ref<!fir.array<10x10xf32>>) {name = "b", structured = false}
!CHECK: acc.detach accPtr(%[[DEVPTR_D]] : !fir.ref<!fir.box<!fir.ptr<f32>>>) {name = "d", structured = false}

  !$acc exit data delete(a) async
!CHECK: %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {asyncOnly = [#acc.device_type<none>], dataClause = #acc<data_clause acc_delete>, name = "a", structured = false}
!CHECK: acc.exit_data dataOperands(%[[DEVPTR]] : !fir.ref<!fir.array<10x10xf32>>) attributes {async}
!CHECK: acc.delete accPtr(%[[DEVPTR]] : !fir.ref<!fir.array<10x10xf32>>) {asyncOnly = [#acc.device_type<none>], name = "a", structured = false}

  !$acc exit data delete(a) wait
!CHECK: %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_delete>, name = "a", structured = false}
!CHECK: acc.exit_data dataOperands(%[[DEVPTR]] : !fir.ref<!fir.array<10x10xf32>>) attributes {wait}
!CHECK: acc.delete accPtr(%[[DEVPTR]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc exit data delete(a) async wait
!CHECK: %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {asyncOnly = [#acc.device_type<none>], dataClause = #acc<data_clause acc_delete>, name = "a", structured = false}
!CHECK: acc.exit_data dataOperands(%[[DEVPTR]] : !fir.ref<!fir.array<10x10xf32>>) attributes {async, wait}
!CHECK: acc.delete accPtr(%[[DEVPTR]] : !fir.ref<!fir.array<10x10xf32>>) {asyncOnly = [#acc.device_type<none>], name = "a", structured = false}

  !$acc exit data delete(a) async(1)
!CHECK: %[[ASYNC1:.*]] = arith.constant 1 : i32
!CHECK: %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) async(%[[ASYNC1]] : i32) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_delete>, name = "a", structured = false}
!CHECK: acc.exit_data async(%[[ASYNC1]] : i32) dataOperands(%[[DEVPTR]] : !fir.ref<!fir.array<10x10xf32>>)
!CHECK: acc.delete accPtr(%[[DEVPTR]] : !fir.ref<!fir.array<10x10xf32>>) async(%[[ASYNC1]] : i32) {name = "a", structured = false}


  !$acc exit data delete(a) async(async)
!CHECK: %[[ASYNC2:.*]] = fir.load %{{.*}} : !fir.ref<i32>
!CHECK: %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) async(%[[ASYNC2]] : i32) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_delete>, name = "a", structured = false}
!CHECK: acc.exit_data async(%[[ASYNC2]] : i32) dataOperands(%[[DEVPTR]] : !fir.ref<!fir.array<10x10xf32>>)
!CHECK: acc.delete accPtr(%[[DEVPTR]] : !fir.ref<!fir.array<10x10xf32>>) async(%[[ASYNC2]] : i32) {name = "a", structured = false}

  !$acc exit data delete(a) wait(1)
!CHECK: %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_delete>, name = "a", structured = false}
!CHECK: %[[WAIT1:.*]] = arith.constant 1 : i32
!CHECK: acc.exit_data wait(%[[WAIT1]] : i32) dataOperands(%[[DEVPTR]] : !fir.ref<!fir.array<10x10xf32>>)
!CHECK: acc.delete accPtr(%[[DEVPTR]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc exit data delete(a) wait(queues: 1, 2)
!CHECK: %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_delete>, name = "a", structured = false}
!CHECK: %[[WAIT2:.*]] = arith.constant 1 : i32
!CHECK: %[[WAIT3:.*]] = arith.constant 2 : i32
!CHECK: acc.exit_data wait(%[[WAIT2]], %[[WAIT3]] : i32, i32) dataOperands(%[[DEVPTR]] : !fir.ref<!fir.array<10x10xf32>>)
!CHECK: acc.delete accPtr(%[[DEVPTR]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

  !$acc exit data delete(a) wait(devnum: 1: queues: 1, 2)
!CHECK: %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[DECLA]]#0 : !fir.ref<!fir.array<10x10xf32>>) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = #acc<data_clause acc_delete>, name = "a", structured = false}
!CHECK: %[[WAIT4:.*]] = arith.constant 1 : i32
!CHECK: %[[WAIT5:.*]] = arith.constant 2 : i32
!CHECK: %[[WAIT6:.*]] = arith.constant 1 : i32
!CHECK: acc.exit_data wait_devnum(%[[WAIT6]] : i32) wait(%[[WAIT4]], %[[WAIT5]] : i32, i32) dataOperands(%[[DEVPTR]] : !fir.ref<!fir.array<10x10xf32>>)
!CHECK: acc.delete accPtr(%[[DEVPTR]] : !fir.ref<!fir.array<10x10xf32>>) {name = "a", structured = false}

end subroutine acc_exit_data
