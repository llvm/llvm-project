! This test checks lowering of OpenACC enter data directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s

subroutine acc_enter_data
  integer :: async = 1
  real, dimension(10, 10) :: a, b, c
  real, pointer :: d
  logical :: ifCondition = .TRUE.

!CHECK: %[[EXTENT_C10:.*]] = arith.constant 10 : index
!CHECK: %[[A:.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ea"}
!CHECK: %[[B:.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Eb"}
!CHECK: %[[C:.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ec"}
!CHECK: %[[D:.*]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "d", uniq_name = "{{.*}}Ed"}

  !$acc enter data create(a)
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>)   -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc enter data create(a) if(.true.)
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>)   -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: [[IF1:%.*]] = arith.constant true
!CHECK: acc.enter_data if([[IF1]]) dataOperands(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc enter data create(a) if(ifCondition)
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>)   -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: [[IFCOND:%.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
!CHECK: [[IF2:%.*]] = fir.convert [[IFCOND]] : (!fir.logical<4>) -> i1
!CHECK: acc.enter_data if([[IF2]]) dataOperands(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc enter data create(a) create(b) create(c)
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>)   -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: %[[CREATE_B:.*]] = acc.create varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>)   -> !fir.ref<!fir.array<10x10xf32>> {name = "b", structured = false}
!CHECK: %[[CREATE_C:.*]] = acc.create varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>)   -> !fir.ref<!fir.array<10x10xf32>> {name = "c", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE_A]], %[[CREATE_B]], %[[CREATE_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc enter data create(a) create(b) create(zero: c)
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>)   -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: %[[CREATE_B:.*]] = acc.create varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>)   -> !fir.ref<!fir.array<10x10xf32>> {name = "b", structured = false}
!CHECK: %[[CREATE_C:.*]] = acc.create varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>)   -> !fir.ref<!fir.array<10x10xf32>> {dataClause = 8 : i64, name = "c", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE_A]], %[[CREATE_B]], %[[CREATE_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc enter data copyin(a) create(b) attach(d)
!CHECK: %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>)   -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: %[[CREATE_B:.*]] = acc.create varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>)   -> !fir.ref<!fir.array<10x10xf32>> {name = "b", structured = false}
!CHECK: %[[ATTACH_D:.*]] = acc.attach varPtr(%[[D]] : !fir.ref<!fir.box<!fir.ptr<f32>>>)   -> !fir.ref<!fir.box<!fir.ptr<f32>>> {name = "d", structured = false}
!CHECK: acc.enter_data dataOperands(%[[COPYIN_A]], %[[CREATE_B]], %[[ATTACH_D]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.box<!fir.ptr<f32>>>){{$}}

  !$acc enter data create(a) async
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>)   -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>) attributes {async}

  !$acc enter data create(a) wait
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>)   -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>) attributes {wait}

  !$acc enter data create(a) async wait
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>)   -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>) attributes {async, wait}

  !$acc enter data create(a) async(1)
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>)   -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: %[[ASYNC1:.*]] = arith.constant 1 : i32
!CHECK: acc.enter_data async(%[[ASYNC1]] : i32) dataOperands(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc enter data create(a) async(async)
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>)   -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: %[[ASYNC2:.*]] = fir.load %{{.*}} : !fir.ref<i32>
!CHECK: acc.enter_data async(%[[ASYNC2]] : i32) dataOperands(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc enter data create(a) wait(1)
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>)   -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: %[[WAIT1:.*]] = arith.constant 1 : i32
!CHECK: acc.enter_data wait(%[[WAIT1]] : i32) dataOperands(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc enter data create(a) wait(queues: 1, 2)
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>)   -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: %[[WAIT2:.*]] = arith.constant 1 : i32
!CHECK: %[[WAIT3:.*]] = arith.constant 2 : i32
!CHECK: acc.enter_data wait(%[[WAIT2]], %[[WAIT3]] : i32, i32) dataOperands(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc enter data create(a) wait(devnum: 1: queues: 1, 2)
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>)   -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: %[[WAIT4:.*]] = arith.constant 1 : i32
!CHECK: %[[WAIT5:.*]] = arith.constant 2 : i32
!CHECK: %[[WAIT6:.*]] = arith.constant 1 : i32
!CHECK: acc.enter_data wait_devnum(%[[WAIT6]] : i32) wait(%[[WAIT4]], %[[WAIT5]] : i32, i32) dataOperands(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc enter data copyin(a(1:10,1:5))
!CHECK: %[[LB1:.*]] = arith.constant 1 : i64
!CHECK: %[[UB1:.*]] = arith.constant 10 : i64
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB1]] : i64) upperbound(%[[UB1]] : i64)
!CHECK: %[[LB2:.*]] = arith.constant 1 : i64
!CHECK: %[[UB2:.*]] = arith.constant 5 : i64
!CHECK: %[[BOUND2:.*]] = acc.bounds lowerbound(%[[LB2]] : i64) upperbound(%[[UB2]] : i64)
!CHECK: %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND1]], %[[BOUND2]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a(1:10,1:5)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[COPYIN_A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc enter data copyin(a(1:,1:5))
!CHECK: %[[LB1:.*]] = arith.constant 1 : i64
!CHECK: %[[BOUND1:.*]] = acc.bounds   lowerbound(%[[LB1]] : i64) extent(%[[EXTENT_C10]] : index)
!CHECK: %[[LB2:.*]] = arith.constant 1 : i64
!CHECK: %[[UB2:.*]] = arith.constant 5 : i64
!CHECK: %[[BOUND2:.*]] = acc.bounds   lowerbound(%[[LB2]] : i64) upperbound(%[[UB2]] : i64)
!CHECK: %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>)   bounds(%[[BOUND1]], %[[BOUND2]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a(1:,1:5)", structured = false}
!CHECK: acc.enter_data   dataOperands(%[[COPYIN_A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc enter data copyin(a(:10,1:5))
!CHECK: %[[UB1:.*]] = arith.constant 10 : i64
!CHECK: %[[BOUND1:.*]] = acc.bounds   upperbound(%[[UB1]] : i64)
!CHECK: %[[LB2:.*]] = arith.constant 1 : i64
!CHECK: %[[UB2:.*]] = arith.constant 5 : i64
!CHECK: %[[BOUND2:.*]] = acc.bounds   lowerbound(%[[LB2]] : i64) upperbound(%[[UB2]] : i64)
!CHECK: %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>)   bounds(%[[BOUND1]], %[[BOUND2]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a(:10,1:5)", structured = false}
!CHECK: acc.enter_data   dataOperands(%[[COPYIN_A]] : !fir.ref<!fir.array<10x10xf32>>)

end subroutine acc_enter_data


subroutine acc_enter_data_dummy(a)
  real :: a(1:10)

!CHECK-LABEL: func.func @_QPacc_enter_data_dummy
!CHECK-SAME:    %[[A:.*]]: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "a"}

  !$acc enter data create(a(5:10))
!CHECK: %[[LB1:.*]] = arith.constant 5 : i64
!CHECK: %[[UB1:.*]] = arith.constant 10 : i64
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB1]] : i64) upperbound(%[[UB1]] : i64)
!CHECK: %[[CREATE1:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>)   bounds(%[[BOUND1]]) -> !fir.ref<!fir.array<10xf32>> {name = "a(5:10)", structured = false}
!CHECK: acc.enter_data   dataOperands(%[[CREATE1]] : !fir.ref<!fir.array<10xf32>>)

end subroutine
