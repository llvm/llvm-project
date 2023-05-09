! This test checks lowering of OpenACC enter data directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s

subroutine acc_enter_data
  integer :: async = 1
  real, dimension(10, 10) :: a, b, c
  real, pointer :: d
  logical :: ifCondition = .TRUE.

!CHECK: %[[C10:.*]] = arith.constant 10 : index
!CHECK: %[[EXTENT_C10:.*]] = arith.constant 10 : index
!CHECK: %[[A:.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ea"}
!CHECK: %[[B:.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Eb"}
!CHECK: %[[C:.*]] = fir.alloca !fir.array<10x10xf32> {{{.*}}uniq_name = "{{.*}}Ec"}
!CHECK: %[[D:.*]] = fir.alloca !fir.box<!fir.ptr<f32>> {bindc_name = "d", uniq_name = "{{.*}}Ed"}

  !$acc enter data create(a)
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[LB:.*]] = arith.constant 0 : index
!CHECK: %[[UB:.*]] = arith.subi %[[C10]], %[[ONE]] : index
!CHECK: %[[BOUND0:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[ONE]] : index) startIdx(%[[ONE]] : index)
!CHECK: %[[LB:.*]] = arith.constant 0 : index
!CHECK: %[[UB:.*]] = arith.subi %[[EXTENT_C10]], %[[ONE]] : index
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[ONE]] : index) startIdx(%[[ONE]] : index)
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND0]], %[[BOUND1]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc enter data create(a) if(.true.)
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[LB:.*]] = arith.constant 0 : index
!CHECK: %[[UB:.*]] = arith.subi %[[C10]], %[[ONE]] : index
!CHECK: %[[BOUND0:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[ONE]] : index) startIdx(%[[ONE]] : index)
!CHECK: %[[LB:.*]] = arith.constant 0 : index
!CHECK: %[[UB:.*]] = arith.subi %[[EXTENT_C10]], %[[ONE]] : index
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[ONE]] : index) startIdx(%[[ONE]] : index)
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND0]], %[[BOUND1]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: [[IF1:%.*]] = arith.constant true
!CHECK: acc.enter_data if([[IF1]]) dataOperands(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc enter data create(a) if(ifCondition)
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[LB:.*]] = arith.constant 0 : index
!CHECK: %[[UB:.*]] = arith.subi %[[C10]], %[[ONE]] : index
!CHECK: %[[BOUND0:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[ONE]] : index) startIdx(%[[ONE]] : index)
!CHECK: %[[LB:.*]] = arith.constant 0 : index
!CHECK: %[[UB:.*]] = arith.subi %[[EXTENT_C10]], %[[ONE]] : index
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[ONE]] : index) startIdx(%[[ONE]] : index)
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND0]], %[[BOUND1]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: [[IFCOND:%.*]] = fir.load %{{.*}} : !fir.ref<!fir.logical<4>>
!CHECK: [[IF2:%.*]] = fir.convert [[IFCOND]] : (!fir.logical<4>) -> i1
!CHECK: acc.enter_data if([[IF2]]) dataOperands(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc enter data create(a) create(b) create(c)
!CHECK: %[[BOUND0:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND0]], %[[BOUND1]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: %[[BOUND0:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[CREATE_B:.*]] = acc.create varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND0]], %[[BOUND1]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "b", structured = false}
!CHECK: %[[BOUND0:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[CREATE_C:.*]] = acc.create varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND0]], %[[BOUND1]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "c", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE_A]], %[[CREATE_B]], %[[CREATE_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc enter data create(a) create(b) create(zero: c)
!CHECK: %[[BOUND0:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND0]], %[[BOUND1]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: %[[BOUND0:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[CREATE_B:.*]] = acc.create varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND0]], %[[BOUND1]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "b", structured = false}
!CHECK: %[[BOUND0:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[CREATE_C:.*]] = acc.create varPtr(%[[C]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND0]], %[[BOUND1]]) -> !fir.ref<!fir.array<10x10xf32>> {dataClause = 8 : i64, name = "c", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE_A]], %[[CREATE_B]], %[[CREATE_C]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>){{$}}

  !$acc enter data copyin(a) create(b) attach(d)
!CHECK: %[[BOUND0:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND0]], %[[BOUND1]])  -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: %[[BOUND0:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[CREATE_B:.*]] = acc.create varPtr(%[[B]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND0]], %[[BOUND1]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "b", structured = false}
!CHECK: %[[BOX_D:.*]] = fir.load %[[D]] : !fir.ref<!fir.box<!fir.ptr<f32>>> 
!CHECK: %[[BOX_ADDR_D:.*]] = fir.box_addr %[[BOX_D]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
!CHECK: %[[ATTACH_D:.*]] = acc.attach varPtr(%[[BOX_ADDR_D]] : !fir.ptr<f32>) -> !fir.ptr<f32> {name = "d", structured = false}
!CHECK: acc.enter_data dataOperands(%[[COPYIN_A]], %[[CREATE_B]], %[[ATTACH_D]] : !fir.ref<!fir.array<10x10xf32>>, !fir.ref<!fir.array<10x10xf32>>, !fir.ptr<f32>){{$}}

  !$acc enter data create(a) async
!CHECK: %[[BOUND0:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND0]], %[[BOUND1]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>) attributes {async}

  !$acc enter data create(a) wait
!CHECK: %[[BOUND0:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND0]], %[[BOUND1]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>) attributes {wait}

  !$acc enter data create(a) async wait
!CHECK: %[[BOUND0:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND0]], %[[BOUND1]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>) attributes {async, wait}

  !$acc enter data create(a) async(1)
!CHECK: %[[BOUND0:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND0]], %[[BOUND1]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: %[[ASYNC1:.*]] = arith.constant 1 : i32
!CHECK: acc.enter_data async(%[[ASYNC1]] : i32) dataOperands(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc enter data create(a) async(async)
!CHECK: %[[BOUND0:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND0]], %[[BOUND1]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: %[[ASYNC2:.*]] = fir.load %{{.*}} : !fir.ref<i32>
!CHECK: acc.enter_data async(%[[ASYNC2]] : i32) dataOperands(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc enter data create(a) wait(1)
!CHECK: %[[BOUND0:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND0]], %[[BOUND1]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: %[[WAIT1:.*]] = arith.constant 1 : i32
!CHECK: acc.enter_data wait(%[[WAIT1]] : i32) dataOperands(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc enter data create(a) wait(queues: 1, 2)
!CHECK: %[[BOUND0:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND0]], %[[BOUND1]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: %[[WAIT2:.*]] = arith.constant 1 : i32
!CHECK: %[[WAIT3:.*]] = arith.constant 2 : i32
!CHECK: acc.enter_data wait(%[[WAIT2]], %[[WAIT3]] : i32, i32) dataOperands(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc enter data create(a) wait(devnum: 1: queues: 1, 2)
!CHECK: %[[BOUND0:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[CREATE_A:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND0]], %[[BOUND1]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a", structured = false}
!CHECK: %[[WAIT4:.*]] = arith.constant 1 : i32
!CHECK: %[[WAIT5:.*]] = arith.constant 2 : i32
!CHECK: %[[WAIT6:.*]] = arith.constant 1 : i32
!CHECK: acc.enter_data wait_devnum(%[[WAIT6]] : i32) wait(%[[WAIT4]], %[[WAIT5]] : i32, i32) dataOperands(%[[CREATE_A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc enter data copyin(a(1:10,1:5))
!CHECK: %[[BOUND0:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND0]], %[[BOUND1]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a(1:10,1:5)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[COPYIN_A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc enter data copyin(a(1:,1:5))
!CHECK: %[[ONE:.*]] = arith.constant 1 : index 
!CHECK: %[[LB1:.*]] = arith.constant 0 : index
!CHECK: %[[LBEXT:.*]] = arith.addi %c10{{.*}}, %[[ONE]] : index
!CHECK: %[[UB1:.*]] = arith.subi %[[LBEXT]], %[[ONE]] : index 
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB1]] : index) upperbound(%[[UB1]] : index) stride(%[[ONE]] : index) startIdx(%c1{{.*}} : index)
!CHECK: %[[LB2:.*]] = arith.constant 0 : index
!CHECK: %[[UB2:.*]] = arith.constant 4 : index
!CHECK: %[[BOUND2:.*]] = acc.bounds lowerbound(%[[LB2]] : index) upperbound(%[[UB2]] : index) stride(%[[ONE]] : index) startIdx(%c1{{.*}} : index)
!CHECK: %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND1]], %[[BOUND2]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a(1:,1:5)", structured = false}
!CHECK: acc.enter_data   dataOperands(%[[COPYIN_A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc enter data copyin(a(:10,1:5))
!CHECK: %[[LB:.*]] = arith.constant 0 : index
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[UB1:.*]] = arith.constant 9 : index
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB1]] : index) stride(%[[ONE]] : index) startIdx(%[[ONE]] : index)
!CHECK: %[[LB:.*]] = arith.constant 0 : index
!CHECK: %[[UB2:.*]] = arith.constant 4 : index
!CHECK: %[[BOUND2:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB2]] : index) stride(%[[ONE]] : index) startIdx(%[[ONE]] : index)
!CHECK: %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND1]], %[[BOUND2]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a(:10,1:5)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[COPYIN_A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc enter data copyin(a(:,:))
!CHECK: %[[LB:.*]] = arith.constant 0 : index
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[LBEXT:.*]] = arith.addi %c10{{.*}}, %[[ONE]] : index
!CHECK: %[[UB:.*]] = arith.subi %[[LBEXT]], %[[ONE]] : index
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[ONE]] : index) startIdx(%[[ONE]] : index)
!CHECK: %[[LBEXT:.*]] = arith.addi %c10{{.*}}, %[[ONE]] : index
!CHECK: %[[UB:.*]] = arith.subi %[[LBEXT]], %[[ONE]] : index 
!CHECK: %[[BOUND2:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[ONE]] : index) startIdx(%[[ONE]] : index)
!CHECK: %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND1]], %[[BOUND2]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a(:,:)", structured = false}
end subroutine acc_enter_data

subroutine acc_enter_data_dummy(a, b, n, m)
  integer :: n, m
  real :: a(1:10)
  real :: b(n:m)

!CHECK-LABEL: func.func @_QPacc_enter_data_dummy
!CHECK-SAME:    %[[A:.*]]: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "a"}, %[[B:.*]]: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "b"}, %[[N:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}, %[[M:.*]]: !fir.ref<i32> {fir.bindc_name = "m"}

!CHECK: %[[C10:.*]] = arith.constant 10 : index
!CHECK: %[[LOAD_N:.*]] = fir.load %[[N]] : !fir.ref<i32>
!CHECK: %[[N_I64:.*]] = fir.convert %[[LOAD_N]] : (i32) -> i64
!CHECK: %[[N_IDX:.*]] = fir.convert %[[N_I64]] : (i64) -> index
!CHECK: %[[LOAD_M:.*]] = fir.load %[[M]] : !fir.ref<i32>
!CHECK: %[[M_I64:.*]] = fir.convert %[[LOAD_M]] : (i32) -> i64
!CHECK: %[[M_IDX:.*]] = fir.convert %[[M_I64]] : (i64) -> index
!CHECK: %[[M_N:.*]] = arith.subi %[[M_IDX]], %[[N_IDX]] : index
!CHECK: %[[C1:.*]] = arith.constant 1 : index
!CHECK: %[[M_N_1:.*]] = arith.addi %[[M_N]], %[[C1]] : index
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[CMP:.*]] = arith.cmpi sgt, %[[M_N_1]], %[[C0]] : index
!CHECK: %[[EXT_B:.*]] = arith.select %[[CMP]], %[[M_N_1]], %[[C0]] : index

  !$acc enter data create(a)
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {name = "a", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<10xf32>>)

  !$acc enter data create(b)
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%{{.*}} : index)
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[B]] : !fir.ref<!fir.array<?xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<?xf32>> {name = "b", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<?xf32>>)

  !$acc enter data create(a(5:10))
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[LB1:.*]] = arith.constant 4 : index
!CHECK: %[[UB1:.*]] = arith.constant 9 : index
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB1]] : index) upperbound(%[[UB1]] : index) stride(%[[ONE]] : index) startIdx(%c1{{.*}} : index)
!CHECK: %[[CREATE1:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND1]]) -> !fir.ref<!fir.array<10xf32>> {name = "a(5:10)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE1]] : !fir.ref<!fir.array<10xf32>>)

  !$acc enter data create(b(n:m))
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[LOAD_N:.*]] = fir.load %[[N]] : !fir.ref<i32> 
!CHECK: %[[CONVERT_N:.*]] = fir.convert %[[LOAD_N]] : (i32) -> index
!CHECK: %[[LB:.*]] = arith.subi %[[CONVERT_N]], %[[N_IDX]] : index
!CHECK: %[[LOAD_M:.*]] = fir.load %[[M]] : !fir.ref<i32>
!CHECK: %[[CONVERT_M:.*]] = fir.convert %[[LOAD_M]] : (i32) -> index
!CHECK: %[[UB:.*]] = arith.subi %[[CONVERT_M]], %[[N_IDX]] : index
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[ONE]] : index) startIdx(%[[N_IDX]] : index)
!CHECK: %[[CREATE1:.*]] = acc.create varPtr(%[[B]] : !fir.ref<!fir.array<?xf32>>) bounds(%[[BOUND1]]) -> !fir.ref<!fir.array<?xf32>> {name = "b(n:m)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE1]] : !fir.ref<!fir.array<?xf32>>)

  !$acc enter data create(b(n:))
!CHECK: %[[ONE:.*]] = arith.constant 1 : index 
!CHECK: %[[LOAD_N:.*]] = fir.load %[[N]] : !fir.ref<i32> 
!CHECK: %[[CONVERT_N:.*]] = fir.convert %[[LOAD_N]] : (i32) -> index
!CHECK: %[[LB:.*]] = arith.subi %[[CONVERT_N]], %[[N_IDX]] : index
!CHECK: %[[LBEXT:.*]] = arith.addi %[[EXT_B]], %[[N_IDX]] : index
!CHECK: %[[UB:.*]] = arith.subi %[[LBEXT:.*]], %[[ONE]] : index 
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[ONE]] : index) startIdx(%[[N_IDX]] : index) 
!CHECK: %[[CREATE1:.*]] = acc.create varPtr(%[[B]] : !fir.ref<!fir.array<?xf32>>) bounds(%[[BOUND1]]) -> !fir.ref<!fir.array<?xf32>> {name = "b(n:)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE1]] : !fir.ref<!fir.array<?xf32>>)

  !$acc enter data create(b(:))
!CHECK: %[[ONE:.*]] = arith.constant 1 : index 
!CHECK: %[[LBEXT:.*]] = arith.addi %[[EXT_B]], %[[N_IDX]] : index 
!CHECK: %[[UB:.*]] = arith.subi %[[LBEXT]], %[[ONE]] : index
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[N_IDX]] : index) upperbound(%[[UB]] : index) stride(%[[ONE]] : index) startIdx(%[[N_IDX]] : index)
!CHECK: %[[CREATE1:.*]] = acc.create varPtr(%[[B]] : !fir.ref<!fir.array<?xf32>>) bounds(%[[BOUND1]]) -> !fir.ref<!fir.array<?xf32>> {name = "b(:)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE1]] : !fir.ref<!fir.array<?xf32>>)

end subroutine

! Test lowering of array section for non default lower bound.
subroutine acc_enter_data_non_default_lb()
  integer :: a(0:9)
  integer :: b(11:20)

!CHECK-LABEL: func.func @_QPacc_enter_data_non_default_lb() {
!CHECK: %[[BASELB:.*]] = arith.constant 0 : index
!CHECK: %[[EXTENT_C10:.*]] = arith.constant 10 : index
!CHECK: %[[A:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "a", uniq_name = "_QFacc_enter_data_non_default_lbEa"}

  !$acc enter data create(a(5:9))
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[SECTIONLB:.*]] = arith.constant 5 : index
!CHECK: %[[LB:.*]] = arith.subi %[[SECTIONLB]], %[[BASELB]] : index
!CHECK: %[[SECTIONUB:.*]] = arith.constant 9 : index
!CHECK: %[[UB:.*]] = arith.subi %[[SECTIONUB]], %[[BASELB]] : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[ONE]] : index) startIdx(%[[BASELB]] : index)
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10xi32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xi32>> {name = "a(5:9)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<10xi32>>)

  !$acc enter data create(a(:))
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[LBEXT:.*]] = arith.addi %[[EXTENT_C10]], %[[BASELB]] : index
!CHECK: %[[UB:.*]] = arith.subi %[[LBEXT]], %[[ONE]] : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[BASELB]] : index) upperbound(%[[UB]] : index) stride(%[[ONE]] : index) startIdx(%[[BASELB]] : index)
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10xi32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xi32>> {name = "a(:)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<10xi32>>)

  !$acc enter data create(a(:6))
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[SECTIONUB:.*]] = arith.constant 6 : index
!CHECK: %[[UB:.*]] = arith.subi %[[SECTIONUB]], %[[BASELB]] : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[BASELB]] : index) upperbound(%[[UB]] : index) stride(%[[ONE]] : index) startIdx(%[[BASELB]] : index)
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10xi32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xi32>> {name = "a(:6)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<10xi32>>)

  !$acc enter data create(a(4:))
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[SECTIONLB:.*]] = arith.constant 4 : index
!CHECK: %[[LB:.*]] = arith.subi %[[SECTIONLB]], %[[BASELB]] : index
!CHECK: %[[LBEXT:.*]] = arith.addi %[[EXTENT_C10]], %[[BASELB]] : index
!CHECK: %[[UB:.*]] = arith.subi %[[LBEXT]], %[[ONE]] : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[ONE]] : index) startIdx(%[[BASELB]] : index)
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10xi32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xi32>> {name = "a(4:)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<10xi32>>)

  !$acc enter data create(b)
!CHECK: %[[ONE:.*]] = arith.constant 1 : index 
!CHECK: %[[LB:.*]] = arith.constant 0 : index 
!CHECK: %[[UB:.*]] = arith.subi %c10{{.*}}, %[[ONE]] : index 
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[ONE]] : index) startIdx(%c11{{.*}} : index) 
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%1 : !fir.ref<!fir.array<10xi32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xi32>> {name = "b", structured = false} 
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<10xi32>>)

end subroutine

! Test lowering of assumed size arrays.
subroutine acc_enter_data_assumed(a, b, n, m)
  integer :: n, m
  real :: a(:)
  real :: b(10:)

!CHECK-LABEL: func.func @_QPacc_enter_data_assumed(
!CHECK-SAME: %[[A:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "a"}, %[[B:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "b"}, %[[N:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}, %[[M:.*]]: !fir.ref<i32> {fir.bindc_name = "m"}) {

!CHECK: %[[LB_C10:.*]] = arith.constant 10 : i64
!CHECK: %[[LB_C10_IDX:.*]] = fir.convert %[[LB_C10]] : (i64) -> index

  !$acc enter data create(a)
!CHECK: %[[C1:.*]] = arith.constant 1 : index
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS:.*]]:3 = fir.box_dims %[[A]], %[[C0]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
!CHECK: %[[LB:.*]] = arith.constant 0 : index 
!CHECK: %[[UB:.*]] = arith.subi %[[DIMS]]#1, %[[C1]] : index 
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[DIMS]]#2 : index) startIdx(%[[C1]] : index) {strideInBytes = true} 
!CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[A]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[BOX_ADDR]] : !fir.ref<!fir.array<?xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<?xf32>> {name = "a", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<?xf32>>)

  !$acc enter data create(a(:))
!CHECK: %[[LB:.*]] = arith.constant 0 : index
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS0:.*]]:3 = fir.box_dims %[[A]], %[[C0]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS1:.*]]:3 = fir.box_dims %[[A]], %[[C0]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
!CHECK: %[[LBEXT:.*]] = arith.addi %[[DIMS1]]#1, %[[ONE]] : index
!CHECK: %[[UB:.*]] = arith.subi %[[LBEXT]], %[[ONE]] : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[DIMS0]]#2 : index) startIdx(%[[ONE]] : index) {strideInBytes = true}
!CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[A]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[BOX_ADDR]] : !fir.ref<!fir.array<?xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<?xf32>> {name = "a(:)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<?xf32>>)

  !$acc enter data create(a(2:))
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS0:.*]]:3 = fir.box_dims %[[A]], %[[C0]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
!CHECK: %[[LB:.*]] = arith.constant 1 : index
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS1:.*]]:3 = fir.box_dims %[[A]], %[[C0]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
!CHECK: %[[LBEXT:.*]] = arith.addi %[[DIMS1]]#1, %[[ONE]] : index
!CHECK: %[[UB:.*]] = arith.subi %[[LBEXT]], %[[ONE]] : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[DIMS0]]#2 : index) startIdx(%[[ONE]] : index) {strideInBytes = true}
!CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[A]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[BOX_ADDR]] : !fir.ref<!fir.array<?xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<?xf32>> {name = "a(2:)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<?xf32>>)

  !$acc enter data create(a(:4))
!CHECK: %[[LB:.*]] = arith.constant 0 : index
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS0:.*]]:3 = fir.box_dims %[[A]], %[[C0]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
!CHECK: %[[UB:.*]] = arith.constant 3 : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[DIMS0]]#2 : index) startIdx(%[[ONE]] : index) {strideInBytes = true}
!CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[A]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[BOX_ADDR]] : !fir.ref<!fir.array<?xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<?xf32>> {name = "a(:4)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<?xf32>>)

  !$acc enter data create(a(6:10))
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS0:.*]]:3 = fir.box_dims %[[A]], %[[C0]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
!CHECK: %[[LB:.*]] = arith.constant 5 : index
!CHECK: %[[UB:.*]] = arith.constant 9 : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[DIMS0]]#2 : index) startIdx(%[[ONE]] : index) {strideInBytes = true}
!CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[A]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[BOX_ADDR]] : !fir.ref<!fir.array<?xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<?xf32>> {name = "a(6:10)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<?xf32>>)

  !$acc enter data create(a(n:))
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS0:.*]]:3 = fir.box_dims %[[A]], %[[C0]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
!CHECK: %[[LOAD_N:.*]] = fir.load %[[N]] : !fir.ref<i32>
!CHECK: %[[CONVERT_N:.*]] = fir.convert %[[LOAD_N]] : (i32) -> index
!CHECK: %[[LB:.*]] = arith.subi %[[CONVERT_N]], %[[ONE]] : index
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS:.*]]:3 = fir.box_dims %[[A]], %[[C0]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
!CHECK: %[[LBEXT:.*]] = arith.addi %[[DIMS]]#1, %[[ONE]] : index
!CHECK: %[[UB:.*]] = arith.subi %[[LBEXT]], %[[ONE]] : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[DIMS0]]#2 : index) startIdx(%[[ONE]] : index) {strideInBytes = true}
!CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[A]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[BOX_ADDR]] : !fir.ref<!fir.array<?xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<?xf32>> {name = "a(n:)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<?xf32>>)

  !$acc enter data create(a(:m))
!CHECK: %[[BASELB:.*]] = arith.constant 0 : index
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS0:.*]]:3 = fir.box_dims %[[A]], %[[C0]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
!CHECK: %[[LOAD_M:.*]] = fir.load %[[M]] : !fir.ref<i32>
!CHECK: %[[CONVERT_M:.*]] = fir.convert %[[LOAD_M]] : (i32) -> index
!CHECK: %[[UB:.*]] = arith.subi %[[CONVERT_M]], %[[ONE]] : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[BASELB]] : index) upperbound(%[[UB]] : index) stride(%[[DIMS0]]#2 : index) startIdx(%[[ONE]] : index) {strideInBytes = true}
!CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[A]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[BOX_ADDR]] : !fir.ref<!fir.array<?xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<?xf32>> {name = "a(:m)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<?xf32>>)

  !$acc enter data create(a(n:m))
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS0:.*]]:3 = fir.box_dims %[[A]], %[[C0]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
!CHECK: %[[LOAD_N:.*]] = fir.load %[[N]] : !fir.ref<i32>
!CHECK: %[[CONVERT_N:.*]] = fir.convert %[[LOAD_N]] : (i32) -> index
!CHECK: %[[LB:.*]] = arith.subi %[[CONVERT_N]], %[[ONE]] : index
!CHECK: %[[LOAD_M:.*]] = fir.load %[[M]] : !fir.ref<i32>
!CHECK: %[[CONVERT_M:.*]] = fir.convert %[[LOAD_M]] : (i32) -> index
!CHECK: %[[UB:.*]] = arith.subi %[[CONVERT_M]], %[[ONE]] : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[DIMS0]]#2 : index) startIdx(%[[ONE]] : index) {strideInBytes = true}
!CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[A]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[BOX_ADDR]] : !fir.ref<!fir.array<?xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<?xf32>> {name = "a(n:m)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<?xf32>>)

  !$acc enter data create(b(:m))
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS0:.*]]:3 = fir.box_dims %[[B]], %[[C0]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
!CHECK: %[[LOAD_M:.*]] = fir.load %[[M]] : !fir.ref<i32>
!CHECK: %[[CONVERT_M:.*]] = fir.convert %[[LOAD_M]] : (i32) -> index
!CHECK: %[[UB:.*]] = arith.subi %[[CONVERT_M]], %[[LB_C10_IDX]] : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB_C10_IDX]] : index) upperbound(%[[UB]] : index) stride(%[[DIMS0]]#2 : index) startIdx(%[[LB_C10_IDX]] : index) {strideInBytes = true}
!CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[B]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[BOX_ADDR]] : !fir.ref<!fir.array<?xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<?xf32>> {name = "b(:m)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<?xf32>>)

  !$acc enter data create(b)
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS0:.*]]:3 = fir.box_dims %[[B]], %[[C0]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[UB:.*]] = arith.subi %[[DIMS0]]#1, %[[ONE]] : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[C0]] : index) upperbound(%[[UB]] : index) stride(%[[DIMS0]]#2 : index) startIdx(%[[LB_C10_IDX]] : index) {strideInBytes = true}
!CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[B]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[BOX_ADDR]] : !fir.ref<!fir.array<?xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<?xf32>> {name = "b", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<?xf32>>)

end subroutine

subroutine acc_enter_data_allocatable()
  real, allocatable :: a(:)
  integer, allocatable :: i
  
!CHECK-LABEL: func.func @_QPacc_enter_data_allocatable() {
!CHECK: %[[A:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", uniq_name = "_QFacc_enter_data_allocatableEa"}
!CHECK: %[[I:.*]] = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "i", uniq_name = "_QFacc_enter_data_allocatableEi"}

  !$acc enter data create(a)
!CHECK: %[[BOX_A_0:.*]] = fir.load %[[A]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK: %[[C0_0:.*]] = arith.constant 0 : index
!CHECK: %[[BOX_A_1:.*]] = fir.load %[[A]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK: %[[C0_1:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS0:.*]]:3 = fir.box_dims %[[BOX_A_1]], %[[C0_1]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
!CHECK: %[[DIMS1:.*]]:3 = fir.box_dims %[[BOX_A_0]], %[[C0_0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
!CHECK: %[[UB:.*]] = arith.subi %[[DIMS1]]#1, %c1{{.*}} : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%c0{{.*}} : index) upperbound(%[[UB]] : index) stride(%[[DIMS1]]#2 : index) startIdx(%[[DIMS0]]#0 : index) {strideInBytes = true}
!CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX_A_0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[BOX_ADDR]] : !fir.heap<!fir.array<?xf32>>) bounds(%[[BOUND]]) -> !fir.heap<!fir.array<?xf32>> {name = "a", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.heap<!fir.array<?xf32>>)

  !$acc enter data create(a(:))
!CHECK: %[[BOX_A_0:.*]] = fir.load %[[A]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[BOX_A_1:.*]] = fir.load %[[A]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS0:.*]]:3 = fir.box_dims %[[BOX_A_1]], %[[C0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS1:.*]]:3 = fir.box_dims %[[BOX_A_0]], %[[C0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
!CHECK: %[[BOX_A_2:.*]] = fir.load %[[A]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS2:.*]]:3 = fir.box_dims %[[BOX_A_2]], %[[C0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
!CHECK: %[[LBEXT:.*]] = arith.addi %[[DIMS2]]#1, %[[DIMS0]]#0 : index
!CHECK: %[[UB:.*]] = arith.subi %[[LBEXT]], %[[ONE]] : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[DIMS0]]#0 : index) upperbound(%[[UB:.*]] : index) stride(%[[DIMS1]]#2 : index) startIdx(%[[DIMS0]]#0 : index) {strideInBytes = true}
!CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX_A_0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[BOX_ADDR]] : !fir.heap<!fir.array<?xf32>>) bounds(%[[BOUND]]) -> !fir.heap<!fir.array<?xf32>> {name = "a(:)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.heap<!fir.array<?xf32>>)

  !$acc enter data create(a(2:5))
!CHECK: %[[BOX_A_0:.*]] = fir.load %[[A]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK: %[[BOX_A_1:.*]] = fir.load %[[A]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS0:.*]]:3 = fir.box_dims %[[BOX_A_1]], %[[C0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS1:.*]]:3 = fir.box_dims %[[BOX_A_0]], %[[C0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
!CHECK: %[[C2:.*]] = arith.constant 2 : index
!CHECK: %[[LB:.*]] = arith.subi %[[C2]], %[[DIMS0]]#0 : index
!CHECK: %[[C5:.*]] = arith.constant 5 : index
!CHECK: %[[UB:.*]] = arith.subi %[[C5]], %[[DIMS0]]#0 : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[DIMS1]]#2 : index) startIdx(%[[DIMS0]]#0 : index) {strideInBytes = true}
!CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX_A_0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[BOX_ADDR]] : !fir.heap<!fir.array<?xf32>>) bounds(%[[BOUND]]) -> !fir.heap<!fir.array<?xf32>> {name = "a(2:5)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.heap<!fir.array<?xf32>>)

  !$acc enter data create(a(3:))
!CHECK: %[[BOX_A_0:.*]] = fir.load %[[A]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[BOX_A_1:.*]] = fir.load %[[A]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS0:.*]]:3 = fir.box_dims %[[BOX_A_1]], %[[C0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS1:.*]]:3 = fir.box_dims %[[BOX_A_0]], %[[C0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
!CHECK: %[[C3:.*]] = arith.constant 3 : index
!CHECK: %[[LB:.*]] = arith.subi %[[C3]], %[[DIMS0]]#0 : index
!CHECK: %[[BOX_A_1:.*]] = fir.load %[[A]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS2:.*]]:3 = fir.box_dims %[[BOX_A_1]], %[[C0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
!CHECK: %[[LBEXT:.*]] = arith.addi %[[DIMS2:.*]]#1, %[[DIMS0]]#0 : index
!CHECK: %[[UB:.*]] = arith.subi %[[LBEXT]], %[[ONE]] : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[DIMS1]]#2 : index) startIdx(%[[DIMS0]]#0 : index) {strideInBytes = true}
!CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX_A_0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[BOX_ADDR]] : !fir.heap<!fir.array<?xf32>>) bounds(%[[BOUND]]) -> !fir.heap<!fir.array<?xf32>> {name = "a(3:)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.heap<!fir.array<?xf32>>)

  !$acc enter data create(a(:7))
!CHECK: %[[BOX_A_0:.*]] = fir.load %[[A]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK: %[[BOX_A_1:.*]] = fir.load %[[A]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS0:.*]]:3 = fir.box_dims %[[BOX_A_1]], %[[C0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS1:.*]]:3 = fir.box_dims %[[BOX_A_0]], %[[C0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
!CHECK: %[[C7:.*]] = arith.constant 7 : index
!CHECK: %[[UB:.*]] = arith.subi %[[C7]], %[[DIMS0]]#0 : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[DIMS0]]#0 : index) upperbound(%[[UB]] : index) stride(%[[DIMS1]]#2 : index) startIdx(%[[DIMS0]]#0 : index) {strideInBytes = true}
!CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX_A_0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[BOX_ADDR]] : !fir.heap<!fir.array<?xf32>>) bounds(%[[BOUND]]) -> !fir.heap<!fir.array<?xf32>> {name = "a(:7)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.heap<!fir.array<?xf32>>)

  !$acc enter data create(i)
!CHECK: %[[BOX_I:.*]] = fir.load %[[I]] : !fir.ref<!fir.box<!fir.heap<i32>>>
!CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX_I]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[BOX_ADDR]] : !fir.heap<i32>) -> !fir.heap<i32> {name = "i", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.heap<i32>)

end subroutine

subroutine acc_enter_data_derived_type()
  type :: dt
    real :: data
    real :: array(1:10)
  end type

  type :: t
    type(dt) :: d
  end type

  type :: z
    integer, allocatable :: data(:)
  end type

  type :: tt
    type(dt) :: d(10)
  end type

  type(dt) :: a
  type(t) :: b
  type(dt) :: aa(10)
  type(z) :: c
  type(tt) :: d

!CHECK-LABEL: func.func @_QPacc_enter_data_derived_type() {
!CHECK: %[[A:.*]] = fir.alloca !fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}> {bindc_name = "a", uniq_name = "_QFacc_enter_data_derived_typeEa"}
!CHECK: %[[AA:.*]] = fir.alloca !fir.array<10x!fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>> {bindc_name = "aa", uniq_name = "_QFacc_enter_data_derived_typeEaa"}
!CHECK: %[[B:.*]] = fir.alloca !fir.type<_QFacc_enter_data_derived_typeTt{d:!fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>}> {bindc_name = "b", uniq_name = "_QFacc_enter_data_derived_typeEb"}
!CHECK: %[[C:.*]] = fir.alloca !fir.type<_QFacc_enter_data_derived_typeTz{data:!fir.box<!fir.heap<!fir.array<?xi32>>>}> {bindc_name = "c", uniq_name = "_QFacc_enter_data_derived_typeEc"}
!CHECK: %[[D:.*]] = fir.alloca !fir.type<_QFacc_enter_data_derived_typeTtt{d:!fir.array<10x!fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>>}> {bindc_name = "d", uniq_name = "_QFacc_enter_data_derived_typeEd"}

  !$acc enter data create(a%data)
!CHECK: %[[DATA_FIELD:.*]] = fir.field_index data, !fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}> 
!CHECK: %[[DATA_COORD:.*]] = fir.coordinate_of %[[A]], %[[DATA_FIELD]] : (!fir.ref<!fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>>, !fir.field) -> !fir.ref<f32> 
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[DATA_COORD]] : !fir.ref<f32>) -> !fir.ref<f32> {name = "a%data", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<f32>)

  !$acc enter data create(b%d%data)
!CHECK: %[[D_FIELD:.*]] = fir.field_index d, !fir.type<_QFacc_enter_data_derived_typeTt{d:!fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>}>
!CHECK: %[[DATA_FIELD:.*]] = fir.field_index data, !fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>
!CHECK: %[[DATA_COORD:.*]] = fir.coordinate_of %[[B]], %[[D_FIELD]], %[[DATA_FIELD]] : (!fir.ref<!fir.type<_QFacc_enter_data_derived_typeTt{d:!fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>}>>, !fir.field, !fir.field) -> !fir.ref<f32>
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[DATA_COORD]] : !fir.ref<f32>) -> !fir.ref<f32> {name = "b%d%data", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<f32>)

  !$acc enter data create(a%array)
!CHECK: %[[ARRAY_FIELD:.*]] = fir.field_index array, !fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>
!CHECK: %[[ARRAY_COORD:.*]] = fir.coordinate_of %[[A]], %[[ARRAY_FIELD]] : (!fir.ref<!fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>>, !fir.field) -> !fir.ref<!fir.array<10xf32>>
!CHECK: %[[C10:.*]] = arith.constant 10 : index
!CHECK: %[[C1:.*]] = arith.constant 1 : index
!CHECK: %[[LB:.*]] = arith.constant 0 : index
!CHECK: %[[UB:.*]] = arith.subi %[[C10]], %[[C1]] : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[ARRAY_COORD]] : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {name = "a%array", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<10xf32>>)

  !$acc enter data create(a%array(:))
!CHECK: %[[ARRAY_FIELD:.*]] = fir.field_index array, !fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>
!CHECK: %[[ARRAY_COORD:.*]] = fir.coordinate_of %[[A]], %[[ARRAY_FIELD]] : (!fir.ref<!fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>>, !fir.field) -> !fir.ref<!fir.array<10xf32>>
!CHECK: %[[C10:.*]] = arith.constant 10 : index
!CHECK: %[[LB:.*]] = arith.constant 0 : index
!CHECK: %[[C1:.*]] = arith.constant 1 : index
!CHECK: %[[LBEXT:.*]] = arith.addi %[[C10]], %[[C1]] : index
!CHECK: %[[UB:.*]] = arith.subi %[[LBEXT]], %[[C1]] : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[ARRAY_COORD]] : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {name = "a%array(:)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<10xf32>>)

  !$acc enter data create(a%array(1:5))
!CHECK: %[[ARRAY_FIELD:.*]] = fir.field_index array, !fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>
!CHECK: %[[ARRAY_COORD:.*]] = fir.coordinate_of %[[A]], %[[ARRAY_FIELD]] : (!fir.ref<!fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>>, !fir.field) -> !fir.ref<!fir.array<10xf32>>
!CHECK: %[[C1:.*]] = arith.constant 1 : index
!CHECK: %[[C0:.*]] = arith.constant 0 : index
!CHECK: %[[C4:.*]] = arith.constant 4 : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[C0]] : index) upperbound(%[[C4]] : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[ARRAY_COORD]] : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {name = "a%array(1:5)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<10xf32>>)

  !$acc enter data create(a%array(:5))
!CHECK: %[[ARRAY_FIELD:.*]] = fir.field_index array, !fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>
!CHECK: %[[ARRAY_COORD:.*]] = fir.coordinate_of %[[A]], %[[ARRAY_FIELD]] : (!fir.ref<!fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>>, !fir.field) -> !fir.ref<!fir.array<10xf32>>
!CHECK: %[[LB:.*]] = arith.constant 0 : index 
!CHECK: %[[C1:.*]] = arith.constant 1 : index
!CHECK: %[[C4:.*]] = arith.constant 4 : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[C4]] : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[ARRAY_COORD]] : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {name = "a%array(:5)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<10xf32>>)

  !$acc enter data create(a%array(2:))
!CHECK: %[[ARRAY_FIELD:.*]] = fir.field_index array, !fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>
!CHECK: %[[ARRAY_COORD:.*]] = fir.coordinate_of %[[A]], %[[ARRAY_FIELD]] : (!fir.ref<!fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>>, !fir.field) -> !fir.ref<!fir.array<10xf32>>
!CHECK: %[[C10:.*]] = arith.constant 10 : index
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[LB:.*]] = arith.constant 1 : index
!CHECK: %[[LBEXT:.*]] = arith.addi %[[C10]], %[[ONE]] : index
!CHECK: %[[UB:.*]] = arith.subi %[[LBEXT]], %[[ONE]] : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[ONE]] : index) startIdx(%[[ONE]] : index)
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[ARRAY_COORD]] : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {name = "a%array(2:)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<10xf32>>)

!$acc enter data create(b%d%array)
!CHECK: %[[D_FIELD:.*]] = fir.field_index d, !fir.type<_QFacc_enter_data_derived_typeTt{d:!fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>}>
!CHECK: %[[ARRAY_FIELD:.*]] = fir.field_index array, !fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>
!CHECK: %[[ARRAY_COORD:.*]] = fir.coordinate_of %[[B]], %[[D_FIELD]], %[[ARRAY_FIELD]] : (!fir.ref<!fir.type<_QFacc_enter_data_derived_typeTt{d:!fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>}>>, !fir.field, !fir.field) -> !fir.ref<!fir.array<10xf32>>
!CHECK: %[[C10:.*]] = arith.constant 10 : index
!CHECK: %[[C1:.*]] = arith.constant 1 : index
!CHECK: %[[LB:.*]] = arith.constant 0 : index
!CHECK: %[[UB:.*]] = arith.subi %[[C10]], %[[C1]] : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[ARRAY_COORD]] : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {name = "b%d%array", structured = false}

  !$acc enter data create(c%data)
!CHECK: %[[DATA_FIELD:.*]] = fir.field_index data, !fir.type<_QFacc_enter_data_derived_typeTz{data:!fir.box<!fir.heap<!fir.array<?xi32>>>}>
!CHECK: %[[DATA_COORD:.*]] = fir.coordinate_of %[[C]], %[[DATA_FIELD]] : (!fir.ref<!fir.type<_QFacc_enter_data_derived_typeTz{data:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!CHECK: %[[DATA_BOX:.*]] = fir.load %[[DATA_COORD]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!CHECK: %[[DIM0:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS0:.*]]:3 = fir.box_dims %[[DATA_BOX]], %[[DIM0]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[DIM0_1:.*]] = arith.constant 0 : index
!CHECK: %[[DIMS0_1:.*]]:3 = fir.box_dims %[[DATA_BOX]], %[[DIM0_1]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!CHECK: %[[UB:.*]] = arith.subi %[[DIMS0_1]]#1, %[[ONE]] : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%c0{{.*}} : index) upperbound(%[[UB]] : index) stride(%[[DIMS0_1]]#2 : index) startIdx(%[[DIMS0]]#0 : index) {strideInBytes = true}
!CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[DATA_BOX]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[BOX_ADDR]] : !fir.heap<!fir.array<?xi32>>) bounds(%[[BOUND]]) -> !fir.heap<!fir.array<?xi32>> {name = "c%data", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.heap<!fir.array<?xi32>>)

  !$acc enter data create (d%d(1)%array)
!CHECK: %[[D_FIELD:.*]] = fir.field_index d, !fir.type<_QFacc_enter_data_derived_typeTtt{d:!fir.array<10x!fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>>}>
!CHECK: %[[D_COORD:.*]] = fir.coordinate_of %[[D]], %[[D_FIELD]] : (!fir.ref<!fir.type<_QFacc_enter_data_derived_typeTtt{d:!fir.array<10x!fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>>}>>, !fir.field) -> !fir.ref<!fir.array<10x!fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>>>
!CHECK: %[[IDX:.*]] = arith.constant 1 : i64
!CHECK: %[[ONE:.*]] = arith.constant 1 : i64
!CHECK: %[[NORMALIZED_IDX:.*]] = arith.subi %[[IDX]], %[[ONE]] : i64
!CHECK: %[[D1_COORD:.*]] = fir.coordinate_of %[[D_COORD]], %[[NORMALIZED_IDX]] : (!fir.ref<!fir.array<10x!fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>>>, i64) -> !fir.ref<!fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>>
!CHECK: %[[ARRAY_FIELD:.*]] = fir.field_index array, !fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>
!CHECK: %[[ARRAY_COORD:.*]] = fir.coordinate_of %[[D1_COORD]], %[[ARRAY_FIELD]] : (!fir.ref<!fir.type<_QFacc_enter_data_derived_typeTdt{data:f32,array:!fir.array<10xf32>}>>, !fir.field) -> !fir.ref<!fir.array<10xf32>>
!CHECK: %[[C10:.*]] = arith.constant 10 : index
!CHECK: %[[C1:.*]] = arith.constant 1 : index
!CHECK: %[[LB:.*]] = arith.constant 0 : index
!CHECK: %[[UB:.*]] = arith.subi %[[C10]], %[[C1]] : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[ARRAY_COORD]] : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {name = "d%d(1_8)%array", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<10xf32>>)

end subroutine
