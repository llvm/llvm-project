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
!CHECK: %[[LB1:.*]] = arith.constant 0 : index
!CHECK: %[[UB1:.*]] = arith.constant 9 : index
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB1]] : index) upperbound(%[[UB1]] : index) startIdx(%c1{{.*}} : index)
!CHECK: %[[LB2:.*]] = arith.constant 0 : index
!CHECK: %[[UB2:.*]] = arith.constant 4 : index
!CHECK: %[[BOUND2:.*]] = acc.bounds lowerbound(%[[LB2]] : index) upperbound(%[[UB2]] : index) startIdx(%c1{{.*}} : index) 
!CHECK: %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND1]], %[[BOUND2]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a(1:10,1:5)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[COPYIN_A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc enter data copyin(a(1:,1:5))
!CHECK: %[[LB1:.*]] = arith.constant 0 : index
!CHECK: %[[EXTENT:.*]] = arith.subi %[[EXTENT_C10:.*]], %c0{{.*}} : index 
!CHECK: %[[BOUND1:.*]] = acc.bounds   lowerbound(%[[LB1]] : index) extent(%[[EXTENT]] : index)  startIdx(%c1{{.*}} : index)
!CHECK: %[[LB2:.*]] = arith.constant 0 : index
!CHECK: %[[UB2:.*]] = arith.constant 4 : index
!CHECK: %[[BOUND2:.*]] = acc.bounds lowerbound(%[[LB2]] : index) upperbound(%[[UB2]] : index)  startIdx(%c1{{.*}} : index)
!CHECK: %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND1]], %[[BOUND2]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a(1:,1:5)", structured = false}
!CHECK: acc.enter_data   dataOperands(%[[COPYIN_A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc enter data copyin(a(:10,1:5))
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[UB1:.*]] = arith.constant 9 : index
!CHECK: %[[BOUND1:.*]] = acc.bounds upperbound(%[[UB1]] : index) startIdx(%[[ONE]] : index)
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[LB2:.*]] = arith.constant 0 : index 
!CHECK: %[[UB2:.*]] = arith.constant 4 : index
!CHECK: %[[BOUND2:.*]] = acc.bounds lowerbound(%[[LB2]] : index) upperbound(%[[UB2]] : index)  startIdx(%c1{{.*}} : index)
!CHECK: %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND1]], %[[BOUND2]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a(:10,1:5)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[COPYIN_A]] : !fir.ref<!fir.array<10x10xf32>>)

  !$acc enter data copyin(a(:,:))
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[BOUND1:.*]] = acc.bounds extent(%c10{{.*}} : index) startIdx(%[[ONE]] : index)
!CHECK: %[[ONE:.*]] = arith.constant 1 : index
!CHECK: %[[BOUND2:.*]] = acc.bounds extent(%c10{{.*}} : index) startIdx(%[[ONE]] : index)
!CHECK: %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[A]] : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND1]], %[[BOUND2]]) -> !fir.ref<!fir.array<10x10xf32>> {name = "a(:,:)", structured = false}
end subroutine acc_enter_data

subroutine acc_enter_data_dummy(a, b, n, m)
  integer :: n, m
  real :: a(1:10)
  real :: b(n:m)

!CHECK-LABEL: func.func @_QPacc_enter_data_dummy
!CHECK-SAME:    %[[A:.*]]: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "a"}, %[[B:.*]]: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "b"}, %[[N:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}, %[[M:.*]]: !fir.ref<i32> {fir.bindc_name = "m"}

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

  !$acc enter data create(a(5:10))
!CHECK: %[[LB1:.*]] = arith.constant 4 : index
!CHECK: %[[UB1:.*]] = arith.constant 9 : index
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB1]] : index) upperbound(%[[UB1]] : index)  startIdx(%c1{{.*}} : index)
!CHECK: %[[CREATE1:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND1]]) -> !fir.ref<!fir.array<10xf32>> {name = "a(5:10)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE1]] : !fir.ref<!fir.array<10xf32>>)

  !$acc enter data create(b(n:m))
!CHECK: %[[LOAD_N:.*]] = fir.load %[[N]] : !fir.ref<i32> 
!CHECK: %[[CONVERT_N:.*]] = fir.convert %[[LOAD_N]] : (i32) -> index
!CHECK: %[[LB:.*]] = arith.subi %[[CONVERT_N]], %[[N_IDX]] : index
!CHECK: %[[LOAD_M:.*]] = fir.load %[[M]] : !fir.ref<i32>
!CHECK: %[[CONVERT_M:.*]] = fir.convert %[[LOAD_M]] : (i32) -> index
!CHECK: %[[UB:.*]] = arith.subi %[[CONVERT_M]], %[[N_IDX]] : index
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) startIdx(%[[N_IDX]] : index)
!CHECK: %[[CREATE1:.*]] = acc.create varPtr(%[[B]] : !fir.ref<!fir.array<?xf32>>) bounds(%[[BOUND1]]) -> !fir.ref<!fir.array<?xf32>> {name = "b(n:m)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE1]] : !fir.ref<!fir.array<?xf32>>)

  !$acc enter data create(b(n:))
!CHECK: %[[LOAD_N:.*]] = fir.load %[[N]] : !fir.ref<i32> 
!CHECK: %[[CONVERT_N:.*]] = fir.convert %[[LOAD_N]] : (i32) -> index
!CHECK: %[[LB:.*]] = arith.subi %[[CONVERT_N]], %[[N_IDX]] : index
!CHECK: %[[EXT:.*]] = arith.subi %[[EXT_B]], %[[LB]] : index
!CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB]] : index) extent(%[[EXT]] : index) startIdx(%[[N_IDX]] : index) 
!CHECK: %[[CREATE1:.*]] = acc.create varPtr(%[[B]] : !fir.ref<!fir.array<?xf32>>) bounds(%[[BOUND1]]) -> !fir.ref<!fir.array<?xf32>> {name = "b(n:)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE1]] : !fir.ref<!fir.array<?xf32>>)

  !$acc enter data create(b(:))
!CHECK: %[[BOUND1:.*]] = acc.bounds extent(%[[EXT_B]] : index) startIdx(%[[N_IDX]] : index)
!CHECK: %[[CREATE1:.*]] = acc.create varPtr(%[[B]] : !fir.ref<!fir.array<?xf32>>) bounds(%[[BOUND1]]) -> !fir.ref<!fir.array<?xf32>> {name = "b(:)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE1]] : !fir.ref<!fir.array<?xf32>>)

end subroutine

! Test lowering of array section for non default lower bound.
subroutine acc_enter_data_non_default_lb()
  integer :: a(0:9)

!CHECK-LABEL: func.func @_QPacc_enter_data_non_default_lb() {
!CHECK: %[[BASELB:.*]] = arith.constant 0 : index
!CHECK: %[[A:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "a", uniq_name = "_QFacc_enter_data_non_default_lbEa"}

  !$acc enter data create(a(5:9))
!CHECK: %[[SECTIONLB:.*]] = arith.constant 5 : index
!CHECK: %[[LB:.*]] = arith.subi %[[SECTIONLB]], %[[BASELB]] : index
!CHECK: %[[SECTIONUB:.*]] = arith.constant 9 : index
!CHECK: %[[UB:.*]] = arith.subi %[[SECTIONUB]], %[[BASELB]] : index
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) startIdx(%[[BASELB]] : index)
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10xi32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xi32>> {name = "a(5:9)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<10xi32>>)

  !$acc enter data create(a(:))
!CHECK: %[[BOUND:.*]] = acc.bounds extent(%c10{{.*}} : index) startIdx(%[[BASELB]] : index)
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10xi32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xi32>> {name = "a(:)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<10xi32>>)

  !$acc enter data create(a(:6))
!CHECK: %[[SECTIONUB:.*]] = arith.constant 6 : index
!CHECK: %[[UB:.*]] = arith.subi %[[SECTIONUB]], %[[BASELB]] : index
!CHECK: %[[BOUND:.*]] = acc.bounds upperbound(%[[UB]] : index) startIdx(%[[BASELB]] : index)
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10xi32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xi32>> {name = "a(:6)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<10xi32>>)

  !$acc enter data create(a(4:))
!CHECK: %[[SECTIONLB:.*]] = arith.constant 4 : index
!CHECK: %[[LB:.*]] = arith.subi %[[SECTIONLB]], %[[BASELB]] : index
!CHECK: %[[EXT:.*]] = arith.subi %c10{{.*}}, %[[LB]] : index 
!CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) extent(%[[EXT]] : index) startIdx(%[[BASELB]] : index)
!CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[A]] : !fir.ref<!fir.array<10xi32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xi32>> {name = "a(4:)", structured = false}
!CHECK: acc.enter_data dataOperands(%[[CREATE]] : !fir.ref<!fir.array<10xi32>>)

end subroutine
