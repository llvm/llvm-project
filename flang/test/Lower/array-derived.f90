! RUN: bbc -hlfir=false %s -o - | FileCheck %s

module cs
  type r
     integer n, d
  end type r

  type t2
     integer :: f1(5)
     type(r) :: f2
  end type t2

  type t3
     type(t2) :: f(3,3)
  end type t3

contains

  ! CHECK: func @_QMcsPc1(
  ! CHECK-SAME:   %[[arg0:[^:]+]]: !fir.box<!fir.array<?x!fir.type<_QMcsTr{n:i32,d:i32}>>>{{.*}}, %[[arg1:[^:]+]]: !fir.box<!fir.array<?x!fir.type<_QMcsTr{n:i32,d:i32}>>>{{.*}})
  function c1(e, c)
    type(r), intent(in) :: e(:), c(:)
    ! CHECK-DAG: fir.alloca !fir.logical<1> {bindc_name = "c1", uniq_name = "_QMcsFc1Ec1"}
    logical*1 :: c1
    ! CHECK-DAG: %[[fldn:.*]] = fir.field_index n, !fir.type<_QMcsTr{n:i32,d:i32}>
    ! CHECK: %[[ext1:.*]]:3 = fir.box_dims %[[arg1]], %c0{{.*}} : (!fir.box<!fir.array<?x!fir.type<_QMcsTr{n:i32,d:i32}>>>, index) -> (index, index, index)
    ! CHECK-DAG: %[[slice1:.*]] = fir.slice %c1{{.*}}, %[[ext1]]#1, %c1{{.*}} path %[[fldn]] : (index, index, index, !fir.field) -> !fir.slice<1>
    ! CHECK-DAG: %[[ext0:.*]]:3 = fir.box_dims %[[arg0]], %c0{{.*}} : (!fir.box<!fir.array<?x!fir.type<_QMcsTr{n:i32,d:i32}>>>, index) -> (index, index, index)
    ! CHECK: %[[slice0:.*]] = fir.slice %c1{{.*}}, %[[ext0]]#1, %c1{{.*}} path %[[fldn]] : (index, index, index, !fir.field) -> !fir.slice<1>
    ! CHECK-DAG: = fir.array_coor %[[arg1]] [%[[slice1]]] %[[index:.*]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTr{n:i32,d:i32}>>>, !fir.slice<1>, index) -> !fir.ref<i32>
    ! CHECK-DAG: = fir.array_coor %[[arg0]] [%[[slice0]]] %[[index]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTr{n:i32,d:i32}>>>, !fir.slice<1>, index) -> !fir.ref<i32>
    ! CHECK: = fir.call @_FortranAAllLogical4x1_simplified(
    c1 = all(c%n == e%n)
  end function c1

! CHECK-LABEL: func @_QMcsPtest2(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>>{{.*}}, %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>>{{.*}}) {
! CHECK-DAG:     %[[VAL_2:.*]] = arith.constant 2 : index
! CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 4 : index
! CHECK-DAG:     %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK-DAG:     %[[VAL_5:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_6:.*]] = fir.field_index f2, !fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>
! CHECK:         %[[VAL_7:.*]] = fir.field_index d, !fir.type<_QMcsTr{n:i32,d:i32}>
! CHECK:         %[[VAL_8:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_9:.*]] = fir.slice %[[VAL_5]], %[[VAL_8]]#1, %[[VAL_5]] path %[[VAL_6]], %[[VAL_7]] : (index, index, index, !fir.field, !fir.field) -> !fir.slice<1>
! CHECK:         %[[VAL_8_2:.*]] = arith.cmpi sgt, %[[VAL_8]]#1, %[[VAL_4]] : index
! CHECK:         %[[VAL_8_3:.*]] = arith.select %[[VAL_8_2]], %[[VAL_8]]#1, %[[VAL_4]] : index
! CHECK:         %[[VAL_10:.*]] = fir.field_index f1, !fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>
! CHECK:         %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_4]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_12:.*]] = fir.slice %[[VAL_5]], %[[VAL_11]]#1, %[[VAL_5]] path %[[VAL_10]], %[[VAL_4]] : (index, index, index, !fir.field, index) -> !fir.slice<1>
! CHECK:         %[[VAL_13:.*]] = fir.slice %[[VAL_5]], %[[VAL_11]]#1, %[[VAL_5]] path %[[VAL_10]], %[[VAL_3]] : (index, index, index, !fir.field, index) -> !fir.slice<1>
! CHECK:         %[[VAL_14:.*]] = fir.slice %[[VAL_5]], %[[VAL_11]]#1, %[[VAL_5]] path %[[VAL_10]], %[[VAL_2]] : (index, index, index, !fir.field, index) -> !fir.slice<1>
! CHECK:         br ^bb1(%[[VAL_4]], %[[VAL_8_3]] : index, index)
! CHECK:       ^bb1(%[[VAL_15:.*]]: index, %[[VAL_16:.*]]: index):
! CHECK:         %[[VAL_17:.*]] = arith.cmpi sgt, %[[VAL_16]], %[[VAL_4]] : index
! CHECK:         cond_br %[[VAL_17]], ^bb2, ^bb3
! CHECK:       ^bb2:
! CHECK:         %[[VAL_18:.*]] = arith.addi %[[VAL_15]], %[[VAL_5]] : index
! CHECK:         %[[VAL_19:.*]] = fir.array_coor %[[VAL_1]] {{\[}}%[[VAL_12]]] %[[VAL_18]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>>, !fir.slice<1>, index) -> !fir.ref<i32>
! CHECK:         %[[VAL_20:.*]] = fir.load %[[VAL_19]] : !fir.ref<i32>
! CHECK:         %[[VAL_21:.*]] = fir.array_coor %[[VAL_1]] {{\[}}%[[VAL_13]]] %[[VAL_18]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>>, !fir.slice<1>, index) -> !fir.ref<i32>
! CHECK:         %[[VAL_22:.*]] = fir.load %[[VAL_21]] : !fir.ref<i32>
! CHECK:         %[[VAL_23:.*]] = fir.array_coor %[[VAL_1]] {{\[}}%[[VAL_14]]] %[[VAL_18]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>>, !fir.slice<1>, index) -> !fir.ref<i32>
! CHECK:         %[[VAL_24:.*]] = fir.load %[[VAL_23]] : !fir.ref<i32>
! CHECK:         %[[VAL_25:.*]] = arith.divsi %[[VAL_22]], %[[VAL_24]] : i32
! CHECK:         %[[VAL_26:.*]] = arith.addi %[[VAL_20]], %[[VAL_25]] : i32
! CHECK:         %[[VAL_27:.*]] = fir.array_coor %[[VAL_0]] {{\[}}%[[VAL_9]]] %[[VAL_18]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>>, !fir.slice<1>, index) -> !fir.ref<i32>
! CHECK:         fir.store %[[VAL_26]] to %[[VAL_27]] : !fir.ref<i32>
! CHECK:         %[[VAL_28:.*]] = arith.subi %[[VAL_16]], %[[VAL_5]] : index
! CHECK:         br ^bb1(%[[VAL_18]], %[[VAL_28]] : index, index)
! CHECK:       ^bb3:
! CHECK:         return
! CHECK:       }


  subroutine test2(a1, a2)
    type(t2) :: a1(:), a2(:)
    a1%f2%d = a2%f1(1) + a2%f1(5) / a2%f1(3)
  end subroutine test2

! CHECK-LABEL: func @_QMcsPtest3(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMcsTt3{f:!fir.array<3x3x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>}>>>{{.*}}, %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.type<_QMcsTt3{f:!fir.array<3x3x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>}>>>{{.*}}) {
! CHECK-DAG:     %[[VAL_2:.*]] = arith.constant 2 : index
! CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 3 : index
! CHECK-DAG:     %[[VAL_4:.*]] = arith.constant 4 : i32
! CHECK-DAG:     %[[VAL_5:.*]] = arith.constant 1 : index
! CHECK-DAG:     %[[VAL_6:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_7:.*]] = fir.field_index f, !fir.type<_QMcsTt3{f:!fir.array<3x3x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>}>
! CHECK:         %[[VAL_8:.*]] = fir.field_index f2, !fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>
! CHECK:         %[[VAL_9:.*]] = fir.field_index n, !fir.type<_QMcsTr{n:i32,d:i32}>
! CHECK:         %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_0]], %[[VAL_6]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTt3{f:!fir.array<3x3x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>}>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_11:.*]] = fir.slice %[[VAL_5]], %[[VAL_10]]#1, %[[VAL_5]] path %[[VAL_7]], %[[VAL_6]], %[[VAL_6]], %[[VAL_8]], %[[VAL_9]] : (index, index, index, !fir.field, index, index, !fir.field, !fir.field) -> !fir.slice<1>
! CHECK:         %[[VAL_10_2:.*]] = arith.cmpi sgt, %[[VAL_10]]#1, %[[VAL_6]] : index
! CHECK:         %[[VAL_10_3:.*]] = arith.select %[[VAL_10_2]], %[[VAL_10]]#1, %[[VAL_6]] : index
! CHECK:         %[[VAL_12:.*]] = fir.field_index f1, !fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>
! CHECK:         %[[VAL_13:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_6]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTt3{f:!fir.array<3x3x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>}>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_14:.*]] = fir.slice %[[VAL_5]], %[[VAL_13]]#1, %[[VAL_5]] path %[[VAL_7]], %[[VAL_5]], %[[VAL_5]], %[[VAL_12]], %[[VAL_3]] : (index, index, index, !fir.field, index, index, !fir.field, index) -> !fir.slice<1>
! CHECK:         br ^bb1(%[[VAL_6]], %[[VAL_10_3]] : index, index)
! CHECK:       ^bb1(%[[VAL_15:.*]]: index, %[[VAL_16:.*]]: index):
! CHECK:         %[[VAL_17:.*]] = arith.cmpi sgt, %[[VAL_16]], %[[VAL_6]] : index
! CHECK:         cond_br %[[VAL_17]], ^bb2, ^bb3
! CHECK:       ^bb2:
! CHECK:         %[[VAL_18:.*]] = arith.addi %[[VAL_15]], %[[VAL_5]] : index
! CHECK:         %[[VAL_19:.*]] = fir.array_coor %[[VAL_1]] {{\[}}%[[VAL_14]]] %[[VAL_18]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTt3{f:!fir.array<3x3x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>}>>>, !fir.slice<1>, index) -> !fir.ref<i32>
! CHECK:         %[[VAL_20:.*]] = fir.load %[[VAL_19]] : !fir.ref<i32>
! CHECK:         %[[VAL_21:.*]] = arith.subi %[[VAL_20]], %[[VAL_4]] : i32
! CHECK:         %[[VAL_22:.*]] = fir.array_coor %[[VAL_0]] {{\[}}%[[VAL_11]]] %[[VAL_18]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTt3{f:!fir.array<3x3x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>}>>>, !fir.slice<1>, index) -> !fir.ref<i32>
! CHECK:         fir.store %[[VAL_21]] to %[[VAL_22]] : !fir.ref<i32>
! CHECK:         %[[VAL_23:.*]] = arith.subi %[[VAL_16]], %[[VAL_5]] : index
! CHECK:         br ^bb1(%[[VAL_18]], %[[VAL_23]] : index, index)
! CHECK:       ^bb3:
! CHECK:         %[[VAL_24:.*]] = fir.slice %[[VAL_5]], %[[VAL_13]]#1, %[[VAL_5]] path %[[VAL_7]], %[[VAL_2]], %[[VAL_2]], %[[VAL_12]], %[[VAL_5]] : (index, index, index, !fir.field, index, index, !fir.field, index) -> !fir.slice<1>
! CHECK:         %[[VAL_13_2:.*]] = arith.cmpi sgt, %[[VAL_13]]#1, %[[VAL_6]] : index
! CHECK:         %[[VAL_13_3:.*]] = arith.select %[[VAL_13_2]], %[[VAL_13]]#1, %[[VAL_6]] : index
! CHECK:         %[[VAL_25:.*]] = fir.field_index d, !fir.type<_QMcsTr{n:i32,d:i32}>
! CHECK:         %[[VAL_26:.*]] = fir.slice %[[VAL_5]], %[[VAL_10]]#1, %[[VAL_5]] path %[[VAL_7]], %[[VAL_6]], %[[VAL_5]], %[[VAL_8]], %[[VAL_25]] : (index, index, index, !fir.field, index, index, !fir.field, !fir.field) -> !fir.slice<1>
! CHECK:         br ^bb4(%[[VAL_6]], %[[VAL_13_3]] : index, index)
! CHECK:       ^bb4(%[[VAL_27:.*]]: index, %[[VAL_28:.*]]: index):
! CHECK:         %[[VAL_29:.*]] = arith.cmpi sgt, %[[VAL_28]], %[[VAL_6]] : index
! CHECK:         cond_br %[[VAL_29]], ^bb5, ^bb6
! CHECK:       ^bb5:
! CHECK:         %[[VAL_30:.*]] = arith.addi %[[VAL_27]], %[[VAL_5]] : index
! CHECK:         %[[VAL_31:.*]] = fir.array_coor %[[VAL_0]] {{\[}}%[[VAL_26]]] %[[VAL_30]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTt3{f:!fir.array<3x3x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>}>>>, !fir.slice<1>, index) -> !fir.ref<i32>
! CHECK:         %[[VAL_32:.*]] = fir.load %[[VAL_31]] : !fir.ref<i32>
! CHECK:         %[[VAL_33:.*]] = arith.addi %[[VAL_32]], %[[VAL_4]] : i32
! CHECK:         %[[VAL_34:.*]] = fir.array_coor %[[VAL_1]] {{\[}}%[[VAL_24]]] %[[VAL_30]] : (!fir.box<!fir.array<?x!fir.type<_QMcsTt3{f:!fir.array<3x3x!fir.type<_QMcsTt2{f1:!fir.array<5xi32>,f2:!fir.type<_QMcsTr{n:i32,d:i32}>}>>}>>>, !fir.slice<1>, index) -> !fir.ref<i32>
! CHECK:         fir.store %[[VAL_33]] to %[[VAL_34]] : !fir.ref<i32>
! CHECK:         %[[VAL_35:.*]] = arith.subi %[[VAL_28]], %[[VAL_5]] : index
! CHECK:         br ^bb4(%[[VAL_30]], %[[VAL_35]] : index, index)
! CHECK:       ^bb6:
! CHECK:         return
! CHECK:       }

  subroutine test3(a3, a4)
    type(t3) :: a3(:), a4(:)
    a3%f(1,1)%f2%n = a4%f(2,2)%f1(4) - 4
    a4%f(3,3)%f1(2) = a3%f(1,2)%f2%d + 4
  end subroutine test3
end module cs

! CHECK: func private @_FortranAAll(!fir.box<none>, !fir.ref<i8>, i32, i32) -> i1 attributes {fir.runtime}
