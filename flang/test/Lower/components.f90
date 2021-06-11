! RUN: bbc %s -o - | FileCheck %s

module components_test
  type t1
     integer :: i(6)
     real :: r(5)
  end type t1

  type t2
     type(t1) :: g1(3,3), g2(4,4,4)
     integer :: g3(5)
  end type t2

  type t3
     type(t1) :: h1(3)
     type(t2) :: h2(4)
  end type t3

  type(t3) :: instance

contains

  ! CHECK-LABEL: func @_QMcomponents_testPs1(
  subroutine s1(i,j)
    ! CHECK-DAG: %[[VAL_0:.*]] = constant 1 : i32
    ! CHECK-DAG: %[[VAL_1:.*]] = constant 6 : i32
    ! CHECK-DAG: %[[VAL_2:.*]] = constant 1 : i64
    ! CHECK-DAG: %[[VAL_3:.*]] = constant 2 : i64
    ! CHECK-DAG: %[[VAL_4:.*]] = constant 3 : i64
    ! CHECK: %[[VAL_5:.*]] = fir.address_of(@_QMcomponents_testEinstance) : !fir.ref<!fir.type<_QMcomponents_testTt3{h1:!fir.array<3x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,h2:!fir.array<4x!fir.type<_QMcomponents_testTt2{g1:!fir.array<3x3x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g2:!fir.array<4x4x4x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g3:!fir.array<5xi32>}>>}>>
    ! CHECK: %[[VAL_6:.*]] = fir.load %[[VAL_7:.*]] : !fir.ref<i32>
    ! CHECK: %[[VAL_8:.*]] = cmpi sge, %[[VAL_6]], %[[VAL_0]] : i32
    ! CHECK: %[[VAL_9:.*]] = cmpi sle, %[[VAL_6]], %[[VAL_1]] : i32
    ! CHECK: %[[VAL_10:.*]] = and %[[VAL_8]], %[[VAL_9]] : i1
    ! CHECK: cond_br %[[VAL_10]], ^bb1, ^bb2
    ! CHECK: ^bb1:
    ! CHECK: %[[VAL_11:.*]] = fir.field_index h2, !fir.type<_QMcomponents_testTt3{h1:!fir.array<3x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,h2:!fir.array<4x!fir.type<_QMcomponents_testTt2{g1:!fir.array<3x3x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g2:!fir.array<4x4x4x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g3:!fir.array<5xi32>}>>}>
    ! CHECK: %[[VAL_12:.*]] = fir.coordinate_of %[[VAL_5]], %[[VAL_11]] : (!fir.ref<!fir.type<_QMcomponents_testTt3{h1:!fir.array<3x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,h2:!fir.array<4x!fir.type<_QMcomponents_testTt2{g1:!fir.array<3x3x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g2:!fir.array<4x4x4x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g3:!fir.array<5xi32>}>>}>>, !fir.field) -> !fir.ref<!fir.array<4x!fir.type<_QMcomponents_testTt2{g1:!fir.array<3x3x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g2:!fir.array<4x4x4x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g3:!fir.array<5xi32>}>>>
    ! CHECK: %[[VAL_13:.*]] = fir.coordinate_of %[[VAL_12]], %[[VAL_3]] : (!fir.ref<!fir.array<4x!fir.type<_QMcomponents_testTt2{g1:!fir.array<3x3x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g2:!fir.array<4x4x4x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g3:!fir.array<5xi32>}>>>, i64) -> !fir.ref<!fir.type<_QMcomponents_testTt2{g1:!fir.array<3x3x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g2:!fir.array<4x4x4x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g3:!fir.array<5xi32>}>>
    ! CHECK: %[[VAL_14:.*]] = fir.field_index g2, !fir.type<_QMcomponents_testTt2{g1:!fir.array<3x3x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g2:!fir.array<4x4x4x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g3:!fir.array<5xi32>}>
    ! CHECK: %[[VAL_15:.*]] = fir.coordinate_of %[[VAL_13]], %[[VAL_14]] : (!fir.ref<!fir.type<_QMcomponents_testTt2{g1:!fir.array<3x3x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g2:!fir.array<4x4x4x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>,g3:!fir.array<5xi32>}>>, !fir.field) -> !fir.ref<!fir.array<4x4x4x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>>
    ! CHECK: %[[VAL_16:.*]] = fir.coordinate_of %[[VAL_15]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : (!fir.ref<!fir.array<4x4x4x!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>>, i64, i64, i64) -> !fir.ref<!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>
    ! CHECK: %[[VAL_17:.*]] = fir.field_index i, !fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>
    ! CHECK: %[[VAL_18:.*]] = fir.coordinate_of %[[VAL_16]], %[[VAL_17]] : (!fir.ref<!fir.type<_QMcomponents_testTt1{i:!fir.array<6xi32>,r:!fir.array<5xf32>}>>, !fir.field) -> !fir.ref<!fir.array<6xi32>>
    ! CHECK: %[[VAL_19:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
    ! CHECK: %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i32) -> i64
    ! CHECK: %[[VAL_21:.*]] = fir.coordinate_of %[[VAL_18]], %[[VAL_20]] : (!fir.ref<!fir.array<6xi32>>, i64) -> !fir.ref<i32>
    ! CHECK: %[[VAL_22:.*]] = fir.load %[[VAL_21]] : !fir.ref<i32>
    ! CHECK: fir.store %[[VAL_22]] to %[[VAL_23:.*]] : !fir.ref<i32>
    ! CHECK: br ^bb2
    ! CHECK: ^bb2:
    ! CHECK: return
    if (j >= 1 .and. j <= 6) then
       i = instance%h2(2)%g2(1,2,3)%i(j)
    end if
  end subroutine s1

end module components_test
