! RUN: bbc %s -o - | FileCheck %s

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
  ! CHECK-SAME: %[[arg0:[^:]+]]: !fir.box<!fir.array<?x!fir.type<_QMcsTr{n:i32,d:i32}>>>,
  ! CHECK-SAME: %[[arg1:[^:]+]]: !fir.box<!fir.array<?x!fir.type<_QMcsTr{n:i32,d:i32}>>>)
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
    ! CHECK: = fir.call @_FortranAAll(
    c1 = all(c%n == e%n)
  end function c1
end module cs
