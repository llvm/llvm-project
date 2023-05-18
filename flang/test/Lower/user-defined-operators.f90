! Test use defined operators/assignment
! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test user defined assignment
! CHECK-LABEL: func @_QPuser_assignment(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.type<{{.*}}>>{{.*}}, %[[arg1:.*]]: !fir.ref<i32>{{.*}}) {
subroutine user_assignment(a, i)
  type t
    real :: x
    integer :: i
  end type
  interface assignment(=)
  subroutine my_assign(b, j)
    import :: t
    type(t), INTENT(OUT) :: b
    integer, INTENT(IN) :: j
  end subroutine
 end interface
 type(t) :: a
! CHECK: %[[V_0:[0-9]+]] = fir.alloca i32
! CHECK: %[[V_1:[0-9]+]] = fir.load %arg1 : !fir.ref<i32>
! CHECK: %[[V_2:[0-9]+]] = fir.no_reassoc %[[V_1:[0-9]+]] : i32
! CHECK: fir.store %[[V_2]] to %[[V_0:[0-9]+]] : !fir.ref<i32>
! CHECK: fir.call @_QPmy_assign(%arg0, %[[V_0]]) fastmath<contract> : (!fir.ref<!fir.type<_QFuser_assignmentTt{x:f32,i:i32}>>, !fir.ref<i32>) -> ()
 a = i
end subroutine
