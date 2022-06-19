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
 ! CHECK: fir.call @_QPmy_assign(%[[arg0]], %[[arg1]])
 a = i
end subroutine
