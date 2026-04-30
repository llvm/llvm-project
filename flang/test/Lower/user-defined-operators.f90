! Test use defined operators/assignment
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

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
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[arg0]] {{.*}} {uniq_name = "_QFuser_assignmentEa"}
! CHECK: %[[I:.*]]:2 = hlfir.declare %[[arg1]] {{.*}} {uniq_name = "_QFuser_assignmentEi"}
! CHECK: hlfir.region_assign {
! CHECK:   %[[V_1:.*]] = fir.load %[[I]]#0 : !fir.ref<i32>
! CHECK:   hlfir.yield %[[V_1]] : i32
! CHECK: } to {
! CHECK:   hlfir.yield %[[A]]#0 : !fir.ref<!fir.type<_QFuser_assignmentTt{x:f32,i:i32}>>
! CHECK: } user_defined_assign  (%[[VAL:.*]]: i32) to (%[[VAR:.*]]: !fir.ref<!fir.type<_QFuser_assignmentTt{x:f32,i:i32}>>) {
! CHECK:   %[[ASSOC:.*]]:3 = hlfir.associate %[[VAL]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
! CHECK:   fir.call @_QPmy_assign(%[[VAR]], %[[ASSOC]]#0) fastmath<contract> : (!fir.ref<!fir.type<_QFuser_assignmentTt{x:f32,i:i32}>>, !fir.ref<i32>) -> ()
! CHECK:   hlfir.end_associate %[[ASSOC]]#1, %[[ASSOC]]#2 : !fir.ref<i32>, i1
! CHECK: }
 a = i
end subroutine
