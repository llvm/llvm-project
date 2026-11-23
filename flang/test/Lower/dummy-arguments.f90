! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: _QQmain
program test1
  ! CHECK: %[[C10:.*]] = arith.constant 10 : i32
  ! CHECK: %[[ASSOC:.*]]:3 = hlfir.associate %[[C10]] {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
  ! CHECK: fir.call @_QFPfoo(%[[ASSOC]]#0) {{.*}} : (!fir.ref<i32>) -> ()
  call foo(10)
contains

! CHECK-LABEL: func private @_QFPfoo
subroutine foo(avar1)
  integer :: avar1
!  integer :: my_data, my_data2
!  DATA my_data / 150 /
!  DATA my_data2 / 150 /
!  print *, my_data, my_data2
  print *, avar1
end subroutine
! CHECK: }
end program test1

! CHECK-LABEL: func @_QPsub2
function sub2(r)
  real :: r(20)
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare %arg0
  ! CHECK: %[[ELT:.*]] = hlfir.designate %[[DECL]]#0 (%c1)  : (!fir.ref<!fir.array<20xf32>>, index) -> !fir.ref<f32>
  ! CHECK: = fir.call @_QPf(%[[ELT]]) {{.*}}: (!fir.ref<f32>) -> f32
  sub2 = f(r(1))
  ! CHECK: return %{{.*}} : f32
end function sub2

! Test TARGET attribute lowering
! CHECK-LABEL: func @_QPtest_target(
! CHECK-SAME: !fir.ref<i32> {fir.bindc_name = "i", fir.target},
! CHECK-SAME: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.target})
subroutine test_target(i, x)
  integer, target :: i
  real, target :: x(:)
  print *, xs, xa
end subroutine
