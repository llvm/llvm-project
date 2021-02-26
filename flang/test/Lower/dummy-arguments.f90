! RUN: bbc %s -o - | FileCheck %s

! CHECK-LABEL: _QQmain
program test1
  ! CHECK-DAG: %[[TMP:.*]] = fir.alloca
  ! CHECK-DAG: %[[TEN:.*]] = constant
  ! CHECK: fir.store %[[TEN]] to %[[TMP]]
  ! CHECK-NEXT: fir.call @_QPfoo
  call foo(10)
contains

! CHECK-LABEL: func @_QPfoo
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
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %arg0
  ! CHECK: = fir.call @_QPf(%[[coor]]) : (!fir.ref<f32>) -> f32
  sub2 = f(r(1))
  ! CHECK: return %{{.*}} : f32
end function sub2

