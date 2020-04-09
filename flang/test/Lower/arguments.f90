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

