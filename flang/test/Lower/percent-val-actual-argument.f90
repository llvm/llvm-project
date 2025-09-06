! RUN: bbc %s -emit-fir -o - | FileCheck %s

program main
  logical::a1
  data a1/.true./
  call sa(%val(a1))
! CHECK: fir.load %3 : !fir.ref<!fir.logical<4>>
! CHECK: fir.convert %13 : (!fir.logical<4>) -> i1
  write(6,*) "a1 = ", a1
end program main

subroutine sa(x1)
  logical::x1
end subroutine sa
