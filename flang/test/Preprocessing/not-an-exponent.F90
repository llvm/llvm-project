!RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
#define e eeeee
module m
  interface operator(.e.)
     module procedure ir,rr
  end interface operator(.e.)
contains
  function ir(k1,k2)
    intent(in)::k1,k2
    ir=k1+k2
  end function ir
  function rr(k1,k2)
    real,intent(in)::k1,k2
    rr=k1+k2
  end function rr
end module m
program main
  use m
!CHECK: IF (real((ir(1_4,5_4)),kind=4)/=6._4) ERROR STOP 1_4
!CHECK: IF ((rr(1._4,5.e-1_4))/=1.5_4) ERROR STOP 2_4
  if((1.e.5)/=6.e0) error stop 1
  if((1..e..5)/=1.5) error stop 2
  print *,'pass'
end program main
