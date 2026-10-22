! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: IF(10>HUGE(1_4).OR.10<-HUGE(1_4)) CALL foo()
#define CHECKSAFEINT(x,k)  IF(x>HUGE(1_  ##  k).OR.x<-HUGE(1_##k)) CALL foo()

program main
  implicit none

  CHECKSAFEINT(10, 4)
end program main
