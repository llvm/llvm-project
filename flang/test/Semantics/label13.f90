! RUN: %f18 -funparse-with-symbols %s 2>&1 | FileCheck %s
! CHECK: branch into loop body from outside
! CHECK: the loop branched into

subroutine s(a)
  integer i
  real a(10)
  do 10 i = 1,10
     if (a(i) < 0.0) then
        goto 20
     end if
30   continue
     a(i) = 1.0
10 end do
  goto 40
20 a(i) = -a(i)
  goto 30
40 continue
end subroutine s
