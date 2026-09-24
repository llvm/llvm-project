! Test to check the option "-fdebug-info-for-profiling".

!RUN: %flang -### -fdebug-info-for-profiling %s 2>&1 | FileCheck %s --check-prefix=DEBUG-INFO
!RUN: %flang -### -fdebug-info-for-profiling -fno-debug-info-for-profiling -fdebug-info-for-profiling %s 2>&1 | FileCheck %s --check-prefix=DEBUG-INFO
!RUN: %flang -### %s 2>&1 | FileCheck %s --check-prefix=NO-DEBUG-INFO
!RUN: %flang -### -fdebug-info-for-profiling -fno-debug-info-for-profiling %s 2>&1 | FileCheck %s --check-prefix=NO-DEBUG-INFO

! DEBUG-INFO: "-fdebug-info-for-profiling"
! NO-DEBUG-INFO-NOT: "-fdebug-info-for-profiling"

program test
   implicit none
   integer :: i, sum
   sum = 0
   do i = 1, 20
      if (mod(i, 2) == 0) then
         sum = sum + compute(i)
      else
         sum = sum + compute(i)*2
      end if
   end do

contains
   integer function compute(x)
      implicit none
      integer, intent(in) :: x
      if (x < 10) then
         compute = x*x
      else
         compute = x + 5
      end if
   end function compute

end program test
