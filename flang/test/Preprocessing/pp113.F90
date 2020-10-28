! RUN: %f18 -E %s 2>&1 | FileCheck %s
! CHECK: res = ((666)+111)
! FLM call split across continuation between name and (, leading &
      integer function IFLM(x)
        integer :: x
        IFLM = x
      end function IFLM
      program main
#define IFLM(x) ((x)+111)
      integer :: res
      res = IFLM&
&(666)
      if (res .eq. 777) then
        print *, 'pp113.F90 yes'
      else
        print *, 'pp113.F90 no: ', res
      end if
      end
