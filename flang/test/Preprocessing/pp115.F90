! RUN: %f18 -E %s 2>&1 | FileCheck %s
! CHECK: res = iflm (666)
! ditto, with & ! comment, no leading &
      integer function IFLM(x)
        integer :: x
        IFLM = x
      end function IFLM
      program main
#define IFLM(x) ((x)+111)
      integer :: res
      res = IFLM& ! comment
(666)
      if (res .eq. 777) then
        print *, 'pp115.F90 yes'
      else
        print *, 'pp115.F90 no: ', res
      end if
      end
