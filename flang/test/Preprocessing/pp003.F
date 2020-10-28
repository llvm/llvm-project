! RUN: %f18 -E %s 2>&1 | FileCheck %s
! CHECK: res=((666)+111)
* function-like macros
      integer function IFLM(x)
        integer :: x
        IFLM = x
      end function IFLM
      program main
#define IFLM(x) ((x)+111)
      integer :: res
      res = IFLM(666)
      if (res .eq. 777) then
        print *, 'pp003.F yes'
      else
        print *, 'pp003.F no: ', res
      end if
      end
