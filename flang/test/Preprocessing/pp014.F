! RUN: %f18 -E %s 2>&1 | FileCheck %s
! CHECK: res=((666)+111)
* FLM call split between name and (, with intervening *comment
      integer function IFLM(x)
        integer :: x
        IFLM = x
      end function IFLM
      program main
#define IFLM(x) ((x)+111)
      integer :: res
      res = IFLM
*comment
     +(666)
      if (res .eq. 777) then
        print *, 'pp014.F yes'
      else
        print *, 'pp014.F no: ', res
      end if
      end
