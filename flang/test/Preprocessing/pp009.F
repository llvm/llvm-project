! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: res = ((666)+111)
* FLM call split across continuation, implicit padding
      integer function IFLM(x)
        integer :: x
        IFLM = x
      end function IFLM
      program main
#define IFLM(x) ((x)+111)
      integer :: res
      res = IFL
     +M(666)
      if (res .eq. 777) then
        print *, 'pp009.F yes'
      else
        print *, 'pp009.F no: ', res
      end if
      end
