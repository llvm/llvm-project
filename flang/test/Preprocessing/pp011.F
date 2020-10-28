! RUN: %f18 -E %s 2>&1 | FileCheck %s
! CHECK: res=iflm(666)
* FLM call name split across continuation, clipped
      integer function IFLM(x)
        integer :: x
        IFLM = x
      end function IFLM
      program main
#define IFLM(x) ((x)+111)
      integer :: res
* 'comment' is in column 73
*        1         2         3         4         5         6         7
*234567890123456789012345678901234567890123456789012345678901234567890123
      res = IFL                                                         comment
     +M(666)
      if (res .eq. 777) then
        print *, 'pp011.F yes'
      else
        print *, 'pp011.F no: ', res
      end if
      end
