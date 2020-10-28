! RUN: %f18 -E %s 2>&1 | FileCheck %s
! CHECK: res=kwm
* KWM split across continuation, clipped after column 72
      integer, parameter :: KWM = 666
#define KWM 777
      integer :: res
* 'comment' is in column 73
*        1         2         3         4         5         6         7
*234567890123456789012345678901234567890123456789012345678901234567890123
      res = KW                                                          comment
     +M
      if (res .eq. 777) then
        print *, 'pp007.F yes'
      else
        print *, 'pp007.F no: ', res
      end if
      end
