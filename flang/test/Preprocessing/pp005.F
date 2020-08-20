! RUN: %f18 -E %s 2>&1 | FileCheck %s
! CHECK: res=777
* KWM split across continuation, implicit padding
      integer, parameter :: KWM = 666
#define KWM 777
      integer :: res
      res = KW
     +M
      if (res .eq. 777) then
        print *, 'pp005.F yes'
      else
        print *, 'pp005.F no: ', res
      end if
      end
