! RUN: %f18 -E %s 2>&1 | FileCheck %s
! CHECK: if(7777.eq.777)then
! CHECK: print*,'pp037.F no: ',7777
* fixed-form clipping NOT applied to #define
      integer, parameter :: KWM = 666
*        1         2         3         4         5         6         7
*234567890123456789012345678901234567890123456789012345678901234567890123
#define KWM                                                          7777
      if (KWM .eq. 777) then
        print *, 'pp037.F yes'
      else
        print *, 'pp037.F no: ', KWM
      end if
      end
