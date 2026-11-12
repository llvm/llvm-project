! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: KWM = 666
! CHECK: if (777 .eq. 777) then
* #DEFINE works in fixed form
      integer, parameter :: KWM = 666
#DEFINE KWM 777
      if (KWM .eq. 777) then
        print *, 'pp027.F yes'
      else
        print *, 'pp027.F no: ', KWM
      end if
      end
