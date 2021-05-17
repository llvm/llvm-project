! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: if(777.eq.777)then
* /* C comment */ erased from #define
      integer, parameter :: KWM = 666
#define KWM 777 /* C comment */
      if (KWM .eq. 777) then
        print *, 'pp030.F yes'
      else
        print *, 'pp030.F no: ', KWM
      end if
      end
