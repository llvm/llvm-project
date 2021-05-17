! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: if(777 .eq. 777) then
! #DEFINE works in free form
      integer, parameter :: KWM = 666
#DEFINE KWM 777
      if (KWM .eq. 777) then
        print *, 'pp125.F90 yes'
      else
        print *, 'pp125.F90 no: ', KWM
      end if
      end
