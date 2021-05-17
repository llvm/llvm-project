! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: if(kwm .eq. 777) then
! KWMs case-sensitive
      integer, parameter :: KWM = 777
#define KWM 666
      if (kwm .eq. 777) then
        print *, 'pp104.F90 yes'
      else
        print *, 'pp104.F90 no: ', kwm
      end if
      end
