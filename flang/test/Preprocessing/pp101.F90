! RUN: %f18 -E %s 2>&1 | FileCheck %s
! CHECK:  if(777 .eq. 777) then
! keyword macros
      integer, parameter :: KWM = 666
#define KWM 777
      if (KWM .eq. 777) then
        print *, 'pp101.F90 yes'
      else
        print *, 'pp101.F90 no: ', KWM
      end if
      end
