! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: if(777 .eq. 777) then
! \ newline works in #define
      integer, parameter :: KWM = 666
#define KWM 77\
7
      if (KWM .eq. 777) then
        print *, 'pp126.F90 yes'
      else
        print *, 'pp126.F90 no: ', KWM
      end if
      end
