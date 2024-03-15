! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: res = (777)
! KWM call name split across continuation, with leading &
      integer, parameter :: KWM = 666
#define KWM 777
      integer :: res
      res = (KW&
&M)
      if (res .eq. 777) then
        print *, 'pp105.F90 yes'
      else
        print *, 'pp105.F90 no: ', res
      end if
      end
