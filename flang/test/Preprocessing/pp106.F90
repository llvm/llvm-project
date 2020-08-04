! RUN: %f18 -E %s 2>&1 | FileCheck %s
! CHECK: res = 777
! ditto, with & ! comment
      integer, parameter :: KWM = 666
#define KWM 777
      integer :: res
      res = KW& ! comment
&M
      if (res .eq. 777) then
        print *, 'pp106.F90 yes'
      else
        print *, 'pp106.F90 no: ', res
      end if
      end
