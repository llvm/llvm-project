! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: res = KWM
! KWM call name split across continuation, no leading &, with & ! comment
      integer, parameter :: KWM = 666
#define KWM 777
      integer :: res
      res = KW& ! comment
M
      if (res .eq. 777) then
        print *, 'pp107.F90 yes'
      else
        print *, 'pp107.F90 no: ', res
      end if
      end
