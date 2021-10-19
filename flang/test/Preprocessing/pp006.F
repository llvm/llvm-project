! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: res = 777
* ditto, but with intervening *comment line
      integer, parameter :: KWM = 666
#define KWM 777
      integer :: res
      res = KW
*comment
     +M
      if (res .eq. 777) then
        print *, 'pp006.F yes'
      else
        print *, 'pp006.F no: ', res
      end if
      end
