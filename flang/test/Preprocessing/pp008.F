! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: res = K W M
* KWM with spaces in name at invocation NOT replaced
      integer, parameter :: KWM = 777
#define KWM 666
      integer :: res
      res = K W M
      if (res .eq. 777) then
        print *, 'pp008.F yes'
      else
        print *, 'pp008.F no: ', res
      end if
      end
