! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: res=ikwm2z
* KWM expansion is before token pasting due to fixed-form space removal
      integer, parameter :: IKWM2Z = 777
#define KWM KWM2
      integer :: res
      res = I KWM Z
      if (res .eq. 777) then
        print *, 'pp025.F yes'
      else
        print *, 'pp025.F no: ', res
      end if
      end
