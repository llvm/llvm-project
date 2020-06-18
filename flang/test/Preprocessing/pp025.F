* KWM expansion is before token pasting due to fixed-form space removal
      integer, parameter :: IKWM2Z = 777
#define KWM KWM2
      integer :: res
      res = I KWM Z
      if (res .eq. 777) then
        print *, 'pp025.F pass'
      else
        print *, 'pp025.F FAIL: ', res
      end if
      end
