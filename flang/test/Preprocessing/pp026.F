* ## token pasting works in FLM
      integer function IFLM(x)
        integer :: x
        IFLM = x
      end function IFLM
      program main
      integer, parameter :: KWM = 668, KWM21 = 111, KWM1 = 669
#define KWM1 111
#define KWM2 33
#define KWM KWM2
#define IFLM(x) ((x##1)+6##6##6)
      integer :: res
      res = IFLM(KWM)
      if (res .eq. 777) then
        print *, 'pp026.F pass'
      else
        print *, 'pp026.F FAIL: ', res
      end if
      end
