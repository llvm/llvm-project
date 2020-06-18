* FLM rescan
      integer function IFLM(x)
        integer :: x
        IFLM = x
      end function IFLM
      program main
      integer, parameter :: KWM = 999
#define KWM 111
#define IFLM(x) ((x)+KWM)
      integer :: res
      res = IFLM(666)
      if (res .eq. 777) then
        print *, 'pp019.F pass'
      else
        print *, 'pp019.F FAIL: ', res
      end if
      end
