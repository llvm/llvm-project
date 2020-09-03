* FLM expansion of argument
      integer function IFLM(x)
        integer :: x
        IFLM = x
      end function IFLM
      program main
      integer, parameter :: KWM = 999
#define KWM 111
#define IFLM(x) ((x)+666)
      integer :: res
      res = IFLM(KWM)
      if (res .eq. 777) then
        print *, 'pp020.F pass'
      else
        print *, 'pp020.F FAIL: ', res
      end if
      end
