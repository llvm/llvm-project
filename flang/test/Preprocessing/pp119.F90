! FLM rescan
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
        print *, 'pp119.F90 pass'
      else
        print *, 'pp119.F90 FAIL: ', res
      end if
      end
