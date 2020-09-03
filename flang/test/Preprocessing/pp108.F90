! ditto, but without & ! comment
      integer, parameter :: KWM = 666
#define KWM 777
      integer :: res
      res = KW&
M
      if (res .eq. 777) then
        print *, 'pp108.F90 pass'
      else
        print *, 'pp108.F90 FAIL: ', res
      end if
      end
