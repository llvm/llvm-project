* KWM split across continuation, implicit padding
      integer, parameter :: KWM = 666
#define KWM 777
      integer :: res
      res = KW
     +M
      if (res .eq. 777) then
        print *, 'pp005.F pass'
      else
        print *, 'pp005.F FAIL: ', res
      end if
      end
