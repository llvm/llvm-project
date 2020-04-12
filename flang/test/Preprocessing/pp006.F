* ditto, but with intervening *comment line
      integer, parameter :: KWM = 666
#define KWM 777
      integer :: res
      res = KW
*comment
     +M
      if (res .eq. 777) then
        print *, 'pp006.F pass'
      else
        print *, 'pp006.F FAIL: ', res
      end if
      end
