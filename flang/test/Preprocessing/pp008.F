* KWM with spaces in name at invocation NOT replaced
      integer, parameter :: KWM = 777
#define KWM 666
      integer :: res
      res = K W M
      if (res .eq. 777) then
        print *, 'pp008.F pass'
      else
        print *, 'pp008.F FAIL: ', res
      end if
      end
