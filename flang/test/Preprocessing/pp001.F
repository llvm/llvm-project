* keyword macros
      integer, parameter :: KWM = 666
#define KWM 777
      if (KWM .eq. 777) then
        print *, 'pp001.F pass'
      else
        print *, 'pp001.F FAIL: ', KWM
      end if
      end
