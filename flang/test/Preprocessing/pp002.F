* #undef
      integer, parameter :: KWM = 777
#define KWM 666
#undef KWM
      if (KWM .eq. 777) then
        print *, 'pp002.F pass'
      else
        print *, 'pp002.F FAIL: ', KWM
      end if
      end
