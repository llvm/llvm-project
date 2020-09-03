* KLM rescan with #undef (so rescan is after expansion)
      integer, parameter :: KWM2 = 777, KWM = 667
#define KWM2 666
#define KWM KWM2
#undef KWM2
      if (KWM .eq. 777) then
        print *, 'pp018.F pass'
      else
        print *, 'pp018.F FAIL: ', KWM
      end if
      end
