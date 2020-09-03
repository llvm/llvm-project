* KLM rescan
      integer, parameter :: KWM = 666, KWM2 = 667
#define KWM2 777
#define KWM KWM2
      if (KWM .eq. 777) then
        print *, 'pp017.F pass'
      else
        print *, 'pp017.F FAIL: ', KWM
      end if
      end
