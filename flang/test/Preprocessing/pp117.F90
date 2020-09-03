! KWM rescan
      integer, parameter :: KWM = 666, KWM2 = 667
#define KWM2 777
#define KWM KWM2
      if (KWM .eq. 777) then
        print *, 'pp117.F90 pass'
      else
        print *, 'pp117.F90 FAIL: ', KWM
      end if
      end
