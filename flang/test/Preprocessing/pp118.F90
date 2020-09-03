! KWM rescan with #undef, proving rescan after expansion
      integer, parameter :: KWM2 = 777, KWM = 667
#define KWM2 666
#define KWM KWM2
#undef KWM2
      if (KWM .eq. 777) then
        print *, 'pp118.F90 pass'
      else
        print *, 'pp118.F90 FAIL: ', KWM
      end if
      end
