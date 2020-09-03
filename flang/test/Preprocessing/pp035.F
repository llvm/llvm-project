* #if 2 .LT. 3 works
      integer, parameter :: KWM = 666
#if 2 .LT. 3
#define KWM 777
#else
#define KWM 667
#endif
      if (KWM .eq. 777) then
        print *, 'pp035.F pass'
      else
        print *, 'pp035.F FAIL: ', KWM
      end if
      end
