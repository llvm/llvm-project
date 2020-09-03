! #undef
      integer, parameter :: KWM = 777
#define KWM 666
#undef KWM
      if (KWM .eq. 777) then
        print *, 'pp102.F90 pass'
      else
        print *, 'pp102.F90 FAIL: ', KWM
      end if
      end
