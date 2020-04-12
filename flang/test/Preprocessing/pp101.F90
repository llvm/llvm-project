! keyword macros
      integer, parameter :: KWM = 666
#define KWM 777
      if (KWM .eq. 777) then
        print *, 'pp101.F90 pass'
      else
        print *, 'pp101.F90 FAIL: ', KWM
      end if
      end
