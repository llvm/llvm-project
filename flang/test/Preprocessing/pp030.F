* /* C comment */ erased from #define
      integer, parameter :: KWM = 666
#define KWM 777 /* C comment */
      if (KWM .eq. 777) then
        print *, 'pp030.F pass'
      else
        print *, 'pp030.F FAIL: ', KWM
      end if
      end
