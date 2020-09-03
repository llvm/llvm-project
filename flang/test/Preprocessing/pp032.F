* /* C comment */ \ newline erased from #define
      integer, parameter :: KWM = 666
#define KWM 77/* C comment */\
7
      if (KWM .eq. 777) then
        print *, 'pp032.F pass'
      else
        print *, 'pp032.F FAIL: ', KWM
      end if
      end
