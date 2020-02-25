* \ newline allowed in #define
      integer, parameter :: KWM = 666
#define KWM 77\
 7
      if (KWM .eq. 777) then
        print *, 'pp029.F pass'
      else
        print *, 'pp029.F FAIL: ', KWM
      end if
      end
