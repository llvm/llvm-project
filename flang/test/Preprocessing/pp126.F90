! \ newline works in #define
      integer, parameter :: KWM = 666
#define KWM 77\
7
      if (KWM .eq. 777) then
        print *, 'pp126.F90 pass'
      else
        print *, 'pp126.F90 FAIL: ', KWM
      end if
      end
