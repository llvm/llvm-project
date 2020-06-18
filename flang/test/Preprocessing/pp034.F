* \ newline allowed in name on KWM definition
      integer, parameter :: KWMC = 666
#define KWM\
C 777
      if (KWMC .eq. 777) then
        print *, 'pp034.F pass'
      else
        print *, 'pp034.F FAIL: ', KWMC
      end if
      end
