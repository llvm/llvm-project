* KWMs case-sensitive
      integer, parameter :: KWM = 777
#define KWM 666
      if (kwm .eq. 777) then
        print *, 'pp004.F pass'
      else
        print *, 'pp004.F FAIL: ', kwm
      end if
      end
