! KWMs case-sensitive
      integer, parameter :: KWM = 777
#define KWM 666
      if (kwm .eq. 777) then
        print *, 'pp104.F90 pass'
      else
        print *, 'pp104.F90 FAIL: ', kwm
      end if
      end
