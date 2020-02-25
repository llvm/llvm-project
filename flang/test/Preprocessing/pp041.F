* use KWM expansion as continuation indicators
#define KWM 0
#define KWM2 1
      integer :: j
      j = 666
     KWM j = j + 1
     KWM2 11
      if (j .eq. 777) then
        print *, 'pp041.F pass'
      else
        print *, 'pp041.F FAIL', j
      end if
      end
