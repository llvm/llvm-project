! #DEFINE works in free form
      integer, parameter :: KWM = 666
#DEFINE KWM 777
      if (KWM .eq. 777) then
        print *, 'pp125.F90 pass'
      else
        print *, 'pp125.F90 FAIL: ', KWM
      end if
      end
