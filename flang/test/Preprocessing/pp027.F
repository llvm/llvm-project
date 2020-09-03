* #DEFINE works in fixed form
      integer, parameter :: KWM = 666
#DEFINE KWM 777
      if (KWM .eq. 777) then
        print *, 'pp027.F pass'
      else
        print *, 'pp027.F FAIL: ', KWM
      end if
      end
