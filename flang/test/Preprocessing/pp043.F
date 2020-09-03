* #define with # in column 6 is a continuation line in fixed-form
      integer, parameter :: defineKWM666 = 555
      integer, parameter :: KWM =
     #define KWM 666
     ++222
      if (KWM .eq. 777) then
        print *, 'pp043.F pass'
      else
        print *, 'pp043.F FAIL: ', KWM
      end if
      end
