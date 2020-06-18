! KWM NOT expanded in Hollerith literal
#define KWM 666
#define HKWM 667
      character(len=3) :: ch
      ch = 3HKWM
      if (ch .eq. 'KWM') then
        print *, 'pp123.F90 pass'
      else
        print *, 'pp123.F90 FAIL: ', ch
      end if
      end
