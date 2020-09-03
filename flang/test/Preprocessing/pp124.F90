! KWM NOT expanded in Hollerith in FORMAT
#define KWM 666
#define HKWM 667
      character(len=3) :: ch
 100  format(3HKWM)
      write(ch, 100)
      if (ch .eq. 'KWM') then
        print *, 'pp124.F90 pass'
      else
        print *, 'pp124.F90 FAIL: ', ch
      end if
      end
