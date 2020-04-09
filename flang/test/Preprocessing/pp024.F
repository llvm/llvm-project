* KWM NOT expanded in Hollerith in FORMAT
#define KWM 666
#define HKWM 667
      character(len=3) :: ch
 100  format(3HKWM)
      write(ch, 100)
      if (ch .eq. 'KWM') then
        print *, 'pp024.F pass'
      else
        print *, 'pp024.F FAIL: ', ch
      end if
      end
