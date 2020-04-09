* KWM NOT expanded in 9HHOLLERITH literal
#define KWM 666
#define HKWM 667
      character(len=3) :: ch
      ch = 3HKWM
      if (ch .eq. 'KWM') then
        print *, 'pp023.F pass'
      else
        print *, 'pp023.F FAIL: ', ch
      end if
      end
