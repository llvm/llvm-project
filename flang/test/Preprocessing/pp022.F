* KWM NOT expanded in "literal"
#define KWM 666
      character(len=3) :: ch
      ch = "KWM"
      if (ch .eq. 'KWM') then
        print *, 'pp022.F pass'
      else
        print *, 'pp022.F FAIL: ', ch
      end if
      end
