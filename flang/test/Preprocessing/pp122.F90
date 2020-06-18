! KWM NOT expanded in "literal"
#define KWM 666
      character(len=3) :: ch
      ch = "KWM"
      if (ch .eq. 'KWM') then
        print *, 'pp122.F90 pass'
      else
        print *, 'pp122.F90 FAIL: ', ch
      end if
      end
