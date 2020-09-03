* KWM NOT expanded in 'literal'
#define KWM 666
      character(len=3) :: ch
      ch = 'KWM'
      if (ch .eq. 'KWM') then
        print *, 'pp021.F pass'
      else
        print *, 'pp021.F FAIL: ', ch
      end if
      end
