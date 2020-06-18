! KWM call name split across continuation, with leading &
      integer, parameter :: KWM = 666
#define KWM 777
      integer :: res
      res = KW&
&M
      if (res .eq. 777) then
        print *, 'pp105.F90 pass'
      else
        print *, 'pp105.F90 FAIL: ', res
      end if
      end
