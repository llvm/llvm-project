! ditto, with & ! comment
      integer, parameter :: KWM = 666
#define KWM 777
      integer :: res
      res = KW& ! comment
&M
      if (res .eq. 777) then
        print *, 'pp106.F90 pass'
      else
        print *, 'pp106.F90 FAIL: ', res
      end if
      end
