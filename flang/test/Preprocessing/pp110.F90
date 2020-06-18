! ditto, with & ! comment
      integer function IFLM(x)
        integer :: x
        IFLM = x
      end function IFLM
      program main
#define IFLM(x) ((x)+111)
      integer :: res
      res = IFL& ! comment
&M(666)
      if (res .eq. 777) then
        print *, 'pp110.F90 pass'
      else
        print *, 'pp110.F90 FAIL: ', res
      end if
      end
