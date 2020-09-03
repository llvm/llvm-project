! FLM call with '(' on next line (not a continuation)
      integer function IFLM(x)
        integer :: x
        IFLM = x
      end function IFLM
      program main
#define IFLM(x) ((x)+111)
      integer :: res
      res = IFLM
(666)
      if (res .eq. 777) then
        print *, 'pp128.F90 pass'
      else
        print *, 'pp128.F90 FAIL: ', res
      end if
      end
