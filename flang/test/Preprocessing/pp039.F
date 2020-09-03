* FLM call with '(' on next line (not a continuation)
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
        print *, 'pp039.F pass'
      else
        print *, 'pp039.F FAIL: ', res
      end if
      end
