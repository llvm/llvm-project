* FLM call split between name and (, with intervening *comment
      integer function IFLM(x)
        integer :: x
        IFLM = x
      end function IFLM
      program main
#define IFLM(x) ((x)+111)
      integer :: res
      res = IFLM
*comment
     +(666)
      if (res .eq. 777) then
        print *, 'pp014.F pass'
      else
        print *, 'pp014.F FAIL: ', res
      end if
      end
