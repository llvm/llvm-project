* FLM call split between name and (
      integer function IFLM(x)
        integer :: x
        IFLM = x
      end function IFLM
      program main
#define IFLM(x) ((x)+111)
      integer :: res
      res = IFLM
     +(666)
      if (res .eq. 777) then
        print *, 'pp013.F pass'
      else
        print *, 'pp013.F FAIL: ', res
      end if
      end
