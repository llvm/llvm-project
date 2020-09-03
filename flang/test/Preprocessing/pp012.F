* FLM call name split across continuation
      integer function IFLM(x)
        integer :: x
        IFLM = x
      end function IFLM
      program main
#define IFLM(x) ((x)+111)
      integer :: res
      res = IFL
     +M(666)
      if (res .eq. 777) then
        print *, 'pp012.F pass'
      else
        print *, 'pp012.F FAIL: ', res
      end if
      end
