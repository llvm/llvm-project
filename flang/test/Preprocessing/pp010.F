* ditto, but with intervening *comment line
      integer function IFLM(x)
        integer :: x
        IFLM = x
      end function IFLM
      program main
#define IFLM(x) ((x)+111)
      integer :: res
      res = IFL
*comment
     +M(666)
      if (res .eq. 777) then
        print *, 'pp010.F pass'
      else
        print *, 'pp010.F FAIL: ', res
      end if
      end
