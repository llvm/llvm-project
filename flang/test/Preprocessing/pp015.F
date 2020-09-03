* FLM call split between name and (, clipped
      integer function IFLM(x)
        integer :: x
        IFLM = x
      end function IFLM
      program main
#define IFLM(x) ((x)+111)
      integer :: res
* 'comment' is in column 73
*        1         2         3         4         5         6         7
*234567890123456789012345678901234567890123456789012345678901234567890123
      res = IFLM                                                        comment
     +(666)
      if (res .eq. 777) then
        print *, 'pp015.F pass'
      else
        print *, 'pp015.F FAIL: ', res
      end if
      end
