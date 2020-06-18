* FLM call split between name and ( and in argument
      integer function IFLM(x)
        integer :: x
        IFLM = x
      end function IFLM
      program main
#define IFLM(x) ((x)+111)
      integer :: res
      res = IFLM
     +(66
     +6)
      if (res .eq. 777) then
        print *, 'pp016.F pass'
      else
        print *, 'pp016.F FAIL: ', res
      end if
      end
