! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: res=((666)+111)
* FLM call with closing ')' on next line (not a continuation)
      integer function IFLM(x)
        integer :: x
        IFLM = x
      end function IFLM
      program main
#define IFLM(x) ((x)+111)
      integer :: res
      res = IFLM(666
)
      if (res .eq. 777) then
        print *, 'pp038.F yes'
      else
        print *, 'pp038.F no: ', res
      end if
      end
