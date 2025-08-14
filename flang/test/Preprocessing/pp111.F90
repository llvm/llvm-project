! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: res = IFLM(666)
! FLM call name split across continuation, no leading &, with & ! comment
      integer function IFLM(x)
        integer :: x
        IFLM = x
      end function IFLM
      program main
#define IFLM(x) ((x)+111)
      integer :: res
      res = IFL& ! comment
M(666)
      if (res .eq. 777) then
        print *, 'pp111.F90 yes'
      else
        print *, 'pp111.F90 no: ', res
      end if
      end
