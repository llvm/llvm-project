! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: res = IFLM (666)
! ditto, but without & ! comment
      integer function IFLM(x)
        integer :: x
        IFLM = x
      end function IFLM
      program main
#define IFLM(x) ((x)+111)
      integer :: res
      res = IFL&
M(666)
      if (res .eq. 777) then
        print *, 'pp112.F90 yes'
      else
        print *, 'pp112.F90 no: ', res
      end if
      end
