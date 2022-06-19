! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: res = ((666)+111)
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
        print *, 'pp010.F yes'
      else
        print *, 'pp010.F no: ', res
      end if
      end
