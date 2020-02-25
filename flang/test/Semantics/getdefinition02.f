! Tests -fget-definition with fixed form.

      module m2
       private :: f
      contains
       pure subroutine s (x, yyy) bind(c)
        intent(in) :: 
     *  x
        intent(inout) :: yyy
       contains
        pure subroutine ss
        end subroutine
       end subroutine
       recursive pure function f() result(x)
        real, allocatable :: x
        x = 1.0
       end function
      end module

! RUN: ${F18} -fget-definition 8 9 10 -fparse-only %s > %t;
! RUN: ${F18} -fget-definition 9 26 29 -fparse-only %s >> %t;
! RUN: ${F18} -fget-definition 16 9 10 -fparse-only %s >> %t;
! RUN: cat %t | ${FileCheck} %s
! CHECK:x:.*getdefinition02.f, 6, 27-28
! CHECK:yyy:.*getdefinition02.f, 6, 30-33
! CHECK:x:.*getdefinition02.f, 15, 30-31
