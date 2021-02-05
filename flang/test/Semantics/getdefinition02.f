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

! RUN and CHECK lines here as test is sensitive to line numbers
! RUN: %f18 -fget-definition 7 9 10 -fsyntax-only %s 2>&1 | FileCheck --check-prefix=CHECK1 %s
! RUN: %f18 -fget-definition 8 26 29 -fsyntax-only %s 2>&1 | FileCheck --check-prefix=CHECK2 %s
! RUN: %f18 -fget-definition 15 9 10 -fsyntax-only %s 2>&1 | FileCheck --check-prefix=CHECK3 %s
! CHECK1: x:{{.*}}getdefinition02.f, 5, 27-28
! CHECK2: yyy:{{.*}}getdefinition02.f, 5, 30-33
! CHECK3: x:{{.*}}getdefinition02.f, 14, 30-31
