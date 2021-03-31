! Tests -fget-definition returning source position of symbol definition.
module m1
 private :: f
contains
 pure subroutine s (x, yyy) bind(c)
  intent(in) :: x
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

! RUN and CHECK lines at the bottom as this test is sensitive to line numbers
! RUN: %f18 -fget-definition 6 17 18 -fsyntax-only %s | FileCheck --check-prefix=CHECK1 %s
! RUN: %f18 -fget-definition 7 20 23 -fsyntax-only %s | FileCheck --check-prefix=CHECK2 %s
! RUN: %f18 -fget-definition 14 3 4 -fsyntax-only %s | FileCheck --check-prefix=CHECK3 %s
! CHECK1: x:{{.*}}getdefinition01.f90, 5, 21-22
! CHECK2: yyy:{{.*}}getdefinition01.f90, 5, 24-27
! CHECK3: x:{{.*}}getdefinition01.f90, 13, 24-25

! RUN: not %f18 -fget-definition -fsyntax-only %s 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
! CHECK-ERROR: Invalid argument to -fget-definitions
