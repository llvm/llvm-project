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
! RUN: %flang_fc1 -fget-definition 6 17 18 %s | FileCheck --check-prefix=CHECK1 %s
! RUN: %flang_fc1 -fget-definition 7 20 23 %s | FileCheck --check-prefix=CHECK2 %s
! RUN: %flang_fc1 -fget-definition 14 3 4 %s | FileCheck --check-prefix=CHECK3 %s
! CHECK1: x:{{.*}}getdefinition01.f90, 5, 21-22
! CHECK2: yyy:{{.*}}getdefinition01.f90, 5, 24-27
! CHECK3: x:{{.*}}getdefinition01.f90, 13, 24-25
