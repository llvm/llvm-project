! Tests -fget-symbols-sources finding all symbols in file.
module mm1
 private :: f
contains
 pure subroutine s (x, y) bind(c)
  intent(in) :: x
  intent(inout) :: y
 contains
  pure subroutine ss
  end subroutine
 end subroutine
 recursive pure function f() result(x)
  real, allocatable :: x
  x = 1.0
 end function
end module

! RUN: %flang_fc1 -fsyntax-only -fget-symbols-sources %s 2>&1 | FileCheck %s
! CHECK-COUNT-1:f:{{.*}}getsymbols01.f90, 12, 26-27
! CHECK-COUNT-1:mm1:{{.*}}getsymbols01.f90, 2, 8-11
! CHECK-COUNT-1:s:{{.*}}getsymbols01.f90, 5, 18-19
! CHECK-COUNT-1:ss:{{.*}}getsymbols01.f90, 9, 19-21
! CHECK-COUNT-1:x:{{.*}}getsymbols01.f90, 5, 21-22
! CHECK-COUNT-1:x:{{.*}}getsymbols01.f90, 13, 24-25
! CHECK-COUNT-1:y:{{.*}}getsymbols01.f90, 5, 24-25
