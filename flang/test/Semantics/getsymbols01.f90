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

! RUN: ${F18} -fget-symbols-sources -fparse-only %s 2>&1 | ${FileCheck} %s
! CHECK-ONCE:mm1:.*getsymbols01.f90, 3, 8-11
! CHECK-ONCE:f:.*getsymbols01.f90, 13, 26-27
! CHECK-ONCE:s:.*getsymbols01.f90, 6, 18-19
! CHECK-ONCE:ss:.*getsymbols01.f90, 10, 19-21
! CHECK-ONCE:x:.*getsymbols01.f90, 6, 21-22
! CHECK-ONCE:y:.*getsymbols01.f90, 6, 24-25
! CHECK-ONCE:x:.*getsymbols01.f90, 14, 24-25
