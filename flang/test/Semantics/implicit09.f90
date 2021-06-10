! RUN: %S/test_errors.sh %s %t %flang_fc1 -fimplicit-none-type-never
! REQUIRES: shell
subroutine s1
  implicit none
  i = j + k  ! would be error without -fimplicit-none-type-never
end

subroutine s2(a, n)
  implicit none
  real :: a(n)  ! would be error without -fimplicit-none-type-never
  integer :: n
end
