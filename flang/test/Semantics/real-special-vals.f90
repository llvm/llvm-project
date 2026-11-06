! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -module-dir %t %t/m.f90
! RUN: %flang_fc1 -fsyntax-only -Werror -module-dir %t %t/use.f90

! Ensure that canonical representations of infinities and NaNs in module files
! can be read without emitting folding exception warnings.

!--- m.f90
module m
  real(4), parameter :: positive_infinity = z'7f800000'
  real(4), parameter :: negative_infinity = z'ff800000'
  real(4), parameter :: quiet_nan = z'7fc00000'
end module

!--- use.f90
program test
  use m
end program
