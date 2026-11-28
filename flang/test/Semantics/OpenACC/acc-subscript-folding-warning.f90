! RUN: %flang_fc1 -fsyntax-only -fopenacc %s 2>&1 | FileCheck %s --implicit-check-not="{{^warning:}}"

! OpenACC analysis builds a path from the subscripts of a reference in a
! region. It must reuse the expression that the analyzer already folded, so a
! folding diagnostic is reported once, with its source location, and not
! repeated without a location (which would be unattributable under -Werror).

subroutine folding_warning_has_location(a)
  real :: a(10)
  real :: r
  !$acc parallel default(none) copy(a) copyout(r)
  r = a(1 / 0)
  !$acc end parallel
end subroutine

! CHECK: acc-subscript-folding-warning.f90:{{[0-9]+}}:7: warning: INTEGER(4) division by zero [-Wfolding-exception]
! CHECK: acc-subscript-folding-warning.f90:{{[0-9]+}}:9: warning: INTEGER(4) division by zero [-Wfolding-exception]
