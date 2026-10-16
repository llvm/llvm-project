! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
! Type-only inquiries of a conditional argument fold in executable statements
! too, except RANK when every consequent-arg is assumed-rank.
subroutine s(flag, x, y, ar1, ar2, i8a, i8b)
  logical :: flag
  real :: x, y
  integer :: ar1(..), ar2(..)
  integer(8) :: i8a, i8b
  !CHECK: PRINT *, 8_4, 24_4
  print *, kind((flag ? i8a : i8b)), digits((flag ? x : y))
  !CHECK: PRINT *, rank(( flag ? ar1 : ar2 ))
  print *, rank((flag ? ar1 : ar2))
end subroutine
