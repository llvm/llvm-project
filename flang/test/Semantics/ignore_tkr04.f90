! RUN: %python %S/test_errors.py %s %flang_fc1
! Tests for ignore_tkr(p)
module ignore_tkr_4_m
interface
  subroutine s(a)
  real, pointer :: a(:)
!dir$ ignore_tkr(p) a
  end subroutine
  subroutine s1(a)
    real, allocatable :: a(:)
!dir$ ignore_tkr(p) a
  end subroutine
end interface
end module
program t
  use ignore_tkr_4_m
  real, allocatable :: x(:)
  real, pointer :: x1(:)
  call s(x)
!CHECK-NOT: error
!CHECK-NOT: warning
  call s1(x1)
!CHECK-NOT: error
!CHECK-NOT: warning
end

