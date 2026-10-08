! RUN: not %flang_fc1 -fopenacc -fsyntax-only %s 2>&1 | FileCheck %s --implicit-check-not="Unexpected DO construct name"

! A labeled DO statement has no construct name, so the END DO statement that
! terminates it must not specify one, also when the loop is associated with an
! OpenACC loop or combined construct.

subroutine combined(a, n)
  integer :: n, i
  real :: a(n)
  !$acc parallel loop
  do 10 i = 1, n
    a(i) = 0
10 end do foo
! CHECK: :[[@LINE-1]]:11: error: Unexpected DO construct name 'foo'
end subroutine

subroutine nested(a, n, m)
  integer :: n, m, i, j
  real :: a(n, m)
  !$acc parallel
  !$acc loop collapse(2)
  do 20 j = 1, m
    do 30 i = 1, n
      a(i, j) = 0
30  end do inner
! CHECK: :[[@LINE-1]]:12: error: Unexpected DO construct name 'inner'
20 end do outer
! CHECK: :[[@LINE-1]]:11: error: Unexpected DO construct name 'outer'
  !$acc end parallel
end subroutine

subroutine shared(a, n, m)
  integer :: n, m, i, j
  real :: a(n, m)
  !$acc parallel loop collapse(2)
  do 40 j = 1, m
    do 40 i = 1, n
      a(i, j) = 0
40 end do bar
! CHECK: :[[@LINE-1]]:11: error: Unexpected DO construct name 'bar'
end subroutine
