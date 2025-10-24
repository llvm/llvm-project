! RUN: %python %S/../test_errors.py %s %flang -fopenacc -fsyntax-only

! Check that loop with force collapse do not break in the semantic step.
subroutine sub3()
  integer :: i, j
  integer, parameter :: n = 100, m = 200
  real, dimension(n, m) :: a
  real, dimension(n) :: bb
  real :: r
  a = 1
  r = 0
  !$acc parallel loop collapse(force:2) copy(a)
  do i = 1, n
    bb(i) = r
    do j = 1, m
      a(i,j) = r * a(i,j)
    enddo
  enddo
end subroutine
