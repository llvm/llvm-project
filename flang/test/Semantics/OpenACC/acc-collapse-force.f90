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

! Check that !$acc loop on a loop inside a collapse nest is rejected.
subroutine sub4()
  integer :: i, j, k
  integer, parameter :: n = 100
  real :: a(n, n, n), tmp
  !$acc parallel loop collapse(force:3) copy(a)
  do i = 1, n
    do j = 1, n
      tmp = real(i + j)
      !ERROR: LOOP directive not expected in COLLAPSE loop nest
      !$acc loop
      do k = 1, n
        a(i, j, k) = tmp + real(k)
      end do
    end do
  end do
end subroutine
