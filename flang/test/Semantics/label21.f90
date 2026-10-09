! RUN: %python %S/test_errors.py %s %flang -fopenacc -fopenmp

! A labeled DO statement whose terminating statement is not in the same block
! cannot be converted into a DO construct.

subroutine outer_ends_inside_acc(a, n, m)
  integer :: n, m, i, k
  real :: a(n)
  !ERROR: Label '100' is not in DO loop scope
  do 100 k = 1, m
  !$acc parallel
  !$acc loop
  do 100 i = 1, n
    a(i) = a(i) + k
100 end do
  a(1) = 0.0
  !$acc end parallel
end subroutine

subroutine outer_ends_inside_omp(a, n, m)
  integer :: n, m, i, k
  real :: a(n)
  !ERROR: Label '200' is not in DO loop scope
  do 200 k = 1, m
  !$omp parallel
  do 200 i = 1, n
    a(i) = a(i) + k
200 continue
  a(1) = 0.0
  !$omp end parallel
end subroutine

subroutine starts_inside_block(a, n)
  integer :: n, i
  real :: a(n)
  block
  !ERROR: Label '300' is not in DO loop scope
  do 300 i = 1, n
    a(i) = 0
  end block
300 continue
end subroutine

subroutine starts_inside_acc(a, n)
  integer :: n, i
  real :: a(n)
  !$acc parallel
  !ERROR: Label '400' is not in DO loop scope
  do 400 i = 1, n
    a(i) = 0
  !$acc end parallel
400 continue
end subroutine

! A labeled DO statement terminated by the last statement of the block of an
! OpenACC or OpenMP construct encloses the construct.

subroutine outer_ends_last_in_acc(a, n, m)
  integer :: n, m, i, k
  real :: a(n)
  do 500 k = 1, m
  !$acc data copy(a)
  do 500 i = 1, n
    a(i) = a(i) + k
500 continue
  !$acc end data
end subroutine

subroutine outer_ends_last_in_omp(a, n, m)
  integer :: n, m, i, k
  real :: a(n)
  do 600 k = 1, m
  !$omp parallel
  do 600 i = 1, n
    a(i) = a(i) + k
600 continue
  !$omp end parallel
end subroutine

! An infinite labeled DO loop has no DO variable, so a terminating statement in
! another block did not even cause an error: the loop was silently dropped.

subroutine infinite_ends_inside_acc(a, i)
  integer :: i
  real :: a(1)
  !ERROR: Label '700' is not in DO loop scope
  do 700
  !$acc parallel
  i = i + 1
700 continue
  a(1) = 0.0
  !$acc end parallel
end subroutine

subroutine infinite_ends_inside_omp(a, i)
  integer :: i
  real :: a(1)
  !ERROR: Label '800' is not in DO loop scope
  do 800
  !$omp parallel
  i = i + 1
800 continue
  a(1) = 0.0
  !$omp end parallel
end subroutine
