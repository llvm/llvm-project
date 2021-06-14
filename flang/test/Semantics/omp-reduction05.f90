! RUN: %S/test_errors.sh %s %t %flang_fc1 -fopenmp
! REQUIRES: shell
! OpenMP Version 4.5
! 2.15.3.6 Reduction Clause

program omp_reduction

  integer :: i
  integer :: k = 10
  integer :: a(10),b(10,10,10)

  !ERROR: 'a' in REDUCTION clause is a zero size array section
  !$omp parallel do reduction(+:a(1:0:2))
  do i = 1, 10
    k = k + 1
  end do
  !$omp end parallel do

  !ERROR: 'a' in REDUCTION clause is a zero size array section
  !$omp parallel do reduction(+:a(1:0))
  do i = 1, 10
    k = k + 1
  end do
  !$omp end parallel do

  !ERROR: 'b' in REDUCTION clause is a zero size array section
  !$omp parallel do reduction(+:b(1:6,5,1:0))
  do i = 1, 10
    k = k + 1
  end do
  !$omp end parallel do

  !ERROR: 'b' in REDUCTION clause is a zero size array section
  !$omp parallel do reduction(+:b(1:6,1:0:5,1:10))
  do i = 1, 10
    k = k + 1
  end do
  !$omp end parallel do
end program omp_reduction
