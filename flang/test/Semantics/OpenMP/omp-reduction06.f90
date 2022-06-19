! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.15.3.6 Reduction Clause

program omp_reduction

  integer :: i
  integer :: k = 10
  integer :: a(10), b(10,10,10)

  !ERROR: A list item that appears in a REDUCTION clause should have a contiguous storage array section.
  !$omp parallel do reduction(+:a(1:10:3))
  do i = 1, 10
    k = k + 1
  end do
  !$omp end parallel do

  !ERROR: A list item that appears in a REDUCTION clause should have a contiguous storage array section.
  !$omp parallel do reduction(+:b(1:10:3,1:8:1,1:5:1))
  do i = 1, 10
    k = k + 1
  end do
  !$omp end parallel do

  !ERROR: A list item that appears in a REDUCTION clause should have a contiguous storage array section.
  !$omp parallel do reduction(+:b(1:10:1,1:8:2,1:5:1))
  do i = 1, 10
    k = k + 1
  end do
  !$omp end parallel do
end program omp_reduction
