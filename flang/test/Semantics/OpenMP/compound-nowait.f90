! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=50
! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51
! RUN: %flang -fopenmp -fopenmp-version=52 -fsyntax-only %s
! RUN: %flang -fopenmp -fopenmp-version=60 -fsyntax-only %s

! NOWAIT applies to TARGET. It may appear on the end directive starting in 5.2.

subroutine begin_form
  integer :: i

  !$omp target parallel do nowait
  do i = 1, 10
  end do
  !$omp end target parallel do

  !$omp target parallel do simd nowait
  do i = 1, 10
  end do
  !$omp end target parallel do simd

  !$omp target parallel loop nowait
  do i = 1, 10
  end do
  !$omp end target parallel loop

  !$omp target teams distribute parallel do nowait
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do

  !$omp target teams distribute parallel do simd nowait
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do simd
end subroutine

subroutine end_form
  integer :: i

  !ERROR: This compound directive does not allow NOWAIT clause
  !$omp target parallel do
  do i = 1, 10
  end do
  !$omp end target parallel do nowait

  !ERROR: This compound directive does not allow NOWAIT clause
  !$omp target parallel do simd
  do i = 1, 10
  end do
  !$omp end target parallel do simd nowait

  !ERROR: This compound directive does not allow NOWAIT clause
  !$omp target parallel loop
  do i = 1, 10
  end do
  !$omp end target parallel loop nowait

  !ERROR: This compound directive does not allow NOWAIT clause
  !$omp target teams distribute parallel do
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do nowait

  !ERROR: This compound directive does not allow NOWAIT clause
  !$omp target teams distribute parallel do simd
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do simd nowait
end subroutine
