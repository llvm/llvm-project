! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=50

subroutine reduction_task_combined(x, y)
  integer :: x, y, i
  !$omp parallel do reduction(task, +: x)
  do i = 1, 4
  end do
  !$omp end parallel do

  !$omp parallel sections reduction(task, +: x)
  !$omp section
  x = x + 1
  !$omp end parallel sections

  !$omp target parallel reduction(task, +: x)
  x = x + 1
  !$omp end target parallel

  !$omp parallel shared(x)
  !$omp do reduction(task, +: x)
  do i = 1, 4
  end do
  !$omp end do
  !$omp end parallel

  !$omp parallel do reduction(task, +: x) reduction(task, +: y)
  do i = 1, 4
  end do
  !$omp end parallel do

  !$omp parallel do reduction(task, +: x) reduction(default, +: y)
  do i = 1, 4
  end do
  !$omp end parallel do

  !$omp parallel do reduction(default, +: x) reduction(task, +: y)
  do i = 1, 4
  end do
  !$omp end parallel do

  !ERROR: Modifier 'TASK' on REDUCTION clause is only allowed with PARALLEL or worksharing directive
  !$omp parallel do simd reduction(task, +: x)
  do i = 1, 4
  end do
  !$omp end parallel do simd

  !ERROR: Modifier 'TASK' on REDUCTION clause is only allowed with PARALLEL or worksharing directive
  !$omp parallel loop reduction(task, +: x)
  do i = 1, 4
  end do
  !$omp end parallel loop
end subroutine
