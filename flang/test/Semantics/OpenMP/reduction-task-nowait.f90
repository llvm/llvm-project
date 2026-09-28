! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=51

! SCOPE requires OpenMP 5.1.

subroutine task_nowait_end(x)
  integer :: x, i
  !$omp parallel shared(x)
  !ERROR: Modifier 'TASK' on REDUCTION clause may not be used with NOWAIT
  !$omp do reduction(task, +: x)
  do i = 1, 4
    x = x + 1
  end do
  !$omp end do nowait
  !$omp end parallel
end subroutine

subroutine task_nowait_start_before(x)
  integer :: x, i
  !$omp parallel shared(x)
  !ERROR: Modifier 'TASK' on REDUCTION clause may not be used with NOWAIT
  !$omp do nowait reduction(task, +: x)
  do i = 1, 4
    x = x + 1
  end do
  !$omp end do
  !$omp end parallel
end subroutine

subroutine repeated_task(x, y)
  integer :: x, y, i
  !$omp parallel shared(x, y)
  !ERROR: Modifier 'TASK' on REDUCTION clause may not be used with NOWAIT
  !ERROR: Modifier 'TASK' on REDUCTION clause may not be used with NOWAIT
  !$omp do reduction(task, +: x) reduction(task, +: y)
  do i = 1, 4
    x = x + 1
    y = y + 1
  end do
  !$omp end do nowait
  !$omp end parallel
end subroutine

subroutine mixed_task_first(x, y)
  integer :: x, y, i
  !$omp parallel shared(x, y)
  !ERROR: Modifier 'TASK' on REDUCTION clause may not be used with NOWAIT
  !$omp do reduction(task, +: x) reduction(+: y) nowait
  do i = 1, 4
    x = x + 1
    y = y + 1
  end do
  !$omp end do
  !$omp end parallel
end subroutine

subroutine outer_nowait_inner_task(x)
  integer :: x, i
  !$omp target if(.false.) map(tofrom: x) nowait
  !$omp parallel shared(x)
  !$omp do reduction(task, +: x)
  do i = 1, 4
    x = x + 1
  end do
  !$omp end do
  !$omp end parallel
  !$omp end target
  !$omp taskwait
end subroutine

subroutine outer_task_inner_nowait(x, y)
  integer :: x, y, i
  !$omp parallel shared(x, y)
  !$omp do reduction(task, +: x)
  do i = 1, 4
    x = x + 1
  end do
  !$omp end do
  !$omp do reduction(+: y)
  do i = 1, 4
    y = y + 1
  end do
  !$omp end do nowait
  !$omp end parallel
end subroutine

subroutine sections_end(x, y)
  integer :: x, y
  !$omp parallel shared(x, y)
  !ERROR: Modifier 'TASK' on REDUCTION clause may not be used with NOWAIT
  !$omp sections reduction(task, +: x)
  !$omp section
  x = x + 1
  y = y + 1
  !$omp end sections nowait
  !$omp end parallel
end subroutine

subroutine sections_start_after(x, y)
  integer :: x, y
  !$omp parallel shared(x, y)
  !ERROR: Modifier 'TASK' on REDUCTION clause may not be used with NOWAIT
  !$omp sections reduction(task, +: x) nowait
  !$omp section
  x = x + 1
  y = y + 1
  !$omp end sections
  !$omp end parallel
end subroutine

subroutine sections_task_only(x, y)
  integer :: x, y
  !$omp parallel shared(x, y)
  !$omp sections reduction(task, +: x)
  !$omp section
  x = x + 1
  y = y + 1
  !$omp end sections
  !$omp end parallel
end subroutine

subroutine scope_end(x, y)
  integer :: x, y
  !$omp parallel shared(x, y)
  !ERROR: Modifier 'TASK' on REDUCTION clause may not be used with NOWAIT
  !$omp scope reduction(task, +: x)
  x = x + 1
  y = y + 1
  !$omp end scope nowait
  !$omp end parallel
end subroutine

subroutine scope_start_after(x, y)
  integer :: x, y
  !$omp parallel shared(x, y)
  !ERROR: Modifier 'TASK' on REDUCTION clause may not be used with NOWAIT
  !$omp scope reduction(task, +: x) nowait
  x = x + 1
  y = y + 1
  !$omp end scope
  !$omp end parallel
end subroutine

subroutine scope_default_nowait(x, y)
  integer :: x, y
  !$omp parallel shared(x, y)
  !$omp scope reduction(+: x) nowait
  x = x + 1
  y = y + 1
  !$omp end scope
  !$omp end parallel
end subroutine
