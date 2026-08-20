!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

subroutine f00(x)
  integer :: x(10)
  integer :: i

! Valid uses on TASK.
!$omp task threadset(omp_pool)
  x = x + 1
!$omp end task

!$omp task threadset(omp_team)
  x = x + 1
!$omp end task

! Valid uses on TASKLOOP.
!$omp taskloop threadset(omp_pool)
  do i = 1, 10
  end do
!$omp end taskloop

!$omp taskloop threadset(omp_team)
  do i = 1, 10
  end do
!$omp end taskloop

! At most one THREADSET clause is allowed on the directive.
!ERROR: At most one THREADSET clause can appear on TASK directive
!$omp task threadset(omp_pool) threadset(omp_team)
  x = x + 1
!$omp end task

! THREADSET is only allowed on TASK and TASKLOOP.
!ERROR: THREADSET clause is not allowed on PARALLEL directive
!$omp parallel threadset(omp_pool)
  x = x + 1
!$omp end parallel
end
