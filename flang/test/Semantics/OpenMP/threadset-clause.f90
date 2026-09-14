!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=45

subroutine f00(x)
  integer :: x(10)
  integer :: i
!ERROR: THREADSET clause is not allowed on TASK directive in OpenMP v4.5, try -fopenmp-version=60
!$omp task threadset(omp_pool)
  x = x + 1
!$omp end task

!ERROR: THREADSET clause is not allowed on TASKLOOP directive in OpenMP v4.5, try -fopenmp-version=60
!$omp taskloop threadset(omp_team)
  do i = 1, 10
  end do
!$omp end taskloop
end
