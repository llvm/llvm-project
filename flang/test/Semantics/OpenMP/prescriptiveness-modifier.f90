!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=61

subroutine f00(x)
  integer :: x, i
  !No diagnostic expected
  !$omp taskloop grainsize(strict: x)
  do i = 1, 10
  enddo
end

subroutine f01(x)
  integer :: x, i
  !ERROR: Only STRICT is allowed as prescriptiveness on this clause
  !$omp taskloop grainsize(fallback: x)
  do i = 1, 10
  enddo
end

subroutine f02(x)
  integer :: x, i
  !No diagnostic expected
  !$omp taskloop num_tasks(strict: x)
  do i = 1, 10
  enddo
end

subroutine f03(x)
  integer :: x, i
  !ERROR: Only STRICT is allowed as prescriptiveness on this clause
  !$omp taskloop num_tasks(fallback: x)
  do i = 1, 10
  enddo
end

subroutine f04(x)
  integer :: x
  !No diagnostic expected
  !$omp target dyn_groupprivate(strict: x)
  !$omp end target
end

subroutine f05(x)
  integer :: x
  !No diagnostic expected
  !$omp target dyn_groupprivate(fallback: x)
  !$omp end target
end
