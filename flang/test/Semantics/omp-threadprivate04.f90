! RUN: %python %S/test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 5.1
! Check OpenMP construct validity for the following directives:
! 2.21.2 Threadprivate Directive

program main
  integer :: i, N = 10
  integer, save :: x
  common /blk/ y

  !$omp threadprivate(x, /blk/)

  !$omp parallel num_threads(x)
  !$omp end parallel

  !$omp single copyprivate(x, /blk/)
  !$omp end single

  !$omp do schedule(static, x)
  do i = 1, N
    y = x
  end do
  !$omp end do

  !$omp parallel copyin(x, /blk/)
  !$omp end parallel

  !$omp parallel if(x > 1)
  !$omp end parallel

  !$omp teams thread_limit(x)
  !$omp end teams

  !ERROR: A THREADPRIVATE variable cannot be in PRIVATE clause
  !ERROR: A THREADPRIVATE variable cannot be in PRIVATE clause
  !$omp parallel private(x, /blk/)
  !$omp end parallel

  !ERROR: A THREADPRIVATE variable cannot be in FIRSTPRIVATE clause
  !ERROR: A THREADPRIVATE variable cannot be in FIRSTPRIVATE clause
  !$omp parallel firstprivate(x, /blk/)
  !$omp end parallel

  !ERROR: A THREADPRIVATE variable cannot be in SHARED clause
  !ERROR: A THREADPRIVATE variable cannot be in SHARED clause
  !$omp parallel shared(x, /blk/)
  !$omp end parallel
end
