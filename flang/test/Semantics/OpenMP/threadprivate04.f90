! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 5.1
! Check OpenMP construct validity for the following directives:
! 2.21.2 Threadprivate Directive

program main
  integer :: i, N = 10
  integer, save :: x1, x2, x3, x4, x5, x6, x7, x8, x9
  common /blk1/ y1, /blk2/ y2, /blk3/ y3, /blk4/ y4, /blk5/ y5

  !$omp threadprivate(x1, x2, x3, x4, x5, x6, x7, x8, x9)
  !$omp threadprivate(/blk1/, /blk2/, /blk3/, /blk4/, /blk5/)

  !$omp parallel num_threads(x1)
  !$omp end parallel

  !$omp single copyprivate(x2, /blk1/)
  !$omp end single

  !$omp do schedule(static, x3)
  do i = 1, N
    y1 = x3
  end do
  !$omp end do

  !$omp parallel copyin(x4, /blk2/)
  !$omp end parallel

  !$omp parallel if(x5 > 1)
  !$omp end parallel

  !$omp teams thread_limit(x6)
  !$omp end teams

  !ERROR: A THREADPRIVATE variable cannot be in PRIVATE clause
  !ERROR: A THREADPRIVATE variable cannot be in PRIVATE clause
  !$omp parallel private(x7, /blk3/)
  !$omp end parallel

  !ERROR: A THREADPRIVATE variable cannot be in FIRSTPRIVATE clause
  !ERROR: A THREADPRIVATE variable cannot be in FIRSTPRIVATE clause
  !$omp parallel firstprivate(x8, /blk4/)
  !$omp end parallel

  !ERROR: A THREADPRIVATE variable cannot be in SHARED clause
  !ERROR: A THREADPRIVATE variable cannot be in SHARED clause
  !$omp parallel shared(x9, /blk5/)
  !$omp end parallel
end
