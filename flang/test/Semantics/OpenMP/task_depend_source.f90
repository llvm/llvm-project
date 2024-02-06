! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 4.5
! 2.13.9 Depend Clause
! depend(source) can be used only with the ordered construct
program main
      implicit none
      integer :: number = 0

      !ERROR: DEPEND(SOURCE) or DEPEND(SINK : vec) can be used only with the ordered directive. Used here in the TASK construct.
      !$omp task depend(source)
      number = 1
      !$omp end task


      !$omp task
      number = number + 1
      !$omp end task

      !$omp task
      print*, number
      !$omp end task
end program main
