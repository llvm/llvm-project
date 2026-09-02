.. title:: flang-tidy - openmp-accumulator-race

openmp-accumulator-race
=======================

Detects potential race conditions in OpenMP parallel regions where variables are modified without proper protection. This check identifies assignments to shared variables that could lead to data races and inconsistent results in parallel execution.

.. code-block:: fortran

    program example
      implicit none
      integer :: sum = 0
      integer :: i

      !$omp parallel do
      do i = 1, 100
        sum = sum + i  ! This will trigger a warning - race condition
      end do
      !$omp end parallel do

      !$omp parallel do
      do i = 1, 100
        !$omp atomic update
        sum = sum + i  ! This is safe
      end do
      !$omp end parallel do

      print *, "Sum:", sum
    end program
