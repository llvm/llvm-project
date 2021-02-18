! RUN: %S/test_errors.sh %s %t %flang -fopenmp

! 2.17.8 Flush construct [OpenMP 5.0]
!        memory-order-clause ->
!                               acq_rel
!                               release
!                               acquire
use omp_lib
  implicit none

  integer :: i, a, b
  real, DIMENSION(10) :: array

  a = 1.0
  !$omp parallel num_threads(4)
  !Only memory-order-clauses.
  if (omp_get_thread_num() == 1) then
    ! Allowed clauses.
    !$omp flush acq_rel
    array = (/1, 2, 3, 4, 5, 6, 7, 8, 9, 10/)
    !$omp flush release
    array = (/1, 2, 3, 4, 5, 6, 7, 8, 9, 10/)
    !$omp flush acquire

    !ERROR: expected end of line
    !ERROR: expected end of line
    !$omp flush private(array)
    !ERROR: expected end of line
    !ERROR: expected end of line
    !$omp flush num_threads(4)

    ! Mix allowed and not allowed clauses.
    !ERROR: expected end of line
    !ERROR: expected end of line
    !$omp flush num_threads(4) acquire
  end if
  !$omp end parallel
end

