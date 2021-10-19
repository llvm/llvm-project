! RUN: %python %S/test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.15.3.6 Reduction Clause

subroutine omp_target(p)
  integer, pointer, intent(in) :: p

  integer :: i
  integer :: k = 10

  !ERROR: Pointer 'p' with the INTENT(IN) attribute may not appear in a REDUCTION clause
  !$omp parallel do reduction(+:p)
  do i = 1, 10
    k= k + 1
  end do
  !$omp end parallel do

end subroutine omp_target
