! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! Test that a pointer with the PROTECTED_TARGET attribute is accepted in
! a LASTPRIVATE clause (updated as if by pointer assignment per OpenMP 5.2
! section 5.4.5), while REDUCTION is rejected (defines target) and INTENT(IN)
! remains rejected.

subroutine omp_lastprivate_pt(p_in)
  integer, pointer, protected_target, intent(in) :: p_in
  integer, pointer, protected_target :: p
  integer, pointer, protected_target :: q
  integer, target :: t(10)
  integer :: i

  ! A pointer with PROTECTED_TARGET is permitted on LASTPRIVATE
  !$omp parallel do lastprivate(p)
  do i = 1, 10
    p => t(i)
    p => q
  end do
  !$omp end parallel do

  !$omp parallel
  !$omp do lastprivate(p)
  do i = 1, 10
    p => t(i)
  end do
  !$omp end do
  !$omp end parallel

  !$omp sections lastprivate(p)
  !$omp section
    p => t(1)
  !$omp end sections

  ! The PROTECTED_TARGET attribute of p also applies inside data-sharing
  ! regions, so it may be associated with another PROTECTED_TARGET pointer
  !$omp parallel private(p)
  p => q
  !$omp end parallel

  !$omp parallel
  p => q
  !$omp end parallel

  ! REDUCTION defines the target, so it must be rejected
  !ERROR: Variable 'p' on the REDUCTION clause is not definable
  !BECAUSE: The target of 'p' with the PROTECTED_TARGET attribute may not be defined
  !$omp parallel do reduction(+:p)
  do i = 1, 10
  end do

  ! INTENT(IN) pointer remains rejected on LASTPRIVATE
  !ERROR: Pointer 'p_in' with the INTENT(IN) attribute may not appear in a LASTPRIVATE clause
  !$omp parallel do lastprivate(p_in)
  do i = 1, 10
  end do
end subroutine
