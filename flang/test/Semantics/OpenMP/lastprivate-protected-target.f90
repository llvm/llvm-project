! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! A LASTPRIVATE pointer is updated as if by pointer assignment, so a
! PROTECTED_TARGET pointer (F2028 draft J3/26-007r1) may appear in the clause.

subroutine s(x)
  real, target :: x
  real, pointer, protected_target :: p
  integer :: i
  !$omp parallel do lastprivate(p)
  do i = 1, 2
    p => x
  end do
  !$omp end parallel do
  print *, associated(p)
end
