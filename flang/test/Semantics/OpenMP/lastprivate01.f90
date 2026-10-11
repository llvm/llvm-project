! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.15.3.5 lastprivate Clause
! A variable that appears in a lastprivate clause must be definable.

module protected_var
  integer, protected :: p
  integer, protected, target :: pt
  integer, pointer, protected_target :: ptp
  integer, pointer, protected, protected_target :: pptp
end module protected_var

program omp_lastprivate
  use protected_var
  integer :: i, a(10), b(10), c(10)
  integer, parameter :: k = 10

  a = 10
  b = 20

  !ERROR: 'k' must be a variable list item
  !$omp parallel do lastprivate(k)
  do i = 1, 10
    c(i) = a(i) + b(i) + k
  end do
  !$omp end parallel do

  !ERROR: Variable 'p' on the LASTPRIVATE clause is not definable
  !BECAUSE: 'p' is protected in this scope
  !$omp parallel do lastprivate(p)
  do i = 1, 10
    c(i) = a(i) + b(i) + k
  end do
  !$omp end parallel do

  ! A POINTER list item is updated by pointer assignment.
  !$omp parallel do lastprivate(ptp)
  do i = 1, 10
    ptp => pptp
    !ERROR: A PROTECTED_TARGET pointer may not be associated with USE-associated PROTECTED target 'pt'
    ptp => pt
  end do
  !$omp end parallel do

  !ERROR: Variable 'pptp' on the LASTPRIVATE clause is not definable
  !BECAUSE: 'pptp' is protected in this scope
  !$omp parallel do lastprivate(pptp)
  do i = 1, 10
    c(i) = a(i) + b(i) + k
  end do
  !$omp end parallel do

  call omp_lastprivate_sb(i)

  print *, c

end program omp_lastprivate

subroutine omp_lastprivate_sb(m)
  integer :: i, a(10), b(10), c(10)
  integer, intent(in) :: m

  a = 10
  b = 20

  !ERROR: Variable 'm' on the LASTPRIVATE clause is not definable
  !BECAUSE: 'm' is an INTENT(IN) dummy argument
  !$omp parallel do lastprivate(m)
  do i = 1, 10
    c(i) = a(i) + b(i) + m
  end do
  !$omp end parallel do

  print *, c

end subroutine omp_lastprivate_sb

subroutine omp_lastprivate_protected_target(ip)
  integer, pointer, protected_target, intent(in) :: ip
  integer, pointer, protected_target :: np
  integer :: i

  !$omp parallel do lastprivate(np)
  do i = 1, 10
    np => ip
  end do
  !$omp end parallel do

  !ERROR: Pointer 'ip' with the INTENT(IN) attribute may not appear in a LASTPRIVATE clause
  !$omp parallel do lastprivate(ip)
  do i = 1, 10
  end do
  !$omp end parallel do
end subroutine omp_lastprivate_protected_target
