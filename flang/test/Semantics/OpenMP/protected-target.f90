! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=52

module protected_target_lastprivate
  integer, pointer, protected_target :: module_pointer
contains
  subroutine host_association(target)
    integer, target :: target
    !$omp parallel sections lastprivate(module_pointer)
    !$omp section
    module_pointer => target
    !$omp end parallel sections
  end subroutine
end module

subroutine pointer_association(p, ordinary, target)
  integer, pointer, protected_target :: p
  integer, pointer :: ordinary
  integer, target :: target
  integer :: i
  ! LASTPRIVATE updates pointer association, not the protected target.
  !$omp parallel do lastprivate(p, ordinary)
  do i = 1, 2
    p => target
    ordinary => target
  end do
  !$omp end parallel do
end subroutine

subroutine use_association(target)
  use protected_target_lastprivate, only: renamed => module_pointer
  integer, target :: target
  !$omp parallel sections lastprivate(renamed)
  !$omp section
  renamed => target
  !$omp end parallel sections
end subroutine

subroutine region_association(p, q, ordinary, target, readonly_target)
  integer, pointer, protected_target :: p, q
  integer, pointer :: ordinary
  integer, target :: target
  integer, target, intent(in) :: readonly_target
  integer :: i
  !$omp parallel
  p => q
  p => readonly_target
  ordinary => target
  !ERROR: Pointer 'ordinary' must have PROTECTED_TARGET to associate with target of PROTECTED_TARGET pointer 'q'
  ordinary => q
  !$omp end parallel

  !$omp parallel private(p, ordinary) firstprivate(q)
  p => q
  p => ordinary
  ordinary => target
  !ERROR: Pointer 'ordinary' must have PROTECTED_TARGET to associate with target of PROTECTED_TARGET pointer 'q'
  ordinary => q
  !$omp end parallel

  !$omp task firstprivate(p, q, ordinary)
  p => q
  ordinary => target
  !ERROR: Pointer 'ordinary' must have PROTECTED_TARGET to associate with target of PROTECTED_TARGET pointer 'q'
  ordinary => q
  !$omp end task

  !$omp parallel do lastprivate(p, ordinary) firstprivate(q)
  do i = 1, 2
    p => q
    ordinary => target
    !ERROR: Pointer 'ordinary' must have PROTECTED_TARGET to associate with target of PROTECTED_TARGET pointer 'q'
    ordinary => q
  end do
  !$omp end parallel do
end subroutine

subroutine use_region_association(q, ordinary, target)
  use protected_target_lastprivate, only: renamed => module_pointer
  integer, pointer, protected_target :: q
  integer, pointer :: ordinary
  integer, target :: target
  !$omp parallel
  renamed => q
  q => renamed
  renamed => ordinary
  ordinary => target
  !ERROR: Pointer 'ordinary' must have PROTECTED_TARGET to associate with target of PROTECTED_TARGET pointer 'q'
  ordinary => q
  !ERROR: Pointer 'ordinary' must have PROTECTED_TARGET to associate with target of PROTECTED_TARGET pointer 'renamed'
  ordinary => renamed
  !$omp end parallel
end subroutine

subroutine intent_in_pointers(p, ordinary)
  integer, pointer, protected_target, intent(in) :: p
  integer, pointer, intent(in) :: ordinary
  integer :: i
  !ERROR: Pointer 'p' with the INTENT(IN) attribute may not appear in a LASTPRIVATE clause
  !ERROR: Pointer 'ordinary' with the INTENT(IN) attribute may not appear in a LASTPRIVATE clause
  !$omp parallel do lastprivate(p, ordinary)
  do i = 1, 2
  end do
  !$omp end parallel do
end subroutine

subroutine target_definition(p)
  integer, pointer, protected_target :: p
  integer :: i
  ! REDUCTION still defines the target (F2028 C866).
  !ERROR: Variable 'p' on the REDUCTION clause is not definable
  !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
  !$omp parallel do reduction(+:p)
  do i = 1, 2
  end do
  !$omp end parallel do
end subroutine
