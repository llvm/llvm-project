! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! A PROTECTED_TARGET pointer (F'2028 8.5.16) may appear in a clause that
! updates a pointer list item by pointer assignment, such as LASTPRIVATE,
! COPYPRIVATE and COPYIN, but not where its target would be defined.

module m
  integer, pointer, protected_target :: mp
end

subroutine pointer_assignment_clauses(p, q, r)
  integer, pointer, protected_target :: p
  integer, pointer, protected_target, intent(in) :: q
  integer, pointer :: r
  integer, pointer, protected_target, save :: tp
  integer, target :: x(10)
  integer :: i
  !$omp threadprivate(tp)

  !$omp parallel do lastprivate(p)
  do i = 1, 10
    p => x(i)
  end do
  !$omp end parallel do

  !$omp parallel do lastprivate(r)
  do i = 1, 10
    r => x(i)
  end do
  !$omp end parallel do

  !ERROR: Pointer 'q' with the INTENT(IN) attribute may not appear in a LASTPRIVATE clause
  !$omp parallel do lastprivate(q)
  do i = 1, 10
  end do
  !$omp end parallel do

  !$omp parallel private(p)
  !$omp single
  p => x(1)
  !$omp end single copyprivate(p)
  !$omp end parallel

  !$omp parallel copyin(tp)
  print *, tp
  !$omp end parallel

  !$omp parallel do firstprivate(p)
  do i = 1, 10
    print *, p
  end do
  !$omp end parallel do
end

subroutine associated_pointers
  use m
  integer, pointer, protected_target :: hp
  integer, target :: x(10)
  integer :: i
  !$omp parallel do lastprivate(mp)
  do i = 1, 10
    mp => x(i)
  end do
  !$omp end parallel do
  call inner
 contains
  subroutine inner
    integer :: j
    !$omp parallel do lastprivate(hp)
    do j = 1, 10
      hp => x(j)
    end do
    !$omp end parallel do
  end
end

subroutine select_rank(ar)
  integer, pointer, protected_target :: ar(..)
  integer, target :: x(10)
  integer :: i
  select rank (r => ar)
  rank (1)
    !ERROR: Variable 'r' in ASSOCIATE cannot be in a LASTPRIVATE clause
    !$omp parallel do lastprivate(r)
    do i = 1, 10
      r => x(i:i)
    end do
    !$omp end parallel do
  end select
end

subroutine target_definitions(p)
  integer, pointer, protected_target :: p
  integer :: i

  !ERROR: Variable 'p' on the REDUCTION clause is not definable
  !BECAUSE: 'p' has the PROTECTED_TARGET attribute
  !$omp parallel do reduction(+:p)
  do i = 1, 10
  end do
  !$omp end parallel do

  !$omp parallel do lastprivate(p)
  do i = 1, 10
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'p' has the PROTECTED_TARGET attribute
    p = i
  end do
  !$omp end parallel do

  !$omp parallel
  !$omp atomic update
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'p' has the PROTECTED_TARGET attribute
  p = p + 1
  !$omp end parallel
end
