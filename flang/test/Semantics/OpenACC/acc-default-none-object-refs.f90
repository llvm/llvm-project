! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! DEFAULT(NONE) must also diagnose references to variables that appear in the
! parse tree as a bare object rather than as an Expr, Variable, or ArrayElement:
! the objects of ALLOCATE/DEALLOCATE/NULLIFY and the pointer target of a pointer
! assignment.  These reach the parse-tree walk as a Name or StructureComponent,
! not wrapped in an Expr/Variable/ArrayElement, so they need dedicated handling.

! 1. Unlisted objects of allocate/deallocate/nullify are diagnosed.
subroutine test_object_refs_unlisted()
  implicit none
  real, allocatable :: q(:)
  real, pointer :: p(:)
  !$acc parallel default(none)
  !ERROR: The DEFAULT(NONE) clause requires that 'q' must be listed in a data-mapping clause
  allocate(q(10))
  !ERROR: The DEFAULT(NONE) clause requires that 'q' must be listed in a data-mapping clause
  deallocate(q)
  !ERROR: The DEFAULT(NONE) clause requires that 'p' must be listed in a data-mapping clause
  nullify(p)
  !$acc end parallel
end subroutine

! 2. Unlisted pointer target and pointee of a pointer assignment are diagnosed.
!    The left-hand side (the pointer) is the reference the parse tree exposes as
!    a bare DataRef; the right-hand side is an Expr and was already checked.
subroutine test_pointer_assign_unlisted()
  implicit none
  real, pointer :: p(:), tg(:)
  !$acc parallel default(none)
  !ERROR: The DEFAULT(NONE) clause requires that 'p' must be listed in a data-mapping clause
  !ERROR: The DEFAULT(NONE) clause requires that 'tg' must be listed in a data-mapping clause
  p => tg
  !$acc end parallel
end subroutine

! 3. A structure-component object still reports its base variable.
subroutine test_object_ref_component()
  implicit none
  type t
    real, pointer :: p(:)
  end type
  type(t) :: a
  !$acc parallel default(none)
  !ERROR: The DEFAULT(NONE) clause requires that 'a' must be listed in a data-mapping clause
  nullify(a%p)
  !$acc end parallel
end subroutine

! 4. Listed objects do not error.
subroutine test_object_refs_listed()
  implicit none
  real, allocatable :: q(:)
  real, pointer :: p(:)
  !$acc parallel default(none) create(q) copyin(p)
  allocate(q(10))
  deallocate(q)
  nullify(p)
  !$acc end parallel
end subroutine

! 5. Without DEFAULT(NONE) the object references are not flagged.
subroutine test_object_refs_no_default_none()
  implicit none
  real, allocatable :: q(:)
  real, pointer :: p(:), tg(:)
  !$acc parallel
  allocate(q(10))
  deallocate(q)
  nullify(p)
  p => tg
  !$acc end parallel
end subroutine
