! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic

! F2028 C866 applies to output specifiers and construct-associated names.
subroutine output_specifiers(status, text, exists)
  integer, pointer, protected_target :: status
  character(20), pointer, protected_target :: text
  logical, pointer, protected_target :: exists
  integer, pointer :: p
  integer :: value, istat
  character(20) :: message
  logical :: ordinary_exists
  namelist /group/ status
  !ERROR: STAT variable 'status' is not definable
  !BECAUSE: Target of PROTECTED_TARGET pointer 'status' is not definable
  allocate(p, source=1, stat=status)
  !ERROR: ERRMSG variable 'text' is not definable
  !BECAUSE: Target of PROTECTED_TARGET pointer 'text' is not definable
  deallocate(p, stat=istat, errmsg=text)
  !ERROR: EXIST variable 'exists' is not definable
  !BECAUSE: Target of PROTECTED_TARGET pointer 'exists' is not definable
  inquire(file='file', exist=exists)
  !ERROR: IOMSG variable 'text' is not definable
  !BECAUSE: Target of PROTECTED_TARGET pointer 'text' is not definable
  read(*, *, iostat=istat, iomsg=text) value
  !ERROR: NAMELIST input group must not contain undefinable item 'status'
  !BECAUSE: Target of PROTECTED_TARGET pointer 'status' is not definable
  read(*, nml=group)
  write(*, nml=group)
  read(text, *, iostat=istat) value
  inquire(iolength=value) status
  inquire(file='file', exist=ordinary_exists)
  allocate(p, source=status, stat=istat, errmsg=message)
  deallocate(p, stat=istat, errmsg=message)
end subroutine

subroutine locality(readonly, writable, n)
  integer, pointer, protected_target :: readonly
  integer, pointer :: writable
  integer :: n, i
  ! F2028 C1130/C1131 exclude objects forbidden in variable definition contexts.
  !ERROR: 'readonly' may not appear in a locality-spec because it is not definable
  !BECAUSE: Target of PROTECTED_TARGET pointer 'readonly' is not definable
  do concurrent (i=1:n) local(readonly)
  end do
  !ERROR: 'readonly' may not appear in a locality-spec because it is not definable
  !BECAUSE: Target of PROTECTED_TARGET pointer 'readonly' is not definable
  do concurrent (i=1:n) local_init(readonly)
  end do
  !ERROR: 'readonly' may not appear in a locality-spec because it is not definable
  !BECAUSE: Target of PROTECTED_TARGET pointer 'readonly' is not definable
  do concurrent (i=1:n) reduce(+:readonly)
  end do
  do concurrent (i=1:n) shared(readonly)
    print *, readonly
  end do
  do concurrent (i=1:n) local(writable)
  end do
end subroutine

subroutine construct_association(polymorphic, assumed_rank)
  class(*), pointer, protected_target :: polymorphic
  integer, pointer, protected_target :: assumed_rank(..)
  interface
    subroutine ordinary_pointer(p)
      integer, pointer, intent(in) :: p(..)
    end subroutine
    subroutine default_intent(x)
      integer :: x(..)
    end subroutine
  end interface
  select type (value => polymorphic)
  type is (integer)
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'polymorphic' is not definable
    value = 1
    print *, value
  end select
  select rank (value => assumed_rank)
  rank (0)
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'assumed_rank' is not definable
    value = 1
    print *, value
  rank (1)
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'assumed_rank' is not definable
    value(1) = 1
    print *, value
    !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
    !BECAUSE: Target of PROTECTED_TARGET pointer 'assumed_rank' is not definable
    call ordinary_pointer(value)
    call default_intent(value)
  rank default
    ! F2028 C872
    !ERROR: POINTER dummy argument 'p=' must have PROTECTED_TARGET to accept PROTECTED_TARGET pointer 'assumed_rank' or its subobject
    call ordinary_pointer(value)
    ! F2028 C874
    !ERROR: Nonpointer dummy argument 'x=' must have INTENT(IN) to accept PROTECTED_TARGET pointer 'assumed_rank' or its subobject
    call default_intent(value)
  end select
end subroutine

subroutine array_definitions(readonly, writable)
  integer, pointer, protected_target :: readonly(:)
  integer, pointer :: writable(:)
  integer :: i
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: Target of PROTECTED_TARGET pointer 'readonly' is not definable
  forall (i=1:2) readonly(i) = i
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: Target of PROTECTED_TARGET pointer 'readonly' is not definable
  where (readonly > 0) readonly = 0
  forall (i=1:2) writable(i) = readonly(i)
  where (readonly > 0) writable = 0
end subroutine

subroutine selector_bounds(n, m, indices, ordinary_indices, text, &
    readonly_text, array, z, coarray)
  integer, pointer, protected_target :: n, indices(:)
  integer, pointer :: m, ordinary_indices(:)
  character(20) :: text
  character(20), pointer, protected_target :: readonly_text
  integer :: array(10)
  complex :: z(10)
  integer :: coarray[*]
  ! Bounds are read; only the selector's parent is defined.
  associate (a => text(n:n))
    a = 'x'
  end associate
  associate (a => text(m:m))
    a = 'x'
  end associate
  associate (a => readonly_text(n:n))
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'readonly_text' is not definable
    a = 'x'
  end associate
  text(n:n) = 'x'
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: Target of PROTECTED_TARGET pointer 'readonly_text' is not definable
  readonly_text(n:n) = 'x'
  associate (a => array(n))
    a = 1
  end associate
  associate (a => array(n:n:n))
    a = 1
  end associate
  associate (a => array(indices))
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Construct association 'a' has a vector subscript
    a = 1
  end associate
  associate (a => array(ordinary_indices))
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Construct association 'a' has a vector subscript
    a = 1
  end associate
  array(n) = 1
  array(n:n:n) = 1
  array(indices) = 1
  associate (a => z(n)%re)
    a = 1.
  end associate
  z(n)%im = 1.
  !ERROR: Selector must not be a coindexed object
  associate (a => coarray[n])
    print *, a
  end associate
  coarray[n] = 1
end subroutine
