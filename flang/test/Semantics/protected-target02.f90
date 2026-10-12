! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic

module protected_target_checks
  type :: t
    integer :: value
    integer, allocatable :: allocated
    integer, pointer :: writable
    integer, pointer, protected_target :: readonly
  end type
  type :: ordinary
    integer, pointer :: p
  end type
  type :: guarded
    integer, pointer, protected_target :: p
  end type
contains
  subroutine definitions(p, n, text, z, object)
    integer, pointer, protected_target :: p(:), n
    character(20), pointer, protected_target :: text
    complex, pointer, protected_target :: z
    type(t), pointer, protected_target :: object
    integer, target :: target
    integer :: i
    ! F2028 C866 and 8.5.16p2
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
    p = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
    p(1) = 1
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'text' is not definable
    text(1:2) = 'hi'
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'z' is not definable
    z%re = 1.
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'object' is not definable
    object%value = 1
    !ERROR: 'n' may not be used as a DO variable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'n' is not definable
    do n = 1, 2
    end do
    !ERROR: Input variable 'n' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'n' is not definable
    read *, n
    !ERROR: Internal file variable 'text' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'text' is not definable
    write(text, *) 1
    !ERROR: IOSTAT variable 'n' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'n' is not definable
    read(*, *, iostat=n) i
    associate (a => p(1))
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
      a = 1
    end associate
    associate (a => object%value)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: Target of PROTECTED_TARGET pointer 'object' is not definable
      a = 1
    end associate

    ! Pointer targets are not subobjects of their containing object.
    object%writable = 1
    ! Flang treats component association changes as changing the protected target.
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'object' is not definable
    object%writable => target
    !ERROR: 'writable' may not appear in NULLIFY
    !BECAUSE: Target of PROTECTED_TARGET pointer 'object' is not definable
    nullify(object%writable)
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'object' is not definable
    allocate(object%writable, source=1)
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'object' is not definable
    allocate(object%allocated, source=1)
    !ERROR: Name in DEALLOCATE statement is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'object' is not definable
    deallocate(object%allocated)
    print *, p, n, text, z, object%value
  end subroutine

  subroutine associate_names(p, object)
    integer, pointer, protected_target :: p
    type(t), pointer, protected_target :: object
    integer, pointer :: writable
    integer, pointer, protected_target :: readonly
    type(ordinary) :: ordinary_value
    type(guarded) :: guarded_value
    external :: implicit_interface
    ! An associate name does not acquire PROTECTED_TARGET (F2028 11.1.3.3).
    associate (a => p)
      !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
      !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
      writable => a
      !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
      !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
      call pointer_in(a)
      call implicit_interface(a)
      !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
      !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
      ordinary_value = ordinary(a)
      readonly => a
      guarded_value = guarded(a)
      call value_in(a)
      call value_default(a)
      ! 11.1.3.3p5 still prohibits variable definition contexts.
      !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'p=' is not definable
      !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
      call value_inout(a)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
      a = 1
    end associate
    associate (a => object)
      ! A pointer component written in the designator retains its attribute.
      !ERROR: Pointer 'writable' must have PROTECTED_TARGET to associate with target of PROTECTED_TARGET pointer 'readonly'
      writable => a%readonly
      !ERROR: Pointer component 'p' must have PROTECTED_TARGET to associate with target of PROTECTED_TARGET pointer 'readonly'
      ordinary_value = ordinary(a%readonly)
      !ERROR: POINTER dummy argument 'p=' must have PROTECTED_TARGET to accept PROTECTED_TARGET pointer 'readonly' or its subobject
      call pointer_in(a%readonly)
      !ERROR: PROTECTED_TARGET pointer 'readonly' or its subobject may not be passed to a procedure with an implicit interface
      call implicit_interface(a%readonly)
      !ERROR: Nonpointer dummy argument 'p=' must have INTENT(IN) to accept PROTECTED_TARGET pointer 'readonly' or its subobject
      call value_default(a%readonly)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: Target of PROTECTED_TARGET pointer 'object' is not definable
      a%value = 1
      writable => a%writable
      call pointer_in(a%writable)
      call implicit_interface(a%writable)
    end associate
  end subroutine

  subroutine associate_elements(p)
    integer, pointer, protected_target :: p(:)
    integer, pointer :: writable
    external :: implicit_interface
    associate (a => p)
      !WARNING: Pointer target is not a definable variable [-Wpointer-to-undefinable]
      !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
      writable => a(1)
      call implicit_interface(a(1))
      !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'p=' is not definable
      !BECAUSE: Target of PROTECTED_TARGET pointer 'p' is not definable
      call value_inout(a(1))
    end associate
  end subroutine

  subroutine allocation(p, object)
    integer, pointer, protected_target :: p
    type(t) :: object
    ! F2028 C869
    !ERROR: Object in ALLOCATE statement may not be allocated
    !BECAUSE: 'p' has PROTECTED_TARGET and requires SOURCE= in ALLOCATE
    allocate(p)
    !ERROR: Object in ALLOCATE statement may not be allocated
    !BECAUSE: 'p' has PROTECTED_TARGET and requires SOURCE= in ALLOCATE
    allocate(p, mold=1)
    !ERROR: Object in ALLOCATE statement may not be allocated
    !BECAUSE: 'readonly' has PROTECTED_TARGET and requires SOURCE= in ALLOCATE
    allocate(object%readonly)
    allocate(p, source=1)
    allocate(object%readonly, source=1)
    ! F2028 C868
    !ERROR: Object in DEALLOCATE statement is not deallocatable
    !BECAUSE: 'p' has PROTECTED_TARGET and may not be deallocated
    deallocate(p)
    !ERROR: Object in DEALLOCATE statement is not deallocatable
    !BECAUSE: 'readonly' has PROTECTED_TARGET and may not be deallocated
    deallocate(object%readonly)
    nullify(p, object%readonly)
  end subroutine

  subroutine association(p, array, object)
    integer, pointer, protected_target :: p, array(:)
    type(t), pointer, protected_target :: object
    integer, pointer :: writable
    integer, pointer, protected_target :: readonly
    type(ordinary) :: ordinary_value
    type(guarded) :: guarded_value
    ! F2028 C870
    !ERROR: Pointer 'writable' must have PROTECTED_TARGET to associate with target of PROTECTED_TARGET pointer 'p'
    writable => p
    !ERROR: Pointer 'writable' must have PROTECTED_TARGET to associate with target of PROTECTED_TARGET pointer 'array'
    writable => array(1)
    !ERROR: Pointer 'writable' must have PROTECTED_TARGET to associate with target of PROTECTED_TARGET pointer 'object'
    writable => object%value
    !ERROR: Pointer 'writable' must have PROTECTED_TARGET to associate with target of PROTECTED_TARGET pointer 'readonly'
    writable => object%readonly
    writable => object%writable
    readonly => p
    readonly => writable
    readonly => object%value
    nullify(readonly)
    ! F2028 C871
    !ERROR: Pointer component 'p' must have PROTECTED_TARGET to associate with target of PROTECTED_TARGET pointer 'p'
    ordinary_value = ordinary(p)
    !ERROR: Pointer component 'p' must have PROTECTED_TARGET to associate with target of PROTECTED_TARGET pointer 'object'
    ordinary_value = ordinary(object%value)
    guarded_value = guarded(p)
    guarded_value = guarded(object%value)
    ordinary_value = ordinary(object%writable)
  end subroutine

  subroutine arguments(p, object)
    integer, pointer, protected_target :: p
    type(t), pointer, protected_target :: object
    external :: implicit_interface
    ! F2028 C872
    !ERROR: POINTER dummy argument 'p=' must have PROTECTED_TARGET to accept PROTECTED_TARGET pointer 'p' or its subobject
    call pointer_in(p)
    !ERROR: POINTER dummy argument 'p=' must have PROTECTED_TARGET to accept PROTECTED_TARGET pointer 'p' or its subobject
    call pointer_out(p)
    !ERROR: POINTER dummy argument 'p=' must have PROTECTED_TARGET to accept PROTECTED_TARGET pointer 'object' or its subobject
    call pointer_in(object%value)
    call pointer_in(object%writable)
    ! Flang checks the container when the component's association may change.
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'p=' is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'object' is not definable
    call pointer_inout(object%writable)
    call protected_in(p)
    call protected_out(p)
    call protected_default(p)
    call protected_in(object%value)
    ! F2028 C873
    !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may not be passed to a procedure with an implicit interface
    call implicit_interface(p)
    !ERROR: PROTECTED_TARGET pointer 'object' or its subobject may not be passed to a procedure with an implicit interface
    call implicit_interface(object%value)
    call implicit_interface(p + 1)
    call implicit_interface((p))
    ! F2028 C874
    !ERROR: Nonpointer dummy argument 'p=' must have INTENT(IN) to accept PROTECTED_TARGET pointer 'p' or its subobject
    call value_out(p)
    !ERROR: Nonpointer dummy argument 'p=' must have INTENT(IN) to accept PROTECTED_TARGET pointer 'p' or its subobject
    call value_inout(p)
    !ERROR: Nonpointer dummy argument 'p=' must have INTENT(IN) to accept PROTECTED_TARGET pointer 'p' or its subobject
    call value_default(p)
    !ERROR: Nonpointer dummy argument 'p=' must have INTENT(IN) to accept PROTECTED_TARGET pointer 'object' or its subobject
    call value_default(object%value)
    call value_in(p)
    call value_in(object%value)
    call value_default(p + 1)
    call value_default((p))
  end subroutine

  subroutine status(p)
    integer, pointer, protected_target, intent(in) :: p
    integer, target :: target
    ! PROTECTED_TARGET does not relax INTENT(IN) on pointer association.
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: 'p' is an INTENT(IN) dummy argument
    p => target
    !ERROR: 'p' may not appear in NULLIFY
    !BECAUSE: 'p' is an INTENT(IN) dummy argument
    nullify(p)
    !ERROR: Name in ALLOCATE statement is not definable
    !BECAUSE: 'p' is an INTENT(IN) dummy argument
    allocate(p, source=1)
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'p=' is not definable
    !BECAUSE: 'p' is an INTENT(IN) dummy argument
    call protected_out(p)
  end subroutine

  subroutine pointer_in(p)
    integer, pointer, intent(in) :: p
  end subroutine
  subroutine pointer_out(p)
    integer, pointer, intent(out) :: p
  end subroutine
  subroutine pointer_inout(p)
    integer, pointer, intent(inout) :: p
  end subroutine
  subroutine protected_in(p)
    integer, pointer, protected_target, intent(in) :: p
  end subroutine
  subroutine protected_out(p)
    integer, pointer, protected_target, intent(out) :: p
  end subroutine
  subroutine protected_default(p)
    integer, pointer, protected_target :: p
  end subroutine
  subroutine value_in(p)
    integer, intent(in) :: p
  end subroutine
  subroutine value_out(p)
    integer, intent(out) :: p
  end subroutine
  subroutine value_inout(p)
    integer, intent(inout) :: p
  end subroutine
  subroutine value_default(p)
    integer :: p
  end subroutine
end module
