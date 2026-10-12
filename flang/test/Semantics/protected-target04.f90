! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic

module protected_target_calls
  type :: item
    integer :: value
  contains
    procedure :: set
    procedure :: get
  end type
  type :: value_item
    integer :: value
  contains
    procedure :: assign_value_object
    generic :: assignment(=) => assign_value_object
  end type
  type :: value_parent
    type(value_item) :: child
  end type
  interface update
    module procedure update_whole, read_each
  end interface
  interface pointer_or_value
    module procedure by_pointer, each_value
  end interface
  interface assignment(=)
    module procedure assign_item, assign_value
  end interface
contains
  subroutine generic_calls(readonly, writable)
    integer, pointer, protected_target :: readonly(:)
    integer, pointer :: writable(:)
    ! F2028 C874 does not change which specific is selected.
    !ERROR: Nonpointer dummy argument 'whole=' must have INTENT(IN) to accept PROTECTED_TARGET pointer 'readonly' or its subobject
    call update(readonly)
    call update(writable)
    ! F2028 C872 does not change which specific is selected.
    !ERROR: POINTER dummy argument 'whole=' must have PROTECTED_TARGET to accept PROTECTED_TARGET pointer 'readonly' or its subobject
    call pointer_or_value(readonly)
    call pointer_or_value(writable)
  end subroutine

  subroutine update_whole(whole)
    integer, intent(inout) :: whole(:)
    whole = 0
  end subroutine
  elemental subroutine read_each(element)
    integer, intent(in) :: element
  end subroutine
  subroutine by_pointer(whole)
    integer, pointer, intent(in) :: whole(:)
  end subroutine
  elemental subroutine each_value(element)
    integer, intent(in) :: element
  end subroutine

  subroutine assignments(readonly, writable)
    type(item), pointer, protected_target :: readonly
    type(item), pointer :: writable
    ! Defined assignment resolves even though the LHS violates F2028 C866/C874.
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: Target of PROTECTED_TARGET pointer 'readonly' is not definable
    !ERROR: Nonpointer dummy argument 'lhs=' must have INTENT(IN) to accept PROTECTED_TARGET pointer 'readonly' or its subobject
    readonly = 1
    writable = 1
  end subroutine
  subroutine assign_item(lhs, rhs)
    type(item), intent(out) :: lhs
    integer, intent(in) :: rhs
    lhs%value = rhs
  end subroutine

  subroutine assignment_rhs(readonly, writable, object, values, mask)
    integer, pointer, protected_target :: readonly(:)
    integer, pointer :: writable(:)
    type(value_parent), pointer, protected_target :: object
    type(value_item) :: value, values(:)
    type(item) :: in_value
    logical :: mask(:)
    integer :: i
    ! F2028 15.4.3.4.3 parenthesizes the RHS even for a VALUE dummy.
    value = readonly(1)
    value = writable(1)
    in_value = readonly(1)
    values = readonly
    where (mask) values = readonly
    forall (i = 1:size(readonly)) values(i) = readonly(i)
    value = object%child
    call assign_value(value, (readonly(1)))
    call assign_item(in_value, readonly(1))
    call value%assign_value_object((object%child))
    ! An explicit call with an unparenthesized actual still follows F2028 C874.
    !ERROR: Nonpointer dummy argument 'rhs=' must have INTENT(IN) to accept PROTECTED_TARGET pointer 'readonly' or its subobject
    call assign_value(value, readonly(1))
    !ERROR: Nonpointer dummy argument 'rhs=' must have INTENT(IN) to accept PROTECTED_TARGET pointer 'object' or its subobject
    call value%assign_value_object(object%child)
  end subroutine
  elemental subroutine assign_value(lhs, rhs)
    type(value_item), intent(out) :: lhs
    integer, value :: rhs
    lhs%value = rhs
  end subroutine
  subroutine assign_value_object(lhs, rhs)
    class(value_item), intent(out) :: lhs
    type(value_item), value :: rhs
    lhs%value = rhs%value
  end subroutine

  subroutine bound_calls(readonly, writable)
    type(item), pointer, protected_target :: readonly
    type(item), pointer :: writable
    integer :: value
    ! F2028 C874 also applies to the passed-object actual argument.
    !ERROR: Nonpointer dummy argument 'this=' must have INTENT(IN) to accept PROTECTED_TARGET pointer 'readonly' or its subobject
    call readonly%set(1)
    value = readonly%get()
    call writable%set(value)
  end subroutine
  subroutine set(this, value)
    class(item), intent(inout) :: this
    integer, intent(in) :: value
    this%value = value
  end subroutine
  integer function get(this)
    class(item), intent(in) :: this
    get = this%value
  end function

  subroutine statement_function(p)
    integer, pointer, protected_target :: p
    integer :: sf, i, value
    sf(i) = i + 1
    ! F2028 C873: a statement function has an implicit interface.
    !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may not be passed to a procedure with an implicit interface
    value = sf(p)
    value = sf(p + 1)
    value = sf((p))
  end subroutine

  subroutine implicit_call_location(p, ordinary)
    integer, pointer, protected_target :: p
    integer, pointer :: ordinary
    external :: implicit_subroutine
    call implicit_subroutine( &
        !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may not be passed to a procedure with an implicit interface
        p)
    call implicit_subroutine(ordinary)
  end subroutine

  subroutine implicit_function(p, dummy_function)
    integer, pointer, protected_target :: p
    integer, external :: f
    procedure(integer) :: dummy_function
    procedure(integer), pointer :: function_pointer
    integer :: value
    ! F2028 C873 applies to function references as well as CALL statements.
    !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may not be passed to a procedure with an implicit interface
    value = f(p)
    !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may not be passed to a procedure with an implicit interface
    value = dummy_function(p)
    !ERROR: PROTECTED_TARGET pointer 'p' or its subobject may not be passed to a procedure with an implicit interface
    value = function_pointer(p)
    value = f((p))
  end subroutine

  subroutine value_arguments(p)
    integer, pointer, protected_target :: p
    ! F2028 C874 requires INTENT(IN) even for a VALUE dummy.
    !ERROR: Nonpointer dummy argument 'x=' must have INTENT(IN) to accept PROTECTED_TARGET pointer 'p' or its subobject
    call by_value(p)
    call by_value_in(p)
  end subroutine
  subroutine by_value(x)
    integer, value :: x
  end subroutine
  subroutine by_value_in(x)
    integer, value, intent(in) :: x
  end subroutine

  subroutine intrinsic_calls(p, array, text)
    use iso_c_binding, only: c_ptr, c_loc, c_f_pointer
    integer, pointer, protected_target :: p
    real, pointer, protected_target :: array(:)
    character(10), pointer, protected_target :: text
    type(c_ptr) :: address
    logical :: is_associated
    integer :: extent
    ! F2028 C874 applies to intrinsic INTENT(OUT) arguments.
    !ERROR: Nonpointer dummy argument 'harvest=' must have INTENT(IN) to accept PROTECTED_TARGET pointer 'array' or its subobject
    call random_number(array)
    !ERROR: Nonpointer dummy argument 'date=' must have INTENT(IN) to accept PROTECTED_TARGET pointer 'text' or its subobject
    call date_and_time(date=text)
    ! FPTR is a pointer dummy without PROTECTED_TARGET (F2028 C872).
    !ERROR: POINTER dummy argument 'fptr=' must have PROTECTED_TARGET to accept PROTECTED_TARGET pointer 'p' or its subobject
    call c_f_pointer(address, p)
    address = c_loc(p)
    is_associated = associated(p)
    extent = size(array) + len(text)
    p => null(mold=p)
  end subroutine

  subroutine ordinary_actual(writable)
    integer, pointer :: writable
    ! F2028 C872 constrains PROTECTED_TARGET actuals, not ordinary pointers.
    call protected_out(writable)
  end subroutine
  subroutine protected_out(p)
    integer, pointer, protected_target, intent(out) :: p
    nullify(p)
  end subroutine
end module
