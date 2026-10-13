! RUN: %python %S/test_errors.py %s %flang_fc1
! F2028 draft C870-C874, including subobjects and pointer function results.

module protected_target_associations
  type node
    integer :: value
    integer, pointer :: ordinary
    integer, pointer, protected_target :: protected
    type(node), pointer :: next
  end type
  type mutable_box
    integer, pointer :: ptr
  end type
  type protected_box
    integer, pointer, protected_target :: ptr
  end type
  type value_box
    integer :: value
  end type
contains
  subroutine pointer_assignment(p, record)
    integer, pointer, protected_target :: p(:), protected_array(:), protected_scalar
    type(node), pointer, protected_target :: record
    integer, pointer :: ordinary, array(:)
    type(mutable_box) :: mutable
    type(protected_box) :: protected
    type(value_box) :: copy
    !ERROR: A pointer without PROTECTED_TARGET may not be associated with PROTECTED_TARGET pointer 'p' or its subobject
    array => p
    !ERROR: A pointer without PROTECTED_TARGET may not be associated with PROTECTED_TARGET pointer 'p' or its subobject
    array => p(1:2)
    !ERROR: A pointer without PROTECTED_TARGET may not be associated with PROTECTED_TARGET pointer 'p' or its subobject
    array(1:2) => p(1:2)
    !ERROR: A pointer without PROTECTED_TARGET may not be associated with PROTECTED_TARGET pointer 'record' or its subobject
    ordinary => record%value
    !ERROR: A pointer without PROTECTED_TARGET may not be associated with PROTECTED_TARGET pointer 'protected' or its subobject
    ordinary => record%protected
    !ERROR: Pointer component 'ptr' must have PROTECTED_TARGET when its value is a PROTECTED_TARGET pointer or subobject
    mutable = mutable_box(p(1))
    !ERROR: Pointer component 'ptr' must have PROTECTED_TARGET when its value is a PROTECTED_TARGET pointer or subobject
    mutable = mutable_box(record%value)
    protected = protected_box(p(1))
    protected = protected_box(record%value)
    copy = value_box(p(1))
    protected_array => p
    protected_array(1:2) => p(1:2)
    protected_scalar => record%value
    protected_scalar => ordinary
    ! The target of a nested ordinary pointer is not part of record's target.
    ordinary => record%ordinary
    ordinary => record%next%value
    record%next%value = 1
    record%next%protected => null()
    ordinary => null(record%protected)
  end subroutine

  subroutine calls(p, record)
    integer, pointer, protected_target :: p(:)
    type(node), pointer, protected_target :: record
    integer, target :: data(2) = [1,2]
    external implicit
    call scalar_in(record%value)
    call array_in(p)
    call protected_pointer_array(p)
    !ERROR: PROTECTED_TARGET pointer or subobject 'p' may not be associated with dummy argument 'arg=' without PROTECTED_TARGET
    call ordinary_pointer_array(p)
    !ERROR: PROTECTED_TARGET pointer or subobject 'record%protected' may not be associated with dummy argument 'arg=' without PROTECTED_TARGET
    call ordinary_pointer_scalar(record%protected)
    ! A pointer dummy also receives the descriptor within record's target.
    !ERROR: PROTECTED_TARGET pointer or subobject 'record%ordinary' may not be associated with dummy argument 'arg=' without PROTECTED_TARGET
    call ordinary_pointer_scalar(record%ordinary)
    call protected_pointer_scalar(record%ordinary)
    call protected_pointer_scalar(record%value)
    call scalar_out(record%ordinary)
    !ERROR: PROTECTED_TARGET pointer or subobject 'record%value' requires an INTENT(IN) nonpointer dummy argument 'arg='
    call scalar_out(record%value)
    !ERROR: PROTECTED_TARGET pointer or subobject 'record%value' requires an INTENT(IN) nonpointer dummy argument 'arg='
    call scalar_inout(record%value)
    !ERROR: PROTECTED_TARGET pointer or subobject 'record%value' requires an INTENT(IN) nonpointer dummy argument 'arg='
    call scalar_default(record%value)
    ! VALUE does not replace the C874 requirement for INTENT(IN).
    !ERROR: PROTECTED_TARGET pointer or subobject 'record%value' requires an INTENT(IN) nonpointer dummy argument 'arg='
    call scalar_value(record%value)
    !ERROR: PROTECTED_TARGET actual argument 'p' requires an explicit interface
    call implicit(p)
    !ERROR: PROTECTED_TARGET actual argument 'record%value' requires an explicit interface
    call implicit(record%value)
    ! Expressions copy values, and subscripts are not the designated object.
    call scalar_default((record%value))
    call implicit(record%value + 1)
    call scalar_default(data(p(1)))
    associate(section => p(1:2))
      call array_in(section)
      !ERROR: PROTECTED_TARGET pointer or subobject 'section' may not be associated with dummy argument 'arg=' without PROTECTED_TARGET
      call ordinary_pointer_array(section)
      !ERROR: PROTECTED_TARGET actual argument 'section' requires an explicit interface
      call implicit(section)
    end associate
  end subroutine

  function get() result(result)
    integer, pointer, protected_target :: result
    integer, target, save :: data = 1
    result => data
  end function

  subroutine function_results
    integer, pointer :: ordinary
    integer, pointer, protected_target :: protected
    type(mutable_box) :: mutable
    type(protected_box) :: box
    external implicit
    protected => get()
    !ERROR: A pointer without PROTECTED_TARGET may not be associated with PROTECTED_TARGET pointer 'result' or its subobject
    ordinary => get()
    !ERROR: Pointer component 'ptr' must have PROTECTED_TARGET when its value is a PROTECTED_TARGET pointer or subobject
    mutable = mutable_box(get())
    box = protected_box(get())
    call scalar_in(get())
    call protected_pointer_scalar(get())
    !ERROR: PROTECTED_TARGET pointer or subobject 'get()' may not be associated with dummy argument 'arg=' without PROTECTED_TARGET
    call ordinary_pointer_scalar(get())
    !ERROR: PROTECTED_TARGET pointer or subobject 'get()' requires an INTENT(IN) nonpointer dummy argument 'arg='
    call scalar_out(get())
    !ERROR: PROTECTED_TARGET actual argument 'get()' requires an explicit interface
    call implicit(get())
    associate(alias => get())
      call scalar_in(alias)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: 'result' is a PROTECTED_TARGET pointer whose target is not definable
      alias = 1
      !ERROR: A pointer without PROTECTED_TARGET may not be associated with PROTECTED_TARGET pointer 'result' or its subobject
      ordinary => alias
    end associate
  end subroutine

  subroutine type_guard(p)
    class(node), pointer, protected_target :: p
    select type(alias => p)
    type is(node)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target is not definable
      alias%value = 1
      !ERROR: PROTECTED_TARGET pointer or subobject 'alias%value' requires an INTENT(IN) nonpointer dummy argument 'arg='
      call scalar_default(alias%value)
    end select
  end subroutine

  subroutine rank_guard(p)
    integer, pointer, protected_target :: p(..)
    select rank(alias => p)
    rank(0)
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target is not definable
      alias = 1
    rank(1)
      !ERROR: PROTECTED_TARGET pointer or subobject 'alias' requires an INTENT(IN) nonpointer dummy argument 'arg='
      call array_default(alias)
    end select
  end subroutine

  subroutine scalar_in(arg)
    integer, intent(in) :: arg
  end subroutine
  subroutine scalar_out(arg)
    integer, intent(out) :: arg
  end subroutine
  subroutine scalar_inout(arg)
    integer, intent(inout) :: arg
  end subroutine
  subroutine scalar_default(arg)
    integer :: arg
  end subroutine
  subroutine scalar_value(arg)
    integer, value :: arg
  end subroutine
  subroutine array_in(arg)
    integer, intent(in) :: arg(:)
  end subroutine
  subroutine array_default(arg)
    integer :: arg(:)
  end subroutine
  subroutine ordinary_pointer_scalar(arg)
    integer, pointer, intent(in) :: arg
  end subroutine
  subroutine ordinary_pointer_array(arg)
    integer, pointer, intent(in) :: arg(:)
  end subroutine
  subroutine protected_pointer_scalar(arg)
    integer, pointer, protected_target, intent(in) :: arg
  end subroutine
  subroutine protected_pointer_array(arg)
    integer, pointer, protected_target, intent(in) :: arg(:)
  end subroutine
end module
