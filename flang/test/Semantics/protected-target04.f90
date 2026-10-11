! RUN: %python %S/test_errors.py %s %flang_fc1
! J3/26-007r1 C872-C874 apply even when a procedure never modifies its dummy.

module calls
  type t
    integer :: value
    integer, pointer :: separate
  end type
  integer, target :: storage
contains
  subroutine test(p, array, object)
    integer, pointer, protected_target :: p, array(:)
    type(t), pointer, protected_target :: object
    integer, pointer :: ordinary
    call read_pointer(p)
    call read_pointer(ordinary)
    call change_pointer(p)
    call read_value(p)
    call read_value(array(1))
    call read_value(object%value)
    call change_value(object%separate)
    call no_intent((p))
    !ERROR: A PROTECTED_TARGET pointer or a subobject of its target may not be associated with dummy argument 'a=' without PROTECTED_TARGET
    call ordinary_pointer(p)
    !ERROR: A PROTECTED_TARGET pointer or a subobject of its target may not be associated with dummy argument 'a=' without PROTECTED_TARGET
    call ordinary_pointer(array(1))
    !ERROR: A PROTECTED_TARGET pointer or a subobject of its target may not be associated with dummy argument 'a=' without PROTECTED_TARGET
    call ordinary_pointer(object%value)
    !ERROR: A PROTECTED_TARGET pointer or a subobject of its target may be associated only with a nonpointer dummy argument 'a=' that has INTENT(IN)
    call change_value(p)
    !ERROR: A PROTECTED_TARGET pointer or a subobject of its target may be associated only with a nonpointer dummy argument 'a=' that has INTENT(IN)
    call no_intent(p)
    !ERROR: A PROTECTED_TARGET pointer or a subobject of its target may be associated only with a nonpointer dummy argument 'a=' that has INTENT(IN)
    call no_intent(array(1))
    !ERROR: A PROTECTED_TARGET pointer or a subobject of its target may be associated only with a nonpointer dummy argument 'a=' that has INTENT(IN)
    call value_no_intent(p)
    !ERROR: A PROTECTED_TARGET pointer or a subobject of its target requires an explicit interface
    call implicit(p)
    !ERROR: A PROTECTED_TARGET pointer or a subobject of its target requires an explicit interface
    call implicit_array(array(1:2))
    !ERROR: A PROTECTED_TARGET pointer or a subobject of its target requires an explicit interface
    call implicit(object%value)
    call implicit(object%separate)
    call implicit((p))
    associate(alias => array(1))
      call read_value(alias)
      !ERROR: A PROTECTED_TARGET pointer or a subobject of its target may be associated only with a nonpointer dummy argument 'a=' that has INTENT(IN)
      call no_intent(alias)
    end associate
    print *, associated(p), size(array), p + 1
    p => null(p)
    p => null(read_only_result())
    call read_value(read_only_result())
    !ERROR: A PROTECTED_TARGET pointer or a subobject of its target may be associated only with a nonpointer dummy argument 'a=' that has INTENT(IN)
    call no_intent(read_only_result())
    !ERROR: A PROTECTED_TARGET pointer or a subobject of its target requires an explicit interface
    call implicit(read_only_result())
  end subroutine

  subroutine read_pointer(a)
    integer, pointer, protected_target, intent(in) :: a
  end subroutine
  subroutine change_pointer(a)
    integer, pointer, protected_target, intent(out) :: a
    nullify(a)
  end subroutine
  subroutine ordinary_pointer(a)
    integer, pointer, intent(in) :: a
  end subroutine
  subroutine read_value(a)
    integer, intent(in) :: a
  end subroutine
  subroutine change_value(a)
    integer, intent(out) :: a
  end subroutine
  subroutine no_intent(a)
    integer :: a
  end subroutine
  subroutine value_no_intent(a)
    integer, value :: a
  end subroutine
  subroutine select_rank_calls(p)
    integer, pointer, protected_target :: p(..)
    select rank(p)
    rank(0)
      call read_value(p)
      !ERROR: A PROTECTED_TARGET pointer or a subobject of its target may not be associated with dummy argument 'a=' without PROTECTED_TARGET
      call ordinary_pointer(p)
      !ERROR: A PROTECTED_TARGET pointer or a subobject of its target requires an explicit interface
      call implicit(p)
    rank(1)
      call read_value(p(1))
      !ERROR: A PROTECTED_TARGET pointer or a subobject of its target may be associated only with a nonpointer dummy argument 'a=' that has INTENT(IN)
      call no_intent(p(1))
    end select
  end subroutine
  subroutine null_allocatable(p, ordinary)
    type with_allocatable
      integer, allocatable :: a(:)
    end type
    type(with_allocatable), pointer, protected_target :: p
    type(with_allocatable), pointer :: ordinary
    type(with_allocatable) :: object
    object = with_allocatable(null(p%a))
    object = with_allocatable(null(ordinary%a))
  end subroutine
  function read_only_result() result(p)
    integer, pointer, protected_target :: p
    p => storage
  end function
end module
