! RUN: %python %S/test_errors.py %s %flang_fc1
! F2028 C872-C874, including generic resolution and pointer function results.

module calls
  integer, target :: target
  type record
    integer :: value
  end type
  interface generic
    module procedure ordinary_array, elemental_read
  end interface
  interface generic_values
    module procedure no_intent_array, elemental_read
  end interface
contains
  subroutine test(p, a, object, flag)
    integer, pointer, protected_target :: p, a(:)
    type(record), pointer, protected_target :: object
    real, pointer, protected_target :: real_pointer
    logical :: flag
    integer, pointer :: ordinary
    external implicit
    !ERROR: PROTECTED_TARGET pointer or subobject 'real_pointer' may be associated only with an INTENT(IN) nonpointer dummy argument 'harvest='
    call random_number(real_pointer)
    !ERROR: PROTECTED_TARGET pointer or subobject 'p' requires an explicit interface
    call implicit(p)
    !ERROR: PROTECTED_TARGET pointer or subobject 'a(1_8:2_8:1_8)' requires an explicit interface
    call implicit(a(1:2))
    !ERROR: dummy argument 'x=' must have the PROTECTED_TARGET attribute when associated with the target of PROTECTED_TARGET pointer 'p'
    call ordinary_pointer(p)
    !ERROR: dummy argument 'x=' must have the PROTECTED_TARGET attribute when associated with the target of PROTECTED_TARGET pointer 'a'
    call ordinary_array(a(1:2))
    !ERROR: dummy argument 'x=' must have the PROTECTED_TARGET attribute when associated with the target of PROTECTED_TARGET pointer 'object'
    call ordinary_pointer(object%value)
    !ERROR: PROTECTED_TARGET pointer or subobject 'p' may be associated only with an INTENT(IN) nonpointer dummy argument 'x='
    call no_intent(p)
    !ERROR: PROTECTED_TARGET pointer or subobject 'a(1_8)' may be associated only with an INTENT(IN) nonpointer dummy argument 'x='
    call no_intent(a(1))
    !ERROR: PROTECTED_TARGET pointer or subobject 'object%value' may be associated only with an INTENT(IN) nonpointer dummy argument 'x='
    call no_intent(object%value)
    !ERROR: PROTECTED_TARGET pointer or subobject 'p' may be associated only with an INTENT(IN) nonpointer dummy argument 'x='
    call write_value(p)
    !ERROR: PROTECTED_TARGET pointer or subobject 'p' may be associated only with an INTENT(IN) nonpointer dummy argument 'x='
    call update_value(p)
    !ERROR: PROTECTED_TARGET pointer or subobject 'p' may be associated only with an INTENT(IN) nonpointer dummy argument 'x='
    call value_no_intent(p)
    !ERROR: dummy argument 'x=' must have the PROTECTED_TARGET attribute when associated with the target of PROTECTED_TARGET pointer 'a'
    call generic(a)
    !ERROR: PROTECTED_TARGET pointer or subobject 'a' may be associated only with an INTENT(IN) nonpointer dummy argument 'x='
    call generic_values(a)
    !ERROR: dummy argument 'x=' must have the PROTECTED_TARGET attribute when associated with the target of PROTECTED_TARGET pointer 'p'
    call ordinary_pointer((flag ? ordinary : p))
    !ERROR: PROTECTED_TARGET pointer or subobject 'p' may be associated only with an INTENT(IN) nonpointer dummy argument 'x='
    call no_intent((flag ? ordinary : p))
    call protected_pointer((flag ? ordinary : p))
    call read_value(p)
    call protected_pointer(p)
    call protected_out(p)
    call read_value(object%value)
    call no_intent((p))
    call no_intent(p + 1)
    call implicit(p + 1)
    call read_value(get())
    call protected_pointer(get())
    !ERROR: pointer 'ordinary' must have the PROTECTED_TARGET attribute when associated with the target of PROTECTED_TARGET pointer 'r'
    ordinary => get()
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: The target of PROTECTED_TARGET pointer 'r' is not definable
    get() = 1
    !ERROR: PROTECTED_TARGET pointer or subobject 'get()' requires an explicit interface
    call implicit(get())
    !ERROR: dummy argument 'x=' must have the PROTECTED_TARGET attribute when associated with the target of PROTECTED_TARGET pointer 'r'
    call ordinary_pointer(get())
    !ERROR: PROTECTED_TARGET pointer or subobject 'get()' may be associated only with an INTENT(IN) nonpointer dummy argument 'x='
    call no_intent(get())
  end subroutine
  function get() result(r)
    integer, pointer, protected_target :: r
    r => target
  end function
  subroutine ordinary_pointer(x)
    integer, pointer, intent(in) :: x
  end subroutine
  subroutine c_pointer(cptr, p)
    use iso_c_binding, only: c_ptr, c_f_pointer
    type(c_ptr) :: cptr
    integer, pointer, protected_target :: p
    !ERROR: dummy argument 'fptr=' must have the PROTECTED_TARGET attribute when associated with the target of PROTECTED_TARGET pointer 'p'
    call c_f_pointer(cptr, p)
  end subroutine
  subroutine ordinary_array(x)
    integer, pointer, intent(in) :: x(:)
  end subroutine
  elemental subroutine elemental_read(x)
    integer, intent(in) :: x
  end subroutine
  subroutine protected_pointer(x)
    integer, pointer, protected_target, intent(in) :: x
  end subroutine
  subroutine protected_out(x)
    integer, pointer, protected_target, intent(out) :: x
    nullify(x)
  end subroutine
  subroutine no_intent(x)
    integer :: x
  end subroutine
  subroutine no_intent_array(x)
    integer :: x(:)
  end subroutine
  subroutine value_no_intent(x)
    integer, value :: x
  end subroutine
  subroutine read_value(x)
    integer, intent(in) :: x
  end subroutine
  subroutine write_value(x)
    integer, intent(out) :: x
  end subroutine
  subroutine update_value(x)
    integer, intent(inout) :: x
  end subroutine
end module
