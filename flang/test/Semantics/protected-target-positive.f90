! RUN: %python %S/test_errors.py %s %flang_fc1
! RUN: %flang_fc1 -emit-hlfir -o /dev/null %s

module protected_targets
  integer, protected, target :: protected_value = 1
  integer, target :: mutable_value = 1
  integer, pointer, protected_target :: module_pointer
  type t
    integer, pointer, protected_target :: component
  end type
contains
  function get() result(p)
    integer, pointer, protected_target :: p
    p => protected_value
  end function
  subroutine read_pointer(p)
    integer, pointer, protected_target, intent(in) :: p
    print *, p
  end subroutine

  subroutine keyword_as_name
    integer :: protected_target
    protected_target = 1
  end subroutine
  subroutine implicit_pointer_declaration
    pointer p
    protected_target p
  end subroutine
  subroutine reset_pointer(p)
    integer, pointer, protected_target, intent(out) :: p
    p => protected_value
  end subroutine
  subroutine read_value(x)
    integer, intent(in) :: x
  end subroutine
end module

subroutine test(x, ordinary_in)
  use protected_targets
  integer, target, intent(in) :: x
  integer, pointer, intent(in) :: ordinary_in
  integer, pointer, protected_target :: p, q
  integer, pointer :: ordinary
  integer, pointer :: statement1, statement2
  protected_target statement1
  protected_target :: statement2
  type(t) :: object
  p => x
  q => p
  p => null(p)
  q => null(p)
  ordinary => mutable_value
  p => ordinary
  ! Follow F2028 8.5.16 NOTE 1 despite the draft's conflicting C864.
  p => protected_value
  p => get()
  module_pointer => p
  statement1 => p
  statement2 => statement1
  object = t(p)
  object%component => p
  call read_pointer(p)
  call reset_pointer(p)
  call read_value(p)
  call read_value(p + 1)
  print *, associated(p), associated(p, protected_value), p
  allocate(q, source=42)
  nullify(p, q, module_pointer, object%component)
  mutable_value = 2
  ordinary_in = 2
end subroutine
