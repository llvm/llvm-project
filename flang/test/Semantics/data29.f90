! RUN: %python %S/test_errors.py %s %flang_fc1
! Check DATA-style declaration initializers before accumulating their values.

subroutine allocatable_object
  !ERROR: Allocatable 'a' may not have a DATA-style initializer
  integer, allocatable :: a /1/
end subroutine

subroutine dummy_argument(x)
  !ERROR: Dummy argument 'x' may not have a DATA-style initializer
  integer :: x /1/
end subroutine

function function_result()
  !ERROR: Function result 'function_result' may not have a DATA-style initializer
  integer :: function_result /1/
end function

function explicit_result() result(r)
  !ERROR: Function result 'r' may not have a DATA-style initializer
  integer :: r /1/
end function

integer function entry_host()
  !ERROR: Function result 'r' may not have a DATA-style initializer
  integer :: r /1/
  entry_host = 0
  return
  entry entry_result() result(r)
  r = 0
end function

subroutine automatic_character(n)
  integer :: n
  !ERROR: Automatic variable 'c' may not have a DATA-style initializer
  character(len=n) :: c /'a'/
end subroutine

subroutine dummy_allocatable(a)
  !ERROR: Dummy argument 'a' may not have a DATA-style initializer
  integer, allocatable :: a /1/
end subroutine

subroutine two_allocatables
  !ERROR: Allocatable 'a' may not have a DATA-style initializer
  !ERROR: Allocatable 'b' may not have a DATA-style initializer
  integer, allocatable :: a /1/, b /2/
end subroutine

subroutine valid_local
  integer :: sum /1/
end subroutine

subroutine valid_default_component
  type :: t
    integer :: i = 1
  end type
  type(t) :: x /t(2)/
end subroutine

subroutine procedure_pointer_slash
  integer, external, pointer :: p /null()/
end subroutine
