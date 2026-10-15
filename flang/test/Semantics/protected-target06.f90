! RUN: %python %S/test_errors.py %s %flang_fc1
! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! F2028 15.3.2.2/15.3.3 do not make PROTECTED_TARGET a characteristic.
! Interface compatibility ignores it; call checking retains it.

module characteristics
  integer, target :: target
  interface
    subroutine pointer_interface(x)
      integer, pointer, protected_target :: x
    end subroutine
    subroutine ordinary_pointer_interface(x)
      integer, pointer :: x
    end subroutine
    function function_interface() result(r)
      integer, pointer, protected_target :: r
    end function
    function ordinary_function_interface() result(r)
      integer, pointer :: r
    end function
    module subroutine separate(protected_arg, ordinary_arg, protected_proc, &
        ordinary_proc, protected_result_proc, ordinary_result_proc)
      integer, pointer, protected_target :: protected_arg
      integer, pointer :: ordinary_arg
      procedure(pointer_interface) :: protected_proc
      procedure(ordinary_pointer_interface) :: ordinary_proc
      procedure(function_interface) :: protected_result_proc
      procedure(ordinary_function_interface) :: ordinary_result_proc
    end subroutine
    module function separate_protected_result() result(r)
      integer, pointer, protected_target :: r
    end function
    module function separate_ordinary_result() result(r)
      integer, pointer :: r
    end function
  end interface
  type base
  contains
    procedure, nopass :: sub => protected_subroutine
    procedure, nopass :: func => protected_function
  end type
  type, extends(base) :: derived
  contains
    procedure, nopass :: sub => ordinary_subroutine
    procedure, nopass :: func => ordinary_function
  end type
  type ordinary_base
  contains
    procedure, nopass :: sub => ordinary_subroutine
    procedure, nopass :: func => ordinary_function
  end type
  type, extends(ordinary_base) :: protected_derived
  contains
    procedure, nopass :: sub => protected_subroutine
    procedure, nopass :: func => protected_function
  end type
contains
  subroutine test
    procedure(pointer_interface), pointer :: subprogram
    procedure(ordinary_pointer_interface), pointer :: ordinary_subprogram
    procedure(function_interface), pointer :: function_pointer
    procedure(ordinary_function_interface), pointer :: ordinary_function_pointer
    integer, pointer :: ordinary
    subprogram => protected_subroutine
    function_pointer => protected_function
    subprogram => ordinary_subroutine
    function_pointer => ordinary_function
    ordinary_subprogram => ordinary_subroutine
    ordinary_subprogram => protected_subroutine
    ordinary_function_pointer => ordinary_function
    ordinary_function_pointer => protected_function
    call take_protected_pointer(ordinary_subroutine)
    call take_ordinary_pointer(protected_subroutine)
    call take_protected_function(ordinary_function)
    call take_ordinary_function(protected_function)
    associate(a => protected_function())
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: The target of PROTECTED_TARGET pointer 'r' is not definable
      a = 1
      !ERROR: pointer 'ordinary' must have the PROTECTED_TARGET attribute when associated with the target of PROTECTED_TARGET pointer 'r'
      ordinary => a
    end associate
  end subroutine
  subroutine protected_subroutine(x)
    integer, pointer, protected_target :: x
  end subroutine
  subroutine ordinary_subroutine(x)
    integer, pointer :: x
  end subroutine
  function protected_function() result(r)
    integer, pointer, protected_target :: r
    r => target
  end function
  function ordinary_function() result(r)
    integer, pointer :: r
    r => target
  end function
  subroutine take_protected_pointer(proc)
    procedure(pointer_interface) :: proc
  end subroutine
  subroutine take_ordinary_pointer(proc)
    procedure(ordinary_pointer_interface) :: proc
  end subroutine
  subroutine take_protected_function(proc)
    procedure(function_interface) :: proc
  end subroutine
  subroutine take_ordinary_function(proc)
    procedure(ordinary_function_interface) :: proc
  end subroutine
end module

submodule(characteristics) bodies
contains
  module subroutine separate(protected_arg, ordinary_arg, protected_proc, &
      ordinary_proc, protected_result_proc, ordinary_result_proc)
    integer, pointer :: protected_arg
    integer, pointer, protected_target :: ordinary_arg
    procedure(ordinary_pointer_interface) :: protected_proc
    procedure(pointer_interface) :: ordinary_proc
    procedure(ordinary_function_interface) :: protected_result_proc
    procedure(function_interface) :: ordinary_result_proc
  end subroutine
  module function separate_protected_result() result(r)
    integer, pointer :: r
    r => target
  end function
  module function separate_ordinary_result() result(r)
    integer, pointer, protected_target :: r
    r => target
  end function
end submodule
