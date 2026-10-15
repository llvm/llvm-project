! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! F'2028 15.3.2.2 and 15.3.3 do not make PROTECTED_TARGET a
! procedure characteristic. Interface mismatches are accepted in both
! directions, while call-site constraints still use the declared interface.

module protected_target_interfaces
  abstract interface
    subroutine protected_interface(x)
      integer, pointer, protected_target, intent(in) :: x
    end subroutine
    subroutine ordinary_interface(x)
      integer, pointer, intent(in) :: x
    end subroutine
    function protected_result_interface() result(res)
      integer, pointer, protected_target :: res
    end function
    function ordinary_result_interface() result(res)
      integer, pointer :: res
    end function
  end interface
  interface
    module subroutine drop_dummy(x)
      integer, pointer, protected_target, intent(in) :: x
    end subroutine
    module subroutine add_dummy(x)
      integer, pointer, intent(in) :: x
    end subroutine
    module function drop_result() result(res)
      integer, pointer, protected_target :: res
    end function
    module function add_result() result(res)
      integer, pointer :: res
    end function
    module subroutine drop_procedure(proc)
      procedure(protected_interface) :: proc
    end subroutine
    module subroutine add_procedure(proc)
      procedure(ordinary_interface) :: proc
    end subroutine
  end interface
  type base
  contains
    procedure, nopass :: drop_dummy => protected_sub
    procedure, nopass :: add_dummy => ordinary_sub
    procedure, nopass :: drop_result => protected_fun
    procedure, nopass :: add_result => ordinary_fun
  end type
  type, extends(base) :: extension
  contains
    procedure, nopass :: drop_dummy => ordinary_sub
    procedure, nopass :: add_dummy => protected_sub
    procedure, nopass :: drop_result => ordinary_fun
    procedure, nopass :: add_result => protected_fun
  end type
contains
  subroutine test(p)
    integer, pointer, protected_target :: p
    procedure(protected_interface), pointer :: sub
    procedure(ordinary_interface), pointer :: ordinary
    procedure(protected_result_interface), pointer :: fun
    procedure(ordinary_result_interface), pointer :: ordinary_result
    sub => protected_sub
    sub => ordinary_sub
    ordinary => ordinary_sub
    ordinary => protected_sub
    fun => protected_fun
    fun => ordinary_fun
    ordinary_result => ordinary_fun
    ordinary_result => protected_fun
    call take_protected(ordinary_sub)
    call take_ordinary(protected_sub)
    call take_protected_result(ordinary_fun)
    call take_ordinary_result(protected_fun)
    call sub(p)
    !ERROR: A PROTECTED_TARGET pointer or its subobject may not be associated with dummy argument 'x=' without PROTECTED_TARGET
    call ordinary(p)
  end subroutine
  subroutine protected_sub(x)
    integer, pointer, protected_target, intent(in) :: x
  end subroutine
  subroutine ordinary_sub(x)
    integer, pointer, intent(in) :: x
  end subroutine
  subroutine take_protected(proc)
    procedure(protected_interface) :: proc
  end subroutine
  subroutine take_ordinary(proc)
    procedure(ordinary_interface) :: proc
  end subroutine
  subroutine take_protected_result(proc)
    procedure(protected_result_interface) :: proc
  end subroutine
  subroutine take_ordinary_result(proc)
    procedure(ordinary_result_interface) :: proc
  end subroutine
  function protected_fun() result(res)
    integer, pointer, protected_target :: res
    nullify(res)
  end function
  function ordinary_fun() result(res)
    integer, pointer :: res
    nullify(res)
  end function
end module

submodule (protected_target_interfaces) implementation
contains
  module subroutine drop_dummy(x)
    integer, pointer, intent(in) :: x
  end subroutine
  module subroutine add_dummy(x)
    integer, pointer, protected_target, intent(in) :: x
  end subroutine
  module function drop_result() result(res)
    integer, pointer :: res
    nullify(res)
  end function
  module function add_result() result(res)
    integer, pointer, protected_target :: res
    nullify(res)
  end function
  module subroutine drop_procedure(proc)
    procedure(ordinary_interface) :: proc
  end subroutine
  module subroutine add_procedure(proc)
    procedure(protected_interface) :: proc
  end subroutine
end submodule
