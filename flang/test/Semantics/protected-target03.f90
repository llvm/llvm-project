! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -module-dir %t %t/module.f90
! RUN: %python %S/test_errors.py %t/use.f90 %flang_fc1 -I%t
! RUN: %python %S/test_errors.py %t/characteristics.f90 %flang_fc1

!--- module.f90
module protected_target_module
  type box
    integer, pointer, protected_target :: ptr
  end type
  integer, pointer, protected_target :: p
contains
  subroutine read_pointer(arg)
    integer, pointer, protected_target, intent(in) :: arg
  end subroutine
  subroutine mutable_pointer(arg)
    integer, pointer, intent(in) :: arg
  end subroutine
  subroutine write_value(arg)
    integer, intent(out) :: arg
  end subroutine
  function get() result(result)
    integer, pointer, protected_target :: result
    result => p
  end function
end module

!--- use.f90
subroutine use_module
  use protected_target_module
  integer, pointer :: ordinary
  integer, pointer, protected_target :: protected
  type(box) :: b
  protected => p
  protected => get()
  b = box(p)
  call read_pointer(p)
  call read_pointer(b%ptr)
  call read_pointer(get())
  !ERROR: A pointer without PROTECTED_TARGET may not be associated with PROTECTED_TARGET pointer 'p' or its subobject
  ordinary => p
  !ERROR: A pointer without PROTECTED_TARGET may not be associated with PROTECTED_TARGET pointer 'result' or its subobject
  ordinary => get()
  !ERROR: Left-hand side of assignment is not definable
  !BECAUSE: 'p' is a PROTECTED_TARGET pointer whose target is not definable
  p = 1
  !ERROR: PROTECTED_TARGET pointer or subobject 'p' may not be associated with dummy argument 'arg=' without PROTECTED_TARGET
  call mutable_pointer(p)
  !ERROR: PROTECTED_TARGET pointer or subobject 'b%ptr' may not be associated with dummy argument 'arg=' without PROTECTED_TARGET
  call mutable_pointer(b%ptr)
  !ERROR: PROTECTED_TARGET pointer or subobject 'p' requires an INTENT(IN) nonpointer dummy argument 'arg='
  call write_value(p)
  !ERROR: PROTECTED_TARGET pointer or subobject 'get()' requires an INTENT(IN) nonpointer dummy argument 'arg='
  call write_value(get())
end subroutine

!--- characteristics.f90
subroutine procedure_characteristics
  interface
    subroutine mutable_dummy(arg)
      integer, pointer, intent(in) :: arg
    end subroutine
    subroutine protected_dummy(arg)
      integer, pointer, protected_target, intent(in) :: arg
    end subroutine
    function mutable_result() result(result)
      integer, pointer :: result
    end function
    function protected_result() result(result)
      integer, pointer, protected_target :: result
    end function
  end interface
  procedure(mutable_dummy), pointer :: mutable_proc
  procedure(protected_dummy), pointer :: protected_proc
  procedure(mutable_result), pointer :: mutable_func
  procedure(protected_result), pointer :: protected_func
  mutable_proc => mutable_dummy
  protected_proc => protected_dummy
  mutable_func => mutable_result
  protected_func => protected_result
  !ERROR: Procedure pointer 'mutable_proc' associated with incompatible procedure designator 'protected_dummy': incompatible dummy argument #1: incompatible dummy data object attributes: ProtectedTarget
  mutable_proc => protected_dummy
  !ERROR: Procedure pointer 'protected_proc' associated with incompatible procedure designator 'mutable_dummy': incompatible dummy argument #1: incompatible dummy data object attributes: ProtectedTarget
  protected_proc => mutable_dummy
  !ERROR: Function pointer 'mutable_func' associated with incompatible function designator 'protected_result': function results have incompatible attributes
  mutable_func => protected_result
  !ERROR: Function pointer 'protected_func' associated with incompatible function designator 'mutable_result': function results have incompatible attributes
  protected_func => mutable_result
end subroutine
