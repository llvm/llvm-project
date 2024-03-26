! RUN: %python %S/test_modfile.py %s %flang_fc1
module m
  use iso_c_binding, only: c_ptr, c_null_ptr
  type foo
    type(c_ptr) :: p = c_null_ptr
  end type
  interface foo ! same name as derived type
    procedure f
  end interface
 contains
  type(foo) function f()
  end
end

!Expect: m.mod
!module m
!use,intrinsic::__fortran_builtins,only:__builtin_c_ptr
!use,intrinsic::iso_c_binding,only:c_ptr
!use,intrinsic::iso_c_binding,only:c_null_ptr
!private::__builtin_c_ptr
!type::foo
!type(c_ptr)::p=__builtin_c_ptr(__address=0_8)
!end type
!interface foo
!procedure::f
!end interface
!contains
!function f()
!type(foo)::f
!end
!end
