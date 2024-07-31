! RUN: %python %S/test_modfile.py %s %flang_fc1
! Test derived type renaming in initializers necessary to avoid
! clashing with local names
module m
  use, intrinsic :: iso_c_binding, only: &
    c_ptr, c_funptr, c_null_ptr, c_null_funptr
  real, private :: __builtin_c_ptr, __builtin_c_funptr
  type mydt
    type(c_funptr) :: component = c_null_funptr
  end type
  type(c_ptr), parameter :: namedConst = c_null_ptr
end

!Expect: m.mod
!module m
!use,intrinsic::__fortran_builtins,only:__fortran_builtins$__builtin_c_ptr=>__builtin_c_ptr
!use,intrinsic::__fortran_builtins,only:__fortran_builtins$__builtin_c_funptr=>__builtin_c_funptr
!use,intrinsic::iso_c_binding,only:c_ptr
!use,intrinsic::iso_c_binding,only:c_funptr
!use,intrinsic::iso_c_binding,only:c_null_ptr
!use,intrinsic::iso_c_binding,only:c_null_funptr
!private::__fortran_builtins$__builtin_c_ptr
!private::__fortran_builtins$__builtin_c_funptr
!real(4),private::__builtin_c_ptr
!real(4),private::__builtin_c_funptr
!type::mydt
!type(c_funptr)::component=__fortran_builtins$__builtin_c_funptr(__address=0_8)
!end type
!type(c_ptr),parameter::namedconst=__fortran_builtins$__builtin_c_ptr(__address=0_8)
!end
