! RUN: %python %S/test_modfile.py %s %flang_fc1
module m1
  use iso_c_binding, only : c_ptr, c_null_ptr
  private
  public :: t1
  type :: t1
    type(c_ptr) :: c_ptr = c_null_ptr
  end type
end

!Expect: m1.mod
!module m1
!use,intrinsic::__fortran_builtins,only:__builtin_c_ptr
!use,intrinsic::iso_c_binding,only:c_ptr
!use,intrinsic::iso_c_binding,only:c_null_ptr
!private::__builtin_c_ptr
!private::c_ptr
!private::c_null_ptr
!type::t1
!type(c_ptr)::c_ptr=__builtin_c_ptr(__address=0_8)
!end type
!end

module m2
  use m1, only : t1
  private
  public :: t2
  type :: t2
    type(t1) :: x = t1()
  end type
end

!Expect: m2.mod
!module m2
!use,intrinsic::__fortran_builtins,only:__builtin_c_ptr
!use m1,only:t1
!private::__builtin_c_ptr
!private::t1
!type::t2
!type(t1)::x=t1(c_ptr=__builtin_c_ptr(__address=0_8))
!end type
!end
