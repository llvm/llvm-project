! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! Enforce C_F_STRPOINTER semantics (18.2.3.5)

program test
  use iso_c_binding
  type(c_ptr) :: cptr
  character(len=:), pointer :: fstrptr
  character(len=1, kind=c_char), dimension(100), target :: cstrarray
  character(len=10), pointer :: fstrptr_not_deferred
  integer :: nchars

  ! Valid calls
  call c_f_strpointer(cstrarray, fstrptr)  ! ok
  call c_f_strpointer(cstrarray, fstrptr, 50)  ! ok with NCHARS
  call c_f_strpointer(cptr, fstrptr, 100)  ! ok with CSTRPTR form

  ! Error: CSTRPTR form requires NCHARS
  !ERROR: NCHARS= argument is required when CSTRPTR= appears in C_F_STRPOINTER()
  call c_f_strpointer(cptr, fstrptr)

  ! Error: FSTRPTR must have deferred length
  !ERROR: FSTRPTR= argument to C_F_STRPOINTER() must have deferred length
  call c_f_strpointer(cstrarray, fstrptr_not_deferred)

  ! Error: NCHARS must be non-negative
  !ERROR: NCHARS= argument to C_F_STRPOINTER() must be non-negative
  call c_f_strpointer(cstrarray, fstrptr, -5)

  ! Error: NCHARS greater than array size (compile-time check)
  !ERROR: NCHARS=150 is greater than the size of CSTRARRAY=100 in C_F_STRPOINTER()
  call c_f_strpointer(cstrarray, fstrptr, 150)

end program

subroutine test_assumed_size(cstrarray_assumed, fstrptr)
  use iso_c_binding
  character(len=1, kind=c_char), dimension(*), target, intent(in) :: cstrarray_assumed
  character(len=:), pointer :: fstrptr

  ! Error: Assumed-size requires NCHARS
  !ERROR: NCHARS= argument is required when CSTRARRAY= is assumed-size in C_F_STRPOINTER()
  call c_f_strpointer(cstrarray_assumed, fstrptr)

  ! Valid: Assumed-size with NCHARS
  call c_f_strpointer(cstrarray_assumed, fstrptr, 100)
end subroutine
