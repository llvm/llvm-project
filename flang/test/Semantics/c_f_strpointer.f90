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
  call c_f_strpointer(CSTRARRAY=cstrarray, FSTRPTR=fstrptr)  ! ok with CSTRARRAY keyword
  call c_f_strpointer(CSTRARRAY=cstrarray, FSTRPTR=fstrptr, NCHARS=50)  ! ok with all keywords
  call c_f_strpointer(CSTRPTR=cptr, FSTRPTR=fstrptr, NCHARS=50)  ! ok with all keywords

  ! Error: CSTRPTR form requires NCHARS
  !ERROR: NCHARS= argument is required when CSTRPTR= appears in C_F_STRPOINTER()
  call c_f_strpointer(cptr, fstrptr)

  ! Error: CSTRPTR form requires NCHARS (with explicit keyword)
  !ERROR: NCHARS= argument is required when CSTRPTR= appears in C_F_STRPOINTER()
  call c_f_strpointer(CSTRPTR=cptr, FSTRPTR=fstrptr)

  ! Error: Wrong keyword for C_PTR argument
  !ERROR: Keyword CSTRARRAY= cannot be used with a C_PTR argument; use CSTRPTR= instead
  call c_f_strpointer(CSTRARRAY=cptr, FSTRPTR=fstrptr, NCHARS=10)

  ! Error: Wrong keyword for character array argument
  !ERROR: Keyword CSTRPTR= cannot be used with a character array argument; use CSTRARRAY= instead
  call c_f_strpointer(CSTRPTR=cstrarray, FSTRPTR=fstrptr, NCHARS=50)

  ! Error: FSTRPTR must have deferred length
  !ERROR: FSTRPTR= argument to C_F_STRPOINTER() must have deferred length
  call c_f_strpointer(cstrarray, fstrptr_not_deferred)

  ! Error: NCHARS must be non-negative
  !ERROR: NCHARS= argument to C_F_STRPOINTER() must be non-negative
  call c_f_strpointer(cstrarray, fstrptr, -5)

  ! Error: NCHARS greater than array size (compile-time check)
  !ERROR: NCHARS=150 is greater than the size of CSTRARRAY=100 in C_F_STRPOINTER()
  call c_f_strpointer(cstrarray, fstrptr, 150)

  ! Error: Missing required argument FSTRPTR
  !ERROR: Dummy argument 'fstrptr=' is absent and not OPTIONAL
  call c_f_strpointer(cstrarray)

  ! Error: Missing both required arguments
  !ERROR: Dummy argument 'cstr=' is absent and not OPTIONAL
  !ERROR: Dummy argument 'fstrptr=' is absent and not OPTIONAL
  call c_f_strpointer()

  ! Error: Too many arguments
  !ERROR: Too many actual arguments (4 > 3)
  call c_f_strpointer(cstrarray, fstrptr, 50, 999)

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
