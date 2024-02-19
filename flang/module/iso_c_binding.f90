!===-- module/iso_c_binding.f90 --------------------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

! See Fortran 2018, clause 18.2

module iso_c_binding

  use __fortran_builtins, only: &
    c_associated => __builtin_c_associated, &
    c_funloc => __builtin_c_funloc, &
    c_funptr => __builtin_c_funptr, &
    c_f_pointer => __builtin_c_f_pointer, &
    c_loc => __builtin_c_loc, &
    c_null_funptr => __builtin_c_null_funptr, &
    c_null_ptr => __builtin_c_null_ptr, &
    c_ptr => __builtin_c_ptr, &
    c_sizeof => sizeof, &
    operator(==), operator(/=)

  implicit none

  ! Set PRIVATE by default to explicitly only export what is meant
  ! to be exported by this MODULE.
  private

  public :: c_associated, c_funloc, c_funptr, c_f_pointer, c_loc, &
    c_null_funptr, c_null_ptr, c_ptr, c_sizeof, &
    operator(==), operator(/=)

  ! Table 18.2 (in clause 18.3.1)
  ! TODO: Specialize (via macros?) for alternative targets
  integer, parameter, public :: &
    c_int8_t = 1, &
    c_int16_t = 2, &
    c_int32_t = 4, &
    c_int64_t = 8, &
    c_int128_t = 16 ! anticipating future addition
  integer, parameter, public :: &
    c_int = c_int32_t, &
    c_short = c_int16_t, &
    c_long = c_int64_t, &
    c_long_long = c_int64_t, &
    c_signed_char = c_int8_t, &
    c_size_t = kind(c_sizeof(1)), &
#if __powerpc__
    c_intmax_t = c_int64_t, &
#else
    c_intmax_t = c_int128_t, &
#endif
    c_intptr_t = c_size_t, &
    c_ptrdiff_t = c_size_t
  integer, parameter, public :: &
    c_int_least8_t = c_int8_t, &
    c_int_fast8_t = c_int8_t, &
    c_int_least16_t = c_int16_t, &
    c_int_fast16_t = c_int16_t, &
    c_int_least32_t = c_int32_t, &
    c_int_fast32_t = c_int32_t, &
    c_int_least64_t = c_int64_t, &
    c_int_fast64_t = c_int64_t, &
    c_int_least128_t = c_int128_t, &
    c_int_fast128_t = c_int128_t

  integer, parameter, public :: &
    c_float = 4, &
    c_double = 8, &
#if __x86_64__
    c_long_double = 10
#else
    c_long_double = 16
#endif

  integer, parameter, public :: &
    c_float_complex = c_float, &
    c_double_complex = c_double, &
    c_long_double_complex = c_long_double

  integer, parameter, public :: c_bool = 1
  integer, parameter, public :: c_char = 1

  ! C characters with special semantics
  character(kind=c_char, len=1), parameter, public :: c_null_char = achar(0)
  character(kind=c_char, len=1), parameter, public :: c_alert = achar(7)
  character(kind=c_char, len=1), parameter, public :: c_backspace = achar(8)
  character(kind=c_char, len=1), parameter, public :: c_form_feed = achar(12)
  character(kind=c_char, len=1), parameter, public :: c_new_line = achar(10)
  character(kind=c_char, len=1), parameter, public :: c_carriage_return = achar(13)
  character(kind=c_char, len=1), parameter, public :: c_horizontal_tab = achar(9)
  character(kind=c_char, len=1), parameter, public :: c_vertical_tab =  achar(11)

  interface c_f_procpointer
    module procedure c_f_procpointer
  end interface
  public :: c_f_procpointer

  ! gfortran extensions
  integer, parameter, public :: &
    c_float128 = 16, &
    c_float128_complex = c_float128

 contains

  subroutine c_f_procpointer(cptr, fptr)
    type(c_funptr), intent(in) :: cptr
    procedure(), pointer, intent(out) :: fptr
    ! TODO: implement
  end subroutine c_f_procpointer

end module iso_c_binding
