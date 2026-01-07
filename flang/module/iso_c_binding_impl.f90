!===-- module/iso_c_binding.f90 --------------------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

submodule (iso_c_binding) iso_c_binding_impl
  implicit none

contains

  ! F_C_STRING - Convert Fortran string to C null-terminated string
  ! Fortran 2023 standard intrinsic
  module function f_c_string(string, asis) result(res)
    character(kind=c_char, len=*), intent(in) :: string
    logical, optional, intent(in) :: asis
    character(kind=c_char, len=:), allocatable :: res
    logical :: use_asis
    
    use_asis = .false.
    if (present(asis)) use_asis = asis
    
    if (use_asis) then
      res = string // c_null_char
    else
      res = trim(string) // c_null_char
    end if
  end function f_c_string

end submodule iso_c_binding_impl