!===-- module/iso_fortran_env_impl.f90 --=--------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

! This MODULE implements part of the ISO_FORTRAN_ENV module file, which
! partially requires linkable symbols for some entities defined
! (e.g., real_kinds).

module iso_fortran_env_impl
  implicit none
  private

  ! INTEGER types
  integer, parameter, public :: &
    selectedInt8 = selected_int_kind(2), &
    selectedInt16 = selected_int_kind(4), &
    selectedInt32 = selected_int_kind(9), &
    selectedInt64 = selected_int_kind(18),&
    selectedInt128 = selected_int_kind(38), &
    safeInt8 = merge(selectedInt8, selected_int_kind(0), &
                     selectedInt8 >= 0), &
    safeInt16 = merge(selectedInt16, selected_int_kind(0), &
                      selectedInt16 >= 0), &
    safeInt32 = merge(selectedInt32, selected_int_kind(0), &
                      selectedInt32 >= 0), &
    safeInt64 = merge(selectedInt64, selected_int_kind(0), &
                      selectedInt64 >= 0), &
    safeInt128 = merge(selectedInt128, selected_int_kind(0), &
                       selectedInt128 >= 0)

  integer, parameter, public :: &
    int8 = merge(selectedInt8, merge(-2, -1, selectedInt8 >= 0), &
                 digits(int(0,kind=safeInt8)) == 7), &
    int16 = merge(selectedInt16, merge(-2, -1, selectedInt16 >= 0), &
                  digits(int(0,kind=safeInt16)) == 15), &
    int32 = merge(selectedInt32, merge(-2, -1, selectedInt32 >= 0), &
                  digits(int(0,kind=safeInt32)) == 31), &
    int64 = merge(selectedInt64, merge(-2, -1, selectedInt64 >= 0), &
                  digits(int(0,kind=safeInt64)) == 63), &
    int128 = merge(selectedInt128, merge(-2, -1, selectedInt128 >= 0), &
                   digits(int(0,kind=safeInt128)) == 127)

  integer, parameter, dimension(*), public :: __builtin_integer_kinds = [ &
      selected_int_kind(0), &
      [(pack([selected_int_kind(k)], &
             selected_int_kind(k) >= 0 .and. &
               selected_int_kind(k) /= selected_int_kind(k-1)), &
        integer :: k=1, 39)]]

  ! LOGICAL TYPES
  integer, parameter, public :: &
    logical8 = int8, logical16 = int16, logical32 = int32, logical64 = int64

  integer, parameter, dimension(*), public :: __builtin_logical_kinds = [ &
      pack([logical8],  logical8 >= 0), &
      pack([logical16], logical16 >= 0), &
      pack([logical32], logical32 >= 0), &
      pack([logical64], logical64 >= 0) &
    ]

  ! REAL types
  integer, parameter, public :: &
    selectedReal16 = selected_real_kind(3, 4), &      ! IEEE half
    selectedBfloat16 = selected_real_kind(2, 37), &   ! truncated IEEE single
    selectedReal32 = selected_real_kind(6, 37), &     ! IEEE single
    selectedReal64 = selected_real_kind(15, 307), &   ! IEEE double
    selectedReal80 = selected_real_kind(18, 4931), &  ! 80x87 extended
    selectedReal64x2 = selected_real_kind(31, 307), & ! "double-double"
    selectedReal128 = selected_real_kind(33, 4931), & ! IEEE quad
    safeReal16 = merge(selectedReal16, selected_real_kind(0,0), &
                       selectedReal16 >= 0), &
    safeBfloat16 = merge(selectedBfloat16, selected_real_kind(0,0), &
                         selectedBfloat16 >= 0), &
    safeReal32 = merge(selectedReal32, selected_real_kind(0,0), &
                       selectedReal32 >= 0), &
    safeReal64 = merge(selectedReal64, selected_real_kind(0,0), &
                       selectedReal64 >= 0), &
    safeReal80 = merge(selectedReal80, selected_real_kind(0,0), &
                       selectedReal80 >= 0), &
    safeReal64x2 = merge(selectedReal64x2, selected_real_kind(0,0), &
                         selectedReal64x2 >= 0), &
    safeReal128 = merge(selectedReal128, selected_real_kind(0,0), &
                        selectedReal128 >= 0)

  integer, parameter, public :: &
    real16 = merge(selectedReal16, merge(-2, -1, selectedReal16 >= 0), &
                   digits(real(0,kind=safeReal16)) == 11), &
    bfloat16 = merge(selectedBfloat16, merge(-2, -1, selectedBfloat16 >= 0), &
                     digits(real(0,kind=safeBfloat16)) == 8), &
    real32 = merge(selectedReal32, merge(-2, -1, selectedReal32 >= 0), &
                   digits(real(0,kind=safeReal32)) == 24), &
    real64 = merge(selectedReal64, merge(-2, -1, selectedReal64 >= 0), &
                   digits(real(0,kind=safeReal64)) == 53), &
    real80 = merge(selectedReal80, merge(-2, -1, selectedReal80 >= 0), &
                   digits(real(0,kind=safeReal80)) == 64), &
    real64x2 = merge(selectedReal64x2, merge(-2, -1, selectedReal64x2 >= 0), &
                     digits(real(0,kind=safeReal64x2)) == 106), &
    real128 = merge(selectedReal128, merge(-2, -1, selectedReal128 >= 0), &
                    digits(real(0,kind=safeReal128)) == 113)

  integer, parameter, dimension(*), public :: __builtin_real_kinds = [ &
      pack([real16], real16 >= 0), &
      pack([bfloat16], bfloat16 >= 0), &
      pack([real32], real32 >= 0), &
      pack([real64], real64 >= 0), &
      pack([real80], real80 >= 0), &
      pack([real64x2], real64x2 >= 0), &
      pack([real128], real128 >= 0) &
    ]
end module iso_fortran_env_impl
