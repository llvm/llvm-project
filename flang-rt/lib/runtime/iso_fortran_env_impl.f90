!===-- module/iso_fortran_env_impl.f90 --=----------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

! This MODULE implements part of the ISO_FORTRAN_ENV module file.
!
! The available kinds depend on compiler options (e.g. -fdisable-integer-16,
! -fdisable-real-16) of the compilation that uses this module, so the
! __builtin_selected_*_kind intrinsics are only folded when the module file is
! read. For the same reason, the kinds cannot be validated with
! digits(int(0,kind=...)) etc.; instead, the selected kind is compared with the
! Flang kind that has the expected number of digits.
!
! There is no object file providing linkable symbols for these constants;
! Flang defines them in the using program from the values folded when reading
! the module file.

module iso_fortran_env_impl
  implicit none
  private

  ! INTEGER types
  intrinsic :: __builtin_selected_int_kind
  integer, parameter, public :: &
    selectedInt8 = __builtin_selected_int_kind(2), &
    selectedInt16 = __builtin_selected_int_kind(4), &
    selectedInt32 = __builtin_selected_int_kind(9), &
    selectedInt64 = __builtin_selected_int_kind(18),&
    selectedInt128 = __builtin_selected_int_kind(38), &
    safeInt8 = merge(selectedInt8, __builtin_selected_int_kind(0), &
                     selectedInt8 >= 0), &
    safeInt16 = merge(selectedInt16, __builtin_selected_int_kind(0), &
                      selectedInt16 >= 0), &
    safeInt32 = merge(selectedInt32, __builtin_selected_int_kind(0), &
                      selectedInt32 >= 0), &
    safeInt64 = merge(selectedInt64, __builtin_selected_int_kind(0), &
                      selectedInt64 >= 0), &
    safeInt128 = merge(selectedInt128, __builtin_selected_int_kind(0), &
                       selectedInt128 >= 0)

  integer, parameter, public :: &
    int8 = merge(selectedInt8, merge(-2, -1, selectedInt8 >= 0), &
                 selectedInt8 == 1), &
    int16 = merge(selectedInt16, merge(-2, -1, selectedInt16 >= 0), &
                  selectedInt16 == 2), &
    int32 = merge(selectedInt32, merge(-2, -1, selectedInt32 >= 0), &
                  selectedInt32 == 4), &
    int64 = merge(selectedInt64, merge(-2, -1, selectedInt64 >= 0), &
                  selectedInt64 == 8), &
    int128 = merge(selectedInt128, merge(-2, -1, selectedInt128 >= 0), &
                   selectedInt128 == 16)

  ! UNSIGNED types
  intrinsic :: __builtin_selected_unsigned_kind
  integer, parameter, public :: &
    selectedUInt8 = __builtin_selected_unsigned_kind(2), &
    selectedUInt16 = __builtin_selected_unsigned_kind(4), &
    selectedUInt32 = __builtin_selected_unsigned_kind(9), &
    selectedUInt64 = __builtin_selected_unsigned_kind(18),&
    selectedUInt128 = __builtin_selected_unsigned_kind(38), &
    safeUInt8 = merge(selectedUInt8, __builtin_selected_unsigned_kind(0), &
                     selectedUInt8 >= 0), &
    safeUInt16 = merge(selectedUInt16, __builtin_selected_unsigned_kind(0), &
                      selectedUInt16 >= 0), &
    safeUInt32 = merge(selectedUInt32, __builtin_selected_unsigned_kind(0), &
                      selectedUInt32 >= 0), &
    safeUInt64 = merge(selectedUInt64, __builtin_selected_unsigned_kind(0), &
                      selectedUInt64 >= 0), &
    safeUInt128 = merge(selectedUInt128, __builtin_selected_unsigned_kind(0), &
                       selectedUInt128 >= 0)

  integer, parameter, public :: &
    uint8 = merge(selectedUInt8, merge(-2, -1, selectedUInt8 >= 0), &
                 selectedUInt8 == 1), &
    uint16 = merge(selectedUInt16, merge(-2, -1, selectedUInt16 >= 0), &
                  selectedUInt16 == 2), &
    uint32 = merge(selectedUInt32, merge(-2, -1, selectedUInt32 >= 0), &
                  selectedUInt32 == 4), &
    uint64 = merge(selectedUInt64, merge(-2, -1, selectedUInt64 >= 0), &
                  selectedUInt64 == 8), &
    uint128 = merge(selectedUInt128, merge(-2, -1, selectedUInt128 >= 0), &
                   selectedUInt128 == 16)

  integer, parameter, dimension(*), public :: __builtin_integer_kinds = [ &
      __builtin_selected_int_kind(0), &
      [(pack([__builtin_selected_int_kind(k)], &
             __builtin_selected_int_kind(k) >= 0 .and. &
               __builtin_selected_int_kind(k) /= &
                 __builtin_selected_int_kind(k-1)), &
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
  intrinsic :: __builtin_selected_real_kind
  integer, parameter, public :: &
    selectedReal16 = __builtin_selected_real_kind(3, 4), &      ! IEEE half
    selectedBfloat16 = __builtin_selected_real_kind(2, 37), &   ! truncated IEEE single
    selectedReal32 = __builtin_selected_real_kind(6, 37), &     ! IEEE single
    selectedReal64 = __builtin_selected_real_kind(15, 307), &   ! IEEE double
    selectedReal80 = __builtin_selected_real_kind(18, 4931), &  ! 80x87 extended
    selectedReal64x2 = __builtin_selected_real_kind(31, 307), & ! "double-double"
    selectedReal128 = __builtin_selected_real_kind(33, 4931), & ! IEEE quad
    safeReal16 = merge(selectedReal16, __builtin_selected_real_kind(0,0), &
                       selectedReal16 >= 0), &
    safeBfloat16 = merge(selectedBfloat16, __builtin_selected_real_kind(0,0), &
                         selectedBfloat16 >= 0), &
    safeReal32 = merge(selectedReal32, __builtin_selected_real_kind(0,0), &
                       selectedReal32 >= 0), &
    safeReal64 = merge(selectedReal64, __builtin_selected_real_kind(0,0), &
                       selectedReal64 >= 0), &
    safeReal80 = merge(selectedReal80, __builtin_selected_real_kind(0,0), &
                       selectedReal80 >= 0), &
    safeReal64x2 = merge(selectedReal64x2, __builtin_selected_real_kind(0,0), &
                         selectedReal64x2 >= 0), &
    safeReal128 = merge(selectedReal128, __builtin_selected_real_kind(0,0), &
                        selectedReal128 >= 0)

  integer, parameter, public :: &
    real16 = merge(selectedReal16, merge(-2, -1, selectedReal16 >= 0), &
                   selectedReal16 == 2), &
    bfloat16 = merge(selectedBfloat16, merge(-2, -1, selectedBfloat16 >= 0), &
                     selectedBfloat16 == 3), &
    real32 = merge(selectedReal32, merge(-2, -1, selectedReal32 >= 0), &
                   selectedReal32 == 4), &
    real64 = merge(selectedReal64, merge(-2, -1, selectedReal64 >= 0), &
                   selectedReal64 == 8), &
    real80 = merge(selectedReal80, merge(-2, -1, selectedReal80 >= 0), &
                   selectedReal80 == 10), &
    real64x2 = merge(-2, -1, selectedReal64x2 >= 0), & ! no such Flang kind
    real128 = merge(selectedReal128, merge(-2, -1, selectedReal128 >= 0), &
                    selectedReal128 == 16)

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
