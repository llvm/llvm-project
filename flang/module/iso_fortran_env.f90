!===-- module/iso_fortran_env.f90 ------------------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

! See Fortran 2023, subclause 16.10.2

include '../include/flang/Runtime/magic-numbers.h'

module iso_fortran_env

  use __fortran_builtins, only: &
    event_type => __builtin_event_type, &
    notify_type => __builtin_notify_type, &
    lock_type => __builtin_lock_type, &
    team_type => __builtin_team_type, &
    atomic_int_kind => __builtin_atomic_int_kind, &
    atomic_logical_kind => __builtin_atomic_logical_kind, &
    compiler_options => __builtin_compiler_options, &
    compiler_version => __builtin_compiler_version

  implicit none
  private

  public :: event_type, notify_type, lock_type, team_type, &
    atomic_int_kind, atomic_logical_kind, compiler_options, &
    compiler_version

  integer, parameter :: &
    selectedASCII = selected_char_kind('ASCII'), &
    selectedUCS_2 = selected_char_kind('UCS-2'), &
    selectedUnicode = selected_char_kind('ISO_10646')
  integer, parameter, public :: character_kinds(*) = [ &
    pack([selectedASCII], selectedASCII >= 0), &
    pack([selectedUCS_2], selectedUCS_2 >= 0), &
    pack([selectedUnicode], selectedUnicode >= 0)]

  integer, parameter :: &
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

  integer, parameter, public :: integer_kinds(*) = [ &
    selected_int_kind(0), &
    [(pack([selected_int_kind(k)], &
           selected_int_kind(k) >= 0 .and. &
             selected_int_kind(k) /= selected_int_kind(k-1)), &
      integer :: k=1, 39)]]

  integer, parameter, public :: &
    logical8 = int8, logical16 = int16, logical32 = int32, logical64 = int64
  integer, parameter, public :: logical_kinds(*) = [ &
    pack([logical8],  logical8 >= 0), &
    pack([logical16], logical16 >= 0), &
    pack([logical32], logical32 >= 0), &
    pack([logical64], logical64 >= 0)]

  integer, parameter :: &
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

  integer, parameter, public :: real_kinds(*) = [ &
    pack([real16], real16 >= 0), &
    pack([bfloat16], bfloat16 >= 0), &
    pack([real32], real32 >= 0), &
    pack([real64], real64 >= 0), &
    pack([real80], real80 >= 0), &
    pack([real64x2], real64x2 >= 0), &
    pack([real128], real128 >= 0)]

  integer, parameter, public :: current_team = -1, &
    initial_team = -2, &
    parent_team = -3

  integer, parameter, public :: character_storage_size = 8
  integer, parameter, public :: file_storage_size = 8

  intrinsic :: __builtin_numeric_storage_size
  ! This value depends on any -fdefault-integer-N and -fdefault-real-N
  ! compiler options that are active when the module file is read.
  integer, parameter, public :: numeric_storage_size = &
    __builtin_numeric_storage_size()

  ! From Runtime/magic-numbers.h:
  integer, parameter, public :: &
    output_unit = FORTRAN_DEFAULT_OUTPUT_UNIT, &
    input_unit = FORTRAN_DEFAULT_INPUT_UNIT, &
    error_unit = FORTRAN_ERROR_UNIT, &
    iostat_end = FORTRAN_RUNTIME_IOSTAT_END, &
    iostat_eor = FORTRAN_RUNTIME_IOSTAT_EOR, &
    iostat_inquire_internal_unit = FORTRAN_RUNTIME_IOSTAT_INQUIRE_INTERNAL_UNIT, &
    stat_failed_image = FORTRAN_RUNTIME_STAT_FAILED_IMAGE, &
    stat_locked = FORTRAN_RUNTIME_STAT_LOCKED, &
    stat_locked_other_image = FORTRAN_RUNTIME_STAT_LOCKED_OTHER_IMAGE, &
    stat_stopped_image = FORTRAN_RUNTIME_STAT_STOPPED_IMAGE, &
    stat_unlocked = FORTRAN_RUNTIME_STAT_UNLOCKED, &
    stat_unlocked_failed_image = FORTRAN_RUNTIME_STAT_UNLOCKED_FAILED_IMAGE

end module iso_fortran_env
