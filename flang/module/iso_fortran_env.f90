!===-- module/iso_fortran_env.f90 ------------------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

! See Fortran 2023, subclause 16.10.2

#include '../include/flang/Runtime/magic-numbers.h'

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

  use iso_fortran_env_impl, only: &
    selectedInt8, selectedInt16, selectedInt32, selectedInt64, selectedInt128, &
    safeInt8, safeInt16, safeInt32, safeInt64, safeInt128, &
    int8, int16, int32, int64, int128, &
    selectedUInt8, selectedUInt16, selectedUInt32, selectedUInt64, selectedUInt128, &
    safeUInt8, safeUInt16, safeUInt32, safeUInt64, safeUInt128, &
    uint8, uint16, uint32, uint64, uint128, &
    logical8, logical16, logical32, logical64, &
    selectedReal16, selectedBfloat16, selectedReal32, &
    selectedReal64, selectedReal80, selectedReal64x2, &
    selectedReal128, &
    safeReal16, safeBfloat16, safeReal32, &
    safeReal64, safeReal80, safeReal64x2, &
    safeReal128, &
    real16, bfloat16, real32, real64, &
    real80, real64x2, real128, &
    integer_kinds => __builtin_integer_kinds, &
    real_kinds => __builtin_real_kinds, &
    logical_kinds => __builtin_logical_kinds

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

  public :: selectedInt8, selectedInt16, selectedInt32, selectedInt64, selectedInt128, &
    safeInt8, safeInt16, safeInt32, safeInt64, safeInt128, &
    int8, int16, int32, int64, int128

  public :: selectedUInt8, selectedUInt16, selectedUInt32, selectedUInt64, selectedUInt128, &
    safeUInt8, safeUInt16, safeUInt32, safeUInt64, safeUInt128, &
    uint8, uint16, uint32, uint64, uint128

  public :: logical8, logical16, logical32, logical64

  public :: selectedReal16, selectedBfloat16, selectedReal32, &
    selectedReal64, selectedReal80, selectedReal64x2, &
    selectedReal128, &
    safeReal16, safeBfloat16, safeReal32, &
    safeReal64, safeReal80, safeReal64x2, &
    safeReal128, &
    real16, bfloat16, real32, real64, &
    real80, real64x2, real128

  public :: integer_kinds, real_kinds, logical_kinds

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
