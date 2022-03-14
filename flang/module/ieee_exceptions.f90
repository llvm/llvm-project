!===-- module/ieee_exceptions.f90 ------------------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

! See Fortran 2018, clause 17
module ieee_exceptions

  type :: ieee_flag_type ! Fortran 2018, 17.2 & 17.3
    private
    integer(kind=1) :: flag = 0
  end type ieee_flag_type

  type(ieee_flag_type), parameter :: &
    ieee_invalid = ieee_flag_type(1), &
    ieee_overflow = ieee_flag_type(2), &
    ieee_divide_by_zero = ieee_flag_type(4), &
    ieee_underflow = ieee_flag_type(8), &
    ieee_inexact = ieee_flag_type(16), &
    ieee_denorm = ieee_flag_type(32) ! PGI extension

  type(ieee_flag_type), parameter :: &
    ieee_usual(*) = [ &
      ieee_overflow, ieee_divide_by_zero, ieee_invalid ], &
    ieee_all(*) = [ &
      ieee_usual, ieee_underflow, ieee_inexact, ieee_denorm ]

  type :: ieee_modes_type ! Fortran 2018, 17.7
    private
  end type ieee_modes_type

  type :: ieee_status_type ! Fortran 2018, 17.7
    private
  end type ieee_status_type

  private :: ieee_support_flag_2, ieee_support_flag_3, &
      ieee_support_flag_4, ieee_support_flag_8, ieee_support_flag_10, &
      ieee_support_flag_16
  interface ieee_support_flag
    module procedure :: ieee_support_flag, &
      ieee_support_flag_2, ieee_support_flag_3, &
      ieee_support_flag_4, ieee_support_flag_8, ieee_support_flag_10, &
      ieee_support_flag_16
  end interface

 contains
  elemental subroutine ieee_get_flag(flag, flag_value)
    type(ieee_flag_type), intent(in) :: flag
    logical, intent(out) :: flag_value
  end subroutine ieee_get_flag

  elemental subroutine ieee_get_halting_mode(flag, halting)
    type(ieee_flag_type), intent(in) :: flag
    logical, intent(out) :: halting
  end subroutine ieee_get_halting_mode

  subroutine ieee_get_modes(modes)
    type(ieee_modes_type), intent(out) :: modes
  end subroutine ieee_get_modes

  subroutine ieee_get_status(status)
    type(ieee_status_type), intent(out) :: status
  end subroutine ieee_get_status

  pure subroutine ieee_set_flag(flag, flag_value)
    type(ieee_flag_type), intent(in) :: flag
    logical, intent(in) :: flag_value
  end subroutine ieee_set_flag

  pure subroutine ieee_set_halting_mode(flag, halting)
    type(ieee_flag_type), intent(in) :: flag
    logical, intent(in) :: halting
  end subroutine ieee_set_halting_mode

  subroutine ieee_set_modes(modes)
    type(ieee_modes_type), intent(in) :: modes
  end subroutine ieee_set_modes

  subroutine ieee_set_status(status)
    type(ieee_status_type), intent(in) :: status
  end subroutine ieee_set_status

  pure logical function ieee_support_flag(flag)
    type(ieee_flag_type), intent(in) :: flag
    ieee_support_flag = .true.
  end function
  pure logical function ieee_support_flag_2(flag, x)
    type(ieee_flag_type), intent(in) :: flag
    real(kind=2), intent(in) :: x(..)
    ieee_support_flag_2 = .true.
  end function
  pure logical function ieee_support_flag_3(flag, x)
    type(ieee_flag_type), intent(in) :: flag
    real(kind=3), intent(in) :: x(..)
    ieee_support_flag_3 = .true.
  end function
  pure logical function ieee_support_flag_4(flag, x)
    type(ieee_flag_type), intent(in) :: flag
    real(kind=4), intent(in) :: x(..)
    ieee_support_flag_4 = .true.
  end function
  pure logical function ieee_support_flag_8(flag, x)
    type(ieee_flag_type), intent(in) :: flag
    real(kind=8), intent(in) :: x(..)
    ieee_support_flag_8 = .true.
  end function
  pure logical function ieee_support_flag_10(flag, x)
    type(ieee_flag_type), intent(in) :: flag
    real(kind=10), intent(in) :: x(..)
    ieee_support_flag_10 = .true.
  end function
  pure logical function ieee_support_flag_16(flag, x)
    type(ieee_flag_type), intent(in) :: flag
    real(kind=16), intent(in) :: x(..)
    ieee_support_flag_16 = .true.
  end function

  pure logical function ieee_support_halting(flag)
    type(ieee_flag_type), intent(in) :: flag
  end function ieee_support_halting

end module ieee_exceptions
