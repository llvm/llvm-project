!===-- module/ieee_features.f90 --------------------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

! See Fortran 2018, clause 17.2

module ieee_features

  type :: ieee_features_type
    private
    integer(kind=1) :: feature = 0
  end type ieee_features_type

  type(ieee_features_type), parameter :: &
    ieee_datatype = ieee_features_type(1), &
    ieee_denormal = ieee_features_type(2), &
    ieee_divide = ieee_features_type(3), &
    ieee_halting = ieee_features_type(4), &
    ieee_inexact_flag = ieee_features_type(5), &
    ieee_inf = ieee_features_type(6), &
    ieee_invalid_flag = ieee_features_type(7), &
    ieee_nan = ieee_features_type(8), &
    ieee_rounding = ieee_features_type(9), &
    ieee_sqrt = ieee_features_type(10), &
    ieee_subnormal = ieee_features_type(11), &
    ieee_underflow_flag = ieee_features_type(12)

end module ieee_features
