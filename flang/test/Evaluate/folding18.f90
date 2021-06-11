! RUN: %S/test_folding.sh %s %t %flang_fc1
! REQUIRES: shell
! Test implementations of IEEE inquiry functions
module m
  use ieee_arithmetic
  logical, parameter :: test_ieee_support_datatype = ieee_support_datatype() &
    .and. ieee_support_datatype(1.0_2) &
    .and. ieee_support_datatype(1.0_3) &
    .and. ieee_support_datatype(1.0_4) &
    .and. ieee_support_datatype(1.0_8) &
    .and. ieee_support_datatype(1.0_10) &
    .and. ieee_support_datatype(1.0_16)
  logical, parameter :: test_ieee_support_denormal = ieee_support_denormal() &
    .and. ieee_support_denormal(1.0_2) &
    .and. ieee_support_denormal(1.0_3) &
    .and. ieee_support_denormal(1.0_4) &
    .and. ieee_support_denormal(1.0_8) &
    .and. ieee_support_denormal(1.0_10) &
    .and. ieee_support_denormal(1.0_16)
  logical, parameter :: test_ieee_support_divide = ieee_support_divide() &
    .and. ieee_support_divide(1.0_2) &
    .and. ieee_support_divide(1.0_3) &
    .and. ieee_support_divide(1.0_4) &
    .and. ieee_support_divide(1.0_8) &
    .and. ieee_support_divide(1.0_10) &
    .and. ieee_support_divide(1.0_16)
  logical, parameter :: test_ieee_support_inf = ieee_support_inf() &
    .and. ieee_support_inf(1.0_2) &
    .and. ieee_support_inf(1.0_3) &
    .and. ieee_support_inf(1.0_4) &
    .and. ieee_support_inf(1.0_8) &
    .and. ieee_support_inf(1.0_10) &
    .and. ieee_support_inf(1.0_16)
  logical, parameter :: test_ieee_support_io = ieee_support_io() &
    .and. ieee_support_io(1.0_2) &
    .and. ieee_support_io(1.0_3) &
    .and. ieee_support_io(1.0_4) &
    .and. ieee_support_io(1.0_8) &
    .and. ieee_support_io(1.0_10) &
    .and. ieee_support_io(1.0_16)
  logical, parameter :: test_ieee_support_nan = ieee_support_nan() &
    .and. ieee_support_nan(1.0_2) &
    .and. ieee_support_nan(1.0_3) &
    .and. ieee_support_nan(1.0_4) &
    .and. ieee_support_nan(1.0_8) &
    .and. ieee_support_nan(1.0_10) &
    .and. ieee_support_nan(1.0_16)
  logical, parameter :: test_ieee_support_sqrt = ieee_support_sqrt() &
    .and. ieee_support_sqrt(1.0_2) &
    .and. ieee_support_sqrt(1.0_3) &
    .and. ieee_support_sqrt(1.0_4) &
    .and. ieee_support_sqrt(1.0_8) &
    .and. ieee_support_sqrt(1.0_10) &
    .and. ieee_support_sqrt(1.0_16)
  logical, parameter :: test_ieee_support_standard = ieee_support_standard() &
    .and. ieee_support_standard(1.0_2) &
    .and. ieee_support_standard(1.0_3) &
    .and. ieee_support_standard(1.0_4) &
    .and. ieee_support_standard(1.0_8) &
    .and. ieee_support_standard(1.0_10) &
    .and. ieee_support_standard(1.0_16)
  logical, parameter :: test_ieee_support_subnormal = ieee_support_subnormal() &
    .and. ieee_support_subnormal(1.0_2) &
    .and. ieee_support_subnormal(1.0_3) &
    .and. ieee_support_subnormal(1.0_4) &
    .and. ieee_support_subnormal(1.0_8) &
    .and. ieee_support_subnormal(1.0_10) &
    .and. ieee_support_subnormal(1.0_16)
  logical, parameter :: test_ieee_support_underflow_control = ieee_support_underflow_control() &
    .and. ieee_support_underflow_control(1.0_2) &
    .and. ieee_support_underflow_control(1.0_3) &
    .and. ieee_support_underflow_control(1.0_4) &
    .and. ieee_support_underflow_control(1.0_8) &
    .and. ieee_support_underflow_control(1.0_10) &
    .and. ieee_support_underflow_control(1.0_16)
end module
