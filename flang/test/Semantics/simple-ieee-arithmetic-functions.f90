! Test that IEEE_ARITHMETIC module functions are classified SIMPLE
! RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s

program test_ieee_arithmetic_simple
  use, intrinsic :: ieee_arithmetic, only: ieee_class, ieee_class_type, &
      ieee_copy_sign, ieee_is_finite, ieee_logb, ieee_max, ieee_max_mag, &
      ieee_max_num, ieee_max_num_mag, ieee_min, ieee_min_mag, ieee_min_num, &
      ieee_min_num_mag, ieee_positive_normal, ieee_quiet_eq, ieee_quiet_ge, &
      ieee_quiet_gt, ieee_quiet_le, ieee_quiet_lt, ieee_quiet_ne, ieee_rem, &
      ieee_rint, ieee_signaling_eq, ieee_signaling_ge, ieee_signaling_gt, &
      ieee_signaling_le, ieee_signaling_lt, ieee_signaling_ne, ieee_signbit, &
      ieee_unordered, ieee_value
  implicit none
  real :: x, y, r
  logical :: l
  type(ieee_class_type) :: c
  x = 1.0
  y = 2.0
  c = ieee_class(x)
  r = ieee_copy_sign(x, y)
  l = ieee_is_finite(x)
  r = ieee_logb(x)
  r = ieee_max(x, y)
  r = ieee_max_mag(x, y)
  r = ieee_max_num(x, y)
  r = ieee_max_num_mag(x, y)
  r = ieee_min(x, y)
  r = ieee_min_mag(x, y)
  r = ieee_min_num(x, y)
  r = ieee_min_num_mag(x, y)
  l = ieee_quiet_eq(x, y)
  l = ieee_quiet_ge(x, y)
  l = ieee_quiet_gt(x, y)
  l = ieee_quiet_le(x, y)
  l = ieee_quiet_lt(x, y)
  l = ieee_quiet_ne(x, y)
  r = ieee_rem(x, y)
  r = ieee_rint(x)
  l = ieee_signaling_eq(x, y)
  l = ieee_signaling_ge(x, y)
  l = ieee_signaling_gt(x, y)
  l = ieee_signaling_le(x, y)
  l = ieee_signaling_lt(x, y)
  l = ieee_signaling_ne(x, y)
  l = ieee_signbit(x)
  l = ieee_unordered(x, y)
  r = ieee_value(x, ieee_positive_normal)
end program
! CHECK: ieee_class_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_copy_sign_a4_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_is_finite_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_logb_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_max_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_max_mag_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_max_num_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_max_num_mag_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_min_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_min_mag_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_min_num_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_min_num_mag_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_quiet_eq_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_quiet_ge_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_quiet_gt_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_quiet_le_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_quiet_lt_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_quiet_ne_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_rem_a4_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_rint_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_signaling_eq_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_signaling_ge_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_signaling_gt_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_signaling_le_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_signaling_lt_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_signaling_ne_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_signbit_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_unordered_a4_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_value_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
