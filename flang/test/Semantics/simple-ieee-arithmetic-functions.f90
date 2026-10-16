! Test that IEEE_ARITHMETIC module functions are classified SIMPLE
! RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s

program test_ieee_arithmetic_simple
  use, intrinsic :: ieee_arithmetic
  implicit none
end program
! CHECK-DAG: ieee_class_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_copy_sign_a4_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_is_finite_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_logb_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_max_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_max_mag_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_max_num_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_max_num_mag_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_min_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_min_mag_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_min_num_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_min_num_mag_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_quiet_eq_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_quiet_ge_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_quiet_gt_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_quiet_le_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_quiet_lt_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_quiet_ne_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_rem_a4_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_rint_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_signaling_eq_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_signaling_ge_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_signaling_gt_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_signaling_le_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_signaling_lt_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_signaling_ne_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_signbit_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_unordered_a4_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK-DAG: ieee_value_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
