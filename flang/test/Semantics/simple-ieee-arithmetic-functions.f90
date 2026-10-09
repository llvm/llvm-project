! Test that IEEE_ARITHMETIC module functions are classified SIMPLE
! RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s

program test_ieee_arithmetic_simple
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  logical :: x
  x = ieee_is_finite(1.0)
end program
! CHECK: ieee_is_finite_a4, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
