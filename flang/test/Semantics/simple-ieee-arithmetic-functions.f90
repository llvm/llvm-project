! Test that IEEE_ARITHMETIC module functions are classified SIMPLE.
! RUN: %python %S/test_symbols.py %s %flang_fc1

!DEF: /TEST_IEEE_ARITHMETIC_SIMPLE MainProgram
program TEST_IEEE_ARITHMETIC_SIMPLE
  !DEF: /ieee_arithmetic INTRINSIC (ModFile) Module
  !DEF: /ieee_arithmetic/ieee_is_finite PUBLIC (Function) Generic
  use :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  !DEF: /TEST_IEEE_ARITHMETIC_SIMPLE/x ObjectEntity LOGICAL(4)
  logical x
  !REF: /TEST_IEEE_ARITHMETIC_SIMPLE/x
  !DEF: /TEST_IEEE_ARITHMETIC_SIMPLE/ieee_arithmetic$ieee_arithmetic$ieee_is_finite_a4 ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE Use LOGICAL(4)
  x = ieee_is_finite(1.0)
end program
