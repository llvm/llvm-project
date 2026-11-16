! Test that IEEE_ARITHMETIC equality and inequality operators are classified SIMPLE
! RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s

program test_ieee_arithmetic_operators
  use, intrinsic :: ieee_arithmetic, only: ieee_class_type, ieee_round_type, &
      operator(==), operator(/=)
  implicit none
  type(ieee_class_type) :: class1, class2
  type(ieee_round_type) :: round1, round2
  logical :: result
  result = class1 == class2
  result = class1 /= class2
  result = round1 == round2
  result = round1 /= round2
end program

! CHECK: ieee_class_eq, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_class_ne, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_round_eq, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
! CHECK: ieee_round_ne, ELEMENTAL, EXTERNAL, PRIVATE, SIMPLE (Function):
