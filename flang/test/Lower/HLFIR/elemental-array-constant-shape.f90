! Test that when a binary elemental operation has one operand with a
! compile-time constant shape and another with only a dynamic (descriptor-based)
! shape, lowering infers the constant shape for the operation result.
! Conforming elemental operands have identical extents, so preferring the
! constant shape is safe and yields a statically shaped, more precise result.

! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! Constant-shape operand on the right ("a" is assumed-shape/dynamic, "x" is
! explicit-shape/constant): the result takes x's constant shape.
subroutine const_right_operand(a)
  real :: a(:)
  real :: x(3)
  a = a + x
end subroutine
! CHECK-LABEL:   func.func @_QPconst_right_operand(
! CHECK:           %[[ELEM:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<3xf32>
! CHECK:           hlfir.assign %[[ELEM]] to %{{.*}} : !hlfir.expr<3xf32>, !fir.box<!fir.array<?xf32>>

! Constant-shape operand on the left: the result still takes the constant shape.
subroutine const_left_operand(a)
  real :: a(:)
  real :: x(3)
  a = x + a
end subroutine
! CHECK-LABEL:   func.func @_QPconst_left_operand(
! CHECK:           hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<3xf32>

! No constant-shape operand: the result stays dynamically shaped.
subroutine both_dynamic(a, b)
  real :: a(:), b(:)
  a = a + b
end subroutine
! CHECK-LABEL:   func.func @_QPboth_dynamic(
! CHECK:           hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32>

! Scalar operand with a constant-shape array: the scalar has no shape, so the
! result takes the array's constant shape.
subroutine scalar_times_const(a, s)
  real :: a(:), s
  real :: x(3)
  a = s * x
end subroutine
! CHECK-LABEL:   func.func @_QPscalar_times_const(
! CHECK:           hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<3xf32>

! Scalar operand with a dynamic array: the result stays dynamically shaped.
subroutine scalar_times_dyn(a, b, s)
  real :: a(:), b(:), s
  a = s * b
end subroutine
! CHECK-LABEL:   func.func @_QPscalar_times_dyn(
! CHECK:           hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32>
