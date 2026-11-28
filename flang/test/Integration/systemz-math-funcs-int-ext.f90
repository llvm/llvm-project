! Check that MLIR math power helpers get internal linkage on SystemZ.
! The SystemZ ABI verifier skips internal functions.

! REQUIRES: systemz-registered-target

! RUN: %flang_fc1 -triple s390x-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
! Only the -S run reaches the SystemZ ABI verifier, which also checks calls
! and returns with no CHECK line (assertions builds only).
! RUN: %flang_fc1 -triple s390x-unknown-linux-gnu -S %s -o /dev/null

! CHECK-DAG: define internal i8 @__mlir_math_ipowi_i8(
! CHECK-DAG: define internal i16 @__mlir_math_ipowi_i16(
! CHECK-DAG: define internal i32 @__mlir_math_ipowi_i32(
! CHECK-DAG: define internal float @__mlir_math_fpowi_f32_i64(
subroutine s(x1, y1, x2, y2, x4, y4, r, k)
  integer(1) :: x1, y1
  integer(2) :: x2, y2
  integer(4) :: x4, y4
  real(4) :: r
  integer(8) :: k
  x1 = x1 ** y1
  x2 = x2 ** y2
  x4 = x4 ** y4
  r = r ** k
end subroutine
