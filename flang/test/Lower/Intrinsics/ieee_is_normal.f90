! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: flang-new -fc1 -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: ieee_is_normal_f16
subroutine ieee_is_normal_f16(r)
  use ieee_arithmetic
  real(KIND=2) :: r
  i = ieee_is_normal(r)
  ! CHECK: %[[test:.*]] = arith.constant 360 : i32
  ! CHECK: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}, %[[test]]) : (f16, i32) -> i1
  ! CHECK: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine ieee_is_normal_f16

! CHECK-LABEL: ieee_is_normal_bf16
subroutine ieee_is_normal_bf16(r)
  use ieee_arithmetic
  real(KIND=3) :: r
  i = ieee_is_normal(r)
  ! CHECK: %[[test:.*]] = arith.constant 360 : i32
  ! CHECK: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}, %[[test]]) : (bf16, i32) -> i1
  ! CHECK: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine ieee_is_normal_bf16



! CHECK-LABEL: ieee_is_normal_f32
subroutine ieee_is_normal_f32(r)
  use ieee_arithmetic
  real :: r
  i = ieee_is_normal(r)
  ! CHECK: %[[test:.*]] = arith.constant 360 : i32
  ! CHECK: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}, %[[test]]) : (f32, i32) -> i1
  ! CHECK: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine ieee_is_normal_f32

! CHECK-LABEL: ieee_is_normal_f64
subroutine ieee_is_normal_f64(r)
  use ieee_arithmetic
  real(KIND=8) :: r
  i = ieee_is_normal(r)
  ! CHECK: %[[test:.*]] = arith.constant 360 : i32
  ! CHECK: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}, %[[test]]) : (f64, i32) -> i1
  ! CHECK: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine ieee_is_normal_f64

! CHECK-LABEL: ieee_is_normal_f80
subroutine ieee_is_normal_f80(r)
  use ieee_arithmetic
  real(KIND=10) :: r
  i = ieee_is_normal(r)
  ! CHECK: %[[test:.*]] = arith.constant 360 : i32
  ! CHECK: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}, %[[test]]) : (f80, i32) -> i1
  ! CHECK: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine ieee_is_normal_f80

! CHECK-LABEL: ieee_is_normal_f128
subroutine ieee_is_normal_f128(r)
  use ieee_arithmetic
  real(KIND=16) :: r
  i = ieee_is_normal(r)
  ! CHECK: %[[test:.*]] = arith.constant 360 : i32
  ! CHECK: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}, %[[test]]) : (f128, i32) -> i1
  ! CHECK: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine ieee_is_normal_f128
