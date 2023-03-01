! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: flang-new -fc1 -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: isnan_f32
subroutine isnan_f32(r)
  real :: r
  i = isnan(r)
  ! CHECK: %[[test:.*]] = arith.constant 3 : i32
  ! CHECK: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}, %[[test]]) : (f32, i32) -> i1
  ! CHECK: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine isnan_f32

! CHECK-LABEL: ieee_is_nan_f32
subroutine ieee_is_nan_f32(r)
  use ieee_arithmetic
  real :: r
  i = ieee_is_nan(r)
  ! CHECK: %[[test:.*]] = arith.constant 3 : i32
  ! CHECK: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}, %[[test]]) : (f32, i32) -> i1
  ! CHECK: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine ieee_is_nan_f32

! CHECK-LABEL: isnan_f64
subroutine isnan_f64(r)
  real(KIND=8) :: r
  i = isnan(r)
  ! CHECK: %[[test:.*]] = arith.constant 3 : i32
  ! CHECK: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}, %[[test]]) : (f64, i32) -> i1
  ! CHECK: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine isnan_f64

! CHECK-LABEL: ieee_is_nan_f64
subroutine ieee_is_nan_f64(r)
  use ieee_arithmetic
  real(KIND=8) :: r
  i = ieee_is_nan(r)
  ! CHECK: %[[test:.*]] = arith.constant 3 : i32
  ! CHECK: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}, %[[test]]) : (f64, i32) -> i1
  ! CHECK: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine ieee_is_nan_f64

! CHECK-LABEL: isnan_f80
subroutine isnan_f80(r)
  real(KIND=10) :: r
  i = isnan(r)
  ! CHECK: %[[test:.*]] = arith.constant 3 : i32
  ! CHECK: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}, %[[test]]) : (f80, i32) -> i1
  ! CHECK: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine isnan_f80

! CHECK-LABEL: ieee_is_nan_f80
subroutine ieee_is_nan_f80(r)
  use ieee_arithmetic
  real(KIND=10) :: r
  i = ieee_is_nan(r)
  ! CHECK: %[[test:.*]] = arith.constant 3 : i32
  ! CHECK: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}, %[[test]]) : (f80, i32) -> i1
  ! CHECK: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine ieee_is_nan_f80

! CHECK-LABEL: isnan_f128
subroutine isnan_f128(r)
  real(KIND=16) :: r
  i = isnan(r)
  ! CHECK: %[[test:.*]] = arith.constant 3 : i32
  ! CHECK: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}, %[[test]]) : (f128, i32) -> i1
  ! CHECK: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine isnan_f128

! CHECK-LABEL: ieee_is_nan_f128
subroutine ieee_is_nan_f128(r)
  use ieee_arithmetic
  real(KIND=16) :: r
  i = ieee_is_nan(r)
  ! CHECK: %[[test:.*]] = arith.constant 3 : i32
  ! CHECK: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}, %[[test]]) : (f128, i32) -> i1
  ! CHECK: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine ieee_is_nan_f128
