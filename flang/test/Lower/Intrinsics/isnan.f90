! RUN: bbc -emit-fir %s -o - | FileCheck %s --check-prefixes=CHECK%if target=x86_64{{.*}} %{,CHECK-KIND10%}%if flang-supports-f128-math %{,CHECK-KIND16%}

! CHECK-LABEL: isnan_f32
subroutine isnan_f32(r)
  real :: r
  i = isnan(r)
  ! CHECK: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 3 : i32}> : (f32) -> i1
  ! CHECK: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine isnan_f32

! CHECK-LABEL: ieee_is_nan_f32
subroutine ieee_is_nan_f32(r)
  use ieee_arithmetic
  real :: r
  i = ieee_is_nan(r)
  ! CHECK: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 3 : i32}> : (f32) -> i1
  ! CHECK: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine ieee_is_nan_f32

! CHECK-LABEL: isnan_f64
subroutine isnan_f64(r)
  real(KIND=8) :: r
  i = isnan(r)
  ! CHECK: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 3 : i32}> : (f64) -> i1
  ! CHECK: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine isnan_f64

! CHECK-LABEL: ieee_is_nan_f64
subroutine ieee_is_nan_f64(r)
  use ieee_arithmetic
  real(KIND=8) :: r
  i = ieee_is_nan(r)
  ! CHECK: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 3 : i32}> : (f64) -> i1
  ! CHECK: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine ieee_is_nan_f64

! CHECK-KIND10-LABEL: isnan_f80
subroutine isnan_f80(r)
  integer, parameter :: kind10 = merge(10, 4, selected_real_kind(p=18).eq.10)
  real(KIND=kind10) :: r
  i = isnan(r)
  ! CHECK-KIND10: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 3 : i32}> : (f80) -> i1
  ! CHECK-KIND10: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine isnan_f80

! CHECK-KIND10-LABEL: ieee_is_nan_f80
subroutine ieee_is_nan_f80(r)
  use ieee_arithmetic
  integer, parameter :: kind10 = merge(10, 4, selected_real_kind(p=18).eq.10)
  real(KIND=kind10) :: r
  i = ieee_is_nan(r)
  ! CHECK-KIND10: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 3 : i32}> : (f80) -> i1
  ! CHECK-KIND10: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine ieee_is_nan_f80

! CHECK-KIND16-LABEL: isnan_f128
subroutine isnan_f128(r)
  integer, parameter :: kind16 = merge(16, 4, selected_real_kind(p=33).eq.16)
  real(KIND=kind16) :: r
  i = isnan(r)
  ! CHECK-KIND16: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 3 : i32}> : (f128) -> i1
  ! CHECK-KIND16: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine isnan_f128

! CHECK-KIND16-LABEL: ieee_is_nan_f128
subroutine ieee_is_nan_f128(r)
  use ieee_arithmetic
  integer, parameter :: kind16 = merge(16, 4, selected_real_kind(p=33).eq.16)
  real(KIND=kind16) :: r
  i = ieee_is_nan(r)
  ! CHECK-KIND16: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 3 : i32}> : (f128) -> i1
  ! CHECK-KIND16: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine ieee_is_nan_f128
