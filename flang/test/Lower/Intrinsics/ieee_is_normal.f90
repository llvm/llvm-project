! RUN: bbc -emit-fir %s -o - | FileCheck %s --check-prefixes=CHECK%if target=x86_64{{.*}} %{,CHECK-KIND10%}%if flang-supports-f128-math %{,CHECK-KIND16%}

! CHECK-LABEL: ieee_is_normal_f16
subroutine ieee_is_normal_f16(r)
  use ieee_arithmetic
  real(KIND=2) :: r
  i = ieee_is_normal(r)
  ! CHECK: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 360 : i32}> : (f16) -> i1
  ! CHECK: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine ieee_is_normal_f16

! CHECK-LABEL: ieee_is_normal_bf16
subroutine ieee_is_normal_bf16(r)
  use ieee_arithmetic
  real(KIND=3) :: r
  i = ieee_is_normal(r)
  ! CHECK: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 360 : i32}> : (bf16) -> i1
  ! CHECK: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine ieee_is_normal_bf16



! CHECK-LABEL: ieee_is_normal_f32
subroutine ieee_is_normal_f32(r)
  use ieee_arithmetic
  real :: r
  i = ieee_is_normal(r)
  ! CHECK: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 360 : i32}> : (f32) -> i1
  ! CHECK: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine ieee_is_normal_f32

! CHECK-LABEL: ieee_is_normal_f64
subroutine ieee_is_normal_f64(r)
  use ieee_arithmetic
  real(KIND=8) :: r
  i = ieee_is_normal(r)
  ! CHECK: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 360 : i32}> : (f64) -> i1
  ! CHECK: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine ieee_is_normal_f64

! CHECK-KIND10-LABEL: ieee_is_normal_f80
subroutine ieee_is_normal_f80(r)
  use ieee_arithmetic
  integer, parameter :: kind10 = merge(10, 4, selected_real_kind(p=18).eq.10)
  real(KIND=kind10) :: r
  i = ieee_is_normal(r)
  ! CHECK-KIND10: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 360 : i32}> : (f80) -> i1
  ! CHECK-KIND10: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine ieee_is_normal_f80

! CHECK-KIND16-LABEL: ieee_is_normal_f128
subroutine ieee_is_normal_f128(r)
  use ieee_arithmetic
  integer, parameter :: kind16 = merge(16, 4, selected_real_kind(p=33).eq.16)
  real(KIND=kind16) :: r
  i = ieee_is_normal(r)
  ! CHECK-KIND16: %[[l:.*]] = "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 360 : i32}> : (f128) -> i1
  ! CHECK-KIND16: fir.convert %[[l]] : (i1) -> !fir.logical<4>
end subroutine ieee_is_normal_f128
