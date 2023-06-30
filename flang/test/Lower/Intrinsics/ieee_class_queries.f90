! RUN: bbc -emit-fir -o - %s | FileCheck %s

  ! CHECK-LABEL: func @_QQmain
  use ieee_arithmetic, only: ieee_is_finite, ieee_is_nan, ieee_is_negative, &
                             ieee_is_normal
  real(2) :: x2 = -2.0
  real(3) :: x3 = -3.0
  real(4) :: x4 = -4.0
  real(8) :: x8 = -8.0
  real(10) :: x10 = -10.0
  real(16) :: x16 = -16.0

  ! CHECK:     "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 504 : i32}> : (f16) -> i1
  ! CHECK:     "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 60 : i32}> : (f16) -> i1
  ! CHECK:     "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 360 : i32}> : (f16) -> i1
  ! CHECK:     "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 3 : i32}> : (f16) -> i1
  print*, ieee_is_finite(x2), ieee_is_negative(x2), ieee_is_normal(x2), &
          ieee_is_nan(x2)

  ! CHECK:     "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 504 : i32}> : (bf16) -> i1
  ! CHECK:     "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 60 : i32}> : (bf16) -> i1
  ! CHECK:     "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 360 : i32}> : (bf16) -> i1
  ! CHECK:     "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 3 : i32}> : (bf16) -> i1
  print*, ieee_is_finite(x3), ieee_is_negative(x3), ieee_is_normal(x3), &
          ieee_is_nan(x3)

  ! CHECK:     "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 504 : i32}> : (f32) -> i1
  ! CHECK:     "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 60 : i32}> : (f32) -> i1
  ! CHECK:     "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 360 : i32}> : (f32) -> i1
  ! CHECK:     "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 3 : i32}> : (f32) -> i1
  print*, ieee_is_finite(x4), ieee_is_negative(x4), ieee_is_normal(x4), &
          ieee_is_nan(x4)

  ! CHECK:     "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 504 : i32}> : (f64) -> i1
  ! CHECK:     "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 60 : i32}> : (f64) -> i1
  ! CHECK:     "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 360 : i32}> : (f64) -> i1
  ! CHECK:     "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 3 : i32}> : (f64) -> i1
  print*, ieee_is_finite(x8), ieee_is_negative(x8), ieee_is_normal(x8), &
          ieee_is_nan(x8)

  ! CHECK:     "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 504 : i32}> : (f128) -> i1
  ! CHECK:     "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 60 : i32}> : (f128) -> i1
  ! CHECK:     "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 360 : i32}> : (f128) -> i1
  ! CHECK:     "llvm.intr.is.fpclass"(%{{.*}}) <{bit = 3 : i32}> : (f128) -> i1
  print*, ieee_is_finite(x16), ieee_is_negative(x16), ieee_is_normal(x16), &
          ieee_is_nan(x16)

  end
