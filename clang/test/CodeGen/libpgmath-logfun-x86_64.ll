; REQUIRES: x86-registered-target

; RUN: %clang -target x86_64-unknown-linux-gnu -msse -Ofast -S %s -o - | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define void @fun_(i64* nocapture %z) local_unnamed_addr #0 {
L.entry:
  %0 = bitcast i64* %z to i8*
  %1 = bitcast i64* %z to float*
  %2 = load float, float* %1, align 4
  %3 = fpext float %2 to double
  %4 = fadd double %3, 5.000000e-01
  %5 = tail call double @__pd_log_1(double %4) #1
  %6 = fptrunc double %5 to float
  %7 = tail call float @__ps_exp_1(float %6) #2
  store float %7, float* %1, align 4
  %8 = getelementptr i8, i8* %0, i64 4
  %9 = bitcast i8* %8 to float*
  %10 = load float, float* %9, align 4
  %11 = fpext float %10 to double
  %12 = fadd double %11, 5.000000e-01
  %13 = tail call double @__pd_log_1(double %12) #1
  %14 = fptrunc double %13 to float
  %15 = tail call float @__ps_exp_1(float %14) #2
  store float %15, float* %9, align 4
  %16 = getelementptr i64, i64* %z, i64 1
  %17 = bitcast i64* %16 to float*
  %18 = load float, float* %17, align 4
  %19 = fpext float %18 to double
  %20 = fadd double %19, 5.000000e-01
  %21 = tail call double @__pd_log_1(double %20) #1
  %22 = fptrunc double %21 to float
  %23 = tail call float @__ps_exp_1(float %22) #2
  store float %23, float* %17, align 4
  %24 = getelementptr i8, i8* %0, i64 12
  %25 = bitcast i8* %24 to float*
  %26 = load float, float* %25, align 4
  %27 = fpext float %26 to double
  %28 = fadd double %27, 5.000000e-01
  %29 = tail call double @__pd_log_1(double %28) #1
  %30 = fptrunc double %29 to float
  %31 = tail call float @__ps_exp_1(float %30) #2
  store float %31, float* %25, align 4
  ret void

; CHECK-NOT: __pd_log_1
; CHECK: __pd_log_4
}

; Function Attrs: nounwind readnone willreturn
declare float @__ps_exp_1(float) #0

; Function Attrs: nounwind readnone willreturn
declare double @__pd_log_1(double) #0

attributes #0 = { nounwind readnone willreturn }
