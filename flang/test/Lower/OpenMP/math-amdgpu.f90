!REQUIRES: amdgpu-registered-target
!RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -emit-llvm -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s

subroutine omp_pow_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__ocml_pow_f32(float {{.*}}, float {{.*}})
  y = x ** x
end subroutine omp_pow_f32

subroutine omp_pow_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__ocml_pow_f64(double {{.*}}, double {{.*}})
  y = x ** x
end subroutine omp_pow_f64

subroutine omp_sin_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__ocml_sin_f32(float {{.*}})
  y = sin(x)
end subroutine omp_sin_f32

subroutine omp_sin_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__ocml_sin_f64(double {{.*}})
  y = sin(x)
end subroutine omp_sin_f64

subroutine omp_abs_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call contract float @llvm.fabs.f32(float {{.*}})
  y = abs(x)
end subroutine omp_abs_f32

subroutine omp_abs_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call contract double @llvm.fabs.f64(double {{.*}})
  y = abs(x)
end subroutine omp_abs_f64

subroutine omp_atan_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__ocml_atan_f32(float {{.*}})
  y = atan(x)
end subroutine omp_atan_f32

subroutine omp_atan_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__ocml_atan_f64(double {{.*}})
  y = atan(x)
end subroutine omp_atan_f64

subroutine omp_atan2_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__ocml_atan2_f32(float {{.*}}, float {{.*}})
  y = atan2(x, x)
end subroutine omp_atan2_f32

subroutine omp_atan2_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__ocml_atan2_f64(double {{.*}}, double {{.*}})
  y = atan2(x ,x)
end subroutine omp_atan2_f64

subroutine omp_cos_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__ocml_cos_f32(float {{.*}})
  y = cos(x)
end subroutine omp_cos_f32

subroutine omp_cos_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__ocml_cos_f64(double {{.*}})
  y = cos(x)
end subroutine omp_cos_f64

subroutine omp_erf_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__ocml_erf_f32(float {{.*}})
  y = erf(x)
end subroutine omp_erf_f32

subroutine omp_erf_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__ocml_erf_f64(double {{.*}})
  y = erf(x)
end subroutine omp_erf_f64

subroutine omp_exp_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call contract float @llvm.exp.f32(float {{.*}})
  y = exp(x)
end subroutine omp_exp_f32

subroutine omp_exp_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__ocml_exp_f64(double {{.*}})
  y = exp(x)
end subroutine omp_exp_f64

subroutine omp_log_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call contract float @llvm.log.f32(float {{.*}})
  y = log(x)
end subroutine omp_log_f32

subroutine omp_log_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__ocml_log_f64(double {{.*}})
  y = log(x)
end subroutine omp_log_f64

subroutine omp_log10_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__ocml_log10_f32(float {{.*}})
  y = log10(x)
end subroutine omp_log10_f32

subroutine omp_log10_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__ocml_log10_f64(double {{.*}})
  y = log10(x)
end subroutine omp_log10_f64

subroutine omp_sqrt_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call contract float @llvm.sqrt.f32(float {{.*}})
  y = sqrt(x)
end subroutine omp_sqrt_f32

subroutine omp_sqrt_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call contract double @llvm.sqrt.f64(double {{.*}})
  y = sqrt(x)
end subroutine omp_sqrt_f64

subroutine omp_tan_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__ocml_tan_f32(float {{.*}})
  y = tan(x)
end subroutine omp_tan_f32

subroutine omp_tan_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__ocml_tan_f64(double {{.*}})
  y = tan(x)
end subroutine omp_tan_f64

subroutine omp_tanh_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__ocml_tanh_f32(float {{.*}})
  y = tanh(x)
end subroutine omp_tanh_f32

subroutine omp_tanh_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__ocml_tanh_f64(double {{.*}})
  y = tanh(x)
end subroutine omp_tanh_f64
