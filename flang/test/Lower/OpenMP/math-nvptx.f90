!REQUIRES: nvptx-registered-target
!RUN: %flang_fc1 -triple nvptx64-nvidia-cuda -emit-llvm -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s

subroutine omp_pow_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__nv_powf(float {{.*}}, float {{.*}})
  y = x ** x
end subroutine omp_pow_f32

subroutine omp_pow_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__nv_pow(double {{.*}}, double {{.*}})
  y = x ** x
end subroutine omp_pow_f64

subroutine omp_sin_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__nv_sinf(float {{.*}})
  y = sin(x)
end subroutine omp_sin_f32

subroutine omp_sin_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__nv_sin(double {{.*}})
  y = sin(x)
end subroutine omp_sin_f64

subroutine omp_abs_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__nv_fabsf(float {{.*}})
  y = abs(x)
end subroutine omp_abs_f32

subroutine omp_abs_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__nv_fabs(double {{.*}})
  y = abs(x)
end subroutine omp_abs_f64

subroutine omp_atan_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__nv_atanf(float {{.*}})
  y = atan(x)
end subroutine omp_atan_f32

subroutine omp_atan_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__nv_atan(double {{.*}})
  y = atan(x)
end subroutine omp_atan_f64

subroutine omp_atanh_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__nv_atanhf(float {{.*}})
  y = atanh(x)
end subroutine omp_atanh_f32

subroutine omp_atanh_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__nv_atanh(double {{.*}})
  y = atanh(x)
end subroutine omp_atanh_f64

subroutine omp_atan2_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__nv_atan2f(float {{.*}}, float {{.*}})
  y = atan2(x, x)
end subroutine omp_atan2_f32

subroutine omp_atan2_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__nv_atan2(double {{.*}}, double {{.*}})
  y = atan2(x, x)
end subroutine omp_atan2_f64

subroutine omp_cos_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__nv_cosf(float {{.*}})
  y = cos(x)
end subroutine omp_cos_f32

subroutine omp_cos_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__nv_cos(double {{.*}})
  y = cos(x)
end subroutine omp_cos_f64

subroutine omp_erf_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__nv_erff(float {{.*}})
  y = erf(x)
end subroutine omp_erf_f32

subroutine omp_erf_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__nv_erf(double {{.*}})
  y = erf(x)
end subroutine omp_erf_f64

subroutine omp_erfc_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__nv_erfcf(float {{.*}})
  y = erfc(x)
end subroutine omp_erfc_f32

subroutine omp_erfc_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__nv_erfc(double {{.*}})
  y = erfc(x)
end subroutine omp_erfc_f64

subroutine omp_exp_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__nv_expf(float {{.*}})
  y = exp(x)
end subroutine omp_exp_f32

subroutine omp_exp_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__nv_exp(double {{.*}})
  y = exp(x)
end subroutine omp_exp_f64

subroutine omp_log_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__nv_logf(float {{.*}})
  y = log(x)
end subroutine omp_log_f32

subroutine omp_log_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__nv_log(double {{.*}})
  y = log(x)
end subroutine omp_log_f64

subroutine omp_log10_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__nv_log10f(float {{.*}})
  y = log10(x)
end subroutine omp_log10_f32

subroutine omp_log10_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__nv_log10(double {{.*}})
  y = log10(x)
end subroutine omp_log10_f64

subroutine omp_sqrt_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__nv_sqrtf(float {{.*}})
  y = sqrt(x)
end subroutine omp_sqrt_f32

subroutine omp_sqrt_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__nv_sqrt(double {{.*}})
  y = sqrt(x)
end subroutine omp_sqrt_f64

subroutine omp_tan_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__nv_tanf(float {{.*}})
  y = tan(x)
end subroutine omp_tan_f32

subroutine omp_tan_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__nv_tan(double {{.*}})
  y = tan(x)
end subroutine omp_tan_f64

subroutine omp_tanh_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__nv_tanhf(float {{.*}})
  y = tanh(x)
end subroutine omp_tanh_f32

subroutine omp_tanh_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__nv_tanh(double {{.*}})
  y = tanh(x)
end subroutine omp_tanh_f64

subroutine omp_acos_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__nv_acosf(float {{.*}})
  y = acos(x)
end subroutine omp_acos_f32

subroutine omp_acos_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__nv_acos(double {{.*}})
  y = acos(x)
end subroutine omp_acos_f64

subroutine omp_acosh_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__nv_acoshf(float {{.*}})
  y = acosh(x)
end subroutine omp_acosh_f32

subroutine omp_acosh_f64(x, y)
!$omp declare target
    real(8) :: x, y
!CHECK: call double @__nv_acosh(double {{.*}})
    y = acosh(x)
end subroutine omp_acosh_f64

subroutine omp_asin_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__nv_asinf(float {{.*}})
  y = asin(x)
end subroutine omp_asin_f32

subroutine omp_asin_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__nv_asin(double {{.*}})
  y = asin(x)
end subroutine omp_asin_f64

subroutine omp_asinh_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__nv_asinhf(float {{.*}})
  y = asinh(x)
end subroutine omp_asinh_f32

subroutine omp_asinh_f64(x, y)
!$omp declare target
    real(8) :: x, y
!CHECK: call double @__nv_asinh(double {{.*}})
    y = asinh(x)
end subroutine omp_asinh_f64

subroutine omp_cosh_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__nv_coshf(float {{.*}})
  y = cosh(x)
end subroutine omp_cosh_f32

subroutine omp_cosh_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__nv_cosh(double {{.*}})
  y = cosh(x)
end subroutine omp_cosh_f64

subroutine omp_sinh_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__nv_sinhf(float {{.*}})
  y = sinh(x)
end subroutine omp_sinh_f32

subroutine omp_sinh_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__nv_sinh(double {{.*}})
  y = sinh(x)
end subroutine omp_sinh_f64

subroutine omp_ceiling_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__nv_ceilf(float {{.*}})
  y = ceiling(x)
end subroutine omp_ceiling_f32

subroutine omp_ceiling_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__nv_ceil(double {{.*}})
  y = ceiling(x)
end subroutine omp_ceiling_f64

subroutine omp_floor_f32(x, y)
!$omp declare target
  real :: x, y
!CHECK: call float @__nv_floorf(float {{.*}})
  y = floor(x)
end subroutine omp_floor_f32    

subroutine omp_floor_f64(x, y)
!$omp declare target
  real(8) :: x, y
!CHECK: call double @__nv_floor(double {{.*}})
  y = floor(x)
end subroutine omp_floor_f64

subroutine omp_sign_f32(x, y)
!$omp declare target
    real :: x, y, z
!CHECK: call float @__nv_copysignf(float {{.*}}, float {{.*}})
    y = sign(x, z)
end subroutine omp_sign_f32

subroutine omp_sign_f64(x, y)
!$omp declare target
    real(8) :: x, y, z
!CHECK: call double @__nv_copysign(double {{.*}}, double {{.*.}})
    y = sign(x, z)
end subroutine omp_sign_f64
