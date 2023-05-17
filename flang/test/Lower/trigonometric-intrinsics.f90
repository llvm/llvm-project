! RUN: bbc -emit-fir -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CHECK,CMPLX-FAST"
! RUN: bbc --math-runtime=precise -emit-fir -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CMPLX-PRECISE"
! RUN: %flang_fc1 -emit-fir -mllvm -outline-intrinsics %s -o - | FileCheck %s

! CHECK-LABEL: tan_testr
subroutine tan_testr(a, b)
  real :: a, b
! CHECK: fir.call @fir.tan.f32.f32
  b = tan(a)
end subroutine

! CHECK-LABEL: tan_testd
subroutine tan_testd(a, b)
  real(kind=8) :: a, b
! CHECK: fir.call @fir.tan.f64.f64
  b = tan(a)
end subroutine

! CHECK-LABEL: tan_testc
subroutine tan_testc(z)
  complex :: z
! CHECK: fir.call @fir.tan.z4.z4
  z = tan(z)
end subroutine

! CHECK-LABEL: tan_testcd
subroutine tan_testcd(z)
  complex(kind=8) :: z
! CHECK: fir.call @fir.tan.z8.z8
  z = tan(z)
end subroutine

! CHECK-LABEL: atan_testr
subroutine atan_testr(a, b)
  real :: a, b
! CHECK: fir.call @fir.atan.f32.f32
  b = atan(a)
end subroutine

! CHECK-LABEL: atan_testd
subroutine atan_testd(a, b)
  real(kind=8) :: a, b
! CHECK: fir.call @fir.atan.f64.f64
  b = atan(a)
end subroutine

! CHECK-LABEL: atan_testc
subroutine atan_testc(z)
  complex :: z
! CHECK: fir.call @fir.atan.z4.z4
  z = atan(z)
end subroutine

! CHECK-LABEL: atan_testcd
subroutine atan_testcd(z)
  complex(kind=8) :: z
! CHECK: fir.call @fir.atan.z8.z8
  z = atan(z)
end subroutine

! CHECK-LABEL: cos_testr
subroutine cos_testr(a, b)
  real :: a, b
! CHECK: fir.call @fir.cos.f32.f32
  b = cos(a)
end subroutine

! CHECK-LABEL: cos_testd
subroutine cos_testd(a, b)
  real(kind=8) :: a, b
! CHECK: fir.call @fir.cos.f64.f64
  b = cos(a)
end subroutine

! CHECK-LABEL: cos_testc
subroutine cos_testc(z)
  complex :: z
! CHECK: fir.call @fir.cos.z4.z4
  z = cos(z)
end subroutine

! CHECK-LABEL: cos_testcd
subroutine cos_testcd(z)
  complex(kind=8) :: z
! CHECK: fir.call @fir.cos.z8.z8
  z = cos(z)
end subroutine

! CHECK-LABEL: cosh_testr
subroutine cosh_testr(a, b)
  real :: a, b
! CHECK: fir.call @fir.cosh.f32.f32
  b = cosh(a)
end subroutine

! CHECK-LABEL: cosh_testd
subroutine cosh_testd(a, b)
  real(kind=8) :: a, b
! CHECK: fir.call @fir.cosh.f64.f64
  b = cosh(a)
end subroutine

! CHECK-LABEL: cosh_testc
subroutine cosh_testc(z)
  complex :: z
! CHECK: fir.call @fir.cosh.z4.z4
  z = cosh(z)
end subroutine

! CHECK-LABEL: cosh_testcd
subroutine cosh_testcd(z)
  complex(kind=8) :: z
! CHECK: fir.call @fir.cosh.z8.z8
  z = cosh(z)
end subroutine

! CHECK-LABEL: sin_testr
subroutine sin_testr(a, b)
  real :: a, b
! CHECK: fir.call @fir.sin.f32.f32
  b = sin(a)
end subroutine

! CHECK-LABEL: sin_testd
subroutine sin_testd(a, b)
  real(kind=8) :: a, b
! CHECK: fir.call @fir.sin.f64.f64
  b = sin(a)
end subroutine

! CHECK-LABEL: sin_testc
subroutine sin_testc(z)
  complex :: z
! CHECK: fir.call @fir.sin.z4.z4
  z = sin(z)
end subroutine

! CHECK-LABEL: sin_testcd
subroutine sin_testcd(z)
  complex(kind=8) :: z
! CHECK: fir.call @fir.sin.z8.z8
  z = sin(z)
end subroutine

! CHECK-LABEL: sinh_testr
subroutine sinh_testr(a, b)
  real :: a, b
! CHECK: fir.call @fir.sinh.f32.f32
  b = sinh(a)
end subroutine

! CHECK-LABEL: sinh_testd
subroutine sinh_testd(a, b)
  real(kind=8) :: a, b
! CHECK: fir.call @fir.sinh.f64.f64
  b = sinh(a)
end subroutine

! CHECK-LABEL: sinh_testc
subroutine sinh_testc(z)
  complex :: z
! CHECK: fir.call @fir.sinh.z4.z4
  z = sinh(z)
end subroutine

! CHECK-LABEL: sinh_testcd
subroutine sinh_testcd(z)
  complex(kind=8) :: z
! CHECK: fir.call @fir.sinh.z8.z8
  z = sinh(z)
end subroutine

! CHECK-LABEL: @fir.tan.f32.f32
! CHECK: math.tan %{{.*}} : f32

! CHECK-LABEL: @fir.tan.f64.f64
! CHECK: math.tan %{{.*}} : f64

! CHECK-LABEL: @fir.tan.z4.z4
! CMPLX-FAST: complex.tan %{{.*}} : complex<f32>
! CMPLX-PRECISE: fir.call @ctanf

! CHECK-LABEL: @fir.tan.z8.z8
! CMPLX-FAST: complex.tan %{{.*}} : complex<f64>
! CMPLX-PRECISE: fir.call @ctan

! CHECK-LABEL: @fir.atan.f32.f32
! CHECK: math.atan %{{.*}} : f32

! CHECK-LABEL: @fir.atan.f64.f64
! CHECK: math.atan %{{.*}} : f64

! CHECK-LABEL: @fir.atan.z4.z4
! CHECK: fir.call @catanf

! CHECK-LABEL: @fir.atan.z8.z8
! CHECK: fir.call @catan

! CHECK-LABEL: @fir.cos.f32.f32
! CHECK: math.cos %{{.*}} : f32

! CHECK-LABEL: @fir.cos.f64.f64
! CHECK: math.cos %{{.*}} : f64

! CHECK-LABEL: @fir.cos.z4.z4
! CMPLX-FAST: complex.cos %{{.*}} : complex<f32>
! CMPLX-PRECISE: fir.call @ccosf

! CHECK-LABEL: @fir.cos.z8.z8
! CMPLX-FAST: complex.cos %{{.*}} : complex<f64>
! CMPLX-PRECISE: fir.call @ccos

! CHECK-LABEL: @fir.cosh.f32.f32
! CHECK: fir.call {{.*}}cosh

! CHECK-LABEL: @fir.cosh.f64.f64
! CHECK: fir.call {{.*}}cosh

! CHECK-LABEL: @fir.cosh.z4.z4
! CHECK: fir.call @ccoshf

! CHECK-LABEL: @fir.cosh.z8.z8
! CHECK: fir.call @ccosh

! CHECK-LABEL: @fir.sin.f32.f32
! CHECK: math.sin %{{.*}} : f32

! CHECK-LABEL: @fir.sin.f64.f64
! CHECK: math.sin %{{.*}} : f64

! CHECK-LABEL: @fir.sin.z4.z4
! CMPLX-FAST: complex.sin %{{.*}} : complex<f32>
! CMPLX-PRECISE: fir.call @csinf

! CHECK-LABEL: @fir.sin.z8.z8
! CMPLX-FAST: complex.sin %{{.*}} : complex<f64>
! CMPLX-PRECISE: fir.call @csin

! CHECK-LABEL: @fir.sinh.f32.f32
! CHECK: fir.call {{.*}}sinh

! CHECK-LABEL: @fir.sinh.f64.f64
! CHECK: fir.call {{.*}}sinh

! CHECK-LABEL: @fir.sinh.z4.z4
! CHECK: fir.call @csinhf

! CHECK-LABEL: @fir.sinh.z8.z8
! CHECK: fir.call @csinh
