! RUN: bbc -emit-fir -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CHECK,CMPLX-PRECISE"
! RUN: bbc --math-runtime=precise -emit-fir -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CMPLX-PRECISE"
! RUN: %flang_fc1 -emit-fir -mllvm -outline-intrinsics %s -o - | FileCheck %s
! RUN: %flang_fc1 -fapprox-func -emit-fir -mllvm -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CMPLX-FAST"

! CHECK-LABEL: tan_testr
subroutine tan_testr(a, b)
  real :: a, b
! CHECK: fir.call @fir.tan.contract.f32.f32
  b = tan(a)
end subroutine

! CHECK-LABEL: tan_testd
subroutine tan_testd(a, b)
  real(kind=8) :: a, b
! CHECK: fir.call @fir.tan.contract.f64.f64
  b = tan(a)
end subroutine

! CHECK-LABEL: tan_testc
subroutine tan_testc(z)
  complex :: z
! CHECK: fir.call @fir.tan.contract.z32.z32
  z = tan(z)
end subroutine

! CHECK-LABEL: tan_testcd
subroutine tan_testcd(z)
  complex(kind=8) :: z
! CHECK: fir.call @fir.tan.contract.z64.z64
  z = tan(z)
end subroutine

! CHECK-LABEL: atan_testr
subroutine atan_testr(a, b)
  real :: a, b
! CHECK: fir.call @fir.atan.contract.f32.f32
  b = atan(a)
end subroutine

! CHECK-LABEL: atan_testd
subroutine atan_testd(a, b)
  real(kind=8) :: a, b
! CHECK: fir.call @fir.atan.contract.f64.f64
  b = atan(a)
end subroutine

! CHECK-LABEL: atan_testc
subroutine atan_testc(z)
  complex :: z
! CHECK: fir.call @fir.atan.contract.z32.z32
  z = atan(z)
end subroutine

! CHECK-LABEL: atan_testcd
subroutine atan_testcd(z)
  complex(kind=8) :: z
! CHECK: fir.call @fir.atan.contract.z64.z64
  z = atan(z)
end subroutine

! CHECK-LABEL: cos_testr
subroutine cos_testr(a, b)
  real :: a, b
! CHECK: fir.call @fir.cos.contract.f32.f32
  b = cos(a)
end subroutine

! CHECK-LABEL: cos_testd
subroutine cos_testd(a, b)
  real(kind=8) :: a, b
! CHECK: fir.call @fir.cos.contract.f64.f64
  b = cos(a)
end subroutine

! CHECK-LABEL: cos_testc
subroutine cos_testc(z)
  complex :: z
! CHECK: fir.call @fir.cos.contract.z32.z32
  z = cos(z)
end subroutine

! CHECK-LABEL: cos_testcd
subroutine cos_testcd(z)
  complex(kind=8) :: z
! CHECK: fir.call @fir.cos.contract.z64.z64
  z = cos(z)
end subroutine

! CHECK-LABEL: acos_testr
subroutine acos_testr(a, b)
  real :: a, b
! CHECK: fir.call @fir.acos.contract.f32.f32
  b = acos(a)
end subroutine

! CHECK-LABEL: acos_testd
subroutine acos_testd(a, b)
  real(kind=8) :: a, b
! CHECK: fir.call @fir.acos.contract.f64.f64
  b = acos(a)
end subroutine

! CHECK-LABEL: acos_testc
subroutine acos_testc(z)
  complex :: z
! CHECK: fir.call @fir.acos.contract.z32.z32
  z = acos(z)
end subroutine

! CHECK-LABEL: acos_testcd
subroutine acos_testcd(z)
  complex(kind=8) :: z
! CHECK: fir.call @fir.acos.contract.z64.z64
  z = acos(z)
end subroutine

! CHECK-LABEL: cosh_testr
subroutine cosh_testr(a, b)
  real :: a, b
! CHECK: fir.call @fir.cosh.contract.f32.f32
  b = cosh(a)
end subroutine

! CHECK-LABEL: cosh_testd
subroutine cosh_testd(a, b)
  real(kind=8) :: a, b
! CHECK: fir.call @fir.cosh.contract.f64.f64
  b = cosh(a)
end subroutine

! CHECK-LABEL: cosh_testc
subroutine cosh_testc(z)
  complex :: z
! CHECK: fir.call @fir.cosh.contract.z32.z32
  z = cosh(z)
end subroutine

! CHECK-LABEL: cosh_testcd
subroutine cosh_testcd(z)
  complex(kind=8) :: z
! CHECK: fir.call @fir.cosh.contract.z64.z64
  z = cosh(z)
end subroutine

! CHECK-LABEL: sin_testr
subroutine sin_testr(a, b)
  real :: a, b
! CHECK: fir.call @fir.sin.contract.f32.f32
  b = sin(a)
end subroutine

! CHECK-LABEL: sin_testd
subroutine sin_testd(a, b)
  real(kind=8) :: a, b
! CHECK: fir.call @fir.sin.contract.f64.f64
  b = sin(a)
end subroutine

! CHECK-LABEL: sin_testc
subroutine sin_testc(z)
  complex :: z
! CHECK: fir.call @fir.sin.contract.z32.z32
  z = sin(z)
end subroutine

! CHECK-LABEL: sin_testcd
subroutine sin_testcd(z)
  complex(kind=8) :: z
! CHECK: fir.call @fir.sin.contract.z64.z64
  z = sin(z)
end subroutine

! CHECK-LABEL: sinh_testr
subroutine sinh_testr(a, b)
  real :: a, b
! CHECK: fir.call @fir.sinh.contract.f32.f32
  b = sinh(a)
end subroutine

! CHECK-LABEL: sinh_testd
subroutine sinh_testd(a, b)
  real(kind=8) :: a, b
! CHECK: fir.call @fir.sinh.contract.f64.f64
  b = sinh(a)
end subroutine

! CHECK-LABEL: sinh_testc
subroutine sinh_testc(z)
  complex :: z
! CHECK: fir.call @fir.sinh.contract.z32.z32
  z = sinh(z)
end subroutine

! CHECK-LABEL: sinh_testcd
subroutine sinh_testcd(z)
  complex(kind=8) :: z
! CHECK: fir.call @fir.sinh.contract.z64.z64
  z = sinh(z)
end subroutine

! CHECK-LABEL: @fir.tan.contract.f32.f32
! CHECK: math.tan %{{.*}} : f32

! CHECK-LABEL: @fir.tan.contract.f64.f64
! CHECK: math.tan %{{.*}} : f64

! CHECK-LABEL: @fir.tan.contract.z32.z32
! CMPLX-FAST: complex.tan %{{.*}} : complex<f32>
! CMPLX-PRECISE: fir.call @ctanf

! CHECK-LABEL: @fir.tan.contract.z64.z64
! CMPLX-FAST: complex.tan %{{.*}} : complex<f64>
! CMPLX-PRECISE: fir.call @ctan

! CHECK-LABEL: @fir.atan.contract.f32.f32
! CHECK: math.atan %{{.*}} : f32

! CHECK-LABEL: @fir.atan.contract.f64.f64
! CHECK: math.atan %{{.*}} : f64

! CHECK-LABEL: @fir.atan.contract.z32.z32
! CHECK: fir.call @catanf

! CHECK-LABEL: @fir.atan.contract.z64.z64
! CHECK: fir.call @catan

! CHECK-LABEL: @fir.cos.contract.f32.f32
! CHECK: math.cos %{{.*}} : f32

! CHECK-LABEL: @fir.cos.contract.f64.f64
! CHECK: math.cos %{{.*}} : f64

! CHECK-LABEL: @fir.cos.contract.z32.z32
! CMPLX-FAST: complex.cos %{{.*}} : complex<f32>
! CMPLX-PRECISE: fir.call @ccosf

! CHECK-LABEL: @fir.cos.contract.z64.z64
! CMPLX-FAST: complex.cos %{{.*}} : complex<f64>
! CMPLX-PRECISE: fir.call @ccos

! CHECK-LABEL: @fir.acos.contract.f32.f32
! CHECK: math.acos {{.*}} : f32

! CHECK-LABEL: @fir.acos.contract.f64.f64
! CHECK: math.acos {{.*}} : f64

! CHECK-LABEL: @fir.acos.contract.z32.z32
! CHECK: fir.call @cacosf

! CHECK-LABEL: @fir.acos.contract.z64.z64
! CHECK: fir.call @cacos

! CHECK-LABEL: @fir.cosh.contract.f32.f32
! CHECK: math.cosh {{.*}} : f32

! CHECK-LABEL: @fir.cosh.contract.f64.f64
! CHECK: math.cosh {{.*}} : f64

! CHECK-LABEL: @fir.cosh.contract.z32.z32
! CHECK: fir.call @ccoshf

! CHECK-LABEL: @fir.cosh.contract.z64.z64
! CHECK: fir.call @ccosh

! CHECK-LABEL: @fir.sin.contract.f32.f32
! CHECK: math.sin %{{.*}} : f32

! CHECK-LABEL: @fir.sin.contract.f64.f64
! CHECK: math.sin %{{.*}} : f64

! CHECK-LABEL: @fir.sin.contract.z32.z32
! CMPLX-FAST: complex.sin %{{.*}} : complex<f32>
! CMPLX-PRECISE: fir.call @csinf

! CHECK-LABEL: @fir.sin.contract.z64.z64
! CMPLX-FAST: complex.sin %{{.*}} : complex<f64>
! CMPLX-PRECISE: fir.call @csin

! CHECK-LABEL: @fir.sinh.contract.f32.f32
! CHECK: fir.call {{.*}}sinh

! CHECK-LABEL: @fir.sinh.contract.f64.f64
! CHECK: fir.call {{.*}}sinh

! CHECK-LABEL: @fir.sinh.contract.z32.z32
! CHECK: fir.call @csinhf

! CHECK-LABEL: @fir.sinh.contract.z64.z64
! CHECK: fir.call @csinh
