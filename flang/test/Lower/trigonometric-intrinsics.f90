! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

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

! CHECK-LABEL: @fir.atan.f32.f32
! CHECK: fir.call {{.*}}atan

! CHECK-LABEL: @fir.atan.f64.f64
! CHECK: fir.call {{.*}}atan

! CHECK-LABEL: @fir.atan.z4.z4
! CHECK: fir.call {{.*}}atan

! CHECK-LABEL: @fir.atan.z8.z8
! CHECK: fir.call {{.*}}atan

! CHECK-LABEL: @fir.cos.f32.f32
! CHECK: fir.call {{.*}}cos

! CHECK-LABEL: @fir.cos.f64.f64
! CHECK: fir.call {{.*}}cos

! CHECK-LABEL: @fir.cos.z4.z4
! CHECK: fir.call {{.*}}cos

! CHECK-LABEL: @fir.cos.z8.z8
! CHECK: fir.call {{.*}}cos

! CHECK-LABEL: @fir.cosh.f32.f32
! CHECK: fir.call {{.*}}cosh

! CHECK-LABEL: @fir.cosh.f64.f64
! CHECK: fir.call {{.*}}cosh

! CHECK-LABEL: @fir.cosh.z4.z4
! CHECK: fir.call {{.*}}cosh

! CHECK-LABEL: @fir.cosh.z8.z8
! CHECK: fir.call {{.*}}cosh

! CHECK-LABEL: @fir.sin.f32.f32
! CHECK: fir.call {{.*}}sin

! CHECK-LABEL: @fir.sin.f64.f64
! CHECK: fir.call {{.*}}sin

! CHECK-LABEL: @fir.sin.z4.z4
! CHECK: fir.call {{.*}}sin

! CHECK-LABEL: @fir.sin.z8.z8
! CHECK: fir.call {{.*}}sin

! CHECK-LABEL: @fir.sinh.f32.f32
! CHECK: fir.call {{.*}}sinh

! CHECK-LABEL: @fir.sinh.f64.f64
! CHECK: fir.call {{.*}}sinh

! CHECK-LABEL: @fir.sinh.z4.z4
! CHECK: fir.call {{.*}}sinh

! CHECK-LABEL: @fir.sinh.z8.z8
! CHECK: fir.call {{.*}}sinh
