! RUN: bbc -emit-fir -outline-intrinsics %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -mllvm -outline-intrinsics %s -o - | FileCheck %s

! CHECK-LABEL: sqrt_testr
subroutine sqrt_testr(a, b)
  real :: a, b
! CHECK: fir.call @fir.sqrt.f32.f32
  b = sqrt(a)
end subroutine

! CHECK-LABEL: sqrt_testd
subroutine sqrt_testd(a, b)
  real(kind=8) :: a, b
! CHECK: fir.call @fir.sqrt.f64.f64
  b = sqrt(a)
end subroutine

! CHECK-LABEL: sqrt_testc
subroutine sqrt_testc(z)
  complex :: z
! CHECK: fir.call @fir.sqrt.z4.z4
  z = sqrt(z)
end subroutine

! CHECK-LABEL: sqrt_testcd
subroutine sqrt_testcd(z)
  complex(kind=8) :: z
! CHECK: fir.call @fir.sqrt.z8.z8
  z = sqrt(z)
end subroutine

! CHECK-LABEL: @fir.sqrt.f32.f32
! CHECK: fir.call {{.*}}mth_i_sqrt

! CHECK-LABEL: @fir.sqrt.f64.f64
! CHECK: fir.call {{.*}}mth_i_dsqrt

! CHECK-LABEL: func private @fir.sqrt.z4.z4
! CHECK: fir.call {{.*}}fc_sqrt

! CHECK-LABEL: @fir.sqrt.z8.z8
! CHECK: fir.call {{.*}}fz_sqrt
