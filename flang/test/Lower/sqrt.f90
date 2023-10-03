! RUN: bbc -emit-fir -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CHECK,CMPLX-PRECISE"
! RUN: bbc --math-runtime=precise -emit-fir -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CMPLX-PRECISE"
! RUN: bbc --force-mlir-complex -emit-fir -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CMPLX-FAST"
! RUN: %flang_fc1 -emit-fir -mllvm -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CHECK,CMPLX-PRECISE"
! RUN: %flang_fc1 -fapprox-func -emit-fir -mllvm -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CMPLX-FAST"
! RUN: %flang_fc1 -emit-fir -mllvm --math-runtime=precise -mllvm -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CMPLX-PRECISE"
! RUN: %flang_fc1 -emit-fir -mllvm --force-mlir-complex -mllvm -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CMPLX-FAST"

! CHECK-LABEL: sqrt_testr
subroutine sqrt_testr(a, b)
  real :: a, b
! CHECK: fir.call @fir.sqrt.contract.f32.f32
  b = sqrt(a)
end subroutine

! CHECK-LABEL: sqrt_testd
subroutine sqrt_testd(a, b)
  real(kind=8) :: a, b
! CHECK: fir.call @fir.sqrt.contract.f64.f64
  b = sqrt(a)
end subroutine

! CHECK-LABEL: sqrt_testc
subroutine sqrt_testc(z)
  complex :: z
! CHECK: fir.call @fir.sqrt.contract.z4.z4
  z = sqrt(z)
end subroutine

! CHECK-LABEL: sqrt_testcd
subroutine sqrt_testcd(z)
  complex(kind=8) :: z
! CHECK: fir.call @fir.sqrt.contract.z8.z8
  z = sqrt(z)
end subroutine

! CHECK-LABEL: @fir.sqrt.contract.f32.f32
! CHECK: math.sqrt %{{.*}} : f32

! CHECK-LABEL: @fir.sqrt.contract.f64.f64
! CHECK: math.sqrt %{{.*}} : f64

! CHECK-LABEL: func private @fir.sqrt.contract.z4.z4
! CMPLX-FAST: complex.sqrt %{{.*}} : complex<f32>
! CMPLX-PRECISE: fir.call @csqrtf

! CHECK-LABEL: @fir.sqrt.contract.z8.z8
! CMPLX-FAST: complex.sqrt %{{.*}} : complex<f64>
! CMPLX-PRECISE: fir.call @csqrt
