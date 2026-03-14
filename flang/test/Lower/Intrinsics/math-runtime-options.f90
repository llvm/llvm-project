! RUN: bbc -emit-fir --math-runtime=fast -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="FIR,FAST"
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=fast -mllvm -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="FIR,FAST"
! RUN: bbc -emit-fir --math-runtime=relaxed -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="FIR,RELAXED"
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=relaxed -mllvm -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="FIR,RELAXED"
! RUN: bbc -emit-fir --math-runtime=precise -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="FIR,PRECISE"
! RUN: %flang_fc1 -emit-fir -mllvm -math-runtime=precise -mllvm -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="FIR,PRECISE"

! CHECK-LABEL: cos_testr
subroutine cos_testr(a, b)
  real :: a, b
! FIR: fir.call @fir.cos.contract.f32.f32
  b = cos(a)
end subroutine

! CHECK-LABEL: cos_testd
subroutine cos_testd(a, b)
  real(kind=8) :: a, b
! FIR: fir.call @fir.cos.contract.f64.f64
  b = cos(a)
end subroutine

! FIR: @fir.cos.contract.f32.f32(%arg0: f32) -> f32 attributes
! FAST: math.cos %arg0 fastmath<contract> : f32
! RELAXED: math.cos %arg0 fastmath<contract> : f32
! PRECISE: fir.call @cosf(%arg0) fastmath<contract> : (f32) -> f32
! FIR: @fir.cos.contract.f64.f64(%arg0: f64) -> f64
! FAST: math.cos %arg0 fastmath<contract> : f64
! RELAXED: math.cos %arg0 fastmath<contract> : f64
! PRECISE: fir.call @cos(%arg0) fastmath<contract> : (f64) -> f64
