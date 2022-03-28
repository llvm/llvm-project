! RUN: bbc -emit-fir --math-runtime=fast %s -o - | FileCheck %s --check-prefixes="FIR,FAST"
! RUN: bbc -emit-fir --math-runtime=relaxed %s -o - | FileCheck %s --check-prefixes="FIR,RELAXED"
! RUN: bbc -emit-fir --math-runtime=precise %s -o - | FileCheck %s --check-prefixes="FIR,PRECISE"
! RUN: bbc -emit-fir --math-runtime=llvm %s -o - | FileCheck %s --check-prefixes="FIR,LLVM"

! CHECK-LABEL: cos_testr
subroutine cos_testr(a, b)
  real :: a, b
! FIR: fir.call @fir.cos.f32.f32
  b = cos(a)
end subroutine

! CHECK-LABEL: cos_testd
subroutine cos_testd(a, b)
  real(kind=8) :: a, b
! FIR: fir.call @fir.cos.f64.f64
  b = cos(a)
end subroutine

! FIR: @fir.cos.f32.f32(%arg0: f32) -> f32 attributes
! FAST: fir.call @__fs_cos_1(%arg0) : (f32) -> f32
! RELAXED: fir.call @__rs_cos_1(%arg0) : (f32) -> f32
! PRECISE: fir.call @__ps_cos_1(%arg0) : (f32) -> f32
! LLVM: fir.call @llvm.cos.f32(%arg0) : (f32) -> f32
! FIR: @fir.cos.f64.f64(%arg0: f64) -> f64
! FAST: fir.call @__fd_cos_1(%arg0) : (f64) -> f64
! RELAXED: fir.call @__rd_cos_1(%arg0) : (f64) -> f64
! PRECISE: fir.call @__pd_cos_1(%arg0) : (f64) -> f64
! LLVM: fir.call @llvm.cos.f64(%arg0) : (f64) -> f64
