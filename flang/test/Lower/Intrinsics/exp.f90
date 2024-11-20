! RUN: bbc -emit-fir -hlfir=false -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CHECK,CMPLX,CMPLX-PRECISE"
! RUN: bbc -emit-fir -hlfir=false --math-runtime=precise -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CMPLX,CMPLX-PRECISE"
! RUN: bbc -emit-fir -hlfir=false --force-mlir-complex -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CMPLX,CMPLX-FAST,CMPLX-MLIR"
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -mllvm -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CHECK,CMPLX,CMPLX-PRECISE"
! RUN: %flang_fc1 -fapprox-func -emit-fir -flang-deprecated-no-hlfir -mllvm -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CMPLX,CMPLX-APPROX"
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -mllvm -outline-intrinsics -mllvm --math-runtime=precise %s -o - | FileCheck %s --check-prefixes="CMPLX,CMPLX-PRECISE"
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -mllvm -outline-intrinsics -mllvm --force-mlir-complex %s -o - | FileCheck %s --check-prefixes="CMPLX,CMPLX-FAST,CMPLX-MLIR"

! CHECK-LABEL: exp_testr
! CHECK-SAME: (%[[AREF:.*]]: !fir.ref<f32> {{.*}}, %[[BREF:.*]]: !fir.ref<f32> {{.*}})
subroutine exp_testr(a, b)
  real :: a, b
! CHECK:  %[[A:.*]] = fir.load %[[AREF:.*]] : !fir.ref<f32>
! CHECK:  %[[RES:.*]] = fir.call @fir.exp.contract.f32.f32(%[[A]]) {{.*}}: (f32) -> f32
! CHECK:  fir.store %[[RES]] to %[[BREF]] : !fir.ref<f32>
  b = exp(a)
end subroutine

! CHECK-LABEL: exp_testd
! CHECK-SAME: (%[[AREF:.*]]: !fir.ref<f64> {{.*}}, %[[BREF:.*]]: !fir.ref<f64> {{.*}})
subroutine exp_testd(a, b)
  real(kind=8) :: a, b
! CHECK:  %[[A:.*]] = fir.load %[[AREF:.*]] : !fir.ref<f64>
! CHECK:  %[[RES:.*]] = fir.call @fir.exp.contract.f64.f64(%[[A]]) {{.*}}: (f64) -> f64
! CHECK:  fir.store %[[RES]] to %[[BREF]] : !fir.ref<f64>
  b = exp(a)
end subroutine

! CHECK-LABEL: exp_testc
! CHECK-SAME: (%[[AREF:.*]]: !fir.ref<complex<f32>> {{.*}}, %[[BREF:.*]]: !fir.ref<complex<f32>> {{.*}})
subroutine exp_testc(a, b)
  complex :: a, b
! CHECK:  %[[A:.*]] = fir.load %[[AREF:.*]] : !fir.ref<complex<f32>>
! CHECK:  %[[RES:.*]] = fir.call @fir.exp.contract.z32.z32(%[[A]]) {{.*}}: (complex<f32>) -> complex<f32>
! CHECK:  fir.store %[[RES]] to %[[BREF]] : !fir.ref<complex<f32>>
  b = exp(a)
end subroutine

! CHECK-LABEL: exp_testcd
! CHECK-SAME: (%[[AREF:.*]]: !fir.ref<complex<f64>> {{.*}}, %[[BREF:.*]]: !fir.ref<complex<f64>> {{.*}})
subroutine exp_testcd(a, b)
  complex(kind=8) :: a, b
! CHECK:  %[[A:.*]] = fir.load %[[AREF:.*]] : !fir.ref<complex<f64>>
! CHECK:  %[[RES:.*]] = fir.call @fir.exp.contract.z64.z64(%[[A]]) {{.*}}: (complex<f64>) -> complex<f64>
! CHECK:  fir.store %[[RES]] to %[[BREF]] : !fir.ref<complex<f64>>
  b = exp(a)
end subroutine

! CHECK-LABEL: private @fir.exp.contract.f32.f32
! CHECK-SAME: (%[[ARG32_OUTLINE:.*]]: f32) -> f32
! CHECK: %[[RESULT32_OUTLINE:.*]] = math.exp %[[ARG32_OUTLINE]] fastmath<contract> : f32
! CHECK: return %[[RESULT32_OUTLINE]] : f32

! CHECK-LABEL: private @fir.exp.contract.f64.f64
! CHECK-SAME: (%[[ARG64_OUTLINE:.*]]: f64) -> f64
! CHECK: %[[RESULT64_OUTLINE:.*]] = math.exp %[[ARG64_OUTLINE]] fastmath<contract> : f64
! CHECK: return %[[RESULT64_OUTLINE]] : f64

! CMPLX-APPROX-LABEL: private @fir.exp.contract_afn.z32.z32
! CMPLX-PRECISE-LABEL: private @fir.exp.contract.z32.z32
! CMPLX-MLIR-LABEL: private @fir.exp.contract.z32.z32
! CMPLX-SAME: (%[[C:.*]]: complex<f32>) -> complex<f32>
! CMPLX-FAST: %[[E:.*]] = complex.exp %[[C]] fastmath<contract> : complex<f32>
! CMPLX-APPROX: %[[E:.*]] = complex.exp %[[C]] fastmath<contract,afn> : complex<f32>
! CMPLX-PRECISE: %[[E:.*]] = fir.call @cexpf(%[[C]]) fastmath<contract> : (complex<f32>) -> complex<f32>
! CMPLX: return %[[E]] : complex<f32>

! CMPLX-APPROX-LABEL: private @fir.exp.contract_afn.z64.z64
! CMPLX-PRECISE-LABEL: private @fir.exp.contract.z64.z64
! CMPLX-MLIR-LABEL: private @fir.exp.contract.z64.z64
! CMPLX-SAME: (%[[C:.*]]: complex<f64>) -> complex<f64>
! CMPLX-FAST: %[[E:.*]] = complex.exp %[[C]] fastmath<contract> : complex<f64>
! CMPLX-APPROX: %[[E:.*]] = complex.exp %[[C]] fastmath<contract,afn> : complex<f64>
! CMPLX-PRECISE: %[[E:.*]] = fir.call @cexp(%[[C]]) fastmath<contract> : (complex<f64>) -> complex<f64>
! CMPLX: return %[[E]] : complex<f64>
