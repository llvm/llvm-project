! RUN: bbc -emit-fir -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CHECK,CMPLX,CMPLX-PRECISE"
! RUN: bbc -emit-fir --math-runtime=precise -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CMPLX,CMPLX-PRECISE"
! RUN: bbc -emit-fir --force-mlir-complex -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CMPLX,CMPLX-FAST,CMPLX-MLIR"
! RUN: %flang_fc1 -emit-fir -mllvm -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CHECK,CMPLX,CMPLX-PRECISE"
! RUN: %flang_fc1 -fapprox-func -emit-fir -mllvm -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CMPLX,CMPLX-FAST,CMPLX-APPROX"
! RUN: %flang_fc1 -emit-fir -mllvm -outline-intrinsics -mllvm --math-runtime=precise %s -o - | FileCheck %s --check-prefixes="CMPLX,CMPLX-PRECISE"
! RUN: %flang_fc1 -emit-fir -mllvm -outline-intrinsics -mllvm --force-mlir-complex %s -o - | FileCheck %s --check-prefixes="CMPLX,CMPLX-FAST,CMPLX-MLIR"

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
! CHECK-SAME: (%[[AREF:.*]]: !fir.ref<!fir.complex<4>> {{.*}}, %[[BREF:.*]]: !fir.ref<!fir.complex<4>> {{.*}})
subroutine exp_testc(a, b)
  complex :: a, b
! CHECK:  %[[A:.*]] = fir.load %[[AREF:.*]] : !fir.ref<!fir.complex<4>>
! CHECK:  %[[RES:.*]] = fir.call @fir.exp.contract.z4.z4(%[[A]]) {{.*}}: (!fir.complex<4>) -> !fir.complex<4>
! CHECK:  fir.store %[[RES]] to %[[BREF]] : !fir.ref<!fir.complex<4>>
  b = exp(a)
end subroutine

! CHECK-LABEL: exp_testcd
! CHECK-SAME: (%[[AREF:.*]]: !fir.ref<!fir.complex<8>> {{.*}}, %[[BREF:.*]]: !fir.ref<!fir.complex<8>> {{.*}})
subroutine exp_testcd(a, b)
  complex(kind=8) :: a, b
! CHECK:  %[[A:.*]] = fir.load %[[AREF:.*]] : !fir.ref<!fir.complex<8>>
! CHECK:  %[[RES:.*]] = fir.call @fir.exp.contract.z8.z8(%[[A]]) {{.*}}: (!fir.complex<8>) -> !fir.complex<8>
! CHECK:  fir.store %[[RES]] to %[[BREF]] : !fir.ref<!fir.complex<8>>
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

! CMPLX-APPROX-LABEL: private @fir.exp.contract_afn.z4.z4
! CMPLX-PRECISE-LABEL: private @fir.exp.contract.z4.z4
! CMPLX-MLIR-LABEL: private @fir.exp.contract.z4.z4
! CMPLX-SAME: (%[[ARG32_OUTLINE:.*]]: !fir.complex<4>) -> !fir.complex<4>
! CMPLX-FAST: %[[C:.*]] = fir.convert %[[ARG32_OUTLINE]] : (!fir.complex<4>) -> complex<f32>
! CMPLX-FAST: %[[E:.*]] = complex.exp %[[C]] : complex<f32>
! CMPLX-FAST: %[[RESULT32_OUTLINE:.*]] = fir.convert %[[E]] : (complex<f32>) -> !fir.complex<4>
! CMPLX-PRECISE: %[[RESULT32_OUTLINE:.*]] = fir.call @cexpf(%[[ARG32_OUTLINE]]) fastmath<contract> : (!fir.complex<4>) -> !fir.complex<4>
! CMPLX: return %[[RESULT32_OUTLINE]] : !fir.complex<4>

! CMPLX-APPROX-LABEL: private @fir.exp.contract_afn.z8.z8
! CMPLX-PRECISE-LABEL: private @fir.exp.contract.z8.z8
! CMPLX-MLIR-LABEL: private @fir.exp.contract.z8.z8
! CMPLX-SAME: (%[[ARG64_OUTLINE:.*]]: !fir.complex<8>) -> !fir.complex<8>
! CMPLX-FAST: %[[C:.*]] = fir.convert %[[ARG64_OUTLINE]] : (!fir.complex<8>) -> complex<f64>
! CMPLX-FAST: %[[E:.*]] = complex.exp %[[C]] : complex<f64>
! CMPLX-FAST: %[[RESULT64_OUTLINE:.*]] = fir.convert %[[E]] : (complex<f64>) -> !fir.complex<8>
! CMPLX-PRECISE: %[[RESULT64_OUTLINE:.*]] = fir.call @cexp(%[[ARG64_OUTLINE]]) fastmath<contract> : (!fir.complex<8>) -> !fir.complex<8>
! CMPLX: return %[[RESULT64_OUTLINE]] : !fir.complex<8>
