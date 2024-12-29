! RUN: bbc -emit-fir -hlfir=false -outline-intrinsics %s -o - | FileCheck %s --check-prefixes=%if system-aix %{"CHECK,CMPLX,CMPLX-PRECISE,AIX-LOG"%} %else %{"CHECK,CMPLX,CMPLX-PRECISE,COMMON-LOG"%}
! RUN: bbc -emit-fir -hlfir=false --math-runtime=precise -outline-intrinsics %s -o - | FileCheck %s --check-prefixes=%if system-aix %{"CMPLX,CMPLX-PRECISE,AIX-LOG"%} %else %{"CMPLX,CMPLX-PRECISE,COMMON-LOG"%}
! RUN: bbc -emit-fir -hlfir=false --force-mlir-complex -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CMPLX,CMPLX-FAST,CMPLX-MLIR"
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -mllvm -outline-intrinsics %s -o - | FileCheck %s --check-prefixes=%if system-aix %{"CHECK,CMPLX,CMPLX-PRECISE,AIX-LOG"%} %else %{"CHECK,CMPLX,CMPLX-PRECISE,COMMON-LOG"%}
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -mllvm -outline-intrinsics -mllvm --math-runtime=precise %s -o - | FileCheck %s --check-prefixes=%if system-aix %{"CMPLX,CMPLX-PRECISE,AIX-LOG"%} %else %{"CMPLX,CMPLX-PRECISE,COMMON-LOG"%}
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -mllvm -outline-intrinsics -mllvm --force-mlir-complex %s -o - | FileCheck %s --check-prefixes="CMPLX,CMPLX-FAST,CMPLX-MLIR"
! RUN: %flang_fc1 -fapprox-func -emit-fir -flang-deprecated-no-hlfir -mllvm -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CMPLX,CMPLX-APPROX"

! CHECK-LABEL: log_testr
! CHECK-SAME: (%[[AREF:.*]]: !fir.ref<f32> {{.*}}, %[[BREF:.*]]: !fir.ref<f32> {{.*}})
subroutine log_testr(a, b)
  real :: a, b
! CHECK:  %[[A:.*]] = fir.load %[[AREF:.*]] : !fir.ref<f32>
! CHECK:  %[[RES:.*]] = fir.call @fir.log.contract.f32.f32(%[[A]]) {{.*}}: (f32) -> f32
! CHECK:  fir.store %[[RES]] to %[[BREF]] : !fir.ref<f32>
  b = log(a)
end subroutine

! CHECK-LABEL: log_testd
! CHECK-SAME: (%[[AREF:.*]]: !fir.ref<f64> {{.*}}, %[[BREF:.*]]: !fir.ref<f64> {{.*}})
subroutine log_testd(a, b)
  real(kind=8) :: a, b
! CHECK:  %[[A:.*]] = fir.load %[[AREF:.*]] : !fir.ref<f64>
! CHECK:  %[[RES:.*]] = fir.call @fir.log.contract.f64.f64(%[[A]]) {{.*}}: (f64) -> f64
! CHECK:  fir.store %[[RES]] to %[[BREF]] : !fir.ref<f64>
  b = log(a)
end subroutine

! CHECK-LABEL: log_testc
! CHECK-SAME: (%[[AREF:.*]]: !fir.ref<complex<f32>> {{.*}}, %[[BREF:.*]]: !fir.ref<complex<f32>> {{.*}})
subroutine log_testc(a, b)
  complex :: a, b
! CHECK:  %[[A:.*]] = fir.load %[[AREF:.*]] : !fir.ref<complex<f32>>
! CHECK:  %[[RES:.*]] = fir.call @fir.log.contract.z32.z32(%[[A]]) {{.*}}: (complex<f32>) -> complex<f32>
! CHECK:  fir.store %[[RES]] to %[[BREF]] : !fir.ref<complex<f32>>
  b = log(a)
end subroutine

! CHECK-LABEL: log_testcd
! CHECK-SAME: (%[[AREF:.*]]: !fir.ref<complex<f64>> {{.*}}, %[[BREF:.*]]: !fir.ref<complex<f64>> {{.*}})
subroutine log_testcd(a, b)
  complex(kind=8) :: a, b
! CHECK:  %[[A:.*]] = fir.load %[[AREF:.*]] : !fir.ref<complex<f64>>
! CHECK:  %[[RES:.*]] = fir.call @fir.log.contract.z64.z64(%[[A]]) {{.*}}: (complex<f64>) -> complex<f64>
! CHECK:  fir.store %[[RES]] to %[[BREF]] : !fir.ref<complex<f64>>
  b = log(a)
end subroutine

! CHECK-LABEL: log10_testr
! CHECK-SAME: (%[[AREF:.*]]: !fir.ref<f32> {{.*}}, %[[BREF:.*]]: !fir.ref<f32> {{.*}})
subroutine log10_testr(a, b)
  real :: a, b
! CHECK:  %[[A:.*]] = fir.load %[[AREF:.*]] : !fir.ref<f32>
! CHECK:  %[[RES:.*]] = fir.call @fir.log10.contract.f32.f32(%[[A]]) {{.*}}: (f32) -> f32
! CHECK:  fir.store %[[RES]] to %[[BREF]] : !fir.ref<f32>
  b = log10(a)
end subroutine

! CHECK-LABEL: log10_testd
! CHECK-SAME: (%[[AREF:.*]]: !fir.ref<f64> {{.*}}, %[[BREF:.*]]: !fir.ref<f64> {{.*}})
subroutine log10_testd(a, b)
  real(kind=8) :: a, b
! CHECK:  %[[A:.*]] = fir.load %[[AREF:.*]] : !fir.ref<f64>
! CHECK:  %[[RES:.*]] = fir.call @fir.log10.contract.f64.f64(%[[A]]) {{.*}}: (f64) -> f64
! CHECK:  fir.store %[[RES]] to %[[BREF]] : !fir.ref<f64>
  b = log10(a)
end subroutine

! CHECK-LABEL: private @fir.log.contract.f32.f32
! CHECK-SAME: (%[[ARG32_OUTLINE:.*]]: f32) -> f32
! CHECK: %[[RESULT32_OUTLINE:.*]] = math.log %[[ARG32_OUTLINE]] fastmath<contract> : f32
! CHECK: return %[[RESULT32_OUTLINE]] : f32

! CHECK-LABEL: private @fir.log.contract.f64.f64
! CHECK-SAME: (%[[ARG64_OUTLINE:.*]]: f64) -> f64
! CHECK: %[[RESULT64_OUTLINE:.*]] = math.log %[[ARG64_OUTLINE]] fastmath<contract> : f64
! CHECK: return %[[RESULT64_OUTLINE]] : f64

! CMPLX-APPROX-LABEL: private @fir.log.contract_afn.z32.z32
! CMPLX-PRECISE-LABEL: private @fir.log.contract.z32.z32
! CMPLX-MLIR-LABEL: private @fir.log.contract.z32.z32
! CMPLX-SAME: (%[[C:.*]]: complex<f32>) -> complex<f32>
! CMPLX-FAST: %[[E:.*]] = complex.log %[[C]] fastmath<contract> : complex<f32>
! CMPLX-APPROX: %[[E:.*]] = complex.log %[[C]] fastmath<contract,afn> : complex<f32>
! CMPLX-PRECISE: %[[E:.*]] = fir.call @clogf(%[[C]]) fastmath<contract> : (complex<f32>) -> complex<f32>
! CMPLX: return %[[E]] : complex<f32>

! CMPLX-APPROX-LABEL: private @fir.log.contract_afn.z64.z64
! CMPLX-PRECISE-LABEL: private @fir.log.contract.z64.z64
! CMPLX-MLIR-LABEL: private @fir.log.contract.z64.z64
! CMPLX-SAME: (%[[C:.*]]: complex<f64>) -> complex<f64>
! CMPLX-FAST: %[[E:.*]] = complex.log %[[C]] fastmath<contract> : complex<f64>
! CMPLX-APPROX: %[[E:.*]] = complex.log %[[C]] fastmath<contract,afn> : complex<f64>
! COMMON-LOG: %[[E:.*]] = fir.call @clog(%[[C]]) fastmath<contract> : (complex<f64>) -> complex<f64>
! AIX-LOG: %[[E:.*]] = fir.call @__clog(%[[C]]) fastmath<contract> : (complex<f64>) -> complex<f64>
! CMPLX: return %[[E]] : complex<f64>

! CHECK-LABEL: private @fir.log10.contract.f32.f32
! CHECK-SAME: (%[[ARG32_OUTLINE:.*]]: f32) -> f32
! CHECK: %[[RESULT32_OUTLINE:.*]] = math.log10 %[[ARG32_OUTLINE]] fastmath<contract> : f32
! CHECK: return %[[RESULT32_OUTLINE]] : f32

! CHECK-LABEL: private @fir.log10.contract.f64.f64
! CHECK-SAME: (%[[ARG64_OUTLINE:.*]]: f64) -> f64
! CHECK: %[[RESULT64_OUTLINE:.*]] = math.log10 %[[ARG64_OUTLINE]] fastmath<contract> : f64
! CHECK: return %[[RESULT64_OUTLINE]] : f64
