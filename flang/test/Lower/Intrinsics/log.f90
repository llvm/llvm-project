! RUN: bbc -emit-fir -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CHECK,CMPLX,CMPLX-FAST"
! RUN: bbc -emit-fir --math-runtime=precise -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CMPLX,CMPLX-PRECISE"
! RUN: bbc -emit-fir --disable-mlir-complex -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CMPLX,CMPLX-PRECISE"
! RUN: %flang_fc1 -emit-fir -mllvm -outline-intrinsics %s -o - | FileCheck %s --check-prefixes="CHECK,CMPLX,CMPLX-FAST"
! RUN: %flang_fc1 -emit-fir -mllvm -outline-intrinsics -mllvm --math-runtime=precise %s -o - | FileCheck %s --check-prefixes="CMPLX,CMPLX-PRECISE"
! RUN: %flang_fc1 -emit-fir -mllvm -outline-intrinsics -mllvm --disable-mlir-complex %s -o - | FileCheck %s --check-prefixes="CMPLX,CMPLX-PRECISE"

! CHECK-LABEL: log_testr
! CHECK-SAME: (%[[AREF:.*]]: !fir.ref<f32> {{.*}}, %[[BREF:.*]]: !fir.ref<f32> {{.*}})
subroutine log_testr(a, b)
  real :: a, b
! CHECK:  %[[A:.*]] = fir.load %[[AREF:.*]] : !fir.ref<f32>
! CHECK:  %[[RES:.*]] = fir.call @fir.log.f32.f32(%[[A]]) : (f32) -> f32
! CHECK:  fir.store %[[RES]] to %[[BREF]] : !fir.ref<f32>
  b = log(a)
end subroutine

! CHECK-LABEL: log_testd
! CHECK-SAME: (%[[AREF:.*]]: !fir.ref<f64> {{.*}}, %[[BREF:.*]]: !fir.ref<f64> {{.*}})
subroutine log_testd(a, b)
  real(kind=8) :: a, b
! CHECK:  %[[A:.*]] = fir.load %[[AREF:.*]] : !fir.ref<f64>
! CHECK:  %[[RES:.*]] = fir.call @fir.log.f64.f64(%[[A]]) : (f64) -> f64
! CHECK:  fir.store %[[RES]] to %[[BREF]] : !fir.ref<f64>
  b = log(a)
end subroutine

! CHECK-LABEL: log_testc
! CHECK-SAME: (%[[AREF:.*]]: !fir.ref<!fir.complex<4>> {{.*}}, %[[BREF:.*]]: !fir.ref<!fir.complex<4>> {{.*}})
subroutine log_testc(a, b)
  complex :: a, b
! CHECK:  %[[A:.*]] = fir.load %[[AREF:.*]] : !fir.ref<!fir.complex<4>>
! CHECK:  %[[RES:.*]] = fir.call @fir.log.z4.z4(%[[A]]) : (!fir.complex<4>) -> !fir.complex<4>
! CHECK:  fir.store %[[RES]] to %[[BREF]] : !fir.ref<!fir.complex<4>>
  b = log(a)
end subroutine

! CHECK-LABEL: log_testcd
! CHECK-SAME: (%[[AREF:.*]]: !fir.ref<!fir.complex<8>> {{.*}}, %[[BREF:.*]]: !fir.ref<!fir.complex<8>> {{.*}})
subroutine log_testcd(a, b)
  complex(kind=8) :: a, b
! CHECK:  %[[A:.*]] = fir.load %[[AREF:.*]] : !fir.ref<!fir.complex<8>>
! CHECK:  %[[RES:.*]] = fir.call @fir.log.z8.z8(%[[A]]) : (!fir.complex<8>) -> !fir.complex<8>
! CHECK:  fir.store %[[RES]] to %[[BREF]] : !fir.ref<!fir.complex<8>>
  b = log(a)
end subroutine

! CHECK-LABEL: log10_testr
! CHECK-SAME: (%[[AREF:.*]]: !fir.ref<f32> {{.*}}, %[[BREF:.*]]: !fir.ref<f32> {{.*}})
subroutine log10_testr(a, b)
  real :: a, b
! CHECK:  %[[A:.*]] = fir.load %[[AREF:.*]] : !fir.ref<f32>
! CHECK:  %[[RES:.*]] = fir.call @fir.log10.f32.f32(%[[A]]) : (f32) -> f32
! CHECK:  fir.store %[[RES]] to %[[BREF]] : !fir.ref<f32>
  b = log10(a)
end subroutine

! CHECK-LABEL: log10_testd
! CHECK-SAME: (%[[AREF:.*]]: !fir.ref<f64> {{.*}}, %[[BREF:.*]]: !fir.ref<f64> {{.*}})
subroutine log10_testd(a, b)
  real(kind=8) :: a, b
! CHECK:  %[[A:.*]] = fir.load %[[AREF:.*]] : !fir.ref<f64>
! CHECK:  %[[RES:.*]] = fir.call @fir.log10.f64.f64(%[[A]]) : (f64) -> f64
! CHECK:  fir.store %[[RES]] to %[[BREF]] : !fir.ref<f64>
  b = log10(a)
end subroutine

! CHECK-LABEL: private @fir.log.f32.f32
! CHECK-SAME: (%[[ARG32_OUTLINE:.*]]: f32) -> f32
! CHECK: %[[RESULT32_OUTLINE:.*]] = math.log %[[ARG32_OUTLINE]] : f32
! CHECK: return %[[RESULT32_OUTLINE]] : f32

! CHECK-LABEL: private @fir.log.f64.f64
! CHECK-SAME: (%[[ARG64_OUTLINE:.*]]: f64) -> f64
! CHECK: %[[RESULT64_OUTLINE:.*]] = math.log %[[ARG64_OUTLINE]] : f64
! CHECK: return %[[RESULT64_OUTLINE]] : f64

! CMPLX-LABEL: private @fir.log.z4.z4
! CMPLX-SAME: (%[[ARG32_OUTLINE:.*]]: !fir.complex<4>) -> !fir.complex<4>
! CMPLX-FAST: %[[C:.*]] = fir.convert %[[ARG32_OUTLINE]] : (!fir.complex<4>) -> complex<f32>
! CMPLX-FAST: %[[E:.*]] = complex.log %[[C]] : complex<f32>
! CMPLX-FAST: %[[RESULT32_OUTLINE:.*]] = fir.convert %[[E]] : (complex<f32>) -> !fir.complex<4>
! CMPLX-PRECISE: %[[RESULT32_OUTLINE:.*]] = fir.call @clogf(%[[ARG32_OUTLINE]]) : (!fir.complex<4>) -> !fir.complex<4>
! CMPLX: return %[[RESULT32_OUTLINE]] : !fir.complex<4>

! CMPLX-LABEL: private @fir.log.z8.z8
! CMPLX-SAME: (%[[ARG64_OUTLINE:.*]]: !fir.complex<8>) -> !fir.complex<8>
! CMPLX-FAST: %[[C:.*]] = fir.convert %[[ARG64_OUTLINE]] : (!fir.complex<8>) -> complex<f64>
! CMPLX-FAST: %[[E:.*]] = complex.log %[[C]] : complex<f64>
! CMPLX-FAST: %[[RESULT64_OUTLINE:.*]] = fir.convert %[[E]] : (complex<f64>) -> !fir.complex<8>
! CMPLX-PRECISE: %[[RESULT64_OUTLINE:.*]] = fir.call @clog(%[[ARG64_OUTLINE]]) : (!fir.complex<8>) -> !fir.complex<8>
! CMPLX: return %[[RESULT64_OUTLINE]] : !fir.complex<8>

! CHECK-LABEL: private @fir.log10.f32.f32
! CHECK-SAME: (%[[ARG32_OUTLINE:.*]]: f32) -> f32
! CHECK: %[[RESULT32_OUTLINE:.*]] = math.log10 %[[ARG32_OUTLINE]] : f32
! CHECK: return %[[RESULT32_OUTLINE]] : f32

! CHECK-LABEL: private @fir.log10.f64.f64
! CHECK-SAME: (%[[ARG64_OUTLINE:.*]]: f64) -> f64
! CHECK: %[[RESULT64_OUTLINE:.*]] = math.log10 %[[ARG64_OUTLINE]] : f64
! CHECK: return %[[RESULT64_OUTLINE]] : f64
