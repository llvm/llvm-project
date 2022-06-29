! RUN: bbc -emit-fir -outline-intrinsics %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -mllvm -outline-intrinsics %s -o - | FileCheck %s

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
! CHECK: %[[RESULT32_OUTLINE:.*]] = fir.call @__fs_log_1(%[[ARG32_OUTLINE]]) : (f32) -> f32
! CHECK: return %[[RESULT32_OUTLINE]] : f32

! CHECK-LABEL: private @fir.log.f64.f64
! CHECK-SAME: (%[[ARG64_OUTLINE:.*]]: f64) -> f64
! CHECK: %[[RESULT64_OUTLINE:.*]] = fir.call @__fd_log_1(%[[ARG64_OUTLINE]]) : (f64) -> f64
! CHECK: return %[[RESULT64_OUTLINE]] : f64

! CHECK-LABEL: private @fir.log.z4.z4
! CHECK-SAME: (%[[ARG32_OUTLINE]]: !fir.complex<4>) -> !fir.complex<4>
! CHECK: %[[RESULT32_OUTLINE]] = fir.call @__fc_log_1(%[[ARG32_OUTLINE]]) : (!fir.complex<4>) -> !fir.complex<4>
! CHECK: return %[[RESULT32_OUTLINE]] : !fir.complex<4>

! CHECK-LABEL: private @fir.log.z8.z8
! CHECK-SAME: (%[[ARG64_OUTLINE]]: !fir.complex<8>) -> !fir.complex<8>
! CHECK: %[[RESULT64_OUTLINE]] = fir.call @__fz_log_1(%[[ARG64_OUTLINE]]) : (!fir.complex<8>) -> !fir.complex<8>
! CHECK: return %[[RESULT64_OUTLINE]] : !fir.complex<8>

! CHECK-LABEL: private @fir.log10.f32.f32
! CHECK-SAME: (%[[ARG32_OUTLINE:.*]]: f32) -> f32
! CHECK: %[[RESULT32_OUTLINE:.*]] = fir.call @__fs_log10_1(%[[ARG32_OUTLINE]]) : (f32) -> f32
! CHECK: return %[[RESULT32_OUTLINE]] : f32

! CHECK-LABEL: private @fir.log10.f64.f64
! CHECK-SAME: (%[[ARG64_OUTLINE:.*]]: f64) -> f64
! CHECK: %[[RESULT64_OUTLINE:.*]] = fir.call @__fd_log10_1(%[[ARG64_OUTLINE]]) : (f64) -> f64
! CHECK: return %[[RESULT64_OUTLINE]] : f64
