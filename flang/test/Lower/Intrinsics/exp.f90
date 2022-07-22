! RUN: bbc -emit-fir -outline-intrinsics %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -mllvm -outline-intrinsics %s -o - | FileCheck %s

! CHECK-LABEL: exp_testr
! CHECK-SAME: (%[[AREF:.*]]: !fir.ref<f32> {{.*}}, %[[BREF:.*]]: !fir.ref<f32> {{.*}})
subroutine exp_testr(a, b)
  real :: a, b
! CHECK:  %[[A:.*]] = fir.load %[[AREF:.*]] : !fir.ref<f32>
! CHECK:  %[[RES:.*]] = fir.call @fir.exp.f32.f32(%[[A]]) : (f32) -> f32
! CHECK:  fir.store %[[RES]] to %[[BREF]] : !fir.ref<f32>
  b = exp(a)
end subroutine

! CHECK-LABEL: exp_testd
! CHECK-SAME: (%[[AREF:.*]]: !fir.ref<f64> {{.*}}, %[[BREF:.*]]: !fir.ref<f64> {{.*}})
subroutine exp_testd(a, b)
  real(kind=8) :: a, b
! CHECK:  %[[A:.*]] = fir.load %[[AREF:.*]] : !fir.ref<f64>
! CHECK:  %[[RES:.*]] = fir.call @fir.exp.f64.f64(%[[A]]) : (f64) -> f64
! CHECK:  fir.store %[[RES]] to %[[BREF]] : !fir.ref<f64>
  b = exp(a)
end subroutine

! CHECK-LABEL: exp_testc
! CHECK-SAME: (%[[AREF:.*]]: !fir.ref<!fir.complex<4>> {{.*}}, %[[BREF:.*]]: !fir.ref<!fir.complex<4>> {{.*}})
subroutine exp_testc(a, b)
  complex :: a, b
! CHECK:  %[[A:.*]] = fir.load %[[AREF:.*]] : !fir.ref<!fir.complex<4>>
! CHECK:  %[[RES:.*]] = fir.call @fir.exp.z4.z4(%[[A]]) : (!fir.complex<4>) -> !fir.complex<4>
! CHECK:  fir.store %[[RES]] to %[[BREF]] : !fir.ref<!fir.complex<4>>
  b = exp(a)
end subroutine

! CHECK-LABEL: exp_testcd
! CHECK-SAME: (%[[AREF:.*]]: !fir.ref<!fir.complex<8>> {{.*}}, %[[BREF:.*]]: !fir.ref<!fir.complex<8>> {{.*}})
subroutine exp_testcd(a, b)
  complex(kind=8) :: a, b
! CHECK:  %[[A:.*]] = fir.load %[[AREF:.*]] : !fir.ref<!fir.complex<8>>
! CHECK:  %[[RES:.*]] = fir.call @fir.exp.z8.z8(%[[A]]) : (!fir.complex<8>) -> !fir.complex<8>
! CHECK:  fir.store %[[RES]] to %[[BREF]] : !fir.ref<!fir.complex<8>>
  b = exp(a)
end subroutine

! CHECK-LABEL: private @fir.exp.f32.f32
! CHECK-SAME: (%[[ARG32_OUTLINE:.*]]: f32) -> f32
! CHECK: %[[RESULT32_OUTLINE:.*]] = math.exp %[[ARG32_OUTLINE]] : f32
! CHECK: return %[[RESULT32_OUTLINE]] : f32

! CHECK-LABEL: private @fir.exp.f64.f64
! CHECK-SAME: (%[[ARG64_OUTLINE:.*]]: f64) -> f64
! CHECK: %[[RESULT64_OUTLINE:.*]] = math.exp %[[ARG64_OUTLINE]] : f64
! CHECK: return %[[RESULT64_OUTLINE]] : f64

! CHECK-LABEL: private @fir.exp.z4.z4
! CHECK-SAME: (%[[ARG32_OUTLINE]]: !fir.complex<4>) -> !fir.complex<4>
! CHECK: %[[RESULT32_OUTLINE]] = fir.call @__fc_exp_1(%[[ARG32_OUTLINE]]) : (!fir.complex<4>) -> !fir.complex<4>
! CHECK: return %[[RESULT32_OUTLINE]] : !fir.complex<4>

! CHECK-LABEL: private @fir.exp.z8.z8
! CHECK-SAME: (%[[ARG64_OUTLINE]]: !fir.complex<8>) -> !fir.complex<8>
! CHECK: %[[RESULT64_OUTLINE]] = fir.call @__fz_exp_1(%[[ARG64_OUTLINE]]) : (!fir.complex<8>) -> !fir.complex<8>
! CHECK: return %[[RESULT64_OUTLINE]] : !fir.complex<8>
