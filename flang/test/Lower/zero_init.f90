! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck --check-prefix=CHECK-DEFAULT %s
! RUN: %flang_fc1 -finit-global-zero -emit-hlfir -o - %s | FileCheck --check-prefix=CHECK-DEFAULT %s
! RUN: %flang_fc1 -fno-init-global-zero -emit-hlfir -o - %s | FileCheck --check-prefix=CHECK-NO-ZERO-INIT %s

module m1
  real :: x
end module m1

!CHECK-DEFAULT: fir.global @_QMm1Ex : f32 {
!CHECK-DEFAULT:   %[[UNDEF:.*]] = fir.zero_bits f32
!CHECK-DEFAULT:   fir.has_value %[[UNDEF]] : f32
!CHECK-DEFAULT: }

!CHECK-NO-ZERO-INIT: fir.global @_QMm1Ex : f32 {
!CHECK-NO-ZERO-INIT:   %[[UNDEF:.*]] = fir.undefined f32
!CHECK-NO-ZERO-INIT:   fir.has_value %[[UNDEF]] : f32
!CHECK-NO-ZERO-INIT: }
