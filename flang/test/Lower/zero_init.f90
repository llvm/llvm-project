! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck --check-prefix=CHECK-DEFAULT %s
! RUN: %flang_fc1 -finit-global-zero -emit-hlfir -o - %s | FileCheck --check-prefix=CHECK-DEFAULT %s
! RUN: %flang_fc1 -fno-init-global-zero -emit-hlfir -o - %s | FileCheck --check-prefix=CHECK-NO-ZERO-INIT %s
! RUN: bbc -emit-hlfir -o - %s | FileCheck --check-prefix=CHECK-DEFAULT %s
! RUN: bbc -finit-global-zero -emit-hlfir -o - %s | FileCheck --check-prefix=CHECK-DEFAULT %s
! RUN: bbc -finit-global-zero=false -emit-hlfir -o - %s | FileCheck --check-prefix=CHECK-NO-ZERO-INIT %s

module zeroInitM1
  real :: x
end module zeroInitM1

!CHECK-DEFAULT: fir.global @_QMzeroinitm1Ex : f32 {
!CHECK-DEFAULT:   %[[UNDEF:.*]] = fir.zero_bits f32
!CHECK-DEFAULT:   fir.has_value %[[UNDEF]] : f32
!CHECK-DEFAULT: }

!CHECK-NO-ZERO-INIT: fir.global @_QMzeroinitm1Ex : f32 {
!CHECK-NO-ZERO-INIT:   %[[UNDEF:.*]] = fir.undefined f32
!CHECK-NO-ZERO-INIT:   fir.has_value %[[UNDEF]] : f32
!CHECK-NO-ZERO-INIT: }
