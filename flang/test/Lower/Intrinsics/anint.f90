! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: anint_test
subroutine anint_test(a, b)
  real :: a, b
  ! CHECK: fir.call @llvm.round.f32
  b = anint(a)
end subroutine
  