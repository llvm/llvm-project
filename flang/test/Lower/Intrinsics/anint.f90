! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: anint_test
subroutine anint_test(a, b)
  real :: a, b
  ! CHECK: "llvm.intr.round"
  b = anint(a)
end subroutine
  
