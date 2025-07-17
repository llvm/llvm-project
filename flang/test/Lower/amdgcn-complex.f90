! REQUIRES: amdgpu-registered-target
! RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s

subroutine cabsf_test(a, b)
   complex :: a
   real :: b
   b = abs(a)
end subroutine

! CHECK-LABEL: func @_QPcabsf_test(
! CHECK: complex.abs
! CHECK-NOT: fir.call @cabsf

subroutine cexpf_test(a, b)
   complex :: a, b
   b = exp(a)
end subroutine

! CHECK-LABEL: func @_QPcexpf_test(
! CHECK: complex.exp
! CHECK-NOT: fir.call @cexpf
