! REQUIRES: amdgpu-registered-target
! RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPcabsf_test(
! CHECK: complex.abs
! CHECK-NOT: fir.call @cabsf
subroutine cabsf_test(a, b)
   complex :: a
   real :: b
   b = abs(a)
end subroutine

! CHECK-LABEL: func @_QPcexpf_test(
! CHECK: complex.exp
! CHECK-NOT: fir.call @cexpf
subroutine cexpf_test(a, b)
   complex :: a, b
   b = exp(a)
end subroutine

! CHECK-LABEL: func @_QPpow_test1(
! CHECK: complex.mul
! CHECK-NOT: complex.pow
! CHECK-NOT: fir.call @_FortranAcpowi
subroutine pow_test1(a, b)
   complex :: a, b
   a = b**2
end subroutine pow_test1

! CHECK-LABEL: func @_QPpow_test2(
! CHECK: complex.pow
! CHECK-NOT: fir.call @_FortranAcpowi
subroutine pow_test2(a, b, c)
   complex :: a, b, c
   a = b**c
end subroutine pow_test2
