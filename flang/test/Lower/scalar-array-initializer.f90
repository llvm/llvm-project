! Test lowering of array variables whose initializer is a scalar. Semantics
! keeps such an initializer as a scalar, so each consumer has to either emit
! it as a splat or expand it itself.
! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

module m
  real(8) :: r(-2:2, 3) = 1.5d0
  complex :: c(2) = (1.0, -2.0)
  logical :: l(3) = .true.
  ! Equivalenced: initialized through the combined storage image.
  real :: e1(6) = 2.5
  real :: e2(4)
  equivalence (e1(2), e2(1))
  ! Common block member.
  real :: x(5) = 3.0
  integer :: n = 4
  common /cb/ x, n
end module

subroutine s()
  ! Saved local.
  integer :: k(3) = -1
  call use(k)
end subroutine

! CHECK-LABEL: fir.global @cb_ {{.*}} : tuple<!fir.array<5xf32>, i32> {
! CHECK: %[[X0:.*]] = fir.undefined !fir.array<5xf32>
! CHECK: %[[THREE:.*]] = arith.constant 3.000000e+00 : f32
! CHECK: %[[X:.*]] = fir.insert_on_range %[[X0]], %[[THREE]] from (0) to (4)
! CHECK: %[[T1:.*]] = fir.insert_value %{{.*}}, %[[X]], [0 : index]
! CHECK: %[[FOUR:.*]] = arith.constant 4 : i32
! CHECK: %[[T2:.*]] = fir.insert_value %[[T1]], %[[FOUR]], [1 : index]
! CHECK: fir.has_value %[[T2]]

! 1075838976 is the bit pattern of 2.5 as a real(4).
! CHECK-LABEL: fir.global @_QMmEe1 {{.*}} : !fir.array<6xi32> {
! CHECK: %[[E0:.*]] = fir.undefined !fir.array<6xi32>
! CHECK: %[[BITS:.*]] = arith.constant 1075838976 : i32
! CHECK: %[[E:.*]] = fir.insert_on_range %[[E0]], %[[BITS]] from (0) to (5)
! CHECK: fir.has_value %[[E]]

! CHECK-DAG: fir.global @_QMmEc(dense<(1.000000e+00,-2.000000e+00)> : tensor<2xcomplex<f32>>) {{.*}} : !fir.array<2xcomplex<f32>>
! CHECK-DAG: fir.global @_QMmEl(dense<1> : tensor<3xi32>) {{.*}} : !fir.array<3x!fir.logical<4>>
! CHECK-DAG: fir.global @_QMmEr(dense<1.500000e+00> : tensor<3x5xf64>) {{.*}} : !fir.array<5x3xf64>
! CHECK-DAG: fir.global internal @_QFsEk(dense<-1> : tensor<3xi32>) {{.*}} : !fir.array<3xi32>
