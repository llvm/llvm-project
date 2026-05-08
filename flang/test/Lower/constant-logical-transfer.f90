! Test lowering of non canonical LOGICAL constants.
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

subroutine test_transfer_constant(l4, l8)
  logical(4) :: l4
  logical(8) :: l8
  l4 = transfer(3, .true._4)
  l8 = transfer(7_8, .true._8)
end subroutine

module constant_logical
  logical(4) :: var(3) = [.true., transfer(3, .true._4), .false.]
end module

! CHECK-LABEL:  func.func @_QPtest_transfer_constant(
! CHECK:          %[[CONSTANT_0:.*]] = arith.constant 3 : i32
! CHECK:          %[[BITCAST_0:.*]] = fir.bitcast %[[CONSTANT_0]] : (i32) -> !fir.logical<4>
! CHECK:          hlfir.assign %[[BITCAST_0]] to %{{.*}} : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK:          %[[CONSTANT_1:.*]] = arith.constant 7 : i64
! CHECK:          %[[BITCAST_1:.*]] = fir.bitcast %[[CONSTANT_1]] : (i64) -> !fir.logical<8>
! CHECK:          hlfir.assign %[[BITCAST_1]] to %{{.*}} : !fir.logical<8>, !fir.ref<!fir.logical<8>>

! CHECK:        fir.global @_QMconstant_logicalEvar(dense<[1, 3, 0]> : tensor<3xi32>) : !fir.array<3x!fir.logical<4>>
