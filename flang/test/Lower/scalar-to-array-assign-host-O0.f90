! Scalar-to-array broadcast assignments are expanded inline even at -O0
! on the host (no OpenMP target-device compilation).

! RUN: %flang_fc1 -emit-fir -O0 %s -o - | FileCheck %s --implicit-check-not=_FortranAAssign

! CHECK-LABEL: func @_QPhost_scalar_broadcast
subroutine host_scalar_broadcast(arr)
  integer :: arr(4)
  ! CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
  ! CHECK-DAG: %[[VALUE:.*]] = arith.constant 11 : i32
  ! CHECK-DAG: %[[EXTENT:.*]] = arith.constant 4 : index
  ! CHECK: %[[SHAPE:.*]] = fir.shape %[[EXTENT]]
  ! CHECK: %[[ARR:.*]] = fir.declare %arg0(%[[SHAPE]])
  ! CHECK: fir.do_loop %[[INDEX:.*]] = %[[ONE]] to %[[EXTENT]] step %[[ONE]] unordered {
  ! CHECK-NEXT: %[[ELEMENT:.*]] = fir.array_coor %[[ARR]](%[[SHAPE]]) %[[INDEX]]
  ! CHECK-NEXT: fir.store %[[VALUE]] to %[[ELEMENT]] : !fir.ref<i32>
  ! CHECK-NEXT: }
  arr = 11
end subroutine
