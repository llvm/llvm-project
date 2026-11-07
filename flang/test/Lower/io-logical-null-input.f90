! A LOGICAL input item is passed to the runtime as a bool&. Its current value
! is stored as a bool before the call, so that it is kept when the runtime does
! not set it (null value in list-directed or namelist input). Otherwise the old
! value would be lost on big-endian targets, where the first byte of a
! LOGICAL(4) .TRUE. is zero.
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPread_logical(
subroutine read_logical(l)
  logical :: l
  read (*, *) l
end subroutine
! CHECK:  %[[L:.*]]:2 = hlfir.declare {{.*}}"_QFread_logicalEl"
! CHECK:  %[[ARG:.*]] = fir.convert %[[L]]#0 : (!fir.ref<!fir.logical<4>>) -> !fir.ref<i1>
! CHECK:  %[[OLD:.*]] = fir.load %[[L]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:  %[[B:.*]] = fir.convert %[[OLD]] : (!fir.logical<4>) -> i1
! CHECK:  %[[BA:.*]] = fir.convert %[[L]]#0 : (!fir.ref<!fir.logical<4>>) -> !fir.ref<i1>
! CHECK:  fir.store %[[B]] to %[[BA]] : !fir.ref<i1>
! CHECK:  fir.call @_FortranAioInputLogical(%{{.*}}, %[[ARG]])
! CHECK:  %[[NEW:.*]] = fir.load %{{.*}} : !fir.ref<i1>
! CHECK:  %[[V:.*]] = fir.convert %[[NEW]] : (i1) -> !fir.logical<4>
! CHECK:  fir.store %[[V]] to %[[L]]#0 : !fir.ref<!fir.logical<4>>
