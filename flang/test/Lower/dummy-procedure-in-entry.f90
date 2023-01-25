! Test dummy procedures that are not an argument in every entry.
! This requires creating a mock value in the entries where it is
! not an argument.
! RUN: bbc -emit-fir %s -o - | FileCheck %s

subroutine dummy_with_iface()
  interface
    real function x()
    end function
  end interface
  entry dummy_with_iface_entry(x)
  call takes_real(x())
end subroutine
! CHECK-LABEL: func @_QPdummy_with_iface() {
! CHECK:  %[[VAL_0:.*]] = fir.alloca f32 {adapt.valuebyref}
! CHECK:  %[[VAL_1:.*]] = fir.undefined !fir.boxproc<() -> ()>
! CHECK:  br ^bb1
! CHECK:  ^bb1:
! CHECK:  %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.boxproc<() -> ()>) -> (() -> f32)
! CHECK:  %[[VAL_3:.*]] = fir.call %[[VAL_2]]() {{.*}}: () -> f32
! CHECK:  fir.store %[[VAL_3]] to %[[VAL_0]] : !fir.ref<f32>
! CHECK:  fir.call @_QPtakes_real(%[[VAL_0]]) {{.*}}: (!fir.ref<f32>) -> ()

! CHECK-LABEL: func @_QPdummy_with_iface_entry(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.boxproc<() -> ()>) {
! CHECK:  %[[VAL_1:.*]] = fir.alloca f32 {adapt.valuebyref}
! CHECK:  br ^bb1
! CHECK:  ^bb1:
! CHECK:  %[[VAL_2:.*]] = fir.box_addr %[[VAL_0]] : (!fir.boxproc<() -> ()>) -> (() -> f32)
! CHECK:  %[[VAL_3:.*]] = fir.call %[[VAL_2]]() {{.*}}: () -> f32
! CHECK:  fir.store %[[VAL_3]] to %[[VAL_1]] : !fir.ref<f32>
! CHECK:  fir.call @_QPtakes_real(%[[VAL_1]]) {{.*}}: (!fir.ref<f32>) -> ()

subroutine subroutine_dummy()
  entry subroutine_dummy_entry(x)
  call x()
end subroutine
! CHECK-LABEL: func @_QPsubroutine_dummy() {
! CHECK:  %[[VAL_0:.*]] = fir.undefined !fir.boxproc<() -> ()>
! CHECK:  br ^bb1
! CHECK:  ^bb1:
! CHECK:  %[[VAL_1:.*]] = fir.box_addr %[[VAL_0]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:  fir.call %[[VAL_1]]() {{.*}}: () -> ()

! CHECK-LABEL: func @_QPsubroutine_dummy_entry(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.boxproc<() -> ()>) {
! CHECK:  br ^bb1
! CHECK:  ^bb1:
! CHECK:  %[[VAL_1:.*]] = fir.box_addr %[[VAL_0]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:  fir.call %[[VAL_1]]() {{.*}}: () -> ()
