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
! CHECK:  %[[VAL_3:.*]] = fir.call %[[VAL_2]]() : () -> f32
! CHECK:  fir.store %[[VAL_3]] to %[[VAL_0]] : !fir.ref<f32>
! CHECK:  fir.call @_QPtakes_real(%[[VAL_0]]) : (!fir.ref<f32>) -> ()

! CHECK-LABEL: func @_QPdummy_with_iface_entry(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.boxproc<() -> ()>) {
! CHECK:  %[[VAL_1:.*]] = fir.alloca f32 {adapt.valuebyref}
! CHECK:  br ^bb1
! CHECK:  ^bb1:
! CHECK:  %[[VAL_2:.*]] = fir.box_addr %[[VAL_0]] : (!fir.boxproc<() -> ()>) -> (() -> f32)
! CHECK:  %[[VAL_3:.*]] = fir.call %[[VAL_2]]() : () -> f32
! CHECK:  fir.store %[[VAL_3]] to %[[VAL_1]] : !fir.ref<f32>
! CHECK:  fir.call @_QPtakes_real(%[[VAL_1]]) : (!fir.ref<f32>) -> ()

subroutine subroutine_dummy()
  entry subroutine_dummy_entry(x)
  call x()
end subroutine
! CHECK-LABEL: func @_QPsubroutine_dummy() {
! CHECK:  %[[VAL_0:.*]] = fir.undefined !fir.boxproc<() -> ()>
! CHECK:  br ^bb1
! CHECK:  ^bb1:
! CHECK:  %[[VAL_1:.*]] = fir.box_addr %[[VAL_0]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:  fir.call %[[VAL_1]]() : () -> ()

! CHECK-LABEL: func @_QPsubroutine_dummy_entry(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.boxproc<() -> ()>) {
! CHECK:  br ^bb1
! CHECK:  ^bb1:
! CHECK:  %[[VAL_1:.*]] = fir.box_addr %[[VAL_0]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:  fir.call %[[VAL_1]]() : () -> ()

subroutine character_dummy()
  external :: c
  character(*) :: c
  entry character_dummy_entry(c)
  call takes_char(c())
end subroutine
! CHECK-LABEL: func @_QPcharacter_dummy() {
! CHECK:  %[[VAL_0:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  br ^bb1
! CHECK:  ^bb1:
! CHECK:  %[[VAL_1:.*]] = fir.extract_value %[[VAL_0]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:  %[[VAL_3:.*]] = fir.extract_value %[[VAL_0]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> i64
! CHECK:  %[[VAL_4:.*]] = fir.call @llvm.stacksave() : () -> !fir.ref<i8>
! CHECK:  %[[VAL_5:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_3]] : i64) {bindc_name = ".result"}
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_2]] : (() -> ()) -> ((!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>)
! CHECK:  %[[VAL_7:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
! CHECK:  %[[VAL_8:.*]] = fir.call %[[VAL_6]](%[[VAL_5]], %[[VAL_7]]) : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:  %[[VAL_9:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
! CHECK:  %[[VAL_10:.*]] = fir.emboxchar %[[VAL_5]], %[[VAL_9]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:  fir.call @_QPtakes_char(%[[VAL_10]]) : (!fir.boxchar<1>) -> ()
! CHECK:  fir.call @llvm.stackrestore(%[[VAL_4]]) : (!fir.ref<i8>) -> ()

! CHECK-LABEL: func @_QPcharacter_dummy_entry(
! CHECK-SAME:  %[[VAL_0:.*]]: tuple<!fir.boxproc<() -> ()>, i64> {fir.char_proc}) {
! CHECK:  br ^bb1
! CHECK:  ^bb1:
! CHECK:  %[[VAL_1:.*]] = fir.extract_value %[[VAL_0]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:  %[[VAL_3:.*]] = fir.extract_value %[[VAL_0]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> i64
! CHECK:  %[[VAL_4:.*]] = fir.call @llvm.stacksave() : () -> !fir.ref<i8>
! CHECK:  %[[VAL_5:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_3]] : i64) {bindc_name = ".result"}
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_2]] : (() -> ()) -> ((!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>)
! CHECK:  %[[VAL_7:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
! CHECK:  %[[VAL_8:.*]] = fir.call %[[VAL_6]](%[[VAL_5]], %[[VAL_7]]) : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:  %[[VAL_9:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
! CHECK:  %[[VAL_10:.*]] = fir.emboxchar %[[VAL_5]], %[[VAL_9]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:  fir.call @_QPtakes_char(%[[VAL_10]]) : (!fir.boxchar<1>) -> ()
! CHECK:  fir.call @llvm.stackrestore(%[[VAL_4]]) : (!fir.ref<i8>) -> ()
