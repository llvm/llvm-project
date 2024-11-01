! Test lowering of procedure designators to HLFIR.
! RUN: bbc -emit-fir -hlfir -o - %s | FileCheck %s

module test_proc_designator
  interface
    subroutine simple()
    end subroutine
    character(10) function return_char(x)
       integer :: x
    end function
  end interface
contains

subroutine test_pass_simple()
  call takes_simple(simple)
end subroutine
! CHECK-LABEL: func.func @_QMtest_proc_designatorPtest_pass_simple() {
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QPsimple) : () -> ()
! CHECK:  %[[VAL_1:.*]] = fir.emboxproc %[[VAL_0]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK:  fir.call @_QPtakes_simple(%[[VAL_1]]) {{.*}}: (!fir.boxproc<() -> ()>) -> ()

subroutine test_pass_character()
  call takes_char_proc(return_char)
end subroutine
! CHECK-LABEL: func.func @_QMtest_proc_designatorPtest_pass_character() {
! CHECK:  %[[VAL_0:.*]] = fir.address_of(@_QPreturn_char) : (!fir.ref<!fir.char<1,10>>, index, !fir.ref<i32>) -> !fir.boxchar<1>
! CHECK:  %[[VAL_1:.*]] = arith.constant 10 : i64
! CHECK:  %[[VAL_2:.*]] = fir.emboxproc %[[VAL_0]] : ((!fir.ref<!fir.char<1,10>>, index, !fir.ref<i32>) -> !fir.boxchar<1>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_3:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_4:.*]] = fir.insert_value %[[VAL_3]], %[[VAL_2]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_5:.*]] = fir.insert_value %[[VAL_4]], %[[VAL_1]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  fir.call @_QPtakes_char_proc(%[[VAL_5]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()

subroutine test_pass_simple_dummy(proc)
  procedure(simple) :: proc
  call takes_simple(proc)
end subroutine
! CHECK-LABEL: func.func @_QMtest_proc_designatorPtest_pass_simple_dummy(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxproc<() -> ()>) {
! CHECK:  fir.call @_QPtakes_simple(%[[VAL_0]]) {{.*}}: (!fir.boxproc<() -> ()>) -> ()

subroutine test_pass_character_dummy(proc)
  procedure(return_char) :: proc
  call takes_char_proc(proc)
end subroutine
! CHECK-LABEL: func.func @_QMtest_proc_designatorPtest_pass_character_dummy(
! CHECK-SAME:    %[[VAL_0:.*]]: tuple<!fir.boxproc<() -> ()>, i64> {fir.char_proc}) {
! CHECK:  %[[VAL_1:.*]] = fir.extract_value %[[VAL_0]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:  %[[VAL_3:.*]] = arith.constant 10 : i64
! CHECK:  %[[VAL_4:.*]] = fir.emboxproc %[[VAL_2]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_5:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_6:.*]] = fir.insert_value %[[VAL_5]], %[[VAL_4]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_7:.*]] = fir.insert_value %[[VAL_6]], %[[VAL_3]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  fir.call @_QPtakes_char_proc(%[[VAL_7]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()

subroutine test_pass_character_dummy_2(proc)
  character(*), external :: proc
  call takes_char_proc(proc)
end subroutine
! CHECK-LABEL: func.func @_QMtest_proc_designatorPtest_pass_character_dummy_2(
! CHECK-SAME:    %[[VAL_0:.*]]: tuple<!fir.boxproc<() -> ()>, i64> {fir.char_proc}) {
! CHECK:  %[[VAL_1:.*]] = fir.extract_value %[[VAL_0]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:  %[[VAL_3:.*]] = fir.extract_value %[[VAL_0]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> i64
! CHECK:  %[[VAL_4:.*]] = fir.emboxproc %[[VAL_2]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_5:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_6:.*]] = fir.insert_value %[[VAL_5]], %[[VAL_4]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_7:.*]] = fir.insert_value %[[VAL_6]], %[[VAL_3]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  fir.call @_QPtakes_char_proc(%[[VAL_7]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()

subroutine test_pass_simple_internal()
  integer :: x
  call takes_simple(simple_internal)
contains
subroutine simple_internal()
  x = 42
end subroutine
end subroutine
! CHECK-LABEL: func.func @_QMtest_proc_designatorPtest_pass_simple_internal() {
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}Ex
! CHECK:  %[[VAL_2:.*]] = fir.alloca tuple<!fir.ref<i32>>
! CHECK:  %[[VAL_3:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_3]] : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:  fir.store %[[VAL_1]]#1 to %[[VAL_4]] : !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:  %[[VAL_5:.*]] = fir.address_of(@_QMtest_proc_designatorFtest_pass_simple_internalPsimple_internal) : (!fir.ref<tuple<!fir.ref<i32>>>) -> ()
! CHECK:  %[[VAL_6:.*]] = fir.emboxproc %[[VAL_5]], %[[VAL_2]] : ((!fir.ref<tuple<!fir.ref<i32>>>) -> (), !fir.ref<tuple<!fir.ref<i32>>>) -> !fir.boxproc<() -> ()>
! CHECK:  fir.call @_QPtakes_simple(%[[VAL_6]]) {{.*}}: (!fir.boxproc<() -> ()>) -> ()

subroutine test_pass_character_internal()
  integer :: x
  call takes_char_proc(return_char_internal)
contains
character(10) function return_char_internal()
  return_char_internal = char(x)
end function
end subroutine
! CHECK-LABEL: func.func @_QMtest_proc_designatorPtest_pass_character_internal() {
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}Ex
! CHECK:  %[[VAL_2:.*]] = fir.alloca tuple<!fir.ref<i32>>
! CHECK:  %[[VAL_3:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_3]] : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:  fir.store %[[VAL_1]]#1 to %[[VAL_4]] : !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:  %[[VAL_5:.*]] = fir.address_of(@_QMtest_proc_designatorFtest_pass_character_internalPreturn_char_internal) : (!fir.ref<!fir.char<1,10>>, index, !fir.ref<tuple<!fir.ref<i32>>>) -> !fir.boxchar<1>
! CHECK:  %[[VAL_6:.*]] = arith.constant 10 : i64
! CHECK:  %[[VAL_7:.*]] = fir.emboxproc %[[VAL_5]], %[[VAL_2]] : ((!fir.ref<!fir.char<1,10>>, index, !fir.ref<tuple<!fir.ref<i32>>>) -> !fir.boxchar<1>, !fir.ref<tuple<!fir.ref<i32>>>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_8:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_9:.*]] = fir.insert_value %[[VAL_8]], %[[VAL_7]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_10:.*]] = fir.insert_value %[[VAL_9]], %[[VAL_6]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  fir.call @_QPtakes_char_proc(%[[VAL_10]]) {{.*}}: (tuple<!fir.boxproc<() -> ()>, i64>) -> ()


subroutine test_call_simple_dummy(proc)
  procedure(simple) :: proc
  call proc()
end subroutine
! CHECK-LABEL: func.func @_QMtest_proc_designatorPtest_call_simple_dummy(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxproc<() -> ()>) {
! CHECK:  %[[VAL_1:.*]] = fir.box_addr %[[VAL_0]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:  fir.call %[[VAL_1]]() {{.*}}: () -> ()

subroutine test_call_character_dummy(proc)
  procedure(return_char) :: proc
  call takes_char(proc(42))
end subroutine
! CHECK-LABEL: func.func @_QMtest_proc_designatorPtest_call_character_dummy(
! CHECK-SAME:    %[[VAL_0:.*]]: tuple<!fir.boxproc<() -> ()>, i64> {fir.char_proc}) {
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.char<1,10> {bindc_name = ".result"}
! CHECK:  %[[VAL_4:.*]] = fir.extract_value %[[VAL_0]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_5:.*]] = fir.box_addr %[[VAL_4]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:  %[[VAL_12:.*]] = fir.convert %[[VAL_5]] : (() -> ()) -> ((!fir.ref<!fir.char<1,10>>, index, !fir.ref<i32>) -> !fir.boxchar<1>)
! CHECK:  %[[VAL_13:.*]] = fir.call %[[VAL_12]](%[[VAL_1]], {{.*}}

subroutine test_present_simple_dummy(proc)
  procedure(simple), optional :: proc
  call takes_logical(present(proc))
end subroutine
! CHECK-LABEL: func.func @_QMtest_proc_designatorPtest_present_simple_dummy(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxproc<() -> ()>) {
! CHECK:  %[[VAL_1:.*]] = fir.is_present %[[VAL_0]] : (!fir.boxproc<() -> ()>) -> i1

subroutine test_present_character_dummy(proc)
  procedure(return_char), optional :: proc
  call takes_logical(present(proc))
end subroutine
! CHECK-LABEL: func.func @_QMtest_proc_designatorPtest_present_character_dummy(
! CHECK-SAME:    %[[VAL_0:.*]]: tuple<!fir.boxproc<() -> ()>, i64> {fir.char_proc}) {
! CHECK:  %[[VAL_1:.*]] = fir.extract_value %[[VAL_0]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:  %[[VAL_3:.*]] = arith.constant 10 : i64
! CHECK:  %[[VAL_4:.*]] = fir.emboxproc %[[VAL_2]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_5:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_6:.*]] = fir.insert_value %[[VAL_5]], %[[VAL_4]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_7:.*]] = fir.insert_value %[[VAL_6]], %[[VAL_3]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:  %[[VAL_8:.*]] = fir.extract_value %[[VAL_7]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_9:.*]] = fir.is_present %[[VAL_8]] : (!fir.boxproc<() -> ()>) -> i1

end module
