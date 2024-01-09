! Test ASSOCIATED() with procedure pointers.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine test_proc_pointer_1(p, dummy_proc)
  procedure(), pointer :: p
  procedure() :: dummy_proc
  call takes_log(associated(p, dummy_proc))
end subroutine
! CHECK-LABEL:   func.func @_QPtest_proc_pointer_1(
! CHECK-SAME:                                      %[[VAL_0:.*]]: !fir.ref<!fir.boxproc<() -> ()>>,
! CHECK-SAME:                                      %[[VAL_1:.*]]: !fir.boxproc<() -> ()>) {
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_proc_pointer_1Ep"} : (!fir.ref<!fir.boxproc<() -> ()>>) -> (!fir.ref<!fir.boxproc<() -> ()>>, !fir.ref<!fir.boxproc<() -> ()>>)
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_2]]#1 : !fir.ref<!fir.boxproc<() -> ()>>
! CHECK:           %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:           %[[VAL_5:.*]] = fir.box_addr %[[VAL_1]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_4]] : (() -> ()) -> i64
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_5]] : (() -> ()) -> i64
! CHECK:           %[[VAL_8:.*]] = arith.cmpi eq, %[[VAL_6]], %[[VAL_7]] : i64
! CHECK:           %[[VAL_9:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_10:.*]] = arith.cmpi ne, %[[VAL_9]], %[[VAL_6]] : i64
! CHECK:           %[[VAL_11:.*]] = arith.andi %[[VAL_8]], %[[VAL_10]] : i1
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i1) -> !fir.logical<4>

subroutine test_proc_pointer_2(p, p_target)
  procedure(), pointer :: p, p_target
  call takes_log(associated(p, p_target))
end subroutine
! CHECK-LABEL:   func.func @_QPtest_proc_pointer_2(
! CHECK-SAME:                                      %[[VAL_0:.*]]: !fir.ref<!fir.boxproc<() -> ()>>,
! CHECK-SAME:                                      %[[VAL_1:.*]]: !fir.ref<!fir.boxproc<() -> ()>>) {
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_proc_pointer_2Ep"} : (!fir.ref<!fir.boxproc<() -> ()>>) -> (!fir.ref<!fir.boxproc<() -> ()>>, !fir.ref<!fir.boxproc<() -> ()>>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_proc_pointer_2Ep_target"} : (!fir.ref<!fir.boxproc<() -> ()>>) -> (!fir.ref<!fir.boxproc<() -> ()>>, !fir.ref<!fir.boxproc<() -> ()>>)
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_2]]#1 : !fir.ref<!fir.boxproc<() -> ()>>
! CHECK:           %[[VAL_5:.*]] = fir.box_addr %[[VAL_4]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_3]]#1 : !fir.ref<!fir.boxproc<() -> ()>>
! CHECK:           %[[VAL_7:.*]] = fir.box_addr %[[VAL_6]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_5]] : (() -> ()) -> i64
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_7]] : (() -> ()) -> i64
! CHECK:           %[[VAL_10:.*]] = arith.cmpi eq, %[[VAL_8]], %[[VAL_9]] : i64
! CHECK:           %[[VAL_11:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_12:.*]] = arith.cmpi ne, %[[VAL_11]], %[[VAL_8]] : i64
! CHECK:           %[[VAL_13:.*]] = arith.andi %[[VAL_10]], %[[VAL_12]] : i1
! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i1) -> !fir.logical<4>

subroutine test_proc_pointer_3(p, dummy_proc)
  procedure(), pointer :: p
  procedure(), optional :: dummy_proc
  call takes_log(associated(p, dummy_proc))
end subroutine
! CHECK-LABEL:   func.func @_QPtest_proc_pointer_3(
! CHECK-SAME:                                      %[[VAL_0:.*]]: !fir.ref<!fir.boxproc<() -> ()>>,
! CHECK-SAME:                                      %[[VAL_1:.*]]: !fir.boxproc<() -> ()>) {
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_proc_pointer_3Ep"} : (!fir.ref<!fir.boxproc<() -> ()>>) -> (!fir.ref<!fir.boxproc<() -> ()>>, !fir.ref<!fir.boxproc<() -> ()>>)
! CHECK:           %[[VAL_3:.*]] = fir.load %[[VAL_2]]#1 : !fir.ref<!fir.boxproc<() -> ()>>
! CHECK:           %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:           %[[VAL_5:.*]] = fir.box_addr %[[VAL_1]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:           %[[VAL_6:.*]] = fir.convert %[[VAL_4]] : (() -> ()) -> i64
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_5]] : (() -> ()) -> i64
! CHECK:           %[[VAL_8:.*]] = arith.cmpi eq, %[[VAL_6]], %[[VAL_7]] : i64
! CHECK:           %[[VAL_9:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_10:.*]] = arith.cmpi ne, %[[VAL_9]], %[[VAL_6]] : i64
! CHECK:           %[[VAL_11:.*]] = arith.andi %[[VAL_8]], %[[VAL_10]] : i1
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i1) -> !fir.logical<4>

subroutine test_proc_pointer_4(p)
  procedure(), pointer :: p
  external :: some_external
  call takes_log(associated(p, some_external))
end subroutine
! CHECK-LABEL:   func.func @_QPtest_proc_pointer_4(
! CHECK-SAME:                                      %[[VAL_0:.*]]: !fir.ref<!fir.boxproc<() -> ()>>) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_proc_pointer_4Ep"} : (!fir.ref<!fir.boxproc<() -> ()>>) -> (!fir.ref<!fir.boxproc<() -> ()>>, !fir.ref<!fir.boxproc<() -> ()>>)
! CHECK:           %[[VAL_2:.*]] = fir.address_of(@_QPsome_external) : () -> ()
! CHECK:           %[[VAL_3:.*]] = fir.emboxproc %[[VAL_2]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_1]]#1 : !fir.ref<!fir.boxproc<() -> ()>>
! CHECK:           %[[VAL_5:.*]] = fir.box_addr %[[VAL_4]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:           %[[VAL_6:.*]] = fir.box_addr %[[VAL_3]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_5]] : (() -> ()) -> i64
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_6]] : (() -> ()) -> i64
! CHECK:           %[[VAL_9:.*]] = arith.cmpi eq, %[[VAL_7]], %[[VAL_8]] : i64
! CHECK:           %[[VAL_10:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_11:.*]] = arith.cmpi ne, %[[VAL_10]], %[[VAL_7]] : i64
! CHECK:           %[[VAL_12:.*]] = arith.andi %[[VAL_9]], %[[VAL_11]] : i1
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i1) -> !fir.logical<4>

subroutine test_proc_pointer_5(p, dummy_proc)
  interface
    character(10) function char_func()
    end function
  end interface
  procedure(char_func), pointer :: p
  procedure(char_func) :: dummy_proc
  call takes_log(associated(p, dummy_proc))
end subroutine
! CHECK-LABEL:   func.func @_QPtest_proc_pointer_5(
! CHECK-SAME:                                      %[[VAL_0:.*]]: !fir.ref<!fir.boxproc<() -> ()>>,
! CHECK-SAME:                                      %[[VAL_1:.*]]: tuple<!fir.boxproc<() -> ()>, i64> {fir.char_proc}) {
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_proc_pointer_5Ep"} : (!fir.ref<!fir.boxproc<() -> ()>>) -> (!fir.ref<!fir.boxproc<() -> ()>>, !fir.ref<!fir.boxproc<() -> ()>>)
! CHECK:           %[[VAL_3:.*]] = fir.extract_value %[[VAL_1]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:           %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:           %[[VAL_5:.*]] = arith.constant 10 : i64
! CHECK:           %[[VAL_6:.*]] = fir.emboxproc %[[VAL_4]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK:           %[[VAL_7:.*]] = fir.undefined tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:           %[[VAL_8:.*]] = fir.insert_value %[[VAL_7]], %[[VAL_6]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, !fir.boxproc<() -> ()>) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:           %[[VAL_9:.*]] = fir.insert_value %[[VAL_8]], %[[VAL_5]], [1 : index] : (tuple<!fir.boxproc<() -> ()>, i64>, i64) -> tuple<!fir.boxproc<() -> ()>, i64>
! CHECK:           %[[VAL_10:.*]] = fir.extract_value %[[VAL_9]], [0 : index] : (tuple<!fir.boxproc<() -> ()>, i64>) -> !fir.boxproc<() -> ()>
! CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_2]]#1 : !fir.ref<!fir.boxproc<() -> ()>>
! CHECK:           %[[VAL_12:.*]] = fir.box_addr %[[VAL_11]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:           %[[VAL_13:.*]] = fir.box_addr %[[VAL_10]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (() -> ()) -> i64
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_13]] : (() -> ()) -> i64
! CHECK:           %[[VAL_16:.*]] = arith.cmpi eq, %[[VAL_14]], %[[VAL_15]] : i64
! CHECK:           %[[VAL_17:.*]] = arith.constant 0 : i64
! CHECK:           %[[VAL_18:.*]] = arith.cmpi ne, %[[VAL_17]], %[[VAL_14]] : i64
! CHECK:           %[[VAL_19:.*]] = arith.andi %[[VAL_16]], %[[VAL_18]] : i1
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i1) -> !fir.logical<4>
