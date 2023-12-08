// Test rewrite of character procedure pointer tuple argument to two different
// arguments: one for the function address, and one for the length. The length
// argument is added after other characters.
// RUN: fir-opt --target-rewrite="target=x86_64-unknown-linux-gnu" %s | FileCheck %s

// CHECK:  func private @takes_char_proc(() -> () {fir.char_proc}, i64)
func.func private @takes_char_proc(tuple<() -> (), i64> {fir.char_proc})

func.func private @takes_char(!fir.boxchar<1>)
func.func private @char_proc(!fir.ref<!fir.char<1,7>>, index) -> !fir.boxchar<1>

func.func @_QPcst_len() {
  %0 = fir.address_of(@char_proc) : (!fir.ref<!fir.char<1,7>>, index) -> !fir.boxchar<1>
  %c7_i64 = arith.constant 7 : i64
  %1 = fir.convert %0 : ((!fir.ref<!fir.char<1,7>>, index) -> !fir.boxchar<1>) -> (() -> ())
  %2 = fir.undefined tuple<() -> (), i64>
  %3 = fir.insert_value %2, %1, [0 : index] : (tuple<() -> (), i64>, () -> ()) -> tuple<() -> (), i64>
  %4 = fir.insert_value %3, %c7_i64, [1 : index] : (tuple<() -> (), i64>, i64) -> tuple<() -> (), i64>

  // CHECK:  %[[PROC_ADDR:.*]] = fir.extract_value %{{.*}}, [0 : index] : (tuple<() -> (), i64>) -> (() -> ())
  // CHECK:  %[[LEN:.*]] = fir.extract_value %{{.*}}, [1 : index] : (tuple<() -> (), i64>) -> i64
  // CHECK:  fir.call @takes_char_proc(%[[PROC_ADDR]], %[[LEN]]) : (() -> (), i64) -> ()
  fir.call @takes_char_proc(%4) : (tuple<() -> (), i64>) -> ()
  return
}

// CHECK:  func @test_dummy_proc_that_takes_dummy_char_proc(
// CHECK-SAME: %[[ARG0:.*]]: () -> ()) {
func.func @test_dummy_proc_that_takes_dummy_char_proc(%arg0: () -> ()) {
  %0 = fir.address_of(@char_proc) : (!fir.ref<!fir.char<1,7>>, index) -> !fir.boxchar<1>
  %c7_i64 = arith.constant 7 : i64
  %1 = fir.convert %0 : ((!fir.ref<!fir.char<1,7>>, index) -> !fir.boxchar<1>) -> (() -> ())
  %2 = fir.undefined tuple<() -> (), i64>
  %3 = fir.insert_value %2, %1, [0 : index] : (tuple<() -> (), i64>, () -> ()) -> tuple<() -> (), i64>
  %4 = fir.insert_value %3, %c7_i64, [1 : index] : (tuple<() -> (), i64>, i64) -> tuple<() -> (), i64>
  %5 = fir.convert %arg0 : (() -> ()) -> ((tuple<() -> (), i64>) -> ())

  // CHECK:  %[[ARG_CAST:.*]] = fir.convert %[[ARG0]] : (() -> ()) -> ((() -> (), i64) -> ())
  // CHECK:  %[[PROC_ADDR:.*]] = fir.extract_value %4, [0 : index] : (tuple<() -> (), i64>) -> (() -> ())
  // CHECK:  %[[PROC_LEN:.*]] = fir.extract_value %4, [1 : index] : (tuple<() -> (), i64>) -> i64
  // CHECK: fir.call %[[ARG_CAST]](%[[PROC_ADDR]], %[[PROC_LEN]]) : (() -> (), i64) -> ()
  fir.call %5(%4) : (tuple<() -> (), i64>) -> ()
  return
}

// CHECK:  func @takes_dummy_char_proc_impl(
// CHECK-SAME: %[[PROC_ADDR:.*]]: () -> () {fir.char_proc},
// CHECK-SAME: %[[C_ADDR:.*]]: !fir.ref<!fir.char<1,?>>,
// CHECK-SAME: %[[PROC_LEN:.*]]: i64,
// CHECK-SAME: %[[C_LEN:.*]]: i64) {
func.func @takes_dummy_char_proc_impl(%arg0: tuple<() -> (), i64> {fir.char_proc}, %arg1: !fir.boxchar<1>) {
  // CHECK:    %[[UNDEF:.*]] = fir.undefined tuple<() -> (), i64>
  // CHECK:    %[[TUPLE0:.*]] = fir.insert_value %[[UNDEF]], %[[PROC_ADDR]], [0 : index] : (tuple<() -> (), i64>, () -> ()) -> tuple<() -> (), i64>
  // CHECK:    %[[TUPLE1:.*]] = fir.insert_value %[[TUPLE0]], %[[PROC_LEN]], [1 : index] : (tuple<() -> (), i64>, i64) -> tuple<() -> (), i64>
  %0 = fir.alloca !fir.char<1,7> {bindc_name = ".result"}
  %1:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %c5 = arith.constant 5 : index
  %2 = fir.emboxchar %1#0, %c5 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  %3 = fir.extract_value %arg0, [0 : index] : (tuple<() -> (), i64>) -> (() -> ())
  %c7_i64 = arith.constant 7 : i64
  %4 = fir.convert %c7_i64 : (i64) -> index
  %6 = fir.convert %3 : (() -> ()) -> ((!fir.ref<!fir.char<1,7>>, index, !fir.boxchar<1>) -> !fir.boxchar<1>)
  %7 = fir.call %6(%0, %4, %2) : (!fir.ref<!fir.char<1,7>>, index, !fir.boxchar<1>) -> !fir.boxchar<1>
  %8 = fir.convert %0 : (!fir.ref<!fir.char<1,7>>) -> !fir.ref<!fir.char<1,?>>
  %9 = fir.emboxchar %8, %4 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  fir.call @takes_char(%9) : (!fir.boxchar<1>) -> ()
  return
}

