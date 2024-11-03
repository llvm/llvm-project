// Test hlfir.associate/hlfir.end_associate operation code generation to FIR.

// RUN: fir-opt %s -bufferize-hlfir | FileCheck %s

func.func @associate_int() {
  %c42_i32 = arith.constant 42 : i32
  %0:3 = hlfir.associate %c42_i32 {uniq_name = "x"} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
  fir.call @take_i4(%0#0) : (!fir.ref<i32>) -> ()
  hlfir.end_associate %0#1, %0#2 : !fir.ref<i32>, i1
  return
}
// CHECK-LABEL:   func.func @associate_int() {
// CHECK:  %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "x"}
// CHECK:  %[[VAL_1:.*]] = arith.constant 42 : i32
// CHECK:  fir.store %[[VAL_1]] to %[[VAL_0]] : !fir.ref<i32>
// CHECK:  %[[VAL_2:.*]] = arith.constant false
// CHECK:  fir.call @take_i4(%[[VAL_0]]) : (!fir.ref<i32>) -> ()
// CHECK-NOT: fir.freemem


func.func @associate_real() {
  %cst = arith.constant 4.200000e-01 : f32
  %0:3 = hlfir.associate %cst {uniq_name = "x"} : (f32) -> (!fir.ref<f32>, !fir.ref<f32>, i1)
  fir.call @take_r4(%0#0) : (!fir.ref<f32>) -> ()
  hlfir.end_associate %0#1, %0#2 : !fir.ref<f32>, i1
  return
}
// CHECK-LABEL:   func.func @associate_real() {
// CHECK:  %[[VAL_0:.*]] = fir.alloca f32 {bindc_name = "x"}
// CHECK:  %[[VAL_1:.*]] = arith.constant 4.200000e-01 : f32
// CHECK:  fir.store %[[VAL_1]] to %[[VAL_0]] : !fir.ref<f32>
// CHECK:  %[[VAL_2:.*]] = arith.constant false
// CHECK:  fir.call @take_r4(%[[VAL_0]]) : (!fir.ref<f32>) -> ()
// CHECK-NOT: fir.freemem


func.func @associate_logical() {
  %true = arith.constant true
  %0 = fir.convert %true : (i1) -> !fir.logical<4>
  %1:3 = hlfir.associate %0 {uniq_name = "x"} : (!fir.logical<4>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>, i1)
  fir.call @take_l4(%1#0) : (!fir.ref<!fir.logical<4>>) -> ()
  hlfir.end_associate %1#1, %1#2 : !fir.ref<!fir.logical<4>>, i1
  return
}
// CHECK-LABEL:   func.func @associate_logical() {
// CHECK:  %[[VAL_0:.*]] = fir.alloca !fir.logical<4> {bindc_name = "x"}
// CHECK:  %[[VAL_1:.*]] = arith.constant true
// CHECK:  %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (i1) -> !fir.logical<4>
// CHECK:  fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.logical<4>>
// CHECK:  %[[VAL_3:.*]] = arith.constant false
// CHECK:  fir.call @take_l4(%[[VAL_0]]) : (!fir.ref<!fir.logical<4>>) -> ()
// CHECK-NOT: fir.freemem


func.func @associate_char(%arg0: !fir.boxchar<1> ) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1:2 = hlfir.declare %0#0 typeparams %0#1 {uniq_name = "x"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %2 = arith.addi %0#1, %0#1 : index
  %3 = hlfir.concat %1#0, %1#0 len %2 : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
  %4:3 = hlfir.associate %3 typeparams %2 {uniq_name = "x"} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  fir.call @take_c(%4#0) : (!fir.boxchar<1>) -> ()
  hlfir.end_associate %4#1, %4#2 : !fir.ref<!fir.char<1,?>>, i1
  return
}
// CHECK-LABEL:   func.func @associate_char(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxchar<1>) {
// CHECK:  %[[VAL_1:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]]#0 typeparams %[[VAL_1]]#1 {uniq_name = "x"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:  %[[VAL_3:.*]] = arith.addi %[[VAL_1]]#1, %[[VAL_1]]#1 : index
// CHECK:  %[[VAL_4:.*]] = arith.addi %[[VAL_1]]#1, %[[VAL_1]]#1 : index
// CHECK:  %[[VAL_5:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_4]] : index) {bindc_name = ".chrtmp"}
// CHECK:  fir.call @llvm.memmove.p0.p0.i64
// CHECK:  %[[VAL_21:.*]]:2 = hlfir.declare %[[VAL_5]] typeparams %[[VAL_4]] {uniq_name = "tmp"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:  %[[VAL_22:.*]] = arith.constant false
// CHECK:  %[[VAL_23:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:  %[[VAL_24:.*]] = fir.insert_value %[[VAL_23]], %[[VAL_22]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:  %[[VAL_25:.*]] = fir.insert_value %[[VAL_24]], %[[VAL_21]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:  fir.call @take_c(%[[VAL_21]]#0) : (!fir.boxchar<1>) -> ()
// CHECK-NOT: fir.freemem


func.func @test_end_associate_box(%var: !fir.box<!fir.array<?xf64>>) {
  %true = arith.constant 1 : i1
  hlfir.end_associate %var, %true : !fir.box<!fir.array<?xf64>>, i1
  return
}
// CHECK-LABEL:   func.func @test_end_associate_box(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf64>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant true
// CHECK:  %[[VAL_2:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?xf64>>) -> !fir.heap<!fir.array<?xf64>>
// CHECK:  fir.freemem %[[VAL_2]] : !fir.heap<!fir.array<?xf64>>


func.func @test_end_associate_boxchar(%var: !fir.boxchar<2>) {
  %true = arith.constant 1 : i1
  hlfir.end_associate %var, %true : !fir.boxchar<2>, i1
  return
}
// CHECK-LABEL:   func.func @test_end_associate_boxchar(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxchar<2>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant true
// CHECK:  %[[VAL_2:.*]] = fir.box_addr %[[VAL_0]] : (!fir.boxchar<2>) -> !fir.heap<!fir.char<2,?>>
// CHECK:  fir.freemem %[[VAL_2]] : !fir.heap<!fir.char<2,?>>


func.func @test_end_associate_box_dynamic(%var: !fir.box<!fir.array<?xf64>>, %must_free: i1) {
  hlfir.end_associate %var, %must_free : !fir.box<!fir.array<?xf64>>, i1
  return
}
// CHECK-LABEL:   func.func @test_end_associate_box_dynamic(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xf64>>,
// CHECK-SAME:    %[[VAL_1:.*]]: i1) {
// CHECK:  fir.if %[[VAL_1]] {
// CHECK:    %[[VAL_2:.*]] = fir.box_addr %[[VAL_0]] : (!fir.box<!fir.array<?xf64>>) -> !fir.heap<!fir.array<?xf64>>
// CHECK:    fir.freemem %[[VAL_2]] : !fir.heap<!fir.array<?xf64>>
// CHECK:  }

func.func private @bar(!fir.ref<!fir.array<?xi32>>) -> ()
func.func @test_result_box_addr(%x : !fir.box<!fir.array<?xi32>>) {
  %true = arith.constant 1 : i1
  %expr = hlfir.as_expr %x move %true : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
  %y:3 = hlfir.associate %expr {uniq_name = "y"}: (!hlfir.expr<?xi32>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1)
  fir.call @bar(%y#1) : (!fir.ref<!fir.array<?xi32>>) -> ()
  return
}
// CHECK-LABEL: func.func @test_result_box_addr(
// CHECK-SAME: %[[X:.*]]: !fir.box<!fir.array<?xi32>>) {
// CHECK:  %[[ADDR:.*]] = fir.box_addr %[[X]] : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
// CHECK: fir.call @bar(%[[ADDR]]) : (!fir.ref<!fir.array<?xi32>>) -> ()

func.func private @bar2(!fir.ref<!fir.array<10xi32>>) -> ()
func.func @test_result_convert(%x : !fir.heap<!fir.array<10xi32>>) {
  %true = arith.constant 1 : i1
  %expr = hlfir.as_expr %x move %true : (!fir.heap<!fir.array<10xi32>>, i1) -> !hlfir.expr<10xi32>
  %y:3 = hlfir.associate %expr {uniq_name = "y"}: (!hlfir.expr<10xi32>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>, i1)
  fir.call @bar2(%y#1) : (!fir.ref<!fir.array<10xi32>>) -> ()
  return
}
// CHECK-LABEL: func.func @test_result_convert(
// CHECK-SAME: %[[X:.*]]: !fir.heap<!fir.array<10xi32>>) {
// CHECK: fir.convert
// CHECK:  %[[ADDR:.*]] = fir.convert %[[X]] : (!fir.heap<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
// CHECK: fir.call @bar2(%[[ADDR]]) : (!fir.ref<!fir.array<10xi32>>) -> ()


func.func private @take_i4(!fir.ref<i32>)
func.func private @take_r4(!fir.ref<f32>)
func.func private @take_l4(!fir.ref<!fir.logical<4>>)
func.func private @take_c(!fir.boxchar<1>)
