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
  %c1 = arith.constant 1 : index
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1:2 = hlfir.declare %0#0 typeparams %0#1 {uniq_name = "x"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %2 = arith.addi %0#1, %c1 : index
  %3 = fir.undefined !fir.ref<!fir.char<1>>
  %4:2 = hlfir.declare %3 typeparams %c1 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "char_literal"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
  %5 = hlfir.concat %1#0, %4#0 len %2 : (!fir.boxchar<1>, !fir.ref<!fir.char<1>>, index) -> !hlfir.expr<!fir.char<1,?>>
  %6 = hlfir.no_reassoc %5 : !hlfir.expr<!fir.char<1,?>>
  %7:3 = hlfir.associate %6 typeparams %2 {uniq_name = "x"} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
  fir.call @take_c(%7#0) : (!fir.boxchar<1>) -> ()
  hlfir.end_associate %7#1, %7#2 : !fir.ref<!fir.char<1,?>>, i1
  return
}

// CHECK-LABEL:   func.func @associate_char(
// CHECK-SAME:                              %[[VAL_0:.*]]: !fir.boxchar<1>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]]#0 typeparams %[[VAL_2]]#1 {uniq_name = "x"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_2]]#1, %[[VAL_1]] : index
// CHECK:           %[[VAL_5:.*]] = fir.undefined !fir.ref<!fir.char<1>>
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] typeparams %[[VAL_1]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "char_literal"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
// CHECK:           %[[VAL_7:.*]] = arith.addi %[[VAL_2]]#1, %[[VAL_1]] : index
// CHECK:           %[[VAL_8:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_7]] : index) {bindc_name = ".chrtmp"}
// CHECK:           %[[VAL_9:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_2]]#1 : (index) -> i64
// CHECK:           %[[VAL_11:.*]] = arith.muli %[[VAL_9]], %[[VAL_10]] : i64
// CHECK:           %[[VAL_12:.*]] = arith.constant false
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_3]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:           fir.call @llvm.memmove.p0.p0.i64(%[[VAL_13]], %[[VAL_14]], %[[VAL_11]], %[[VAL_12]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
// CHECK:           %[[VAL_15:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_16:.*]] = arith.subi %[[VAL_7]], %[[VAL_15]] : index
// CHECK:           fir.do_loop %[[VAL_17:.*]] = %[[VAL_2]]#1 to %[[VAL_16]] step %[[VAL_15]] {
// CHECK:             %[[VAL_18:.*]] = arith.subi %[[VAL_17]], %[[VAL_2]]#1 : index
// CHECK:             %[[VAL_19:.*]] = fir.convert %[[VAL_6]]#1 : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.array<1x!fir.char<1>>>
// CHECK:             %[[VAL_20:.*]] = fir.coordinate_of %[[VAL_19]], %[[VAL_18]] : (!fir.ref<!fir.array<1x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:             %[[VAL_21:.*]] = fir.load %[[VAL_20]] : !fir.ref<!fir.char<1>>
// CHECK:             %[[VAL_22:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:             %[[VAL_23:.*]] = fir.coordinate_of %[[VAL_22]], %[[VAL_17]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:             fir.store %[[VAL_21]] to %[[VAL_23]] : !fir.ref<!fir.char<1>>
// CHECK:           }
// CHECK:           %[[VAL_24:.*]]:2 = hlfir.declare %[[VAL_8]] typeparams %[[VAL_7]] {uniq_name = "tmp"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           %[[VAL_25:.*]] = arith.constant false
// CHECK:           %[[VAL_26:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[VAL_27:.*]] = fir.insert_value %[[VAL_26]], %[[VAL_25]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[VAL_28:.*]] = fir.insert_value %[[VAL_27]], %[[VAL_24]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[VAL_29:.*]] = hlfir.no_reassoc %[[VAL_24]]#0 : !fir.boxchar<1>
// CHECK:           %[[VAL_30:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[VAL_31:.*]] = fir.insert_value %[[VAL_30]], %[[VAL_25]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[VAL_32:.*]] = fir.insert_value %[[VAL_31]], %[[VAL_29]], [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[VAL_33:.*]] = fir.box_addr %[[VAL_29]] : (!fir.boxchar<1>) -> !fir.ref<!fir.char<1,?>>
// CHECK:           fir.call @take_c(%[[VAL_29]]) : (!fir.boxchar<1>) -> ()
// CHECK:           return
// CHECK:         }

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
