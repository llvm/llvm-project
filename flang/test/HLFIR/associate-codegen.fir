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


func.func @test_0dim_box(%x : !fir.ref<!fir.box<!fir.heap<i32>>>) {
  %0 = fir.load %x : !fir.ref<!fir.box<!fir.heap<i32>>>
  %1:2 = hlfir.declare %0 {uniq_name = ".tmp.intrinsic_result"} : (!fir.box<!fir.heap<i32>>) -> (!fir.box<!fir.heap<i32>>, !fir.box<!fir.heap<i32>>)
  %true = arith.constant true
  %2 = hlfir.as_expr %1#0 move %true : (!fir.box<!fir.heap<i32>>, i1) -> !hlfir.expr<i32>
  %3:3 = hlfir.associate %2 {adapt.valuebyref} : (!hlfir.expr<i32>) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
  return
}
// CHECK-LABEL:   func.func @test_0dim_box(
// CHECK-SAME:                             %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>>) {
// CHECK:           %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<i32>>>
// CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = ".tmp.intrinsic_result"} : (!fir.box<!fir.heap<i32>>) -> (!fir.box<!fir.heap<i32>>, !fir.box<!fir.heap<i32>>)
// CHECK:           %[[VAL_3:.*]] = arith.constant true
// CHECK:           %[[VAL_4:.*]] = fir.undefined tuple<!fir.box<!fir.heap<i32>>, i1>
// CHECK:           %[[VAL_5:.*]] = fir.insert_value %[[VAL_4]], %[[VAL_3]], [1 : index] : (tuple<!fir.box<!fir.heap<i32>>, i1>, i1) -> tuple<!fir.box<!fir.heap<i32>>, i1>
// CHECK:           %[[VAL_6:.*]] = fir.insert_value %[[VAL_5]], %[[VAL_2]]#0, [0 : index] : (tuple<!fir.box<!fir.heap<i32>>, i1>, !fir.box<!fir.heap<i32>>) -> tuple<!fir.box<!fir.heap<i32>>, i1>
// CHECK:           %[[VAL_7:.*]] = fir.box_addr %[[VAL_2]]#0 : (!fir.box<!fir.heap<i32>>) -> !fir.ref<i32>
// CHECK:           %[[VAL_8:.*]] = fir.box_addr %[[VAL_2]]#1 : (!fir.box<!fir.heap<i32>>) -> !fir.ref<i32>
// CHECK:           return
// CHECK:         }

// test that we support a hlfir.associate operation where the expr is also used in a hlfir.shape_of op
func.func @test_shape_of(%arg0: !fir.ref<!fir.array<4x3xi32>>) {
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %shape = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  // %0 = hlfir.transpose %arg0 : (!fir.ref<!fir.array<4x3xi32>>) -> !hlfir.expr<3x4xi32>
  %0 = hlfir.elemental %shape unordered : (!fir.shape<2>) -> !hlfir.expr<3x4xi32> {
  ^bb0(%arg1: index, %arg2: index):
    %4 = hlfir.designate %arg0 (%arg2, %arg1) : (!fir.ref<!fir.array<4x3xi32>>, index, index) -> !fir.ref<i32>
    %5 = fir.load %4 : !fir.ref<i32>
    hlfir.yield_element %5 : i32
  }
  %1 = hlfir.shape_of %0 : (!hlfir.expr<3x4xi32>) -> !fir.shape<2>
  %2:3 = hlfir.associate %0(%1) {adapt.valuebyref} : (!hlfir.expr<3x4xi32>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x4xi32>>, !fir.ref<!fir.array<3x4xi32>>, i1)
  // ...
  hlfir.end_associate %2#1, %2#2 : !fir.ref<!fir.array<3x4xi32>>, i1
  return
}
// CHECK-LABEL:   func.func @test_shape_of(
// CHECK-SAME:                             %[[VAL_0:.*]]: !fir.ref<!fir.array<4x3xi32>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 3 : index
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]], %[[VAL_1]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_4:.*]] = fir.allocmem !fir.array<3x4xi32> {bindc_name = ".tmp.array", uniq_name = ""}
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]](%[[VAL_3]]) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<3x4xi32>>, !fir.shape<2>) -> (!fir.heap<!fir.array<3x4xi32>>, !fir.heap<!fir.array<3x4xi32>>)
// CHECK:           %[[VAL_6:.*]] = arith.constant true
// CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_8:.*]] = %[[VAL_7]] to %[[VAL_1]] step %[[VAL_7]] unordered {
// CHECK:             fir.do_loop %[[VAL_9:.*]] = %[[VAL_7]] to %[[VAL_2]] step %[[VAL_7]] unordered {
// CHECK:               %[[VAL_10:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_8]], %[[VAL_9]])  : (!fir.ref<!fir.array<4x3xi32>>, index, index) -> !fir.ref<i32>
// CHECK:               %[[VAL_11:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
// CHECK:               %[[VAL_12:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_9]], %[[VAL_8]])  : (!fir.heap<!fir.array<3x4xi32>>, index, index) -> !fir.ref<i32>
// CHECK:               hlfir.assign %[[VAL_11]] to %[[VAL_12]] temporary_lhs : i32, !fir.ref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_13:.*]] = fir.undefined tuple<!fir.heap<!fir.array<3x4xi32>>, i1>
// CHECK:           %[[VAL_14:.*]] = fir.insert_value %[[VAL_13]], %[[VAL_6]], [1 : index] : (tuple<!fir.heap<!fir.array<3x4xi32>>, i1>, i1) -> tuple<!fir.heap<!fir.array<3x4xi32>>, i1>
// CHECK:           %[[VAL_15:.*]] = fir.insert_value %[[VAL_14]], %[[VAL_5]]#0, [0 : index] : (tuple<!fir.heap<!fir.array<3x4xi32>>, i1>, !fir.heap<!fir.array<3x4xi32>>) -> tuple<!fir.heap<!fir.array<3x4xi32>>, i1>
// CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_5]]#0 : (!fir.heap<!fir.array<3x4xi32>>) -> !fir.ref<!fir.array<3x4xi32>>
// CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_5]]#1 : (!fir.heap<!fir.array<3x4xi32>>) -> !fir.ref<!fir.array<3x4xi32>>
// CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (!fir.ref<!fir.array<3x4xi32>>) -> !fir.heap<!fir.array<3x4xi32>>
// CHECK:           fir.freemem %[[VAL_18]] : !fir.heap<!fir.array<3x4xi32>>
// CHECK:           return
// CHECK:         }

func.func @test_multiple_associations(%arg0: !hlfir.expr<1x2xi32>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %shape = fir.shape %c1, %c2 : (index, index) -> !fir.shape<2>
  %0:3 = hlfir.associate %arg0(%shape) {uniq_name = "associate 0"} : (!hlfir.expr<1x2xi32>, !fir.shape<2>) -> (!fir.ref<!fir.array<1x2xi32>>, !fir.ref<!fir.array<1x2xi32>>, i1)
  %1:3 = hlfir.associate %arg0(%shape) {uniq_name = "associate 1"} : (!hlfir.expr<1x2xi32>, !fir.shape<2>) -> (!fir.ref<!fir.array<1x2xi32>>, !fir.ref<!fir.array<1x2xi32>>, i1)
  hlfir.end_associate %0#1, %0#2 : !fir.ref<!fir.array<1x2xi32>>, i1
  hlfir.end_associate %1#1, %1#2 : !fir.ref<!fir.array<1x2xi32>>, i1
  return
}
// CHECK-LABEL:   func.func @test_multiple_associations(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !hlfir.expr<1x2xi32>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
// associate 0:
// CHECK:           %[[VAL_4:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<1x2xi32>) -> !fir.shape<2>
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_7:.*]] = fir.allocmem !fir.array<1x2xi32> {bindc_name = ".tmp", uniq_name = ""}
// CHECK:           %[[VAL_8:.*]] = arith.constant true
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_7]](%[[VAL_4]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<1x2xi32>>, !fir.shape<2>) -> (!fir.heap<!fir.array<1x2xi32>>, !fir.heap<!fir.array<1x2xi32>>)
// CHECK:           hlfir.assign %[[VAL_0]] to %[[VAL_9]]#0 temporary_lhs : !hlfir.expr<1x2xi32>, !fir.heap<!fir.array<1x2xi32>>
// CHECK:           %[[VAL_10:.*]] = fir.undefined tuple<!fir.heap<!fir.array<1x2xi32>>, i1>
// CHECK:           %[[VAL_11:.*]] = fir.insert_value %[[VAL_10]], %[[VAL_8]], [1 : index] : (tuple<!fir.heap<!fir.array<1x2xi32>>, i1>, i1) -> tuple<!fir.heap<!fir.array<1x2xi32>>, i1>
// CHECK:           %[[VAL_12:.*]] = fir.insert_value %[[VAL_11]], %[[VAL_9]]#0, [0 : index] : (tuple<!fir.heap<!fir.array<1x2xi32>>, i1>, !fir.heap<!fir.array<1x2xi32>>) -> tuple<!fir.heap<!fir.array<1x2xi32>>, i1>
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_9]]#0 : (!fir.heap<!fir.array<1x2xi32>>) -> !fir.ref<!fir.array<1x2xi32>>
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_9]]#1 : (!fir.heap<!fir.array<1x2xi32>>) -> !fir.ref<!fir.array<1x2xi32>>
// associate 1:
// CHECK:           %[[VAL_15:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<1x2xi32>) -> !fir.shape<2>
// CHECK:           %[[VAL_16:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_17:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_18:.*]] = fir.allocmem !fir.array<1x2xi32> {bindc_name = ".tmp", uniq_name = ""}
// CHECK:           %[[VAL_19:.*]] = arith.constant true
// CHECK:           %[[VAL_20:.*]]:2 = hlfir.declare %[[VAL_18]](%[[VAL_15]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<1x2xi32>>, !fir.shape<2>) -> (!fir.heap<!fir.array<1x2xi32>>, !fir.heap<!fir.array<1x2xi32>>)
// CHECK:           hlfir.assign %[[VAL_0]] to %[[VAL_20]]#0 temporary_lhs : !hlfir.expr<1x2xi32>, !fir.heap<!fir.array<1x2xi32>>
// CHECK:           %[[VAL_21:.*]] = fir.undefined tuple<!fir.heap<!fir.array<1x2xi32>>, i1>
// CHECK:           %[[VAL_22:.*]] = fir.insert_value %[[VAL_21]], %[[VAL_19]], [1 : index] : (tuple<!fir.heap<!fir.array<1x2xi32>>, i1>, i1) -> tuple<!fir.heap<!fir.array<1x2xi32>>, i1>
// CHECK:           %[[VAL_23:.*]] = fir.insert_value %[[VAL_22]], %[[VAL_20]]#0, [0 : index] : (tuple<!fir.heap<!fir.array<1x2xi32>>, i1>, !fir.heap<!fir.array<1x2xi32>>) -> tuple<!fir.heap<!fir.array<1x2xi32>>, i1>
// CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_20]]#0 : (!fir.heap<!fir.array<1x2xi32>>) -> !fir.ref<!fir.array<1x2xi32>>
// CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_20]]#1 : (!fir.heap<!fir.array<1x2xi32>>) -> !fir.ref<!fir.array<1x2xi32>>
// end associate 0:
// CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_14]] : (!fir.ref<!fir.array<1x2xi32>>) -> !fir.heap<!fir.array<1x2xi32>>
// CHECK:           fir.freemem %[[VAL_26]] : !fir.heap<!fir.array<1x2xi32>>
// end associate 1:
// CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_25]] : (!fir.ref<!fir.array<1x2xi32>>) -> !fir.heap<!fir.array<1x2xi32>>
// CHECK:           fir.freemem %[[VAL_27]] : !fir.heap<!fir.array<1x2xi32>>
// CHECK:           return
// CHECK:         }

// test that we support a hlfir.associate operation where the expr is also used in a hlfir.get_length op
func.func @test_get_length(%arg0: !fir.ref<!fir.char<1,2>>) {
  %0 = hlfir.as_expr %arg0 : (!fir.ref<!fir.char<1,2>>) -> !hlfir.expr<!fir.char<1,2>>
  %1 = hlfir.get_length %0 : (!hlfir.expr<!fir.char<1,2>>)  -> index
  %c2 = arith.constant 2 : index
  %2:3 = hlfir.associate %0 typeparams %c2 {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,2>>, index) -> (!fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,2>>, i1)
  // ...
  hlfir.end_associate %2#1, %2#2 : !fir.ref<!fir.char<1,2>>, i1
  return
}
// CHECK-LABEL:   func.func @test_get_length(
// CHECK-SAME:                               %[[VAL_0:.*]]: !fir.ref<!fir.char<1,2>>) {
// CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.char<1,2> {bindc_name = ".tmp"}
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant false
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] typeparams %[[VAL_2]] {uniq_name = ".tmp"} : (!fir.ref<!fir.char<1,2>>, index) -> (!fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,2>>)
// CHECK:           hlfir.assign %[[VAL_0]] to %[[VAL_4]]#0 temporary_lhs : !fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,2>>
// CHECK:           %[[VAL_5:.*]] = fir.undefined tuple<!fir.ref<!fir.char<1,2>>, i1>
// CHECK:           %[[VAL_6:.*]] = fir.insert_value %[[VAL_5]], %[[VAL_3]], [1 : index] : (tuple<!fir.ref<!fir.char<1,2>>, i1>, i1) -> tuple<!fir.ref<!fir.char<1,2>>, i1>
// CHECK:           %[[VAL_7:.*]] = fir.insert_value %[[VAL_6]], %[[VAL_4]]#0, [0 : index] : (tuple<!fir.ref<!fir.char<1,2>>, i1>, !fir.ref<!fir.char<1,2>>) -> tuple<!fir.ref<!fir.char<1,2>>, i1>
// CHECK:           %[[VAL_8:.*]] = arith.constant 2 : index
// CHECK:           return
// CHECK:         }

func.func private @take_i4(!fir.ref<i32>)
func.func private @take_r4(!fir.ref<f32>)
func.func private @take_l4(!fir.ref<!fir.logical<4>>)
func.func private @take_c(!fir.boxchar<1>)

// Test the hlfir.associate/hlfir.end_associate does not take ownership over
// a single-use hlfir.expr if hlfir.end_associate might be executed more times
// than the producer of hlfir.expr. This might cause double-free effects.
func.func @_QPtest_multiple_expr_uses_inside_elemental() {
  %true = arith.constant true
  %18 = fir.undefined !fir.heap<!fir.char<1,?>>
  %17 = fir.undefined index
  %19:2 = hlfir.declare %18 typeparams %17 {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.heap<!fir.char<1,?>>)
  %20 = hlfir.as_expr %19#0 move %true : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
  %21 = fir.undefined index
  %22 = fir.shape %21 : (index) -> !fir.shape<1>
  %23 = hlfir.elemental %22 unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%arg2: index):
    %35:3 = hlfir.associate %20 typeparams %17 {adapt.valuebyref} : (!hlfir.expr<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>, i1)
    hlfir.end_associate %35#1, %35#2 : !fir.ref<!fir.char<1,?>>, i1
    %ci1 = arith.constant 1 : i1
    %42 = fir.convert %ci1 : (i1) -> !fir.logical<4>
    hlfir.yield_element %42 : !fir.logical<4>
  }
  return
}
// CHECK-LABEL:   func.func @_QPtest_multiple_expr_uses_inside_elemental() {
// CHECK:           %[[VAL_2:.*]] = arith.constant true
// CHECK:           %[[VAL_3:.*]] = fir.undefined !fir.heap<!fir.char<1,?>>
// CHECK:           %[[VAL_4:.*]] = fir.undefined index
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_3]] typeparams %[[VAL_4]] {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.heap<!fir.char<1,?>>)
// CHECK:           %[[VAL_6:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[VAL_7:.*]] = fir.insert_value %[[VAL_6]], %[[VAL_2]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[VAL_8:.*]] = fir.insert_value %[[VAL_7]], %[[VAL_5]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:           %[[VAL_9:.*]] = fir.undefined index
// CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_9]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_11:.*]] = fir.allocmem !fir.array<?x!fir.logical<4>>, %[[VAL_9]] {bindc_name = ".tmp.array", uniq_name = ""}
// CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]](%[[VAL_10]]) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<?x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.heap<!fir.array<?x!fir.logical<4>>>)
// CHECK:           %[[VAL_13:.*]] = arith.constant true
// CHECK:           %[[VAL_14:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_15:.*]] = %[[VAL_14]] to %[[VAL_9]] step %[[VAL_14]] unordered {
// CHECK:             %[[VAL_16:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_4]] : index) {bindc_name = ".tmp"}
// CHECK:             %[[VAL_17:.*]] = arith.constant false
// CHECK:             %[[VAL_18:.*]]:2 = hlfir.declare %[[VAL_16]] typeparams %[[VAL_4]] {uniq_name = ".tmp"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:             hlfir.assign %[[VAL_5]]#0 to %[[VAL_18]]#0 temporary_lhs : !fir.boxchar<1>, !fir.boxchar<1>
// CHECK:             %[[VAL_19:.*]] = fir.undefined tuple<!fir.boxchar<1>, i1>
// CHECK:             %[[VAL_20:.*]] = fir.insert_value %[[VAL_19]], %[[VAL_17]], [1 : index] : (tuple<!fir.boxchar<1>, i1>, i1) -> tuple<!fir.boxchar<1>, i1>
// CHECK:             %[[VAL_21:.*]] = fir.insert_value %[[VAL_20]], %[[VAL_18]]#0, [0 : index] : (tuple<!fir.boxchar<1>, i1>, !fir.boxchar<1>) -> tuple<!fir.boxchar<1>, i1>
// CHECK:             %[[VAL_22:.*]] = arith.constant true
// CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i1) -> !fir.logical<4>
// CHECK:             %[[VAL_24:.*]] = hlfir.designate %[[VAL_12]]#0 (%[[VAL_15]])  : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
// CHECK:             hlfir.assign %[[VAL_23]] to %[[VAL_24]] temporary_lhs : !fir.logical<4>, !fir.ref<!fir.logical<4>>
// CHECK:           }
// CHECK:           %[[VAL_25:.*]] = fir.undefined tuple<!fir.box<!fir.array<?x!fir.logical<4>>>, i1>
// CHECK:           %[[VAL_26:.*]] = fir.insert_value %[[VAL_25]], %[[VAL_13]], [1 : index] : (tuple<!fir.box<!fir.array<?x!fir.logical<4>>>, i1>, i1) -> tuple<!fir.box<!fir.array<?x!fir.logical<4>>>, i1>
// CHECK:           %[[VAL_27:.*]] = fir.insert_value %[[VAL_26]], %[[VAL_12]]#0, [0 : index] : (tuple<!fir.box<!fir.array<?x!fir.logical<4>>>, i1>, !fir.box<!fir.array<?x!fir.logical<4>>>) -> tuple<!fir.box<!fir.array<?x!fir.logical<4>>>, i1>
// CHECK:           return
// CHECK:         }

// Verify that we properly recognize mutliple consequent hlfir.associate using
// the same result of hlfir.elemental.
func.func @_QPtest_multitple_associates_for_same_expr() {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %4 = fir.shape %c10 : (index) -> !fir.shape<1>
  %11 = hlfir.elemental %4 typeparams %c1 unordered : (!fir.shape<1>, index) -> !hlfir.expr<10x!fir.char<1>> {
  ^bb0(%arg1: index):
    %44 = fir.undefined !fir.ref<!fir.char<1>>
    hlfir.yield_element %44 : !fir.ref<!fir.char<1>>
  }
  %12:3 = hlfir.associate %11(%4) typeparams %c1 {adapt.valuebyref} : (!hlfir.expr<10x!fir.char<1>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<10x!fir.char<1>>>, !fir.ref<!fir.array<10x!fir.char<1>>>, i1)
  hlfir.end_associate %12#1, %12#2 : !fir.ref<!fir.array<10x!fir.char<1>>>, i1
  %31:3 = hlfir.associate %11(%4) typeparams %c1 {adapt.valuebyref} : (!hlfir.expr<10x!fir.char<1>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<10x!fir.char<1>>>, !fir.ref<!fir.array<10x!fir.char<1>>>, i1)
  hlfir.end_associate %31#1, %31#2 : !fir.ref<!fir.array<10x!fir.char<1>>>, i1
  hlfir.destroy %11 : !hlfir.expr<10x!fir.char<1>>
  return
}
// CHECK-LABEL:   func.func @_QPtest_multitple_associates_for_same_expr() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_3:.*]] = fir.allocmem !fir.array<10x!fir.char<1>> {bindc_name = ".tmp.array", uniq_name = ""}
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]](%[[VAL_2]]) typeparams %[[VAL_0]] {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<10x!fir.char<1>>>, !fir.shape<1>, index) -> (!fir.heap<!fir.array<10x!fir.char<1>>>, !fir.heap<!fir.array<10x!fir.char<1>>>)
// CHECK:           %[[VAL_5:.*]] = arith.constant true
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_7:.*]] = %[[VAL_6]] to %[[VAL_1]] step %[[VAL_6]] unordered {
// CHECK:             %[[VAL_8:.*]] = fir.undefined !fir.ref<!fir.char<1>>
// CHECK:             %[[VAL_9:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_7]])  typeparams %[[VAL_0]] : (!fir.heap<!fir.array<10x!fir.char<1>>>, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:             hlfir.assign %[[VAL_8]] to %[[VAL_9]] temporary_lhs : !fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>
// CHECK:           }
// CHECK:           %[[VAL_10:.*]] = fir.undefined tuple<!fir.heap<!fir.array<10x!fir.char<1>>>, i1>
// CHECK:           %[[VAL_11:.*]] = fir.insert_value %[[VAL_10]], %[[VAL_5]], [1 : index] : (tuple<!fir.heap<!fir.array<10x!fir.char<1>>>, i1>, i1) -> tuple<!fir.heap<!fir.array<10x!fir.char<1>>>, i1>
// CHECK:           %[[VAL_12:.*]] = fir.insert_value %[[VAL_11]], %[[VAL_4]]#0, [0 : index] : (tuple<!fir.heap<!fir.array<10x!fir.char<1>>>, i1>, !fir.heap<!fir.array<10x!fir.char<1>>>) -> tuple<!fir.heap<!fir.array<10x!fir.char<1>>>, i1>
// CHECK:           %[[VAL_13:.*]] = fir.allocmem !fir.array<10x!fir.char<1>> {bindc_name = ".tmp", uniq_name = ""}
// CHECK:           %[[VAL_14:.*]] = arith.constant true
// CHECK:           %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_13]](%[[VAL_2]]) typeparams %[[VAL_0]] {uniq_name = ".tmp"} : (!fir.heap<!fir.array<10x!fir.char<1>>>, !fir.shape<1>, index) -> (!fir.heap<!fir.array<10x!fir.char<1>>>, !fir.heap<!fir.array<10x!fir.char<1>>>)
// CHECK:           hlfir.assign %[[VAL_4]]#0 to %[[VAL_15]]#0 temporary_lhs : !fir.heap<!fir.array<10x!fir.char<1>>>, !fir.heap<!fir.array<10x!fir.char<1>>>
// CHECK:           %[[VAL_16:.*]] = fir.undefined tuple<!fir.heap<!fir.array<10x!fir.char<1>>>, i1>
// CHECK:           %[[VAL_17:.*]] = fir.insert_value %[[VAL_16]], %[[VAL_14]], [1 : index] : (tuple<!fir.heap<!fir.array<10x!fir.char<1>>>, i1>, i1) -> tuple<!fir.heap<!fir.array<10x!fir.char<1>>>, i1>
// CHECK:           %[[VAL_18:.*]] = fir.insert_value %[[VAL_17]], %[[VAL_15]]#0, [0 : index] : (tuple<!fir.heap<!fir.array<10x!fir.char<1>>>, i1>, !fir.heap<!fir.array<10x!fir.char<1>>>) -> tuple<!fir.heap<!fir.array<10x!fir.char<1>>>, i1>
// CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_15]]#0 : (!fir.heap<!fir.array<10x!fir.char<1>>>) -> !fir.ref<!fir.array<10x!fir.char<1>>>
// CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_15]]#1 : (!fir.heap<!fir.array<10x!fir.char<1>>>) -> !fir.ref<!fir.array<10x!fir.char<1>>>
// CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<!fir.array<10x!fir.char<1>>>) -> !fir.heap<!fir.array<10x!fir.char<1>>>
// CHECK:           fir.freemem %[[VAL_21]] : !fir.heap<!fir.array<10x!fir.char<1>>>
// CHECK:           %[[VAL_22:.*]] = fir.allocmem !fir.array<10x!fir.char<1>> {bindc_name = ".tmp", uniq_name = ""}
// CHECK:           %[[VAL_23:.*]] = arith.constant true
// CHECK:           %[[VAL_24:.*]]:2 = hlfir.declare %[[VAL_22]](%[[VAL_2]]) typeparams %[[VAL_0]] {uniq_name = ".tmp"} : (!fir.heap<!fir.array<10x!fir.char<1>>>, !fir.shape<1>, index) -> (!fir.heap<!fir.array<10x!fir.char<1>>>, !fir.heap<!fir.array<10x!fir.char<1>>>)
// CHECK:           hlfir.assign %[[VAL_4]]#0 to %[[VAL_24]]#0 temporary_lhs : !fir.heap<!fir.array<10x!fir.char<1>>>, !fir.heap<!fir.array<10x!fir.char<1>>>
// CHECK:           %[[VAL_25:.*]] = fir.undefined tuple<!fir.heap<!fir.array<10x!fir.char<1>>>, i1>
// CHECK:           %[[VAL_26:.*]] = fir.insert_value %[[VAL_25]], %[[VAL_23]], [1 : index] : (tuple<!fir.heap<!fir.array<10x!fir.char<1>>>, i1>, i1) -> tuple<!fir.heap<!fir.array<10x!fir.char<1>>>, i1>
// CHECK:           %[[VAL_27:.*]] = fir.insert_value %[[VAL_26]], %[[VAL_24]]#0, [0 : index] : (tuple<!fir.heap<!fir.array<10x!fir.char<1>>>, i1>, !fir.heap<!fir.array<10x!fir.char<1>>>) -> tuple<!fir.heap<!fir.array<10x!fir.char<1>>>, i1>
// CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_24]]#0 : (!fir.heap<!fir.array<10x!fir.char<1>>>) -> !fir.ref<!fir.array<10x!fir.char<1>>>
// CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_24]]#1 : (!fir.heap<!fir.array<10x!fir.char<1>>>) -> !fir.ref<!fir.array<10x!fir.char<1>>>
// CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (!fir.ref<!fir.array<10x!fir.char<1>>>) -> !fir.heap<!fir.array<10x!fir.char<1>>>
// CHECK:           fir.freemem %[[VAL_30]] : !fir.heap<!fir.array<10x!fir.char<1>>>
// CHECK:           fir.freemem %[[VAL_4]]#0 : !fir.heap<!fir.array<10x!fir.char<1>>>
// CHECK:           return
// CHECK:         }

// Test hlfir.associate codegen, when its operand is used
// by hlfir.shape located in a block different from the block
// of the hlfir.end_associate.
func.func @_QPtest(%arg0: index, %arg1: index, %arg2 : i32) {
  %c1_i32 = arith.constant 1 : i32
  %3 = fir.shape %arg0, %arg1 : (index, index) -> !fir.shape<2>
  %4 = hlfir.elemental %3 unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xi32> {
  ^bb0(%arg3: index, %arg4: index):
    %16 = fir.undefined i32
    hlfir.yield_element %16 : i32
  }
  %5 = hlfir.shape_of %4 : (!hlfir.expr<?x?xi32>) -> !fir.shape<2>
  %6:3 = hlfir.associate %4(%5) {adapt.valuebyref} : (!hlfir.expr<?x?xi32>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.ref<!fir.array<?x?xi32>>, i1)
  %13 = arith.cmpi ne, %arg2, %c1_i32 : i32
  cf.cond_br %13, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  fir.unreachable
^bb2:  // pred: ^bb0
  hlfir.end_associate %6#1, %6#2 : !fir.ref<!fir.array<?x?xi32>>, i1
  hlfir.destroy %4 : !hlfir.expr<?x?xi32>
  return
}
