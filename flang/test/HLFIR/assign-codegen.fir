// Test hlfir.assign code generation to FIR

// RUN: fir-opt %s -convert-hlfir-to-fir | FileCheck %s

func.func @scalar_int(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>) {
  hlfir.assign %arg0 to %arg1 : !fir.ref<i32>, !fir.ref<i32>
  return
}
// CHECK-LABEL:   func.func @scalar_int(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<i32>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<i32>) {
// CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
// CHECK:  fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<i32>

func.func @scalar_int_2(%arg0: !fir.ref<i32>) {
  %c42_i32 = arith.constant 42 : i32
  hlfir.assign %c42_i32 to %arg0 : i32, !fir.ref<i32>
  return
}
// CHECK-LABEL:   func.func @scalar_int_2(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<i32>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 42 : i32
// CHECK:  fir.store %[[VAL_1]] to %[[VAL_0]] : !fir.ref<i32>

func.func @scalar_logical(%arg0: !fir.ref<!fir.logical<4>>, %arg1: !fir.ref<!fir.logical<4>>) {
  hlfir.assign %arg0 to %arg1 : !fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>
  return
}
// CHECK-LABEL:   func.func @scalar_logical(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.logical<4>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<!fir.logical<4>>) {
// CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.logical<4>>
// CHECK:  fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<!fir.logical<4>>

func.func @scalar_logical_2(%arg0: !fir.ref<!fir.logical<4>>) {
  %true = arith.constant true
  hlfir.assign %true to %arg0 : i1, !fir.ref<!fir.logical<4>>
  return
}
// CHECK-LABEL:   func.func @scalar_logical_2(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.logical<4>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant true
// CHECK:  %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (i1) -> !fir.logical<4>
// CHECK:  fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.logical<4>>

func.func @scalar_real(%arg0: !fir.ref<f32>, %arg1: !fir.ref<f32>) {
  hlfir.assign %arg0 to %arg1 : !fir.ref<f32>, !fir.ref<f32>
  return
}
// CHECK-LABEL:   func.func @scalar_real(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<f32>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<f32>) {
// CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<f32>
// CHECK:  fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<f32>

func.func @scalar_real_2(%arg0: !fir.ref<f32>) {
  %cst = arith.constant 3.140000e+00 : f32
  hlfir.assign %cst to %arg0 : f32, !fir.ref<f32>
  return
}
// CHECK-LABEL:   func.func @scalar_real_2(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<f32>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 3.140000e+00 : f32
// CHECK:  fir.store %[[VAL_1]] to %[[VAL_0]] : !fir.ref<f32>

func.func @scalar_complex(%arg0: !fir.ref<!fir.complex<4>>, %arg1: !fir.ref<!fir.complex<4>>) {
  hlfir.assign %arg0 to %arg0 : !fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>
  return
}
// CHECK-LABEL:   func.func @scalar_complex(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.complex<4>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<!fir.complex<4>>) {
// CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.complex<4>>
// CHECK:  fir.store %[[VAL_2]] to %[[VAL_0]] : !fir.ref<!fir.complex<4>>

func.func @scalar_complex_2(%arg0: !fir.ref<!fir.complex<4>>, %arg1: !fir.complex<4>) {
  hlfir.assign %arg1 to %arg0 : !fir.complex<4>, !fir.ref<!fir.complex<4>>
  return
}
// CHECK-LABEL:   func.func @scalar_complex_2(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.complex<4>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.complex<4>) {
// CHECK:  fir.store %[[VAL_1]] to %[[VAL_0]] : !fir.ref<!fir.complex<4>>

func.func @scalar_character(%arg0: !fir.boxchar<1>, %arg1: !fir.boxchar<1>) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1:2 = hlfir.declare %0#0 typeparams %0#1 {uniq_name = "x"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3:2 = hlfir.declare %2#0 typeparams %2#1 {uniq_name = "y"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  hlfir.assign %3#0 to %1#0 : !fir.boxchar<1>, !fir.boxchar<1>
  return
}
// CHECK-LABEL:   func.func @scalar_character(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxchar<1>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.boxchar<1>) {
// CHECK:  %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:  %[[VAL_3:.*]] = fir.declare %[[VAL_2]]#0 typeparams %[[VAL_2]]#1 {uniq_name = "x"} : (!fir.ref<!fir.char<1,?>>, index) -> !fir.ref<!fir.char<1,?>>
// CHECK:  %[[VAL_4:.*]] = fir.emboxchar %[[VAL_3]], %[[VAL_2]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
// CHECK:  %[[VAL_5:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:  %[[VAL_6:.*]] = fir.declare %[[VAL_5]]#0 typeparams %[[VAL_5]]#1 {uniq_name = "y"} : (!fir.ref<!fir.char<1,?>>, index) -> !fir.ref<!fir.char<1,?>>
// CHECK:  %[[VAL_7:.*]] = fir.emboxchar %[[VAL_6]], %[[VAL_5]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
// CHECK:  %[[VAL_8:.*]] = arith.cmpi slt, %[[VAL_2]]#1, %[[VAL_5]]#1 : index
// CHECK:  %[[VAL_9:.*]] = arith.select %[[VAL_8]], %[[VAL_2]]#1, %[[VAL_5]]#1 : index
// CHECK:  %[[VAL_10:.*]] = arith.constant 1 : i64
// CHECK:  %[[VAL_11:.*]] = fir.convert %[[VAL_9]] : (index) -> i64
// CHECK:  %[[VAL_12:.*]] = arith.muli %[[VAL_10]], %[[VAL_11]] : i64
// CHECK:  %[[VAL_13:.*]] = arith.constant false
// CHECK:  %[[VAL_14:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:  %[[VAL_15:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
// CHECK:  fir.call @llvm.memmove.p0.p0.i64(%[[VAL_14]], %[[VAL_15]], %[[VAL_12]], %[[VAL_13]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
// CHECK:  %[[VAL_16:.*]] = arith.constant 1 : index
// CHECK:  %[[VAL_17:.*]] = arith.subi %[[VAL_2]]#1, %[[VAL_16]] : index
// CHECK:  %[[VAL_18:.*]] = arith.constant 32 : i8
// CHECK:  %[[VAL_19:.*]] = fir.undefined !fir.char<1>
// CHECK:  %[[VAL_20:.*]] = fir.insert_value %[[VAL_19]], %[[VAL_18]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
// CHECK:  %[[VAL_21:.*]] = arith.constant 1 : index
// CHECK:  fir.do_loop %[[VAL_22:.*]] = %[[VAL_9]] to %[[VAL_17]] step %[[VAL_21]] {
// CHECK:    %[[VAL_23:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
// CHECK:    %[[VAL_24:.*]] = fir.coordinate_of %[[VAL_23]], %[[VAL_22]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
// CHECK:    fir.store %[[VAL_20]] to %[[VAL_24]] : !fir.ref<!fir.char<1>>
// CHECK:  }

func.func @array(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: !fir.ref<!fir.array<100xi32>>) {
  %c100 = arith.constant 100 : index
  %0:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %arg1(%1) {uniq_name = "y"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
  hlfir.assign %2#0 to %0#0 : !fir.ref<!fir.array<100xi32>>, !fir.box<!fir.array<?xi32>>
  return
}
// CHECK-LABEL:   func.func @array(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<!fir.array<100xi32>>) {
// CHECK:  %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
// CHECK:  %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<100xi32>>>
// CHECK:  %[[VAL_4:.*]] = arith.constant 100 : index
// CHECK:  %[[VAL_5:.*]] = fir.declare %[[VAL_0]] {uniq_name = "x"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:  %[[VAL_6:.*]] = fir.rebox %[[VAL_5]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:  %[[VAL_7:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_8:.*]] = fir.declare %[[VAL_1]](%[[VAL_7]]) {uniq_name = "y"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xi32>>
// CHECK:  %[[VAL_9:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_10:.*]] = fir.embox %[[VAL_8]](%[[VAL_9]]) : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<100xi32>>
// CHECK:  %[[VAL_11:.*]] = fir.zero_bits !fir.heap<!fir.array<100xi32>>
// CHECK:  %[[VAL_12:.*]] = arith.constant 0 : index
// CHECK:  %[[VAL_13:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_14:.*]] = fir.embox %[[VAL_11]](%[[VAL_13]]) : (!fir.heap<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<100xi32>>>
// CHECK:  fir.store %[[VAL_14]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<100xi32>>>>
// CHECK:  %[[VAL_18:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<100xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:  %[[VAL_19:.*]] = fir.convert %[[VAL_10]] : (!fir.box<!fir.array<100xi32>>) -> !fir.box<none>
// CHECK:  %[[VAL_21:.*]] = fir.call @_FortranAAssign(%[[VAL_18]], %[[VAL_19]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
// CHECK:  %[[VAL_22:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<100xi32>>>>
// CHECK:  fir.store %[[VAL_5]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK:  %[[VAL_26:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.ref<!fir.box<none>>
// CHECK:  %[[VAL_27:.*]] = fir.convert %[[VAL_22]] : (!fir.box<!fir.heap<!fir.array<100xi32>>>) -> !fir.box<none>
// CHECK:  %[[VAL_29:.*]] = fir.call @_FortranAAssign(%[[VAL_26]], %[[VAL_27]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
// CHECK:  %[[VAL_30:.*]] = fir.box_addr %[[VAL_22]] : (!fir.box<!fir.heap<!fir.array<100xi32>>>) -> !fir.heap<!fir.array<100xi32>>
// CHECK:  fir.freemem %[[VAL_30]] : !fir.heap<!fir.array<100xi32>>
