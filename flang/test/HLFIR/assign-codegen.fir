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
// CHECK:  %[[VAL_4:.*]] = arith.constant 100 : index
// CHECK:  %[[VAL_5:.*]] = fir.declare %[[VAL_0]] {uniq_name = "x"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:  %[[VAL_6:.*]] = fir.rebox %[[VAL_5]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:  %[[VAL_7:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_8:.*]] = fir.declare %[[VAL_1]](%[[VAL_7]]) {uniq_name = "y"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xi32>>
// CHECK:  %[[VAL_9:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_10:.*]] = fir.embox %[[VAL_8]](%[[VAL_9]]) : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<100xi32>>
// CHECK:  fir.store %[[VAL_5]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK:  %[[VAL_26:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.ref<!fir.box<none>>
// CHECK:  %[[VAL_27:.*]] = fir.convert %[[VAL_10]] : (!fir.box<!fir.array<100xi32>>) -> !fir.box<none>
// CHECK:  %[[VAL_29:.*]] = fir.call @_FortranAAssign(%[[VAL_26]], %[[VAL_27]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none


func.func @array_temp(%arg0: !fir.box<!fir.array<?xi32>>, %arg1: !fir.ref<!fir.array<100xi32>>) {
  %c100 = arith.constant 100 : index
  %0:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %1 = fir.shape %c100 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %arg1(%1) {uniq_name = "y"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
  hlfir.assign %2#0 to %0#0 temporary_lhs : !fir.ref<!fir.array<100xi32>>, !fir.box<!fir.array<?xi32>>
  return
}
// CHECK-LABEL:   func.func @array_temp(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME:    %[[VAL_1:.*]]: !fir.ref<!fir.array<100xi32>>) {
// CHECK:  %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
// CHECK:  %[[VAL_4:.*]] = arith.constant 100 : index
// CHECK:  %[[VAL_5:.*]] = fir.declare %[[VAL_0]] {uniq_name = "x"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:  %[[VAL_6:.*]] = fir.rebox %[[VAL_5]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:  %[[VAL_7:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_8:.*]] = fir.declare %[[VAL_1]](%[[VAL_7]]) {uniq_name = "y"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xi32>>
// CHECK:  %[[VAL_9:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_10:.*]] = fir.embox %[[VAL_8]](%[[VAL_9]]) : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<100xi32>>
// CHECK:  fir.store %[[VAL_5]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK:  %[[VAL_26:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.ref<!fir.box<none>>
// CHECK:  %[[VAL_27:.*]] = fir.convert %[[VAL_10]] : (!fir.box<!fir.array<100xi32>>) -> !fir.box<none>
// CHECK:  %[[VAL_29:.*]] = fir.call @_FortranAAssignTemporary(%[[VAL_26]], %[[VAL_27]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none


func.func @test_scalar_to_array(%lhs: !fir.box<!fir.array<?xi32>>, %rhs: i32) {
  hlfir.assign %rhs to %lhs : i32, !fir.box<!fir.array<?xi32>>
  return
}
// CHECK-LABEL:   func.func @test_scalar_to_array(
// CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME:    %[[VAL_1:.*]]: i32) {
// CHECK:  %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
// CHECK:  %[[VAL_5:.*]] = fir.alloca i32
// CHECK:  fir.store %[[VAL_1]] to %[[VAL_5]] : !fir.ref<i32>
// CHECK:  %[[VAL_6:.*]] = fir.embox %[[VAL_5]] : (!fir.ref<i32>) -> !fir.box<i32>
// CHECK:  fir.store %[[VAL_0]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK:  %[[VAL_10:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.ref<!fir.box<none>>
// CHECK:  %[[VAL_11:.*]] = fir.convert %[[VAL_6]] : (!fir.box<i32>) -> !fir.box<none>
// CHECK:  %[[VAL_13:.*]] = fir.call @_FortranAAssign(%[[VAL_10]], %[[VAL_11]]


func.func @test_i1_scalar_to_array(%lhs: !fir.box<!fir.array<?x!fir.logical<4>>>, %rhs: i1) {
  hlfir.assign %rhs to %lhs : i1, !fir.box<!fir.array<?x!fir.logical<4>>>
  return
}
// CHECK-LABEL:   func.func @test_i1_scalar_to_array(
// CHECK:  %[[VAL_5:.*]] = fir.convert %{{.*}} : (i1) -> !fir.logical<4>
// CHECK:  %[[VAL_6:.*]] = fir.alloca !fir.logical<4>
// CHECK:  %[[VAL_7:.*]] = fir.embox %[[VAL_6]] : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
// CHECK:  %[[VAL_12:.*]] = fir.convert %[[VAL_7]] : (!fir.box<!fir.logical<4>>) -> !fir.box<none>
// CHECK:  %[[VAL_14:.*]] = fir.call @_FortranAAssign(%{{.*}}, %[[VAL_12]]

func.func @alloc_assign(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: !fir.box<!fir.array<?xi32>>) {
  hlfir.assign %arg1 to %arg0 realloc : !fir.box<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  return
}
// CHECK-LABEL:  func.func @alloc_assign(
// CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>,
// CHECK-SAME: %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>>) {
// CHECK:  %[[VAL_2:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
// CHECK:  fir.call @_FortranAAssign(%[[VAL_2]], %[[VAL_3]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none

func.func @alloc_assign_temp(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: !fir.box<!fir.array<?xi32>>) {
  hlfir.assign %arg1 to %arg0 realloc temporary_lhs : !fir.box<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  return
}
// CHECK-LABEL:  func.func @alloc_assign_temp(
// CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>,
// CHECK-SAME: %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>>) {
// CHECK:  %[[VAL_2:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
// CHECK:  fir.call @_FortranAAssignTemporary(%[[VAL_2]], %[[VAL_3]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none

func.func @test_alloc_assign_explicit_length_character(%lhs: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>, %rhs: !fir.box<!fir.array<?x!fir.char<1,?>>>) {
  hlfir.assign %rhs to %lhs realloc keep_lhs_len : !fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
  return
}
// CHECK-LABEL:   func.func @test_alloc_assign_explicit_length_character(
// CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>,
// CHECK-SAME:  %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>) {
// CHECK:  %[[VAL_7:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:  %[[VAL_10:.*]] = fir.call @_FortranAAssignExplicitLengthCharacter(%[[VAL_7]], %[[VAL_8]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none

func.func @test_alloc_assign_polymorphic(%lhs: !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<t>>>>>, %rhs: !fir.class<!fir.array<?x!fir.type<t>>>) {
  hlfir.assign %rhs to %lhs realloc : !fir.class<!fir.array<?x!fir.type<t>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<t>>>>>
  return
}
// CHECK-LABEL:   func.func @test_alloc_assign_polymorphic(
// CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<t>>>>>,
// CHECK-SAME:  %[[VAL_1:.*]]: !fir.class<!fir.array<?x!fir.type<t>>>) {
// CHECK:  %[[VAL_7:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<t>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_1]] : (!fir.class<!fir.array<?x!fir.type<t>>>) -> !fir.box<none>
// CHECK:  %[[VAL_10:.*]] = fir.call @_FortranAAssignPolymorphic(%[[VAL_7]], %[[VAL_8]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none

func.func @assing_scalar_int_to_polymorphic(%arg0: !fir.ref<!fir.class<!fir.heap<none>>>) {
  %c123_i32 = arith.constant 123 : i32
  %0:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "x"} : (!fir.ref<!fir.class<!fir.heap<none>>>) -> (!fir.ref<!fir.class<!fir.heap<none>>>, !fir.ref<!fir.class<!fir.heap<none>>>)
  hlfir.assign %c123_i32 to %0#0 realloc : i32, !fir.ref<!fir.class<!fir.heap<none>>>
  return
}

// CHECK-LABEL:   func.func @assing_scalar_int_to_polymorphic(
// CHECK-SAME:        %[[VAL_0:.*]]: !fir.ref<!fir.class<!fir.heap<none>>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 123 : i32
// CHECK:           %[[VAL_2:.*]] = fir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "x"} : (!fir.ref<!fir.class<!fir.heap<none>>>) -> !fir.ref<!fir.class<!fir.heap<none>>>
// CHECK:           %[[VAL_3:.*]] = fir.alloca i32
// CHECK:           fir.store %[[VAL_1]] to %[[VAL_3]] : !fir.ref<i32>
// CHECK:           %[[VAL_4:.*]] = fir.embox %[[VAL_3]] : (!fir.ref<i32>) -> !fir.box<i32>
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.class<!fir.heap<none>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_4]] : (!fir.box<i32>) -> !fir.box<none>
// CHECK:           %[[VAL_11:.*]] = fir.call @_FortranAAssignPolymorphic(%[[VAL_8]], %[[VAL_9]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none

func.func @assign_i1_to_polymorphic(%arg0: !fir.ref<!fir.class<!fir.heap<none>>>) {
  %false = arith.constant false
  %0:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "x"} : (!fir.ref<!fir.class<!fir.heap<none>>>) -> (!fir.ref<!fir.class<!fir.heap<none>>>, !fir.ref<!fir.class<!fir.heap<none>>>)
  %1 = hlfir.no_reassoc %false : i1
  hlfir.assign %1 to %0#0 realloc : i1, !fir.ref<!fir.class<!fir.heap<none>>>
  return
}

// CHECK-LABEL:   func.func @assign_i1_to_polymorphic(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !fir.ref<!fir.class<!fir.heap<none>>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant false
// CHECK:           %[[VAL_2:.*]] = fir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "x"} : (!fir.ref<!fir.class<!fir.heap<none>>>) -> !fir.ref<!fir.class<!fir.heap<none>>>
// CHECK:           %[[VAL_3:.*]] = fir.no_reassoc %[[VAL_1]] : i1
// CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i1) -> !fir.logical<4>
// CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.logical<4>
// CHECK:           fir.store %[[VAL_4]] to %[[VAL_5]] : !fir.ref<!fir.logical<4>>
// CHECK:           %[[VAL_6:.*]] = fir.embox %[[VAL_5]] : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.class<!fir.heap<none>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_6]] : (!fir.box<!fir.logical<4>>) -> !fir.box<none>
// CHECK:           %[[VAL_13:.*]] = fir.call @_FortranAAssignPolymorphic(%[[VAL_10]], %[[VAL_11]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none

func.func @test_alloc_assign_polymorphic_temp(%lhs: !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<t>>>>>, %rhs: !fir.class<!fir.array<?x!fir.type<t>>>) {
  hlfir.assign %rhs to %lhs realloc temporary_lhs : !fir.class<!fir.array<?x!fir.type<t>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<t>>>>>
  return
}
// CHECK-LABEL:   func.func @test_alloc_assign_polymorphic_temp(
// CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<t>>>>>,
// CHECK-SAME:  %[[VAL_1:.*]]: !fir.class<!fir.array<?x!fir.type<t>>>) {
// CHECK:  %[[VAL_7:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<t>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_1]] : (!fir.class<!fir.array<?x!fir.type<t>>>) -> !fir.box<none>
// CHECK:  %[[VAL_10:.*]] = fir.call @_FortranAAssignTemporary(%[[VAL_7]], %[[VAL_8]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none

func.func @test_allocatable_component(%arg0: !fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>> {fir.bindc_name = "x", fir.target}, %arg1: !fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>> {fir.bindc_name = "y", fir.target}) {
  %4:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEx"} : (!fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> (!fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>)
  %5:2 = hlfir.declare %arg1 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEy"} : (!fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> (!fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>)
  hlfir.assign %5#0 to %4#0 : !fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
  return
}
// CHECK-LABEL:   func.func @test_allocatable_component(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>> {fir.bindc_name = "x", fir.target},
// CHECK-SAME:                                          %[[VAL_1:.*]]: !fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>> {fir.bindc_name = "y", fir.target}) {
// CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
// CHECK:           %[[VAL_3:.*]] = fir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEx"} : (!fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
// CHECK:           %[[VAL_4:.*]] = fir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEy"} : (!fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
// CHECK:           %[[VAL_5:.*]] = fir.embox %[[VAL_3]] : (!fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.box<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
// CHECK:           %[[VAL_6:.*]] = fir.embox %[[VAL_4]] : (!fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.box<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
// CHECK:           fir.store %[[VAL_5]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_6]] : (!fir.box<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.box<none>
// CHECK:           %[[VAL_12:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:           %[[VAL_13:.*]] = fir.call @_FortranAAssign(%[[VAL_10]], %[[VAL_11]], %[[VAL_12]], %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
// CHECK:           return
// CHECK:         }

func.func @test_allocatable_component_temp(%arg0: !fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>> {fir.bindc_name = "x", fir.target}, %arg1: !fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>> {fir.bindc_name = "y", fir.target}) {
  %4:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEx"} : (!fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> (!fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>)
  %5:2 = hlfir.declare %arg1 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEy"} : (!fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> (!fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>)
  hlfir.assign %5#0 to %4#0 temporary_lhs : !fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
  return
}
// CHECK-LABEL:   func.func @test_allocatable_component_temp(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>> {fir.bindc_name = "x", fir.target},
// CHECK-SAME:                                          %[[VAL_1:.*]]: !fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>> {fir.bindc_name = "y", fir.target}) {
// CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
// CHECK:           %[[VAL_3:.*]] = fir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEx"} : (!fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
// CHECK:           %[[VAL_4:.*]] = fir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEy"} : (!fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
// CHECK:           %[[VAL_5:.*]] = fir.embox %[[VAL_3]] : (!fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.box<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
// CHECK:           %[[VAL_6:.*]] = fir.embox %[[VAL_4]] : (!fir.ref<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.box<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
// CHECK:           fir.store %[[VAL_5]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_6]] : (!fir.box<!fir.type<_QFtestTt1{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.box<none>
// CHECK:           %[[VAL_12:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
// CHECK:           %[[VAL_13:.*]] = fir.call @_FortranAAssignTemporary(%[[VAL_10]], %[[VAL_11]], %[[VAL_12]], %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
// CHECK:           return
// CHECK:         }

// Check that Destroy() is called for LHS, when hlfir.assign
// is lowered into simple load/store.
func.func @_QFPtest_scalar_lhs_finalization(%arg0: !fir.ref<!fir.type<_QMa8vTt1{val:i32}>> {fir.bindc_name = "s1"}, %arg1: !fir.ref<!fir.type<_QMa8vTt1{val:i32}>> {fir.bindc_name = "s2"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFtest_scalar_lhs_finalizationEs1"} : (!fir.ref<!fir.type<_QMa8vTt1{val:i32}>>) -> (!fir.ref<!fir.type<_QMa8vTt1{val:i32}>>, !fir.ref<!fir.type<_QMa8vTt1{val:i32}>>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFtest_scalar_lhs_finalizationEs2"} : (!fir.ref<!fir.type<_QMa8vTt1{val:i32}>>) -> (!fir.ref<!fir.type<_QMa8vTt1{val:i32}>>, !fir.ref<!fir.type<_QMa8vTt1{val:i32}>>)
  hlfir.assign %1#0 to %0#0 : !fir.ref<!fir.type<_QMa8vTt1{val:i32}>>, !fir.ref<!fir.type<_QMa8vTt1{val:i32}>>
  return
}
// CHECK-LABEL:   func.func @_QFPtest_scalar_lhs_finalization(
// CHECK-SAME:                                                %[[VAL_0:.*]]: !fir.ref<!fir.type<_QMa8vTt1{val:i32}>> {fir.bindc_name = "s1"},
// CHECK-SAME:                                                %[[VAL_1:.*]]: !fir.ref<!fir.type<_QMa8vTt1{val:i32}>> {fir.bindc_name = "s2"}) {
// CHECK:           %[[VAL_2:.*]] = fir.declare %[[VAL_0]] {uniq_name = "_QFFtest_scalar_lhs_finalizationEs1"} : (!fir.ref<!fir.type<_QMa8vTt1{val:i32}>>) -> !fir.ref<!fir.type<_QMa8vTt1{val:i32}>>
// CHECK:           %[[VAL_3:.*]] = fir.declare %[[VAL_1]] {uniq_name = "_QFFtest_scalar_lhs_finalizationEs2"} : (!fir.ref<!fir.type<_QMa8vTt1{val:i32}>>) -> !fir.ref<!fir.type<_QMa8vTt1{val:i32}>>
// CHECK:           %[[VAL_4:.*]] = fir.embox %[[VAL_2]] : (!fir.ref<!fir.type<_QMa8vTt1{val:i32}>>) -> !fir.box<!fir.type<_QMa8vTt1{val:i32}>>
// CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.box<!fir.type<_QMa8vTt1{val:i32}>>) -> !fir.box<none>
// CHECK:           %[[VAL_6:.*]] = fir.call @_FortranADestroy(%[[VAL_5]]) : (!fir.box<none>) -> none
// CHECK:           %[[VAL_7:.*]] = fir.field_index val, !fir.type<_QMa8vTt1{val:i32}>
// CHECK:           %[[VAL_8:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_7]] : (!fir.ref<!fir.type<_QMa8vTt1{val:i32}>>, !fir.field) -> !fir.ref<i32>
// CHECK:           %[[VAL_9:.*]] = fir.field_index val, !fir.type<_QMa8vTt1{val:i32}>
// CHECK:           %[[VAL_10:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_9]] : (!fir.ref<!fir.type<_QMa8vTt1{val:i32}>>, !fir.field) -> !fir.ref<i32>
// CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_8]] : !fir.ref<i32>
// CHECK:           fir.store %[[VAL_11]] to %[[VAL_10]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }

// Check that Destroy() is not called for temporary LHS.
func.func @_QFPtest_scalar_temp_lhs_no_finalization(%arg0: !fir.ref<!fir.type<_QMa8vTt1{val:i32}>> {fir.bindc_name = "s1"}, %arg1: !fir.ref<!fir.type<_QMa8vTt1{val:i32}>> {fir.bindc_name = "s2"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFtest_scalar_lhs_finalizationEs1"} : (!fir.ref<!fir.type<_QMa8vTt1{val:i32}>>) -> (!fir.ref<!fir.type<_QMa8vTt1{val:i32}>>, !fir.ref<!fir.type<_QMa8vTt1{val:i32}>>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "_QFFtest_scalar_lhs_finalizationEs2"} : (!fir.ref<!fir.type<_QMa8vTt1{val:i32}>>) -> (!fir.ref<!fir.type<_QMa8vTt1{val:i32}>>, !fir.ref<!fir.type<_QMa8vTt1{val:i32}>>)
  hlfir.assign %1#0 to %0#0 temporary_lhs : !fir.ref<!fir.type<_QMa8vTt1{val:i32}>>, !fir.ref<!fir.type<_QMa8vTt1{val:i32}>>
  return
}
// CHECK-LABEL:   func.func @_QFPtest_scalar_temp_lhs_no_finalization(
// CHECK-NOT: fir.call @_FortranADestroy
