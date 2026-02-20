// RUN: fir-opt %s --allow-unregistered-dialect --mem2reg --split-input-file | FileCheck %s

// CHECK-LABEL:   func.func @basic() -> i32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 5 : i32
// CHECK:           return %[[CONSTANT_0]] : i32
// CHECK:         }
func.func @basic() -> i32 {
  %0 = arith.constant 5 : i32
  %1 = fir.alloca i32
  fir.store %0 to %1 : !fir.ref<i32>
  %2 = fir.load %1 : !fir.ref<i32>
  return %2 : i32
}

// -----

// CHECK-LABEL:   func.func @default_value() -> i32 {
// CHECK:           %[[UNDEFINED_0:.*]] = fir.undefined i32
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 5 : i32
// CHECK:           return %[[UNDEFINED_0]] : i32
// CHECK:         }
func.func @default_value() -> i32 {
  %0 = arith.constant 5 : i32
  %1 = fir.alloca i32
  %2 = fir.load %1 : !fir.ref<i32>
  fir.store %0 to %1 : !fir.ref<i32>
  return %2 : i32
}

// -----

// CHECK-LABEL:   func.func @basic_float() -> f32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 5.200000e+00 : f32
// CHECK:           return %[[CONSTANT_0]] : f32
// CHECK:         }
func.func @basic_float() -> f32 {
  %0 = arith.constant 5.2 : f32
  %1 = fir.alloca f32
  fir.store %0 to %1 : !fir.ref<f32>
  %2 = fir.load %1 : !fir.ref<f32>
  return %2 : f32
}

// -----

// CHECK-LABEL:   func.func @cycle(
// CHECK-SAME:                     %[[ARG0:.*]]: i64,
// CHECK-SAME:                     %[[ARG1:.*]]: i1,
// CHECK-SAME:                     %[[ARG2:.*]]: i64) {
// CHECK:           cf.cond_br %[[ARG1]], ^bb1(%[[ARG2]] : i64), ^bb2(%[[ARG2]] : i64)
// CHECK:         ^bb1(%[[VAL_0:.*]]: i64):
// CHECK:           "test.use"(%[[VAL_0]]) : (i64) -> ()
// CHECK:           cf.br ^bb2(%[[ARG0]] : i64)
// CHECK:         ^bb2(%[[VAL_1:.*]]: i64):
// CHECK:           cf.br ^bb1(%[[VAL_1]] : i64)
// CHECK:         }
func.func @cycle(%arg0: i64, %arg1: i1, %arg2: i64) {
  %alloca = fir.alloca i64
  fir.store %arg2 to %alloca : !fir.ref<i64>
  cf.cond_br %arg1, ^bb1, ^bb2
^bb1:
  %use = fir.load %alloca : !fir.ref<i64>
  "test.use"(%use) : (i64) -> ()
  fir.store %arg0 to %alloca : !fir.ref<i64>
  cf.br ^bb2
^bb2:
  cf.br ^bb1
}

// -----

// CHECK-LABEL: func.func @test_simple_declare(%arg0: !fir.ref<i32> {fir.bindc_name = "i"}) {
// CHECK: %[[C42:.*]] = arith.constant 42 : i32
// CHECK: %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
// CHECK: %[[ARG_DECL:.*]] = fir.declare %arg0 dummy_scope %[[SCOPE]] arg 1 {uniq_name = "_QFfooEi"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK: fir.declare_value %[[C42]] {uniq_name = "_QFfooEj"} : i32
// CHECK: fir.store %[[C42]] to %[[ARG_DECL]] : !fir.ref<i32>
func.func @test_simple_declare(%arg0: !fir.ref<i32> {fir.bindc_name = "i"}) {
    %c42_i32 = arith.constant 42 : i32
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFfooEi"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
    %2 = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFfooEj"}
    %3 = fir.declare %2 {uniq_name = "_QFfooEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    fir.store %c42_i32 to %3 : !fir.ref<i32>
    %4 = fir.load %3 : !fir.ref<i32>
    fir.store %4 to %1 : !fir.ref<i32>
    return
}

// -----

// CHECK-LABEL:   func.func @test_two_values(
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 43 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 42 : i32
// CHECK:           fir.declare_value %[[CONSTANT_1]] {uniq_name = "_QFfooEjlocal"} : i32
// CHECK:           fir.store %[[CONSTANT_1]] to %{{.*}} : !fir.ref<i32>
// CHECK:           fir.declare_value %[[CONSTANT_0]] {uniq_name = "_QFfooEjlocal"} : i32
// CHECK:           fir.store %[[CONSTANT_0]] to %{{.*}} : !fir.ref<i32>

func.func @test_two_values(%arg0: !fir.ref<i32> {fir.bindc_name = "i"}, %arg1: !fir.ref<i32> {fir.bindc_name = "j"}) {
  %c43_i32 = arith.constant 43 : i32
  %c42_i32 = arith.constant 42 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFfooEi"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %2 = fir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFfooEj"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %3 = fir.alloca i32 {bindc_name = "jlocal", uniq_name = "_QFfooEjlocal"}
  %4 = fir.declare %3 {uniq_name = "_QFfooEjlocal"} : (!fir.ref<i32>) -> !fir.ref<i32>
  fir.store %c42_i32 to %4 : !fir.ref<i32>
  %5 = fir.load %4 : !fir.ref<i32>
  fir.store %5 to %1 : !fir.ref<i32>
  fir.store %c43_i32 to %4 : !fir.ref<i32>
  %6 = fir.load %4 : !fir.ref<i32>
  fir.store %6 to %2 : !fir.ref<i32>
  return
}

// -----

// CHECK-LABEL:   func.func @array_val_not_mem2reg(
// CHECK:           fir.alloca !fir.array<2xi32>
// CHECK:           fir.store
// CHECK:           fir.load
// CHECK:           fir.store

func.func @array_val_not_mem2reg(%arg0: !fir.ref<!fir.array<2xi32>> {fir.bindc_name = "i"}, %arg1: !fir.ref<i32> {fir.bindc_name = "j"}, %arrayval : !fir.array<2xi32>) {
  %c2 = arith.constant 2 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c2 : (index) -> !fir.shape<1>
  %2 = fir.declare %arg0(%1) dummy_scope %0 arg 1 {uniq_name = "_QFarrayEi"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<2xi32>>
  %3 = fir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFarrayEj"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.alloca !fir.array<2xi32> {bindc_name = "jlocal", uniq_name = "_QFarrayEjlocal"}
  %5 = fir.declare %4(%1) {uniq_name = "_QFarrayEjlocal"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<2xi32>>
  fir.store %arrayval to %5 : !fir.ref<!fir.array<2xi32>>
  %val_load = fir.load %5 : !fir.ref<!fir.array<2xi32>>
  fir.store %val_load to %2 : !fir.ref<!fir.array<2xi32>>
  return
}

// -----

// CHECK-LABEL:   func.func @box_not_mem2reg(
// CHECK:           fir.alloca !fir.box<f32>
// CHECK:           fir.store
// CHECK:           fir.load
// CHECK:           fir.store

func.func @box_not_mem2reg(%arg0: !fir.ref<!fir.box<f32>> {fir.bindc_name = "i"}, %arg1: !fir.ref<i32> {fir.bindc_name = "j"}, %arrayval : !fir.box<f32>) {
  %0 = fir.dummy_scope : !fir.dscope
  %2 = fir.declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFarrayEi"} : (!fir.ref<!fir.box<f32>>, !fir.dscope) -> !fir.ref<!fir.box<f32>>
  %3 = fir.declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFarrayEj"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.alloca !fir.box<f32> {bindc_name = "jlocal", uniq_name = "_QFarrayEjlocal"}
  %5 = fir.declare %4 {uniq_name = "_QFarrayEjlocal"} : (!fir.ref<!fir.box<f32>>) -> !fir.ref<!fir.box<f32>>
  fir.store %arrayval to %5 : !fir.ref<!fir.box<f32>>
  %val_load = fir.load %5 : !fir.ref<!fir.box<f32>>
  fir.store %val_load to %2 : !fir.ref<!fir.box<f32>>
  return
}

// -----

// CHECK-LABEL:   func.func @block_argument_value(
// CHECK-SAME:      %[[ARG0:.*]]: i32,
// CHECK-SAME:      %[[ARG1:.*]]: i1) -> i32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 42 : i32
// CHECK:           fir.declare_value %[[CONSTANT_0]] {uniq_name = "_QFfooEjlocal"} : i32
// CHECK:           llvm.cond_br %[[ARG1]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           fir.declare_value %[[ARG0]] {uniq_name = "_QFfooEjlocal"} : i32
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:
// CHECK:           return %[[CONSTANT_0]] : i32
// CHECK:         }
func.func @block_argument_value(%arg0: i32, %cdt: i1) -> i32 {
  %c42_i32 = arith.constant 42 : i32
  %3 = fir.alloca i32 {bindc_name = "jlocal", uniq_name = "_QFfooEjlocal"}
  %4 = fir.declare %3 {uniq_name = "_QFfooEjlocal"} : (!fir.ref<i32>) -> !fir.ref<i32>
  fir.store %c42_i32 to %4 : !fir.ref<i32>
  llvm.cond_br %cdt, ^bb1, ^bb2
^bb1:
  fir.store %arg0 to %4 : !fir.ref<i32>
  llvm.br ^bb2
^bb2:
  %6 = fir.load %4 : !fir.ref<i32>
  return %6 : i32
}
