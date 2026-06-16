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

// Conditional store in a different block through fir.declare is promoted:
// the through-declare store is discovered as a defining block via the
// `getPromotableSlotView` interface and a block argument is added at the
// merge point.

// CHECK-LABEL: func.func @block_argument_value(
// CHECK-SAME: %[[ARG0:.*]]: i32,
// CHECK-SAME: %[[ARG1:.*]]: i1) -> i32 {
// CHECK-NOT: fir.alloca
// CHECK-NOT: fir.declare {{.*}} : (!fir.ref<i32>)
// CHECK: %[[C42:.*]] = arith.constant 42 : i32
// CHECK: fir.declare_value %[[C42]] {{.*}}
// CHECK: llvm.cond_br %[[ARG1]], ^bb1, ^bb2(%[[C42]] : i32)
// CHECK: ^bb1:
// CHECK: fir.declare_value %[[ARG0]] {{.*}}
// CHECK: llvm.br ^bb2(%[[ARG0]] : i32)
// CHECK: ^bb2(%[[MERGE:.*]]: i32):
// CHECK: return %[[MERGE]] : i32
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

// -----

// Conditional store inside a loop through fir.declare is promoted: the
// loop header gets a block argument carrying the conditional update from
// the loop body. This is the case that motivated the previous same-block
// restriction in fir::DeclareOp::canUsesBeRemoved.

// CHECK-LABEL: func.func @loop_conditional_update(
// CHECK-SAME: %[[ARG0:.*]]: i32,
// CHECK-SAME: %[[ARG1:.*]]: i1) -> i32 {
// CHECK-NOT: fir.alloca
// CHECK-NOT: fir.declare {{.*}} : (!fir.ref<i32>)
// CHECK: fir.declare_value %[[ARG0]] {{.*}}
// CHECK: llvm.br ^bb1(%[[ARG0]] : i32)
// CHECK: ^bb1(%[[LOOP_ARG:.*]]: i32):
// CHECK: llvm.cond_br %[[ARG1]], ^bb2, ^bb3
// CHECK: ^bb2:
// CHECK: %[[NEW:.*]] = arith.subi %[[LOOP_ARG]], {{.*}} : i32
// CHECK: fir.declare_value %[[NEW]] {{.*}}
// CHECK: llvm.br ^bb1(%[[NEW]] : i32)
// CHECK: ^bb3:
// CHECK: return %[[LOOP_ARG]] : i32
func.func @loop_conditional_update(%arg0: i32, %cdt: i1) -> i32 {
  %c1 = arith.constant 1 : i32
  %alloca = fir.alloca i32 {bindc_name = "mywatch", uniq_name = "_QFkernelEmywatch"}
  %declare = fir.declare %alloca {uniq_name = "_QFkernelEmywatch"} : (!fir.ref<i32>) -> !fir.ref<i32>
  fir.store %arg0 to %declare : !fir.ref<i32>
  llvm.br ^loop
^loop:
  %val = fir.load %declare : !fir.ref<i32>
  llvm.cond_br %cdt, ^update, ^exit
^update:
  %new = arith.subi %val, %c1 : i32
  fir.store %new to %declare : !fir.ref<i32>
  llvm.br ^loop
^exit:
  %result = fir.load %declare : !fir.ref<i32>
  return %result : i32
}

// -----

// fir.convert at the same element type is a transparent view; the slot is
// fully promoted with no value conversion needed.

// CHECK-LABEL: func.func @convert_same_type(
// CHECK-SAME: %[[ARG0:.*]]: i32) -> i32 {
// CHECK-NOT: fir.alloca
// CHECK-NOT: fir.convert
// CHECK: return %[[ARG0]] : i32
func.func @convert_same_type(%arg0: i32) -> i32 {
  %alloca = fir.alloca i32
  %ptr = fir.convert %alloca : (!fir.ref<i32>) -> !fir.ref<i32>
  fir.store %arg0 to %ptr : !fir.ref<i32>
  %v = fir.load %ptr : !fir.ref<i32>
  return %v : i32
}

// -----

// A type-changing fir.convert exposes the slot at a different element type;
// mem2reg materialises fir.bitcast value conversions at the store and load.

// CHECK-LABEL: func.func @convert_type_changing(
// CHECK-SAME: %[[ARG0:.*]]: f32) -> f32
// CHECK-NOT: fir.alloca
// CHECK: %[[I32:.*]] = fir.bitcast %[[ARG0]] : (f32) -> i32
// CHECK: fir.bitcast %[[I32]] : (i32) -> f32
// CHECK: return %{{.*}} : f32
func.func @convert_type_changing(%arg0: f32) -> f32 {
  %alloca = fir.alloca i32
  %ptr = fir.convert %alloca : (!fir.ref<i32>) -> !fir.ref<f32>
  fir.store %arg0 to %ptr : !fir.ref<f32>
  %v = fir.load %ptr : !fir.ref<f32>
  return %v : f32
}

// -----

// Chained view: alloca -> fir.declare -> fir.convert -> load/store. The
// declare's element type (i32) differs from the value type the store sees
// at the leaf view (f32), so `fir.declare_value` is skipped until
// `fir.declare_value` can carry the declared type independently of the
// value type (see the TODO in fir::DeclareOp::visitReplacedValues).

// CHECK-LABEL: func.func @declare_then_convert(
// CHECK-SAME: %[[ARG0:.*]]: f32) -> f32
// CHECK-NOT: fir.alloca
// CHECK-NOT: fir.declare {{.*}} : (!fir.ref<i32>)
// CHECK-NOT: fir.declare_value
// CHECK: %[[I32:.*]] = fir.bitcast %[[ARG0]] : (f32) -> i32
// CHECK: %[[F32:.*]] = fir.bitcast %[[I32]] : (i32) -> f32
// CHECK: return %[[F32]] : f32
func.func @declare_then_convert(%arg0: f32) -> f32 {
  %alloca = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
  %declare = fir.declare %alloca {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %ptr = fir.convert %declare : (!fir.ref<i32>) -> !fir.ref<f32>
  fir.store %arg0 to %ptr : !fir.ref<f32>
  %v = fir.load %ptr : !fir.ref<f32>
  return %v : f32
}

// -----

// Inverse chain: alloca -> fir.convert -> fir.declare -> load/store. The
// declare's memref is the result of a type-changing convert, so the
// declare sees the slot at f32 — the same type as the store value, so
// `fir.declare_value` is emitted.

// CHECK-LABEL: func.func @convert_then_declare(
// CHECK-SAME: %[[ARG0:.*]]: f32) -> f32
// CHECK-NOT: fir.alloca
// CHECK-NOT: fir.declare {{.*}} : (!fir.ref<f32>)
// CHECK: %[[I32:.*]] = fir.bitcast %[[ARG0]] : (f32) -> i32
// CHECK: fir.declare_value %[[ARG0]] {{.*}} : f32
// CHECK: %[[F32:.*]] = fir.bitcast %[[I32]] : (i32) -> f32
// CHECK: return %[[F32]] : f32
func.func @convert_then_declare(%arg0: f32) -> f32 {
  %alloca = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFEx"}
  %ptr = fir.convert %alloca : (!fir.ref<i32>) -> !fir.ref<f32>
  %declare = fir.declare %ptr {uniq_name = "_QFEx"} : (!fir.ref<f32>) -> !fir.ref<f32>
  fir.store %arg0 to %declare : !fir.ref<f32>
  %v = fir.load %declare : !fir.ref<f32>
  return %v : f32
}

// -----

// A memref.alloca slot with a mixed memref/FIR view chain
// (memref -> fir.convert -> fir.declare -> fir.convert -> memref). All
// element types are the same, so the slot promotes with no value
// conversions; only `fir.declare_value` is emitted from the declare.

// CHECK-LABEL: func.func @memref_with_declare_chain(
// CHECK-SAME: %[[ARG0:.*]]: i32) -> i32
// CHECK-NOT: memref.alloca
// CHECK-NOT: fir.declare {{.*}} : (!fir.ref<i32>)
// CHECK-NOT: fir.convert
// CHECK-NOT: memref.store
// CHECK-NOT: memref.load
// CHECK: fir.declare_value %[[ARG0]] {{.*}} : i32
// CHECK: return %[[ARG0]] : i32
func.func @memref_with_declare_chain(%arg0: i32) -> i32 {
  %alloca = memref.alloca() : memref<i32>
  %fref = fir.convert %alloca : (memref<i32>) -> !fir.ref<i32>
  %decl = fir.declare %fref {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %mref = fir.convert %decl : (!fir.ref<i32>) -> memref<i32>
  memref.store %arg0, %mref[] : memref<i32>
  %v = memref.load %mref[] : memref<i32>
  return %v : i32
}

// -----

// Same mixed memref/FIR chain with a conditional store across blocks. The
// merge-point block argument is at the root slot's element type (i32), and
// `fir.declare_value` is emitted at each store site.

// CHECK-LABEL: func.func @memref_with_declare_chain_blocks(
// CHECK-SAME: %[[ARG0:.*]]: i32,
// CHECK-SAME: %[[COND:.*]]: i1) -> i32
// CHECK-NOT: memref.alloca
// CHECK-NOT: fir.declare {{.*}} : (!fir.ref<i32>)
// CHECK: %[[C42:.*]] = arith.constant 42 : i32
// CHECK: fir.declare_value %[[C42]] {{.*}} : i32
// CHECK: cf.cond_br %[[COND]], ^[[BB1:.*]], ^[[BB2:.*]](%[[C42]] : i32)
// CHECK: ^[[BB1]]:
// CHECK: fir.declare_value %[[ARG0]] {{.*}} : i32
// CHECK: cf.br ^[[BB2]](%[[ARG0]] : i32)
// CHECK: ^[[BB2]](%[[MERGE:.*]]: i32):
// CHECK: return %[[MERGE]] : i32
func.func @memref_with_declare_chain_blocks(%arg0: i32, %cond: i1) -> i32 {
  %c42 = arith.constant 42 : i32
  %alloca = memref.alloca() : memref<i32>
  %fref = fir.convert %alloca : (memref<i32>) -> !fir.ref<i32>
  %decl = fir.declare %fref {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %mref = fir.convert %decl : (!fir.ref<i32>) -> memref<i32>
  memref.store %c42, %mref[] : memref<i32>
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  memref.store %arg0, %mref[] : memref<i32>
  cf.br ^bb2
^bb2:
  %v = memref.load %mref[] : memref<i32>
  return %v : i32
}

// -----

// A ref->integer fir.convert is not a memory view; it must block promotion.

// CHECK-LABEL: func.func @convert_to_integer_blocks_promotion(
// CHECK: %[[ALLOCA:.*]] = fir.alloca i32
// CHECK: fir.store
// CHECK: %[[INT:.*]] = fir.convert %[[ALLOCA]] : (!fir.ref<i32>) -> i64
// CHECK: fir.call @use_addr(%[[INT]])
func.func private @use_addr(%a: i64)
func.func @convert_to_integer_blocks_promotion(%arg0: i32) {
  %alloca = fir.alloca i32
  fir.store %arg0 to %alloca : !fir.ref<i32>
  %addr = fir.convert %alloca : (!fir.ref<i32>) -> i64
  fir.call @use_addr(%addr) : (i64) -> ()
  return
}

// -----

// Make sure we do not generate fir.declare_value for a replaced value
// fir.declare with dummy_scope. This can result in the declare_value being
// inserted before the dummy_scope it uses as would be the case here.

// CHECK-LABEL: func.func @dummy_scope(
// CHECK-NOT: fir.declare_value
func.func @dummy_scope(%arg : i32) {
  %alloca = fir.alloca i32 {adapt.valuebyref}
  fir.store %arg to %alloca : !fir.ref<i32>
  %scope = fir.dummy_scope : !fir.dscope
  %declare = fir.declare %alloca dummy_scope %scope arg 1 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "foo"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %result = fir.load %declare : !fir.ref<i32>
  fir.call @use(%result) : (i32) -> ()
  return
}

// -----

// Do not create block-argument fir.declare_value ops with non-dominating
// dummy scopes.

// CHECK-LABEL: func.func @dummy_scope_block_argument(
// CHECK: ^bb1(%{{.*}}: i32):
// CHECK-NOT: fir.declare_value
func.func @dummy_scope_block_argument(%arg : i32, %cond : i1) {
  %c1 = arith.constant 1 : i32
  %alloca = fir.alloca i32 {adapt.valuebyref}
  fir.store %arg to %alloca : !fir.ref<i32>
  cf.br ^loop
^loop:
  %result = fir.load %alloca : !fir.ref<i32>
  cf.cond_br %cond, ^body, ^exit
^body:
  %scope = fir.dummy_scope : !fir.dscope
  %declare = fir.declare %alloca dummy_scope %scope arg 1 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "foo"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %next = arith.addi %result, %c1 : i32
  fir.store %next to %alloca : !fir.ref<i32>
  cf.br ^loop
^exit:
  fir.call @use(%result) : (i32) -> ()
  return
}
