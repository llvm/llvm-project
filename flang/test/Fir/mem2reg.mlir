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

// Write in another block than the fir.declare: found through the alias, so the
// join point gets a block argument.

// CHECK-LABEL: func.func @block_argument_value(
// CHECK-SAME: %[[ARG0:.*]]: i32,
// CHECK-SAME: %[[ARG1:.*]]: i1) -> i32 {
// CHECK-NOT: fir.alloca
// CHECK: llvm.cond_br %[[ARG1]], ^bb1, ^bb2(%[[C42:.*]] : i32)
// CHECK: llvm.br ^bb2(%[[ARG0]] : i32)
// CHECK: ^bb2(%[[PHI:.*]]: i32):
// CHECK: return %[[PHI]] : i32
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

// Write inside a loop through fir.declare: the header takes a block argument.

// CHECK-LABEL: func.func @loop_conditional_update(
// CHECK-SAME: %[[ARG0:.*]]: i32,
// CHECK-SAME: %[[ARG1:.*]]: i1) -> i32 {
// CHECK-NOT: fir.alloca
// CHECK: llvm.br ^bb1(%[[ARG0]] : i32)
// CHECK: ^bb1(%[[PHI:.*]]: i32):
// CHECK: %[[NEW:.*]] = arith.subi %[[PHI]], %{{.*}} : i32
// CHECK: llvm.br ^bb1(%[[NEW]] : i32)
// CHECK: return %[[PHI]] : i32
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

// -----

// CHECK-LABEL: func.func @convert_preserving_pointee(
// CHECK-NOT: fir.alloca
func.func @convert_preserving_pointee(%arg : i32) {
  %alloca = fir.alloca i32
  %conv = fir.convert %alloca : (!fir.ref<i32>) -> !fir.ref<i32>
  fir.store %arg to %conv : !fir.ref<i32>
  %v = fir.load %conv : !fir.ref<i32>
  fir.call @use(%v) : (i32) -> ()
  return
}

// -----

// A cast that changes the pointee is not a transparent alias.

// CHECK-LABEL: func.func @convert_changing_pointee(
// CHECK: fir.alloca
func.func @convert_changing_pointee(%arg : i32) {
  %alloca = fir.alloca i32
  fir.store %arg to %alloca : !fir.ref<i32>
  %conv = fir.convert %alloca : (!fir.ref<i32>) -> !fir.ref<f32>
  %v = fir.load %conv : !fir.ref<f32>
  fir.call @usef(%v) : (f32) -> ()
  return
}

// -----

// A scalar slot reached through both cast directions, as lowering emits it:
//   memref.alloca -> fir.convert -> fir.declare -> fir.convert -> store/load.

// CHECK-LABEL: func.func @scalar_slot_through_casts(
// CHECK-NOT: memref.alloca
// CHECK-NOT: fir.declare
// CHECK: %[[IDX:.*]] = arith.index_cast %{{.*}} : index to i32
// CHECK: fir.declare_value %[[IDX]]
// CHECK: "test.use"(%[[IDX]])
func.func @scalar_slot_through_casts(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%i) = (%c0) to (%n) step (%c1) {
    %alloca = memref.alloca() {bindc_name = "k"} : memref<i32>
    %r = fir.convert %alloca : (memref<i32>) -> !fir.ref<i32>
    %d = fir.declare %r {uniq_name = "_QFEk"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %v = arith.index_cast %i : index to i32
    %m = fir.convert %d : (!fir.ref<i32>) -> memref<i32>
    memref.store %v, %m[] : memref<i32>
    %l = memref.load %m[] : memref<i32>
    "test.use"(%l) : (i32) -> ()
    scf.reduce
  }
  return
}

// -----

// A ranked memref may be indexed, so the cast is not a transparent alias.

// CHECK-LABEL: func.func @ranked_memref_not_promoted(
// CHECK: fir.alloca
func.func @ranked_memref_not_promoted(%arg: i32) {
  %c0 = arith.constant 0 : index
  %alloca = fir.alloca i32
  %m = fir.convert %alloca : (!fir.ref<i32>) -> memref<1xi32>
  memref.store %arg, %m[%c0] : memref<1xi32>
  %l = memref.load %m[%c0] : memref<1xi32>
  "test.use"(%l) : (i32) -> ()
  return
}

// -----

// A read that no write reaches would take the default value of the allocation,
// which is poison here, so the cast must not report itself as an alias.

// CHECK-LABEL: func.func @read_only_slot_not_promoted(
// CHECK-NOT: ub.poison
// CHECK: memref.alloca
// CHECK: memref.load
func.func @read_only_slot_not_promoted() {
  %alloca = memref.alloca() {bindc_name = "x"} : memref<i32>
  %r = fir.convert %alloca : (memref<i32>) -> !fir.ref<i32>
  %d = fir.declare %r {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %m = fir.convert %d : (!fir.ref<i32>) -> memref<i32>
  %l = memref.load %m[] : memref<i32>
  "test.use"(%l) : (i32) -> ()
  return
}

// -----

// Written inside a conditional region, so no write dominates the read.

// CHECK-LABEL: func.func @partially_written_slot_not_promoted(
// CHECK-NOT: ub.poison
// CHECK: memref.alloca
// CHECK: memref.load
func.func @partially_written_slot_not_promoted(%c: i1, %arg: i32) {
  %alloca = memref.alloca() {bindc_name = "x"} : memref<i32>
  %r = fir.convert %alloca : (memref<i32>) -> !fir.ref<i32>
  %d = fir.declare %r {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %m = fir.convert %d : (!fir.ref<i32>) -> memref<i32>
  scf.if %c {
    memref.store %arg, %m[] : memref<i32>
  }
  %l = memref.load %m[] : memref<i32>
  "test.use"(%l) : (i32) -> ()
  return
}

// -----

// The write is through a different alias than the read, so it is only found by
// walking to the root of the chain.

// CHECK-LABEL: func.func @store_through_sibling_alias(
// CHECK-NOT: memref.alloca
// CHECK: "test.use"(%[[ARG:.*]])
func.func @store_through_sibling_alias(%arg: i32) {
  %alloca = memref.alloca() {bindc_name = "x"} : memref<i32>
  %r = fir.convert %alloca : (memref<i32>) -> !fir.ref<i32>
  %d = fir.declare %r {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %stored = fir.convert %d : (!fir.ref<i32>) -> memref<i32>
  memref.store %arg, %stored[] : memref<i32>
  %loaded = fir.convert %d : (!fir.ref<i32>) -> memref<i32>
  %l = memref.load %loaded[] : memref<i32>
  "test.use"(%l) : (i32) -> ()
  return
}

// -----

// A declare that does not carry the pointee through would need the values of
// the two slots to be converted, so it is not an alias.

// CHECK-LABEL: func.func @declare_changing_pointee(
// CHECK: fir.alloca
// CHECK: fir.declare
// CHECK: fir.load
func.func @declare_changing_pointee() {
  %alloca = fir.alloca i32
  %d = fir.declare %alloca {uniq_name = "x"} : (!fir.ref<i32>) -> !fir.ref<f32>
  %l = fir.load %d : !fir.ref<f32>
  "test.use"(%l) : (f32) -> ()
  return
}

// -----

// A floating-point slot is not aliased, so it is promoted only when all its
// uses are in one block, where no block argument is needed.

// CHECK-LABEL: func.func @fp_declare_single_block(
// CHECK-SAME: %[[ARG0:.*]]: f32) -> f32 {
// CHECK-NOT: fir.alloca
// CHECK: fir.declare_value %[[ARG0]]
// CHECK: return %[[ARG0]] : f32
func.func @fp_declare_single_block(%arg: f32) -> f32 {
  %alloca = fir.alloca f32
  %d = fir.declare %alloca {uniq_name = "_QFEx"} : (!fir.ref<f32>) -> !fir.ref<f32>
  fir.store %arg to %d : !fir.ref<f32>
  %v = fir.load %d : !fir.ref<f32>
  return %v : f32
}

// -----

// Without an alias, mem2reg does not see the write in the other block, so a
// floating-point slot used across blocks stays in memory.

// CHECK-LABEL: func.func @fp_declare_multi_block(
// CHECK: fir.alloca f32
// CHECK: fir.declare
// CHECK: fir.load
func.func @fp_declare_multi_block(%arg: f32, %cdt: i1) -> f32 {
  %alloca = fir.alloca f32
  %d = fir.declare %alloca {uniq_name = "_QFEx"} : (!fir.ref<f32>) -> !fir.ref<f32>
  fir.store %arg to %d : !fir.ref<f32>
  llvm.cond_br %cdt, ^bb1, ^bb2
^bb1:
  fir.store %arg to %d : !fir.ref<f32>
  llvm.br ^bb2
^bb2:
  %v = fir.load %d : !fir.ref<f32>
  return %v : f32
}

// -----

// A memory space lets the cast relocate the storage, so it is not an alias.

// CHECK-LABEL: func.func @memref_memory_space_not_promoted(
// CHECK: memref.alloca
// CHECK: fir.load
func.func @memref_memory_space_not_promoted(%arg: i32) {
  %alloca = memref.alloca() : memref<i32, 1>
  %c = fir.convert %alloca : (memref<i32, 1>) -> !fir.ref<i32>
  fir.store %arg to %c : !fir.ref<i32>
  %l = fir.load %c : !fir.ref<i32>
  "test.use"(%l) : (i32) -> ()
  return
}

// -----

// The address escapes through the cast, so the storage has to be kept.

// CHECK-LABEL: func.func @escape_through_cast(
// CHECK: memref.alloca
// CHECK: fir.call @escapes
// CHECK: fir.load
func.func @escape_through_cast(%arg: i32) {
  %alloca = memref.alloca() : memref<i32>
  %c = fir.convert %alloca : (memref<i32>) -> !fir.ref<i32>
  fir.store %arg to %c : !fir.ref<i32>
  fir.call @escapes(%c) : (!fir.ref<i32>) -> ()
  %l = fir.load %c : !fir.ref<i32>
  "test.use"(%l) : (i32) -> ()
  return
}

// -----

// A cast may add volatility, which the default verifier accepts. Promotion
// would elide the accesses that volatility asks to be kept.

// CHECK-LABEL: func.func @volatile_convert_not_promoted(
// CHECK: memref.alloca
// CHECK: fir.store
// CHECK: fir.load
func.func @volatile_convert_not_promoted(%arg: i32) {
  %alloca = memref.alloca() : memref<i32>
  %c = fir.convert %alloca : (memref<i32>) -> !fir.ref<i32, volatile>
  fir.store %arg to %c : !fir.ref<i32, volatile>
  %l = fir.load %c : !fir.ref<i32, volatile>
  "test.use"(%l) : (i32) -> ()
  return
}

// -----

// The write dominates the read from another block, so the read cannot observe
// the uninitialized slot and no poison is needed.

// CHECK-LABEL: func.func @write_dominating_read_in_another_block(
// CHECK-SAME: %[[ARG:.*]]: i32
// CHECK-NOT: memref.alloca
// CHECK-NOT: ub.poison
// CHECK: "test.use"(%[[ARG]])
func.func @write_dominating_read_in_another_block(%arg: i32) {
  %alloca = memref.alloca() {bindc_name = "x"} : memref<i32>
  %r = fir.convert %alloca : (memref<i32>) -> !fir.ref<i32>
  %d = fir.declare %r {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %m = fir.convert %d : (!fir.ref<i32>) -> memref<i32>
  memref.store %arg, %m[] : memref<i32>
  cf.br ^bb1
^bb1:
  %l = memref.load %m[] : memref<i32>
  "test.use"(%l) : (i32) -> ()
  return
}

// -----

// Initialized before a loop and updated inside it. The initializing write
// dominates the in-loop read, so the slot becomes a block argument.

// CHECK-LABEL: func.func @write_before_loop_updated_in_loop(
// CHECK-SAME: %[[ARG:[^:]*]]: i32, %[[N:[^:]*]]: i32
// CHECK-NOT: memref.alloca
// CHECK-NOT: ub.poison
// CHECK: cf.br ^bb1(%[[ARG]] : i32)
// CHECK: ^bb1(%[[PHI:.*]]: i32)
// CHECK: arith.addi %[[PHI]], %[[ARG]]
func.func @write_before_loop_updated_in_loop(%arg: i32, %n: i32) {
  %alloca = memref.alloca() {bindc_name = "x"} : memref<i32>
  %r = fir.convert %alloca : (memref<i32>) -> !fir.ref<i32>
  %d = fir.declare %r {uniq_name = "_QFEx"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %m = fir.convert %d : (!fir.ref<i32>) -> memref<i32>
  memref.store %arg, %m[] : memref<i32>
  cf.br ^bb1
^bb1:
  %l = memref.load %m[] : memref<i32>
  %next = arith.addi %l, %arg : i32
  memref.store %next, %m[] : memref<i32>
  %c = arith.cmpi slt, %next, %n : i32
  cf.cond_br %c, ^bb1, ^bb2
^bb2:
  %f = memref.load %m[] : memref<i32>
  "test.use"(%f) : (i32) -> ()
  return
}
