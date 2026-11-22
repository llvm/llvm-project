// Test that FIR alias analysis sees through view operations from other
// dialects, e.g. the memref.reinterpret_cast ops created by fir-to-memref.
// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// Two distinct dummy arguments without TARGET do not alias.
// CHECK-LABEL: Testing : "_QPdummies"
// CHECK-DAG: a#0 <-> b#0: NoAlias
func.func @_QPdummies(%arg0: !fir.ref<!fir.array<10xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<10xf64>> {fir.bindc_name = "b"}) {
  %c10 = arith.constant 10 : index
  %scope = fir.dummy_scope : !fir.dscope
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %da = fir.declare %arg0(%shape) dummy_scope %scope arg 1 uniq_name("_QFdummiesEa") : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<10xf64>>
  %db = fir.declare %arg1(%shape) dummy_scope %scope arg 2 uniq_name("_QFdummiesEb") : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<10xf64>>
  %ma = fir.convert %da : (!fir.ref<!fir.array<10xf64>>) -> memref<10xf64>
  %mb = fir.convert %db : (!fir.ref<!fir.array<10xf64>>) -> memref<10xf64>
  %ra = memref.reinterpret_cast %ma to offset: [0], sizes: [10], strides: [1] {test.ptr = "a"} : memref<10xf64> to memref<10xf64, strided<[1]>>
  %rb = memref.reinterpret_cast %mb to offset: [0], sizes: [10], strides: [1] {test.ptr = "b"} : memref<10xf64> to memref<10xf64, strided<[1]>>
  return
}

// -----

// A dummy argument and a local object do not alias.
// CHECK-LABEL: Testing : "_QPdummy_local"
// CHECK-DAG: a#0 <-> loc#0: NoAlias
func.func @_QPdummy_local(%arg0: !fir.ref<!fir.array<10xf64>> {fir.bindc_name = "a"}) {
  %c10 = arith.constant 10 : index
  %scope = fir.dummy_scope : !fir.dscope
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %da = fir.declare %arg0(%shape) dummy_scope %scope arg 1 uniq_name("_QFdummy_localEa") : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<10xf64>>
  %l = fir.alloca !fir.array<10xf64> {bindc_name = "l", uniq_name = "_QFdummy_localEl"}
  %dl = fir.declare %l(%shape) uniq_name("_QFdummy_localEl") : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf64>>
  %ma = fir.convert %da : (!fir.ref<!fir.array<10xf64>>) -> memref<10xf64>
  %ml = fir.convert %dl : (!fir.ref<!fir.array<10xf64>>) -> memref<10xf64>
  %ca = memref.cast %ma {test.ptr = "a"} : memref<10xf64> to memref<?xf64>
  %rl = memref.reinterpret_cast %ml to offset: [0], sizes: [10], strides: [1] {test.ptr = "loc"} : memref<10xf64> to memref<10xf64, strided<[1]>>
  return
}

// -----

// TARGET dummy arguments may alias.
// CHECK-LABEL: Testing : "_QPtarget"
// CHECK-DAG: a#0 <-> b#0: MayAlias
func.func @_QPtarget(%arg0: !fir.ref<!fir.array<10xf64>> {fir.bindc_name = "a", fir.target}, %arg1: !fir.ref<!fir.array<10xf64>> {fir.bindc_name = "b", fir.target}) {
  %c10 = arith.constant 10 : index
  %scope = fir.dummy_scope : !fir.dscope
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %da = fir.declare %arg0(%shape) dummy_scope %scope arg 1 uniq_name("_QFtargetEa") fortran_attrs<target> : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<10xf64>>
  %db = fir.declare %arg1(%shape) dummy_scope %scope arg 2 uniq_name("_QFtargetEb") fortran_attrs<target> : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<10xf64>>
  %ma = fir.convert %da : (!fir.ref<!fir.array<10xf64>>) -> memref<10xf64>
  %mb = fir.convert %db : (!fir.ref<!fir.array<10xf64>>) -> memref<10xf64>
  %ra = memref.reinterpret_cast %ma to offset: [0], sizes: [10], strides: [1] {test.ptr = "a"} : memref<10xf64> to memref<10xf64, strided<[1]>>
  %rb = memref.reinterpret_cast %mb to offset: [0], sizes: [10], strides: [1] {test.ptr = "b"} : memref<10xf64> to memref<10xf64, strided<[1]>>
  return
}

// -----

// Only memref.cast and memref.reinterpret_cast are looked through. Other views
// stop the walk, so the result stays MayAlias.
// CHECK-LABEL: Testing : "_QPnot_opted_in"
// CHECK-DAG: a#0 <-> b#0: MayAlias
func.func @_QPnot_opted_in(%arg0: !fir.ref<!fir.array<10xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<10xf64>> {fir.bindc_name = "b"}) {
  %c10 = arith.constant 10 : index
  %scope = fir.dummy_scope : !fir.dscope
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %da = fir.declare %arg0(%shape) dummy_scope %scope arg 1 uniq_name("_QFnot_opted_inEa") : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<10xf64>>
  %db = fir.declare %arg1(%shape) dummy_scope %scope arg 2 uniq_name("_QFnot_opted_inEb") : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<10xf64>>
  %ma = fir.convert %da : (!fir.ref<!fir.array<10xf64>>) -> memref<10xf64>
  %mb = fir.convert %db : (!fir.ref<!fir.array<10xf64>>) -> memref<10xf64>
  %sa = memref.subview %ma[0] [10] [1] {test.ptr = "a"} : memref<10xf64> to memref<10xf64, strided<[1]>>
  %sb = memref.subview %mb[0] [10] [1] {test.ptr = "b"} : memref<10xf64> to memref<10xf64, strided<[1]>>
  return
}

// -----

// A reinterpret_cast with a zero offset starts at the address of its source,
// whether the offset is written statically or is a constant operand. A
// non-constant offset is unknown. The views are made from the block argument
// so that LocalAliasAnalysis cannot answer for the declare.
// CHECK-LABEL: Testing : "_QPconstant_offset"
// CHECK-DAG: a#0 <-> static#0: MustAlias
// CHECK-DAG: a#0 <-> ssa#0: MustAlias
// CHECK-DAG: a#0 <-> unknown#0: MayAlias
func.func @_QPconstant_offset(%arg0: !fir.ref<!fir.array<10xf64>> {fir.bindc_name = "a"}, %off: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %scope = fir.dummy_scope : !fir.dscope
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %da = fir.declare %arg0(%shape) dummy_scope %scope arg 1 uniq_name("_QFconstant_offsetEa") {test.ptr = "a"} : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<10xf64>>
  %m = fir.convert %arg0 : (!fir.ref<!fir.array<10xf64>>) -> memref<10xf64>
  %static = memref.reinterpret_cast %m to offset: [0], sizes: [10], strides: [1] {test.ptr = "static"} : memref<10xf64> to memref<10xf64, strided<[1]>>
  %ssa = memref.reinterpret_cast %m to offset: [%c0], sizes: [%c10], strides: [%c1] {test.ptr = "ssa"} : memref<10xf64> to memref<?xf64, strided<[?], offset: ?>>
  %unknown = memref.reinterpret_cast %m to offset: [%off], sizes: [%c10], strides: [%c1] {test.ptr = "unknown"} : memref<10xf64> to memref<?xf64, strided<[?], offset: ?>>
  return
}

// -----

// A cast that keeps the address of its source, such as a zero-offset
// reinterpret_cast or a memref.cast, must alias it. A view at a different
// offset, or a view of one, is only known to be part of the same object. The
// views are made from the block argument so that LocalAliasAnalysis cannot
// answer for the declare.
// CHECK-LABEL: Testing : "_QPoffsets"
// CHECK-DAG: a#0 <-> zero#0: MustAlias
// CHECK-DAG: a#0 <-> dyn_layout#0: MustAlias
// CHECK-DAG: a#0 <-> five#0: MayAlias
// CHECK-DAG: b#0 <-> five#0: NoAlias
// CHECK-DAG: a#0 <-> back#0: MayAlias
func.func @_QPoffsets(%arg0: !fir.ref<!fir.array<10xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<10xf64>> {fir.bindc_name = "b"}) {
  %c10 = arith.constant 10 : index
  %scope = fir.dummy_scope : !fir.dscope
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %da = fir.declare %arg0(%shape) dummy_scope %scope arg 1 uniq_name("_QFoffsetsEa") {test.ptr = "a"} : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<10xf64>>
  %db = fir.declare %arg1(%shape) dummy_scope %scope arg 2 uniq_name("_QFoffsetsEb") {test.ptr = "b"} : (!fir.ref<!fir.array<10xf64>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<10xf64>>
  %ma = fir.convert %arg0 : (!fir.ref<!fir.array<10xf64>>) -> memref<10xf64>
  %zero = memref.reinterpret_cast %ma to offset: [0], sizes: [10], strides: [1] {test.ptr = "zero"} : memref<10xf64> to memref<10xf64, strided<[1]>>
  %dyn = memref.cast %ma {test.ptr = "dyn_layout"} : memref<10xf64> to memref<10xf64, strided<[1], offset: ?>>
  %five = memref.reinterpret_cast %ma to offset: [5], sizes: [5], strides: [1] {test.ptr = "five"} : memref<10xf64> to memref<5xf64, strided<[1], offset: 5>>
  %back = memref.reinterpret_cast %five to offset: [0], sizes: [10], strides: [1] {test.ptr = "back"} : memref<5xf64, strided<[1], offset: 5>> to memref<10xf64, strided<[1]>>
  return
}
