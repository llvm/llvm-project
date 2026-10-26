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
