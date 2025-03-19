// Test alias analysis queries for dummy arguments wired
// through fir.pack_array.
// fir.pack_array is a pass-through operation for FIR alias analysis.
// RUN: fir-opt %s --test-fir-alias-analysis -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// CHECK: test1_y(1)#0 <-> test1_x(1)#0: NoAlias
func.func @_QFtest1(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}, %arg1: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "y"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.pack_array %arg0 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %2:2 = hlfir.declare %1 dummy_scope %0 {uniq_name = "_QFtest1Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %3 = fir.pack_array %arg1 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %4:2 = hlfir.declare %3 dummy_scope %0 {uniq_name = "_QFtest1Ey"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %5 = hlfir.designate %4#0 (%c1) {test.ptr = "test1_y(1)"} : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  %6 = fir.load %5 : !fir.ref<f32>
  %7 = hlfir.designate %2#0 (%c1) {test.ptr = "test1_x(1)"} : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  hlfir.assign %6 to %7 : f32, !fir.ref<f32>
  fir.unpack_array %3 to %arg1 heap : !fir.box<!fir.array<?xf32>>
  fir.unpack_array %1 to %arg0 heap : !fir.box<!fir.array<?xf32>>
  return
}
