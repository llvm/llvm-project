// Test alias analysis queries for dummy arguments wired
// through fir.pack_array.
// fir.pack_array is a pass-through operation for FIR alias analysis.
// RUN: fir-opt %s --test-fir-alias-analysis -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// The two pointers referencing two different maybe repacked
// versions of the original dummy arguments do not alias:
// CHECK-DAG: test1_y_repack(1)#0 <-> test1_x_repack(1)#0: NoAlias
// CHECK-DAG: test1_x_orig(1)#0 <-> test1_y_orig(1)#0: NoAlias

// Repacked dummy does not alias with another original dummy:
// CHECK-DAG: test1_y_repack(1)#0 <-> test1_x_orig(1)#0: NoAlias
// CHECK-DAG: test1_x_repack(1)#0 <-> test1_y_orig(1)#0: NoAlias

// Repacked dummy may alias with its original:
// CHECK-DAG: test1_x_repack(1)#0 <-> test1_x_orig(1)#0: MayAlias
// CHECK-DAG: test1_y_repack(1)#0 <-> test1_y_orig(1)#0: MayAlias

// Ideally, these should report MustAlias, but MayAlias
// may work as well:
// CHECK-DAG: test1_y_repack(1)#0 <-> test1_y_repack2(1)#0: MayAlias
// CHECK-DAG: test1_x_repack(1)#0 <-> test1_x_repack2(1)#0: MayAlias


func.func @_QFtest1(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}, %arg1: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "y"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.pack_array %arg0 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %2:2 = hlfir.declare %1 dummy_scope %0 {uniq_name = "_QFtest1Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %3 = fir.pack_array %arg1 heap whole : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %4:2 = hlfir.declare %3 dummy_scope %0 {uniq_name = "_QFtest1Ey"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %5 = fir.box_addr %4#0 {test.ptr = "test1_y_repack(1)"} : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<f32>
  %52 = fir.box_addr %4#0 {test.ptr = "test1_y_repack2(1)"} : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<f32>
  %6 = fir.load %5 : !fir.ref<f32>
  %7 = fir.box_addr %2#0 {test.ptr = "test1_x_repack(1)"} : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<f32>
  %72 = fir.box_addr %2#0 {test.ptr = "test1_x_repack2(1)"} : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<f32>
  hlfir.assign %6 to %7 : f32, !fir.ref<f32>
  fir.unpack_array %3 to %arg1 heap : !fir.box<!fir.array<?xf32>>
  fir.unpack_array %1 to %arg0 heap : !fir.box<!fir.array<?xf32>>
  %8 = fir.box_addr %arg0 {test.ptr = "test1_x_orig(1)"} : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<f32>
  %9 = fir.box_addr %arg1 {test.ptr = "test1_y_orig(1)"} : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<f32>
  return
}
