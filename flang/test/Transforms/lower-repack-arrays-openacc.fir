// RUN: fir-opt --lower-repack-arrays %s | FileCheck %s

// Test the current implementation of #fir.acc_safe_temp_array_copy attribute.
// It results in removal of fir.[un]pack_array operations.
func.func @_QPtest(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.pack_array %arg0 heap whole is_safe [#fir.acc_safe_temp_array_copy] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>>
  %2 = fir.declare %1 dummy_scope %0 {uniq_name = "_QFtestEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  fir.unpack_array %1 to %arg0 heap is_safe [#fir.acc_safe_temp_array_copy] : !fir.box<!fir.array<?xf32>>
  return
}
// CHECK-LABEL:   func.func @_QPtest(
// CHECK-SAME:                       %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
// CHECK-NEXT:      %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
// CHECK-NEXT:      %[[VAL_2:.*]] = fir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QFtestEx"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
