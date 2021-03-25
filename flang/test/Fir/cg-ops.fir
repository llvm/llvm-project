// RUN: fir-opt --pass-pipeline="func(cg-rewrite),fir.global(cg-rewrite),cse" %s | FileCheck %s

// CHECK-LABEL: func @codegen(
// CHECK-SAME: %[[arg:.*]]: !fir
func @codegen(%addr : !fir.ref<!fir.array<?xi32>>) {
  // CHECK: %[[zero:.*]] = constant 0 : index
  %0 = constant 0 : index
  %1 = fir.shape_shift %0, %0 : (index, index) -> !fir.shapeshift<1>
  %2 = fir.slice %0, %0, %0 : (index, index, index) -> !fir.slice<1>
  // CHECK: %[[box:.*]] = fircg.ext_embox %[[arg]](%[[zero]]) origin %[[zero]][%[[zero]], %[[zero]], %[[zero]]] : (!fir.ref<!fir.array<?xi32>>, index, index, index, index, index) -> !fir.box<!fir.array<?xi32>>
  %3 = fir.embox %addr (%1) [%2] : (!fir.ref<!fir.array<?xi32>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xi32>>
  // CHECK: fircg.ext_array_coor %[[arg]](%[[zero]]) origin %[[zero]][%[[zero]], %[[zero]], %[[zero]]]<%[[zero]]> : (!fir.ref<!fir.array<?xi32>>, index, index, index, index, index, index) -> !fir.ref<i32>
  %4 = fir.array_coor %addr (%1) [%2] %0 : (!fir.ref<!fir.array<?xi32>>, !fir.shapeshift<1>, !fir.slice<1>, index) -> !fir.ref<i32>
  // CHECK: fircg.ext_rebox %[[box]](%[[zero]]) origin %[[zero]] : (!fir.box<!fir.array<?xi32>>, index, index) -> !fir.box<!fir.array<?xi32>>
  %5 = fir.rebox %3(%1) : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xi32>>
  return
}

// CHECK-LABEL: fir.global @box_global
fir.global @box_global : !fir.box<!fir.array<?xi32>> {
  // CHECK: %[[arr:.*]] = fir.zero_bits !fir.ref
  %arr = fir.zero_bits !fir.ref<!fir.array<?xi32>>
  // CHECK: %[[zero:.*]] = constant 0 : index
  %0 = constant 0 : index
  %1 = fir.shape_shift %0, %0 : (index, index) -> !fir.shapeshift<1>
  %2 = fir.slice %0, %0, %0 : (index, index, index) -> !fir.slice<1>
  // CHECK: fircg.ext_embox %[[arr]](%[[zero]]) origin %[[zero]][%[[zero]], %[[zero]], %[[zero]]] : (!fir.ref<!fir.array<?xi32>>, index, index, index, index, index) -> !fir.box<!fir.array<?xi32>>
  %3 = fir.embox %arr (%1) [%2] : (!fir.ref<!fir.array<?xi32>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xi32>>
  fir.has_value %3 : !fir.box<!fir.array<?xi32>>
}
