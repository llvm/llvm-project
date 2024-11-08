// RUN: fir-opt %s --abstract-result | FileCheck %s

// Check that the attributes are shifted along with their corresponding arguments

// CHECK-LABEL: func.func @_QMi8Pintrinsic_pack0
// CHECK-SAME:(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %arg1: !fir.box<!fir.array<?x?x?xi32>> {fir.bindc_name = "array"}, %arg2: !fir.ref<!fir.logical<4>> {fir.bindc_name = "mask"}, %arg3: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "vector", fir.optional})

func.func @_QMi8Pintrinsic_pack0(%arg0: !fir.box<!fir.array<?x?x?xi32>> {fir.bindc_name = "array"}, %arg1: !fir.ref<!fir.logical<4>> {fir.bindc_name = "mask"}, %arg2: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "vector", fir.optional}) -> !fir.box<!fir.heap<!fir.array<?xi32>>> {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  %1 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "intrinsic_pack0", uniq_name = "_QMi8Fintrinsic_pack0Eintrinsic_pack0"}
  %2 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
  %c0 = arith.constant 0 : index
  %3 = fir.shape %c0 : (index) -> !fir.shape<1>
  %4 = fir.embox %2(%3) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  fir.store %4 to %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %5 = fir.embox %arg1 : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
  %6 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
  %7 = fir.embox %6(%3) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  fir.store %7 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %8 = fir.convert %0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  %9 = fir.convert %arg0 : (!fir.box<!fir.array<?x?x?xi32>>) -> !fir.box<none>
  %10 = fir.convert %5 : (!fir.box<!fir.logical<4>>) -> !fir.box<none>
  %11 = fir.convert %arg2 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
  %12 = fir.call @_FortranAPack(%8, %9, %10, %11) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>) -> none
  %13 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  return %13 : !fir.box<!fir.heap<!fir.array<?xi32>>>
}
func.func private @_FortranAPack(!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>) -> none attributes {fir.runtime}

// CHECK-LABEL: func.func private @empty
// CHECK-SAME:(!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.array<?x?x?xi32>> {fir.bindc_name = "array"}, !fir.ref<!fir.logical<4>> {fir.bindc_name = "mask"}, !fir.box<!fir.array<?xi32>> {fir.bindc_name = "vector", fir.optional})
func.func private @empty(%arg0: !fir.box<!fir.array<?x?x?xi32>> {fir.bindc_name = "array"}, %arg1: !fir.ref<!fir.logical<4>> {fir.bindc_name = "mask"}, %arg2: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "vector", fir.optional}) -> !fir.box<!fir.heap<!fir.array<?xi32>>> 
