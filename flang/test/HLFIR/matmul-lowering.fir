// Test hlfir.matmul operation lowering to fir runtime call
// RUN: fir-opt %s -lower-hlfir-intrinsics | FileCheck %s

func.func @_QPmatmul1(%arg0: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "lhs"}, %arg1: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "rhs"}, %arg2: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "res"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFmatmul1Elhs"} : (!fir.box<!fir.array<?x?xi32>>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
  %1:2 = hlfir.declare %arg2 {uniq_name = "_QFmatmul1Eres"} : (!fir.box<!fir.array<?x?xi32>>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFmatmul1Erhs"} : (!fir.box<!fir.array<?x?xi32>>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
  %3 = hlfir.matmul %0#0 %2#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>) -> !hlfir.expr<?x?xi32>
  hlfir.assign %3 to %1#0 : !hlfir.expr<?x?xi32>, !fir.box<!fir.array<?x?xi32>>
  hlfir.destroy %3 : !hlfir.expr<?x?xi32>
  return
}
// CHECK-LABEL: func.func @_QPmatmul1(
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "lhs"}
// CHECK:           %[[ARG1:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "rhs"}
// CHECK:           %[[ARG2:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "res"}
// CHECK-DAG:     %[[TRUE:.*]] = arith.constant true
// CHECK-DAG:     %[[LHS_VAR:.*]]:2 = hlfir.declare %[[ARG0]]
// CHECK-DAG:     %[[RHS_VAR:.*]]:2 = hlfir.declare %[[ARG1]]
// CHECK-DAG:     %[[RES_VAR:.*]]:2 = hlfir.declare %[[ARG2]]

// CHECK-DAG:     %[[RET_BOX:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>>
// CHECK-DAG:     %[[RET_ADDR:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xi32>>
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[RET_SHAPE:.*]] = fir.shape %[[C0]], %[[C0]] : (index, index) -> !fir.shape<2>
// CHECK-DAG:     %[[RET_EMBOX:.*]] = fir.embox %[[RET_ADDR]](%[[RET_SHAPE]])
// CHECK-DAG:     fir.store %[[RET_EMBOX]] to %[[RET_BOX]]

// CHECK:         %[[RET_ARG:.*]] = fir.convert %[[RET_BOX]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK-DAG:     %[[LHS_ARG:.*]] = fir.convert %[[LHS_VAR]]#1 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
// CHECK-DAG:     %[[RHS_ARG:.*]] = fir.convert %[[RHS_VAR]]#1 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
// CHECK:         %[[NONE:.*]] = fir.call @_FortranAMatmul(%[[RET_ARG]], %[[LHS_ARG]], %[[RHS_ARG]], %[[LOC_STR:.*]], %[[LOC_N:.*]]) fastmath<contract>

// CHECK:         %[[RET:.*]] = fir.load %[[RET_BOX]]
// CHECK-DAG:     %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[RET]]
// CHECK-DAG:     %[[ADDR:.*]] = fir.box_addr %[[RET]]
// CHECK-NEXT:    %[[SHIFT:.*]] = fir.shape_shift %[[BOX_DIMS]]#0, %[[BOX_DIMS]]#1
// TODO: fix alias analysis in hlfir.assign bufferization
// CHECK-NEXT:    %[[TMP:.*]]:2 = hlfir.declare %[[ADDR]](%[[SHIFT]]) {uniq_name = ".tmp.intrinsic_result"}
// TODO: add shape information from original intrinsic op
// CHECK:         %[[ASEXPR:.*]] = hlfir.as_expr %[[TMP]]#0 move %[[TRUE]] : (!fir.box<!fir.array<?x?xi32>>, i1) -> !hlfir.expr<?x?xi32>
// CHECK:         hlfir.assign %[[ASEXPR]] to %[[RES_VAR]]#0
// CHECK:         hlfir.destroy %[[ASEXPR]]
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// nested matmuls leading to recursive pattern application
func.func @_QPtest(%arg0: !fir.ref<!fir.array<3x3xf32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<3x3xf32>> {fir.bindc_name = "b"}, %arg2: !fir.ref<!fir.array<3x3xf32>> {fir.bindc_name = "c"}, %arg3: !fir.ref<!fir.array<3x3xf32>> {fir.bindc_name = "out"}) {
  %c3 = arith.constant 3 : index
  %c3_0 = arith.constant 3 : index
  %0 = fir.shape %c3, %c3_0 : (index, index) -> !fir.shape<2>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFtestEa"} : (!fir.ref<!fir.array<3x3xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x3xf32>>, !fir.ref<!fir.array<3x3xf32>>)
  %c3_1 = arith.constant 3 : index
  %c3_2 = arith.constant 3 : index
  %2 = fir.shape %c3_1, %c3_2 : (index, index) -> !fir.shape<2>
  %3:2 = hlfir.declare %arg1(%2) {uniq_name = "_QFtestEb"} : (!fir.ref<!fir.array<3x3xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x3xf32>>, !fir.ref<!fir.array<3x3xf32>>)
  %c3_3 = arith.constant 3 : index
  %c3_4 = arith.constant 3 : index
  %4 = fir.shape %c3_3, %c3_4 : (index, index) -> !fir.shape<2>
  %5:2 = hlfir.declare %arg2(%4) {uniq_name = "_QFtestEc"} : (!fir.ref<!fir.array<3x3xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x3xf32>>, !fir.ref<!fir.array<3x3xf32>>)
  %c3_5 = arith.constant 3 : index
  %c3_6 = arith.constant 3 : index
  %6 = fir.shape %c3_5, %c3_6 : (index, index) -> !fir.shape<2>
  %7:2 = hlfir.declare %arg3(%6) {uniq_name = "_QFtestEout"} : (!fir.ref<!fir.array<3x3xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x3xf32>>, !fir.ref<!fir.array<3x3xf32>>)
  %8 = hlfir.matmul %1#0 %3#0 {fastmath = #arith.fastmath<contract>} : (!fir.ref<!fir.array<3x3xf32>>, !fir.ref<!fir.array<3x3xf32>>) -> !hlfir.expr<3x3xf32>
  %9 = hlfir.matmul %8 %5#0 {fastmath = #arith.fastmath<contract>} : (!hlfir.expr<3x3xf32>, !fir.ref<!fir.array<3x3xf32>>) -> !hlfir.expr<3x3xf32>
  hlfir.assign %9 to %7#0 : !hlfir.expr<3x3xf32>, !fir.ref<!fir.array<3x3xf32>>
  hlfir.destroy %9 : !hlfir.expr<3x3xf32>
  hlfir.destroy %8 : !hlfir.expr<3x3xf32>
  return
}
// just check that we apply the patterns successfully. The details are checked above
// CHECK-LABEL: func.func @_QPtest(
// CHECK:         fir.call @_FortranAMatmul({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
// CHECK:         fir.call @_FortranAMatmul({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
// CHECK:         return
// CHECK-NEXT:  }
