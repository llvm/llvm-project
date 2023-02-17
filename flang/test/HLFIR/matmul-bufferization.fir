// Test hlfir.matmul operation lowering to fir runtime call
// RUN: fir-opt %s -bufferize-hlfir | FileCheck %s

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
// CHECK:         %[[NONE:.*]] = fir.call @_FortranAMatmul(%[[RET_ARG]], %[[LHS_ARG]], %[[RHS_ARG]], %[[LOC_STR:.*]], %[[LOC_N:.*]])

// CHECK:         %[[RET:.*]] = fir.load %[[RET_BOX]]
// CHECK-DAG:     %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[RET]]
// CHECK-DAG:     %[[ADDR:.*]] = fir.box_addr %[[RET]]
// CHECK-NEXT:    %[[SHIFT:.*]] = fir.shape_shift %[[BOX_DIMS]]#0, %[[BOX_DIMS]]#1
// TODO: fix alias analysis in hlfir.assign bufferization
// CHECK-NEXT:    %[[TMP:.*]]:2 = hlfir.declare %[[ADDR]](%[[SHIFT]]) {uniq_name = ".tmp.intrinsic_result"}
// CHECK:         %[[TUPLE0:.*]] = fir.undefined tuple<!fir.box<!fir.array<?x?xi32>>, i1>
// CHECK:         %[[TUPLE1:.*]] = fir.insert_value %[[TUPLE0]], %[[TRUE:.*]], [1 : index]
// CHECK:         %[[TUPLE2:.*]] = fir.insert_value %[[TUPLE1]], %[[TMP]]#0, [0 : index]
// CHECK:         hlfir.assign %[[TMP]]#0 to %[[RES_VAR]]#0
// CHECK:         fir.freemem %[[TMP]]#1
// CHECK-NEXT:    return
// CHECK-NEXT:  }
