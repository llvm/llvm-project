// Test hlfir.transpose operation lowering to fir runtime call
// RUN: fir-opt %s -lower-hlfir-intrinsics | FileCheck %s

func.func @_QPtranspose1(%arg0: !fir.ref<!fir.array<1x2xi32>> {fir.bindc_name = "m"}, %arg1: !fir.ref<!fir.array<2x1xi32>> {fir.bindc_name = "res"}) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = fir.shape %c1, %c2 : (index, index) -> !fir.shape<2>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFtranspose1Em"} : (!fir.ref<!fir.array<1x2xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<1x2xi32>>, !fir.ref<!fir.array<1x2xi32>>)
  %c2_0 = arith.constant 2 : index
  %c1_1 = arith.constant 1 : index
  %2 = fir.shape %c2_0, %c1_1 : (index, index) -> !fir.shape<2>
  %3:2 = hlfir.declare %arg1(%2) {uniq_name = "_QFtranspose1Eres"} : (!fir.ref<!fir.array<2x1xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<2x1xi32>>, !fir.ref<!fir.array<2x1xi32>>)
  %4 = hlfir.transpose %1#0 : (!fir.ref<!fir.array<1x2xi32>>) -> !hlfir.expr<2x1xi32>
  hlfir.assign %4 to %3#0 : !hlfir.expr<2x1xi32>, !fir.ref<!fir.array<2x1xi32>>
  hlfir.destroy %4 : !hlfir.expr<2x1xi32>
  return
}
// CHECK-LABEL: func.func @_QPtranspose1(
// CHECK:           %[[ARG0:.*]]: !fir.ref<!fir.array<1x2xi32>> {fir.bindc_name = "m"}
// CHECK:           %[[ARG1:.*]]: !fir.ref<!fir.array<2x1xi32>> {fir.bindc_name = "res"}
// CHECK-DAG:     %[[TRUE:.*]] = arith.constant true
// CHECK-DAG:     %[[M_VAR:.*]]:2 = hlfir.declare %[[ARG0]]
// CHECK-DAG:     %[[RES_VAR:.*]]:2 = hlfir.declare %[[ARG1]]

// CHECK-DAG:     %[[RET_BOX:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>>
// CHECK-DAG:     %[[RET_ADDR:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xi32>>
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[RET_SHAPE:.*]] = fir.shape %[[C0]], %[[C0]] : (index, index) -> !fir.shape<2>
// CHECK-DAG:     %[[RET_EMBOX:.*]] = fir.embox %[[RET_ADDR]](%[[RET_SHAPE]])
// CHECK-DAG:     fir.store %[[RET_EMBOX]] to %[[RET_BOX]]

// CHECK-DAG:     %[[M_BOX:.*]] = fir.embox %[[M_VAR]]#1(%[[M_SHAPE:.*]])

// CHECK:         %[[RET_ARG:.*]] = fir.convert %[[RET_BOX]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>)
// CHECK:         %[[M_ARG:.*]] = fir.convert %[[M_BOX]] : (!fir.box<!fir.array<1x2xi32>>) -> !fir.box<none>
// CHECK:         fir.call @_FortranATranspose(%[[RET_ARG]], %[[M_ARG]], %[[LOC_STR:.*]], %[[LOC_N:.*]])

// CHECK:         %[[RET:.*]] = fir.load %[[RET_BOX]]
// CHECK-DAG:     %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[RET]]
// CHECK-DAG:     %[[ADDR:.*]] = fir.box_addr %[[RET]]
// CHECK-NEXT:    %[[SHIFT:.*]] = fir.shape_shift %[[BOX_DIMS]]#0, %[[BOX_DIMS]]#1
// TODO: fix alias analysis in hlfir.assign bufferization
// CHECK-NEXT:    %[[TMP:.*]]:2 = hlfir.declare %[[ADDR]](%[[SHIFT]]) {uniq_name = ".tmp.intrinsic_result"}
// CHECK:         %[[ASEXPR:.*]] = hlfir.as_expr %[[TMP]]#0 move %[[TRUE]] : (!fir.box<!fir.array<?x?xi32>>, i1) -> !hlfir.expr<?x?xi32>
// CHECK:         hlfir.assign %[[ASEXPR]] to %[[RES_VAR]]#0
// CHECK:         hlfir.destroy %[[ASEXPR]]
// CHECK-NEXT:    return
// CHECK-NEXT:  }
