// Check that an elemental result of a derived type with an allocatable
// component is shallow-copied into the array result.
// RUN: fir-opt %s --bufferize-hlfir | FileCheck %s

func.func @_QMtypesPtest() {
  %false = arith.constant false
  %c1 = arith.constant 1 : index
  %0 = fir.alloca !fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}> {bindc_name = ".result"}
  %11 = fir.shape %c1 : (index) -> !fir.shape<1>
  %18 = fir.alloca !fir.array<1x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>> {bindc_name = "y", uniq_name = "_QMtypesFtestEy"}
  %19:2 = hlfir.declare %18(%11) {uniq_name = "_QMtypesFtestEy"} : (!fir.ref<!fir.array<1x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>, !fir.ref<!fir.array<1x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>)
  %23 = hlfir.elemental %11 : (!fir.shape<1>) -> !hlfir.expr<1x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>> {
  ^bb0(%arg0: index):
    %26:2 = hlfir.declare %0 {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>) -> (!fir.ref<!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>, !fir.ref<!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>)
    %27 = hlfir.as_expr %26#0 move %false : (!fir.ref<!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>, i1) -> !hlfir.expr<!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>
    hlfir.yield_element %27 : !hlfir.expr<!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>
  }
  hlfir.assign %23 to %19#0 : !hlfir.expr<1x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>, !fir.ref<!fir.array<1x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>
  hlfir.destroy %23 : !hlfir.expr<1x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>
  return
}
// CHECK-LABEL:   func.func @_QMtypesPtest() {
// CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}> {bindc_name = ".result"}
// CHECK:           %[[VAL_6:.*]] = fir.allocmem !fir.array<1x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>> {bindc_name = ".tmp.array", uniq_name = ""}
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]](%{{.*}}) {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<1x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>, !fir.shape<1>) -> (!fir.heap<!fir.array<1x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>, !fir.heap<!fir.array<1x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>)
// CHECK:           fir.do_loop %[[VAL_10:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>) -> (!fir.ref<!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>, !fir.ref<!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>)
// CHECK:             %[[VAL_15:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_10]])  : (!fir.heap<!fir.array<1x!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>>, index) -> !fir.ref<!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>
// CHECK:             %[[VAL_16:.*]] = fir.load %[[VAL_11]]#0 : !fir.ref<!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>
// CHECK:             fir.store %[[VAL_16]] to %[[VAL_15]] : !fir.ref<!fir.type<_QMtypesTt{x:!fir.box<!fir.heap<f32>>}>>
// CHECK:           }
