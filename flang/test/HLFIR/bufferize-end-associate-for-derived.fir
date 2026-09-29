// Test buffer destruction for hlfir.end_associate operations with
// operands of derived types.
// RUN: fir-opt --bufferize-hlfir %s | FileCheck %s

func.func @_QPtest1(%arg0: !fir.box<!fir.array<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>> {fir.bindc_name = "x"}) {
  %c0 = arith.constant 0 : index
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFtest1Ex"} : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>>) -> (!fir.box<!fir.array<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>>)
  %1 = hlfir.as_expr %0#0 : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>>) -> !hlfir.expr<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>
  %2:3 = fir.box_dims %0#0, %c0 : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>>, index) -> (index, index, index)
  %3 = fir.shape %2#1 : (index) -> !fir.shape<1>
  %4:3 = hlfir.associate %1(%3) {uniq_name = "adapt.valuebyref"} : (!hlfir.expr<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>, !fir.shape<1>) -> (!fir.box<!fir.array<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>>, !fir.ref<!fir.array<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>>, i1)
  %5 = fir.convert %4#1 : (!fir.ref<!fir.array<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>>) -> !fir.ref<!fir.array<10x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>>
  fir.call @_QPcallee1(%5) fastmath<contract> : (!fir.ref<!fir.array<10x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>>) -> ()
  hlfir.end_associate %4#0, %4#2 : !fir.box<!fir.array<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>>, i1
  return
}
// CHECK-LABEL:   func.func @_QPtest1(
// CHECK-NOT:       fir.call @_Fortran
// CHECK:           fir.call @_FortranADestroyWithoutFinalization(%{{.*}}) : (!fir.box<none>) -> ()
// CHECK-NOT:       fir.call @_Fortran

func.func @_QPtest2(%arg0: !fir.box<!fir.array<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>> {fir.bindc_name = "x"}) {
  %c0 = arith.constant 0 : index
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFtest2Ex"} : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>>) -> (!fir.box<!fir.array<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>>)
  %1 = hlfir.as_expr %0#0 : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>>) -> !hlfir.expr<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>
  %2:3 = fir.box_dims %0#0, %c0 : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>>, index) -> (index, index, index)
  %3 = fir.shape %2#1 : (index) -> !fir.shape<1>
  %4:3 = hlfir.associate %1(%3) {uniq_name = "adapt.valuebyref"} : (!hlfir.expr<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>, !fir.shape<1>) -> (!fir.box<!fir.array<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>>, !fir.ref<!fir.array<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>>, i1)
  %5 = fir.convert %4#1 : (!fir.ref<!fir.array<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>>) -> !fir.ref<!fir.array<10x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>>
  fir.call @_QPcallee2(%5) fastmath<contract> : (!fir.ref<!fir.array<10x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>>) -> ()
  hlfir.end_associate %4#0, %4#2 : !fir.box<!fir.array<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>>, i1
  return
}
// CHECK-LABEL:   func.func @_QPtest2(
// CHECK-NOT:       fir.call @_Fortran
// CHECK:           fir.call @_FortranADestroyWithoutFinalization(%{{.*}}) : (!fir.box<none>) -> ()
// CHECK-NOT:       fir.call @_Fortran

func.func @_QPtest3(%arg0: !fir.box<!fir.array<?x!fir.type<_QMtypesTt3{x:f32}>>> {fir.bindc_name = "x"}) {
  %c0 = arith.constant 0 : index
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFtest3Ex"} : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt3{x:f32}>>>) -> (!fir.box<!fir.array<?x!fir.type<_QMtypesTt3{x:f32}>>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt3{x:f32}>>>)
  %1 = hlfir.as_expr %0#0 : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt3{x:f32}>>>) -> !hlfir.expr<?x!fir.type<_QMtypesTt3{x:f32}>>
  %2:3 = fir.box_dims %0#0, %c0 : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt3{x:f32}>>>, index) -> (index, index, index)
  %3 = fir.shape %2#1 : (index) -> !fir.shape<1>
  %4:3 = hlfir.associate %1(%3) {uniq_name = "adapt.valuebyref"} : (!hlfir.expr<?x!fir.type<_QMtypesTt3{x:f32}>>, !fir.shape<1>) -> (!fir.box<!fir.array<?x!fir.type<_QMtypesTt3{x:f32}>>>, !fir.ref<!fir.array<?x!fir.type<_QMtypesTt3{x:f32}>>>, i1)
  %5 = fir.convert %4#1 : (!fir.ref<!fir.array<?x!fir.type<_QMtypesTt3{x:f32}>>>) -> !fir.ref<!fir.array<10x!fir.type<_QMtypesTt3{x:f32}>>>
  fir.call @_QPcallee3(%5) fastmath<contract> : (!fir.ref<!fir.array<10x!fir.type<_QMtypesTt3{x:f32}>>>) -> ()
  hlfir.end_associate %4#1, %4#2 : !fir.ref<!fir.array<?x!fir.type<_QMtypesTt3{x:f32}>>>, i1
  return
}
// CHECK-LABEL:   func.func @_QPtest3(
// CHECK-NOT:       fir.call @_Fortran
