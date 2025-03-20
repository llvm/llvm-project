// Test buffer destruction for hlfir.destroy operations with
// operands of derived types.
// RUN: fir-opt --bufferize-hlfir %s | FileCheck %s

func.func @_QPtest1(%arg0: !fir.box<!fir.array<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>> {fir.bindc_name = "x"}) {
  %c0 = arith.constant 0 : index
  %0 = fir.alloca !fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}> {bindc_name = ".result"}
  %1:2 = hlfir.declare %arg0 {uniq_name = "_QFtest1Ex"} : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>>) -> (!fir.box<!fir.array<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>>)
  %2:3 = fir.box_dims %1#0, %c0 : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>>, index) -> (index, index, index)
  %3 = fir.shape %2#1 : (index) -> !fir.shape<1>
  %4 = hlfir.elemental %3 unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>> {
  ^bb0(%arg1: index):
    %5 = hlfir.designate %1#0 (%arg1)  : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>>, index) -> !fir.ref<!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>
    %6 = fir.call @_QPelem1(%5) fastmath<contract> : (!fir.ref<!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>) -> !fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>
    fir.save_result %6 to %0 : !fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>, !fir.ref<!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>
    %7:2 = hlfir.declare %0 {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>) -> (!fir.ref<!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>, !fir.ref<!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>)
    hlfir.yield_element %7#0 : !fir.ref<!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>
  }
  hlfir.assign %4 to %1#0 : !hlfir.expr<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>>
  hlfir.destroy %4 : !hlfir.expr<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>
  return
}
// CHECK-LABEL:   func.func @_QPtest1(
// CHECK:             hlfir.assign %{{.*}} to %{{.*}} temporary_lhs : !fir.ref<!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>, !fir.ref<!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>
// CHECK:           hlfir.assign %[[VAL_7:.*]]#0 to %{{.*}}#0 : !fir.box<!fir.array<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>>
// CHECK-NEXT:      %[[VAL_18:.*]] = fir.box_addr %[[VAL_7]]#0 : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>>) -> !fir.heap<!fir.array<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>>
// CHECK-NEXT:      %[[VAL_19:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>>) -> !fir.box<none>
// CHECK-NEXT:      fir.call @_FortranADestroyWithoutFinalization(%[[VAL_19]]) : (!fir.box<none>) -> ()
// CHECK-NEXT:      fir.freemem %[[VAL_18]] : !fir.heap<!fir.array<?x!fir.type<_QMtypesTt1{x:!fir.box<!fir.heap<f32>>}>>>
// CHECK-NEXT:      return
// CHECK-NEXT:    }

func.func @_QPtest2(%arg0: !fir.box<!fir.array<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>> {fir.bindc_name = "x"}) {
  %c0 = arith.constant 0 : index
  %0 = fir.alloca !fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}> {bindc_name = ".result"}
  %1:2 = hlfir.declare %arg0 {uniq_name = "_QFtest2Ex"} : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>>) -> (!fir.box<!fir.array<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>>)
  %2:3 = fir.box_dims %1#0, %c0 : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>>, index) -> (index, index, index)
  %3 = fir.shape %2#1 : (index) -> !fir.shape<1>
  %4 = hlfir.elemental %3 unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>> {
  ^bb0(%arg1: index):
    %5 = hlfir.designate %1#0 (%arg1)  : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>>, index) -> !fir.ref<!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>
    %6 = fir.call @_QPelem2(%5) fastmath<contract> : (!fir.ref<!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>) -> !fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>
    fir.save_result %6 to %0 : !fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>, !fir.ref<!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>
    %7:2 = hlfir.declare %0 {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>) -> (!fir.ref<!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>, !fir.ref<!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>)
    hlfir.yield_element %7#0 : !fir.ref<!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>
  }
  hlfir.assign %4 to %1#0 : !hlfir.expr<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>>
  hlfir.destroy %4 finalize : !hlfir.expr<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>
  return
}
// CHECK-LABEL:   func.func @_QPtest2(
// CHECK:             hlfir.assign %{{.*}}#0 to %{{.*}} temporary_lhs : !fir.ref<!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>, !fir.ref<!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>
// CHECK:           hlfir.assign %[[VAL_7:.*]]#0 to %{{.*}}#0 : !fir.box<!fir.array<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>>
// CHECK-NEXT:      %[[VAL_18:.*]] = fir.box_addr %[[VAL_7]]#0 : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>>) -> !fir.heap<!fir.array<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>>
// CHECK-NEXT:      %[[VAL_19:.*]] = fir.address_of(@_QQclX{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
// CHECK-NEXT:      %[[VAL_20:.*]] = arith.constant {{[0-9]*}} : index
// CHECK-NEXT:      %[[VAL_21:.*]] = arith.constant {{[0-9]*}} : i32
// CHECK-NEXT:      %[[VAL_22:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>>) -> !fir.box<none>
// CHECK-NEXT:      %[[VAL_23:.*]] = fir.convert %[[VAL_19]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
// CHECK-NEXT:      fir.call @_FortranAFinalize(%[[VAL_22]], %[[VAL_23]], %[[VAL_21]]) : (!fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK-NEXT:      %[[VAL_25:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>>) -> !fir.box<none>
// CHECK-NEXT:      fir.call @_FortranADestroyWithoutFinalization(%[[VAL_25]]) : (!fir.box<none>) -> ()
// CHECK-NEXT:      fir.freemem %[[VAL_18]] : !fir.heap<!fir.array<?x!fir.type<_QMtypesTt2{x:!fir.box<!fir.heap<f32>>}>>>
// CHECK-NEXT:      return
// CHECK-NEXT:    }

func.func @_QPtest3(%arg0: !fir.box<!fir.array<?x!fir.type<_QMtypesTt3{x:f32}>>> {fir.bindc_name = "x"}) {
  %c0 = arith.constant 0 : index
  %0 = fir.alloca !fir.type<_QMtypesTt3{x:f32}> {bindc_name = ".result"}
  %1:2 = hlfir.declare %arg0 {uniq_name = "_QFtest3Ex"} : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt3{x:f32}>>>) -> (!fir.box<!fir.array<?x!fir.type<_QMtypesTt3{x:f32}>>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt3{x:f32}>>>)
  %2:3 = fir.box_dims %1#0, %c0 : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt3{x:f32}>>>, index) -> (index, index, index)
  %3 = fir.shape %2#1 : (index) -> !fir.shape<1>
  %4 = hlfir.elemental %3 unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.type<_QMtypesTt3{x:f32}>> {
  ^bb0(%arg1: index):
    %5 = hlfir.designate %1#0 (%arg1)  : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt3{x:f32}>>>, index) -> !fir.ref<!fir.type<_QMtypesTt3{x:f32}>>
    %6 = fir.call @_QPelem3(%5) fastmath<contract> : (!fir.ref<!fir.type<_QMtypesTt3{x:f32}>>) -> !fir.type<_QMtypesTt3{x:f32}>
    fir.save_result %6 to %0 : !fir.type<_QMtypesTt3{x:f32}>, !fir.ref<!fir.type<_QMtypesTt3{x:f32}>>
    %7:2 = hlfir.declare %0 {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.type<_QMtypesTt3{x:f32}>>) -> (!fir.ref<!fir.type<_QMtypesTt3{x:f32}>>, !fir.ref<!fir.type<_QMtypesTt3{x:f32}>>)
    hlfir.yield_element %7#0 : !fir.ref<!fir.type<_QMtypesTt3{x:f32}>>
  }
  hlfir.assign %4 to %1#0 : !hlfir.expr<?x!fir.type<_QMtypesTt3{x:f32}>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt3{x:f32}>>>
  hlfir.destroy %4 finalize : !hlfir.expr<?x!fir.type<_QMtypesTt3{x:f32}>>
  return
}
// CHECK-LABEL:   func.func @_QPtest3(
// CHECK:             hlfir.assign %{{.*}}#0 to %{{.*}} temporary_lhs : !fir.ref<!fir.type<_QMtypesTt3{x:f32}>>, !fir.ref<!fir.type<_QMtypesTt3{x:f32}>>
// CHECK:           hlfir.assign %[[VAL_7:.*]]#0 to %{{.*}}#0 : !fir.box<!fir.array<?x!fir.type<_QMtypesTt3{x:f32}>>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt3{x:f32}>>>
// CHECK-NEXT:      %[[VAL_18:.*]] = fir.box_addr %[[VAL_7]]#0 : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt3{x:f32}>>>) -> !fir.heap<!fir.array<?x!fir.type<_QMtypesTt3{x:f32}>>>
// CHECK-NEXT:      %[[VAL_19:.*]] = fir.address_of(@_QQclX{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]*}}>>
// CHECK-NEXT:      %[[VAL_20:.*]] = arith.constant {{[0-9]*}} : index
// CHECK-NEXT:      %[[VAL_21:.*]] = arith.constant {{[0-9]*}} : i32
// CHECK-NEXT:      %[[VAL_22:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt3{x:f32}>>>) -> !fir.box<none>
// CHECK-NEXT:      %[[VAL_23:.*]] = fir.convert %[[VAL_19]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
// CHECK-NEXT:      fir.call @_FortranAFinalize(%[[VAL_22]], %[[VAL_23]], %[[VAL_21]]) : (!fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK-NEXT:      fir.freemem %[[VAL_18]] : !fir.heap<!fir.array<?x!fir.type<_QMtypesTt3{x:f32}>>>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
