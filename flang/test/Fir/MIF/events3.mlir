// RUN: fir-opt --mif-convert %s | FileCheck %s

func.func @class_wait(%arg0: !fir.class<!fir.heap<!fir.type<event>>, corank:1>) {
  mif.event_wait %arg0 : (!fir.class<!fir.heap<!fir.type<event>>, corank:1>) -> ()
  return
}

// CHECK-LABEL: func.func @class_wait(
// CHECK-SAME:    %[[EVENT:.*]]: !fir.class<!fir.heap<!fir.type<event>>, corank:1>)
// CHECK:         %[[CPTR:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK:         %[[ADDR:.*]] = fir.box_addr %[[EVENT]] : (!fir.class<!fir.heap<!fir.type<event>>, corank:1>) -> !fir.heap<!fir.type<event>>
// CHECK:         %[[INT:.*]] = fir.convert %[[ADDR]] : (!fir.heap<!fir.type<event>>) -> i64
// CHECK:         %[[FIELD:.*]] = fir.coordinate_of %[[CPTR]], __address
// CHECK:         fir.store %[[INT]] to %[[FIELD]] : !fir.ref<i64>
// CHECK:         fir.call @_QMprifPprif_event_wait(%[[CPTR]],
