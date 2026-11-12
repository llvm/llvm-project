// RUN: fir-opt --cse --canonicalize %s | FileCheck %s

// STAT-less mif.event_query ops must be neither merged nor erased

!event_ty = !fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>

func.func private @do_sync()

// Two queries around a fir.call: CSE must not fold the second into the first.
func.func @around_call(%ev: !fir.heap<!event_ty>, %a: !fir.ref<i32>, %b: !fir.ref<i32>) {
  %0 = mif.event_query %ev : (!fir.heap<!event_ty>) -> i32
  fir.store %0 to %a : !fir.ref<i32>
  fir.call @do_sync() : () -> ()
  %1 = mif.event_query %ev : (!fir.heap<!event_ty>) -> i32
  fir.store %1 to %b : !fir.ref<i32>
  return
}
// CHECK-LABEL: func.func @around_call(
// CHECK-SAME:     %[[EV:[^:]+]]: !fir.heap<{{.*}}>, %[[A:[^:]+]]: !fir.ref<i32>, %[[B:[^:]+]]: !fir.ref<i32>)
// CHECK:      %[[Q1:.*]] = mif.event_query %[[EV]]
// CHECK-NEXT: fir.store %[[Q1]] to %[[A]]
// CHECK-NEXT: fir.call @do_sync
// CHECK-NEXT: %[[Q2:.*]] = mif.event_query %[[EV]]
// CHECK-NEXT: fir.store %[[Q2]] to %[[B]]

func.func @around_event_wait(%ev: !fir.heap<!event_ty>, %a: !fir.ref<i32>, %b: !fir.ref<i32>) {
  %0 = mif.event_query %ev : (!fir.heap<!event_ty>) -> i32
  fir.store %0 to %a : !fir.ref<i32>
  mif.event_wait %ev : (!fir.heap<!event_ty>) -> ()
  %1 = mif.event_query %ev : (!fir.heap<!event_ty>) -> i32
  fir.store %1 to %b : !fir.ref<i32>
  return
}
// CHECK-LABEL: func.func @around_event_wait(
// CHECK-SAME:    %[[EV:[^:]+]]: !fir.heap<{{.*}}>, %[[A:[^:]+]]: !fir.ref<i32>, %[[B:[^:]+]]: !fir.ref<i32>)
// CHECK:      %[[Q1:.*]] = mif.event_query %[[EV]]
// CHECK-NEXT: fir.store %[[Q1]] to %[[A]]
// CHECK-NEXT: mif.event_wait %[[EV]]
// CHECK-NEXT: %[[Q2:.*]] = mif.event_query %[[EV]]
// CHECK-NEXT: fir.store %[[Q2]] to %[[B]]

func.func @unused_query(%ev: !fir.heap<!event_ty>) {
  %0 = mif.event_query %ev : (!fir.heap<!event_ty>) -> i32
  return
}
// CHECK-LABEL: func.func @unused_query(
// CHECK-SAME:    %[[EV:[^:]+]]: !fir.heap<{{.*}}>)
// CHECK:      mif.event_query %[[EV]]
// CHECK-NEXT: return

