// RUN: fir-opt --boxed-procedure="use-safe-trampoline=true" %s | FileCheck %s
// RUN: fir-opt --boxed-procedure %s | FileCheck %s

// A module or SAVE procedure pointer with an initializer is lowered to a
// fir.global whose initializer region emboxes a procedure. The boxed-procedure
// pass must rewrite the ops inside that region in both modes: fir.emboxproc has
// no codegen, and the safe-trampoline path used to visit only the fir.global
// op itself, leaving the initializer untouched and the global's type
// inconsistent with its fir.has_value.

// CHECK-LABEL: fir.global @_QMmEnull_ptr : () -> () {
// CHECK:         %[[ZERO:.*]] = fir.zero_bits () -> ()
// CHECK:         %[[NULL:.*]] = fir.convert %[[ZERO]] : (() -> ()) -> (() -> ())
// CHECK:         fir.has_value %[[NULL]] : () -> ()
// CHECK-NOT:     fir.emboxproc
fir.global @_QMmEnull_ptr : !fir.boxproc<() -> ()> {
  %0 = fir.zero_bits () -> ()
  %1 = fir.emboxproc %0 : (() -> ()) -> !fir.boxproc<() -> ()>
  fir.has_value %1 : !fir.boxproc<() -> ()>
}

// CHECK-LABEL: fir.global @_QMmEtarget_ptr : () -> () {
// CHECK:         %[[ADDR:.*]] = fir.address_of(@_QMmPcallee) : () -> ()
// CHECK:         %[[FUNC:.*]] = fir.convert %[[ADDR]] : (() -> ()) -> (() -> ())
// CHECK:         fir.has_value %[[FUNC]] : () -> ()
// CHECK-NOT:     fir.emboxproc
fir.global @_QMmEtarget_ptr : !fir.boxproc<() -> ()> {
  %0 = fir.address_of(@_QMmPcallee) : () -> ()
  %1 = fir.emboxproc %0 : (() -> ()) -> !fir.boxproc<() -> ()>
  fir.has_value %1 : !fir.boxproc<() -> ()>
}

// CHECK-LABEL: func.func @_QMmPcallee()
func.func @_QMmPcallee() {
  return
}

// CHECK-LABEL: func.func @_QMmPcall_through_ptr()
// CHECK:         %[[REF:.*]] = fir.address_of(@_QMmEtarget_ptr) : !fir.ref<() -> ()>
// CHECK:         %[[LOAD:.*]] = fir.load %[[REF]] : !fir.ref<() -> ()>
// CHECK:         %[[CALLEE:.*]] = fir.convert %[[LOAD]] : (() -> ()) -> (() -> ())
// CHECK:         fir.call %[[CALLEE]]() : () -> ()
func.func @_QMmPcall_through_ptr() {
  %0 = fir.address_of(@_QMmEtarget_ptr) : !fir.ref<!fir.boxproc<() -> ()>>
  %1 = fir.load %0 : !fir.ref<!fir.boxproc<() -> ()>>
  %2 = fir.box_addr %1 : (!fir.boxproc<() -> ()>) -> (() -> ())
  fir.call %2() : () -> ()
  return
}
