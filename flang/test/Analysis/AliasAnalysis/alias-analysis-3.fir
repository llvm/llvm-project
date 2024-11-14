// Use --mlir-disable-threading so that the AA queries are serialized
// as well as its diagnostic output.
// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// module m
//   type t
//      real, pointer :: pointer_component
//   end type t
//   type(t) :: a
// contains
//   subroutine test(pointer_dummy, x)
//     real, pointer :: pointer_dummy
//     real, target :: x
//     pointer_dummy => x
//     call test2(a%pointer_component)
//   end subroutine test
// end module m

// A composite with a pointer component may alias with a dummy pointer
// CHECK-LABEL: Testing : "_QMmPtest
// CHECK: a#0 <-> func.region0#0: MayAlias

// FIXME: a's box cannot alias with raw reference to f32 (x), so MayAlias below must be NoAlias:
// CHECK: a#0 <-> func.region0#1: MayAlias

// pointer_dummy's box cannot alias with raw reference to f32 (x)
// CHECK: func.region0#0 <-> func.region0#1: NoAlias

fir.global @_QMmEa : !fir.type<_QMmTt{pointer_component:!fir.box<!fir.ptr<f32>>}> {
  %0 = fir.undefined !fir.type<_QMmTt{pointer_component:!fir.box<!fir.ptr<f32>>}>
  fir.has_value %0 : !fir.type<_QMmTt{pointer_component:!fir.box<!fir.ptr<f32>>}>
}
func.func @_QMmPtest(%arg0: !fir.ref<!fir.box<!fir.ptr<f32>>> {fir.bindc_name = "pointer_dummy"}, %arg1: !fir.ref<f32> {fir.bindc_name = "x", fir.target}) attributes {test.ptr = "func"} {
  %0 = fir.address_of(@_QMmEa) {test.ptr = "a"} : !fir.ref<!fir.type<_QMmTt{pointer_component:!fir.box<!fir.ptr<f32>>}>>
  %1 = fir.embox %arg1 : (!fir.ref<f32>) -> !fir.box<!fir.ptr<f32>>
  fir.store %1 to %arg0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %2 = fir.field_index pointer_component, !fir.type<_QMmTt{pointer_component:!fir.box<!fir.ptr<f32>>}>
  %3 = fir.coordinate_of %0, %2 : (!fir.ref<!fir.type<_QMmTt{pointer_component:!fir.box<!fir.ptr<f32>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %4 = fir.load %3 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %5 = fir.box_addr %4 : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %6 = fir.convert %5 : (!fir.ptr<f32>) -> !fir.ref<f32>
  fir.call @_QPtest2(%6) fastmath<contract> : (!fir.ref<f32>) -> ()
  return
}
func.func private @_QPtest2(!fir.ref<f32>)

// -----

// A composite with a pointer component may alias with a dummy
// argument of composite type with a pointer component:
// module m
//   type t
//      real, pointer :: pointer_component
//   end type t
//   type(t) :: a
// contains
//   subroutine test(b, x)
//     type(t) :: b
//     real, target :: x
//     a%pointer_component => x
//     call test2(b%pointer_component)
//   end subroutine test
// end module m

// CHECK-LABEL: Testing : "_QMmPtest"
// CHECK: a#0 <-> func.region0#0: MayAlias

fir.global @_QMmEa : !fir.type<_QMmTt{pointer_component:!fir.box<!fir.ptr<f32>>}> {
  %0 = fir.undefined !fir.type<_QMmTt{pointer_component:!fir.box<!fir.ptr<f32>>}>
  fir.has_value %0 : !fir.type<_QMmTt{pointer_component:!fir.box<!fir.ptr<f32>>}>
}
func.func @_QMmPtest(%arg0: !fir.ref<!fir.type<_QMmTt{pointer_component:!fir.box<!fir.ptr<f32>>}>> {fir.bindc_name = "b"}, %arg1: !fir.ref<f32> {fir.bindc_name = "x", fir.target}) attributes {test.ptr = "func"} {
  %0 = fir.address_of(@_QMmEa) {test.ptr = "a"} : !fir.ref<!fir.type<_QMmTt{pointer_component:!fir.box<!fir.ptr<f32>>}>>
  %1 = fir.field_index pointer_component, !fir.type<_QMmTt{pointer_component:!fir.box<!fir.ptr<f32>>}>
  %2 = fir.coordinate_of %0, %1 : (!fir.ref<!fir.type<_QMmTt{pointer_component:!fir.box<!fir.ptr<f32>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %3 = fir.embox %arg1 : (!fir.ref<f32>) -> !fir.box<!fir.ptr<f32>>
  fir.store %3 to %2 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %4 = fir.field_index pointer_component, !fir.type<_QMmTt{pointer_component:!fir.box<!fir.ptr<f32>>}>
  %5 = fir.coordinate_of %arg0, %4 : (!fir.ref<!fir.type<_QMmTt{pointer_component:!fir.box<!fir.ptr<f32>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %6 = fir.load %5 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %7 = fir.box_addr %6 : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %8 = fir.convert %7 : (!fir.ptr<f32>) -> !fir.ref<f32>
  fir.call @_QPtest2(%8) fastmath<contract> : (!fir.ref<f32>) -> ()
  return
}
func.func private @_QPtest2(!fir.ref<f32>)

// -----

// Two dummy arguments of composite type with a pointer component
// may alias each other:
// module m
//   type t
//      real, pointer :: pointer_component
//   end type t
// contains
//   subroutine test(a, b, x)
//     type(t) :: a, b
//     real, target :: x
//     a%pointer_component => x
//     call test2(b%pointer_component)
//   end subroutine test
// end module m

// CHECK-LABEL: Testing : "_QMmPtest"
// CHECK: func.region0#0 <-> func.region0#1: MayAlias

func.func @_QMmPtest(%arg0: !fir.ref<!fir.type<_QMmTt{pointer_component:!fir.box<!fir.ptr<f32>>}>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.type<_QMmTt{pointer_component:!fir.box<!fir.ptr<f32>>}>> {fir.bindc_name = "b"}, %arg2: !fir.ref<f32> {fir.bindc_name = "x", fir.target}) attributes {test.ptr = "func"} {
  %0 = fir.field_index pointer_component, !fir.type<_QMmTt{pointer_component:!fir.box<!fir.ptr<f32>>}>
  %1 = fir.coordinate_of %arg0, %0 : (!fir.ref<!fir.type<_QMmTt{pointer_component:!fir.box<!fir.ptr<f32>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %2 = fir.embox %arg2 : (!fir.ref<f32>) -> !fir.box<!fir.ptr<f32>>
  fir.store %2 to %1 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %3 = fir.field_index pointer_component, !fir.type<_QMmTt{pointer_component:!fir.box<!fir.ptr<f32>>}>
  %4 = fir.coordinate_of %arg1, %3 : (!fir.ref<!fir.type<_QMmTt{pointer_component:!fir.box<!fir.ptr<f32>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %5 = fir.load %4 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %6 = fir.box_addr %5 : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %7 = fir.convert %6 : (!fir.ptr<f32>) -> !fir.ref<f32>
  fir.call @_QPtest2(%7) fastmath<contract> : (!fir.ref<f32>) -> ()
  return
}
func.func private @_QPtest2(!fir.ref<f32>)

// -----

// Two dummy arguments of composite type consisting of an allocatable
// component cannot alias:
// module m
//   type t
//      real, allocatable :: allocatable_component
//   end type t
// contains
//   subroutine test(a, b)
//     type(t) :: a, b
//     allocate(a%allocatable_component)
//     call test2(b%allocatable_component)
//   end subroutine test
// end module m

// CHECK-LABEL: Testing : "_QMmPtest"
// FIXME: MayAlias must be NoAlias
// CHECK: func.region0#0 <-> func.region0#1: MayAlias

func.func @_QMmPtest(%arg0: !fir.ref<!fir.type<_QMmTt{allocatable_component:!fir.box<!fir.heap<f32>>}>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.type<_QMmTt{allocatable_component:!fir.box<!fir.heap<f32>>}>> {fir.bindc_name = "b"}) attributes {test.ptr = "func"} {
  %0 = fir.field_index allocatable_component, !fir.type<_QMmTt{allocatable_component:!fir.box<!fir.heap<f32>>}>
  %1 = fir.coordinate_of %arg0, %0 : (!fir.ref<!fir.type<_QMmTt{allocatable_component:!fir.box<!fir.heap<f32>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %2 = fir.allocmem f32 {uniq_name = "_QMmEallocatable_component.alloc"}
  %3 = fir.embox %2 : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
  fir.store %3 to %1 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %4 = fir.field_index allocatable_component, !fir.type<_QMmTt{allocatable_component:!fir.box<!fir.heap<f32>>}>
  %5 = fir.coordinate_of %arg1, %4 : (!fir.ref<!fir.type<_QMmTt{allocatable_component:!fir.box<!fir.heap<f32>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %6 = fir.load %5 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %7 = fir.box_addr %6 : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  %8 = fir.convert %7 : (!fir.heap<f32>) -> !fir.ref<f32>
  fir.call @_QPtest2(%8) fastmath<contract> : (!fir.ref<f32>) -> ()
  return
}
func.func private @_QPtest2(!fir.ref<f32>)
