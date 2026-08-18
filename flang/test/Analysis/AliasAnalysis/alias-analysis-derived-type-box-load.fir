// Exercise AliasAnalysis for fir.load of fir.box values coming from dummy
// arguments, nested derived-type component chains, allocatable variables,
// whole-object pointer/target semantics, and globals.

// RUN: fir-opt %s -split-input-file -o /dev/null --mlir-disable-threading \
// RUN:   -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' \
// RUN:   2>&1 | FileCheck %s

// subroutine test_dummy_ptr(p)
//   real, pointer :: p
//   real :: v
// end subroutine

// CHECK-LABEL: Testing : "_QPtest_dummy_ptr"
// Data through the pointer dummy cannot alias an unrelated local scalar.
// CHECK-DAG: local#0 <-> p_dummy.tgt#0: NoAlias
// CHECK-LABEL: Testing : "_QPtest_dummy_ptr.fir"
// CHECK-DAG: local.fir#0 <-> p_dummy.tgt.fir#0: NoAlias

func.func @_QPtest_dummy_ptr(%arg0: !fir.ref<!fir.box<!fir.ptr<f32>>> {fir.bindc_name = "p"}) {
  %0:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "p_dummy", uniq_name = "_QFtest_dummy_ptrEp"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> (!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.ptr<f32>>>)
  %1 = fir.alloca f32 {bindc_name = "v", uniq_name = "_QFtest_dummy_ptrEv"}
  %2:2 = hlfir.declare %1 {test.ptr = "local", uniq_name = "_QFtest_dummy_ptrEv"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %3 = fir.load %0#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %4 = fir.box_addr %3 {test.ptr = "p_dummy.tgt"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  return
}

func.func @_QPtest_dummy_ptr.fir(%arg0: !fir.ref<!fir.box<!fir.ptr<f32>>> {fir.bindc_name = "p"}) {
  %0 = fir.declare %arg0 {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "p_dummy.fir", uniq_name = "_QFtest_dummy_ptrEp"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %1 = fir.alloca f32 {bindc_name = "v", uniq_name = "_QFtest_dummy_ptrEv"}
  %2 = fir.declare %1 {test.ptr = "local.fir", uniq_name = "_QFtest_dummy_ptrEv"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %3 = fir.load %0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %4 = fir.box_addr %3 {test.ptr = "p_dummy.tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  return
}

// -----

// module m
// type :: inner
//   real, pointer :: p
// end type inner
// type :: outer
//   type(inner) :: in
// end type outer
// contains
// subroutine nested_ptr(x)
//   type(outer) :: x
//   real :: v
// end subroutine
// end module

// CHECK-LABEL: Testing : "_QMmPnested_ptr"
// Target of nested pointer component vs unrelated local.
// CHECK-DAG: nested_local#0 <-> x_in_p.tgt#0: NoAlias
// Whole-variable x may overlap component storage (see load-ptr-designate.fir).
// CHECK-DAG: x#0 <-> x_in_p#0: MayAlias
// CHECK-DAG: x#0 <-> x_in_p.tgt#0: MayAlias
// CHECK-LABEL: Testing : "_QMmPnested_ptr.fir"
// CHECK-DAG: nested_local.fir#0 <-> x_in_p.tgt.fir#0: NoAlias
// CHECK-DAG: x.fir#0 <-> x_in_p.fir#0: MayAlias
// CHECK-DAG: x.fir#0 <-> x_in_p.tgt.fir#0: MayAlias

func.func @_QMmPnested_ptr(%arg0: !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>> {fir.bindc_name = "x"}) {
  %0:2 = hlfir.declare %arg0 {test.ptr = "x", uniq_name = "_QMmFnested_ptrEx"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>, !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>)
  %1 = fir.alloca f32 {bindc_name = "v", uniq_name = "_QMmFnested_ptrEv"}
  %2:2 = hlfir.declare %1 {test.ptr = "nested_local", uniq_name = "_QMmFnested_ptrEv"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %3 = hlfir.designate %0#0{"in"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>
  %4 = hlfir.designate %3{"p"} {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "x_in_p"} : (!fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %5 = fir.load %4 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %6 = fir.box_addr %5 {test.ptr = "x_in_p.tgt"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  return
}

func.func @_QMmPnested_ptr.fir(%arg0: !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>> {fir.bindc_name = "x"}) {
  %0 = fir.declare %arg0 {test.ptr = "x.fir", uniq_name = "_QMmFnested_ptrEx"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %1 = fir.alloca f32 {bindc_name = "v", uniq_name = "_QMmFnested_ptrEv"}
  %2 = fir.declare %1 {test.ptr = "nested_local.fir", uniq_name = "_QMmFnested_ptrEv"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %3 = fir.coordinate_of %0, in : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>
  %4 = fir.coordinate_of %3, p {test.ptr = "x_in_p.fir"} : (!fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %5 = fir.load %4 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %6 = fir.box_addr %5 {test.ptr = "x_in_p.tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  return
}

// -----

// type(outer), allocatable :: y — the variable is a box holding heap-allocated
// outer; y%in%p uses fir.load of the outer box, box_addr to the record, then
// component access and fir.load of the inner pointer's box.
//
// Fortran sketch (module m types as in nested_ptr above):
//   type(outer), allocatable :: y
//   real :: v
//   allocate(y)
//   y%in%p = 2.0

// CHECK-LABEL: Testing : "_QMmPalloc_outer_box"
// Target of component pointer inside allocatable outer vs unrelated local.
// CHECK-DAG: alloc_outer_local#0 <-> y_alloc_in_p.tgt#0: NoAlias
// CHECK-LABEL: Testing : "_QMmPalloc_outer_box.fir"
// CHECK-DAG: alloc_outer_local.fir#0 <-> y_alloc_in_p.tgt.fir#0: NoAlias

func.func @_QMmPalloc_outer_box() {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>> {bindc_name = "y", uniq_name = "_QMmFalloc_outer_boxEy"}
  %1 = fir.allocmem !fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}> {uniq_name = "_QMmFalloc_outer_boxEy.alloc"}
  %2 = fir.embox %1 : (!fir.heap<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.box<!fir.heap<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>
  fir.store %2 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>
  %3:2 = hlfir.declare %0 {fortran_attrs = #fir.var_attrs<allocatable>, test.ptr = "y_alloc", uniq_name = "_QMmFalloc_outer_boxEy"} : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>, !fir.ref<!fir.box<!fir.heap<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>)
  %4 = fir.alloca f32 {bindc_name = "v", uniq_name = "_QMmFalloc_outer_boxEv"}
  %5:2 = hlfir.declare %4 {test.ptr = "alloc_outer_local", uniq_name = "_QMmFalloc_outer_boxEv"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %6 = fir.load %3#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>
  %7 = fir.box_addr %6 : (!fir.box<!fir.heap<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>) -> !fir.heap<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %8 = hlfir.designate %7{"in"} : (!fir.heap<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>
  %9 = hlfir.designate %8{"p"} {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "y_alloc_in_p"} : (!fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %10 = fir.load %9 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %11 = fir.box_addr %10 {test.ptr = "y_alloc_in_p.tgt"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  return
}

func.func @_QMmPalloc_outer_box.fir() {
  %0 = fir.alloca !fir.box<!fir.heap<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>> {bindc_name = "y", uniq_name = "_QMmFalloc_outer_boxEy"}
  %1 = fir.allocmem !fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}> {uniq_name = "_QMmFalloc_outer_boxEy.alloc"}
  %2 = fir.embox %1 : (!fir.heap<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.box<!fir.heap<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>
  fir.store %2 to %0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>
  %3 = fir.declare %0 {fortran_attrs = #fir.var_attrs<allocatable>, test.ptr = "y_alloc.fir", uniq_name = "_QMmFalloc_outer_boxEy"} : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>
  %4 = fir.alloca f32 {bindc_name = "v", uniq_name = "_QMmFalloc_outer_boxEv"}
  %5 = fir.declare %4 {test.ptr = "alloc_outer_local.fir", uniq_name = "_QMmFalloc_outer_boxEv"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %6 = fir.load %3 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>
  %7 = fir.box_addr %6 : (!fir.box<!fir.heap<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>) -> !fir.heap<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %8 = fir.coordinate_of %7, in : (!fir.heap<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>
  %9 = fir.coordinate_of %8, p {test.ptr = "y_alloc_in_p.fir"} : (!fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %10 = fir.load %9 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %11 = fir.box_addr %10 {test.ptr = "y_alloc_in_p.tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  return
}

// -----

// type(outer), pointer :: p, q  — whole derived-type POINTER variables (box of
// ptr-to-type).  MayAlias each other's pointees and scalars with TARGET.
// type(outer), target :: z, u — data via z%in%p may alias a real POINTER dummy
// but currently MayAlias a second target's component data (conservative).
// fir.global outer — disambiguate from locals and from a second global.

// CHECK-LABEL: Testing : "_QMmPtest_derived_ptr_neg"
// CHECK-DAG: p_rec#0 <-> q_rec#0: MayAlias
// CHECK-DAG: t_scalar#0 <-> p_rec#0: MayAlias
// CHECK-LABEL: Testing : "_QMmPtest_derived_ptr_neg_fir"
// CHECK-DAG: p_rec.fir#0 <-> q_rec.fir#0: MayAlias
// CHECK-DAG: t_scalar.fir#0 <-> p_rec.fir#0: MayAlias

func.func @_QMmPtest_derived_ptr_neg(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>> {fir.bindc_name = "p"}, %arg1: !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>> {fir.bindc_name = "q"}, %arg2: !fir.ref<f32> {fir.bindc_name = "tscalar", fir.target}) {
  %p:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "p_op", uniq_name = "_QMmFp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>)
  %q:2 = hlfir.declare %arg1 {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "q_op", uniq_name = "_QMmFq"} : (!fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>)
  %t:2 = hlfir.declare %arg2 {fortran_attrs = #fir.var_attrs<target>, test.ptr = "t_scalar", uniq_name = "_QMmFt"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  %lp = fir.load %p#0 : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>
  %p_rec = fir.box_addr %lp {test.ptr = "p_rec"} : (!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>) -> !fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %lq = fir.load %q#0 : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>
  %q_rec = fir.box_addr %lq {test.ptr = "q_rec"} : (!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>) -> !fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  return
}

func.func @_QMmPtest_derived_ptr_neg_fir(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>> {fir.bindc_name = "p"}, %arg1: !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>> {fir.bindc_name = "q"}, %arg2: !fir.ref<f32> {fir.bindc_name = "tscalar", fir.target}) {
  %p = fir.declare %arg0 {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "p_op.fir", uniq_name = "_QMmFp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>
  %q = fir.declare %arg1 {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "q_op.fir", uniq_name = "_QMmFq"} : (!fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>
  %t = fir.declare %arg2 {fortran_attrs = #fir.var_attrs<target>, test.ptr = "t_scalar.fir", uniq_name = "_QMmFt"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %lp = fir.load %p : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>
  %p_rec = fir.box_addr %lp {test.ptr = "p_rec.fir"} : (!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>) -> !fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %lq = fir.load %q : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>
  %q_rec = fir.box_addr %lq {test.ptr = "q_rec.fir"} : (!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>) -> !fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  return
}

// -----

// CHECK-LABEL: Testing : "_QMmPtest_derived_target_neg"
// TODO: Fortran allows refining z_p_tgt vs u_p_tgt to NoAlias (distinct TARGET).
// CHECK-DAG: z_p_tgt#0 <-> rptr_tgt#0: MayAlias
// CHECK-DAG: z_p_tgt#0 <-> u_p_tgt#0: MayAlias
// CHECK-LABEL: Testing : "_QMmPtest_derived_target_neg_fir"
// CHECK-DAG: z_p_tgt.fir#0 <-> rptr_tgt.fir#0: MayAlias
// CHECK-DAG: z_p_tgt.fir#0 <-> u_p_tgt.fir#0: MayAlias

func.func @_QMmPtest_derived_target_neg(%arg0: !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>> {fir.bindc_name = "z", fir.target}, %arg1: !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>> {fir.bindc_name = "u", fir.target}, %arg2: !fir.ref<!fir.box<!fir.ptr<f32>>> {fir.bindc_name = "rptr"}) {
  %z:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<target>, test.ptr = "z_op", uniq_name = "_QMmFz"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>, !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>)
  %u:2 = hlfir.declare %arg1 {fortran_attrs = #fir.var_attrs<target>, test.ptr = "u_op", uniq_name = "_QMmFu"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>, !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>)
  %r:2 = hlfir.declare %arg2 {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "rptr_op", uniq_name = "_QMmFr"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> (!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.ptr<f32>>>)
  %zin = hlfir.designate %z#0{"in"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>
  %zp = hlfir.designate %zin{"p"} {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "z_in_p"} : (!fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %zpl = fir.load %zp : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %zpt = fir.box_addr %zpl {test.ptr = "z_p_tgt"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %uin = hlfir.designate %u#0{"in"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>
  %up = hlfir.designate %uin{"p"} {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "u_in_p"} : (!fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %upl = fir.load %up : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %upt = fir.box_addr %upl {test.ptr = "u_p_tgt"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %rl = fir.load %r#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %rt = fir.box_addr %rl {test.ptr = "rptr_tgt"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  return
}

func.func @_QMmPtest_derived_target_neg_fir(%arg0: !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>> {fir.bindc_name = "z", fir.target}, %arg1: !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>> {fir.bindc_name = "u", fir.target}, %arg2: !fir.ref<!fir.box<!fir.ptr<f32>>> {fir.bindc_name = "rptr"}) {
  %z = fir.declare %arg0 {fortran_attrs = #fir.var_attrs<target>, test.ptr = "z_op.fir", uniq_name = "_QMmFz"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %u = fir.declare %arg1 {fortran_attrs = #fir.var_attrs<target>, test.ptr = "u_op.fir", uniq_name = "_QMmFu"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %r = fir.declare %arg2 {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "rptr_op.fir", uniq_name = "_QMmFr"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %zin = fir.coordinate_of %z, in : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>
  %zp = fir.coordinate_of %zin, p {test.ptr = "z_in_p.fir"} : (!fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %zpl = fir.load %zp : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %zpt = fir.box_addr %zpl {test.ptr = "z_p_tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %uin = fir.coordinate_of %u, in : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>
  %up = fir.coordinate_of %uin, p {test.ptr = "u_in_p.fir"} : (!fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %upl = fir.load %up : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %upt = fir.box_addr %upl {test.ptr = "u_p_tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %rl = fir.load %r : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %rt = fir.box_addr %rl {test.ptr = "rptr_tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  return
}

// -----

fir.global @_QMmEg_outer : !fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}> {
  %0 = fir.undefined !fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>
  fir.has_value %0 : !fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>
}

// CHECK-LABEL: Testing : "_QMmPtest_global_outer"
// CHECK-DAG: g_outer#0 <-> loc_outer#0: NoAlias
// CHECK-DAG: g_in_p#0 <-> loc_in_p#0: NoAlias
// CHECK-DAG: g_p_tgt#0 <-> loc_in_p#0: NoAlias
// CHECK-LABEL: Testing : "_QMmPtest_global_outer_fir"
// CHECK-DAG: g_outer.fir#0 <-> loc_outer.fir#0: NoAlias
// CHECK-DAG: g_in_p.fir#0 <-> loc_in_p.fir#0: NoAlias
// CHECK-DAG: g_p_tgt.fir#0 <-> loc_in_p.fir#0: NoAlias

func.func @_QMmPtest_global_outer() {
  %0 = fir.address_of(@_QMmEg_outer) : !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %g:2 = hlfir.declare %0 {test.ptr = "g_outer", uniq_name = "_QMmEg_outer"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>, !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>)
  %1 = fir.alloca !fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}> {uniq_name = "_QMmFloc"}
  %loc:2 = hlfir.declare %1 {test.ptr = "loc_outer", uniq_name = "_QMmFloc"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>, !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>)
  %gin = hlfir.designate %g#0{"in"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>
  %gp = hlfir.designate %gin{"p"} {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "g_in_p"} : (!fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %gl = fir.load %gp : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %gt = fir.box_addr %gl {test.ptr = "g_p_tgt"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %lin = hlfir.designate %loc#0{"in"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>
  %lp = hlfir.designate %lin{"p"} {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "loc_in_p"} : (!fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %ll = fir.load %lp : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %lt = fir.box_addr %ll {test.ptr = "loc_p_tgt"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  return
}

func.func @_QMmPtest_global_outer_fir() {
  %0 = fir.address_of(@_QMmEg_outer) : !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %g = fir.declare %0 {test.ptr = "g_outer.fir", uniq_name = "_QMmEg_outer"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %1 = fir.alloca !fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}> {uniq_name = "_QMmFloc"}
  %loc = fir.declare %1 {test.ptr = "loc_outer.fir", uniq_name = "_QMmFloc"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %gin = fir.coordinate_of %g, in : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>
  %gp = fir.coordinate_of %gin, p {test.ptr = "g_in_p.fir"} : (!fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %gl = fir.load %gp : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %gt = fir.box_addr %gl {test.ptr = "g_p_tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %lin = fir.coordinate_of %loc, in : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>
  %lp = fir.coordinate_of %lin, p {test.ptr = "loc_in_p.fir"} : (!fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %ll = fir.load %lp : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %lt = fir.box_addr %ll {test.ptr = "loc_p_tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  return
}

// -----

fir.global @_QMmEg1pair : !fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}> {
  %0 = fir.undefined !fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>
  fir.has_value %0 : !fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>
}
fir.global @_QMmEg2pair : !fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}> {
  %0 = fir.undefined !fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>
  fir.has_value %0 : !fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>
}

// CHECK-LABEL: Testing : "_QMmPtest_two_globals_outer"
// CHECK-DAG: g1_outer#0 <-> g2_outer#0: NoAlias
// CHECK-DAG: g1_p_tgt#0 <-> g2_in_p#0: NoAlias
// CHECK-DAG: g1_p_tgt#0 <-> g2_p_tgt#0: MayAlias
// CHECK-LABEL: Testing : "_QMmPtest_two_globals_outer_fir"
// CHECK-DAG: g1_outer.fir#0 <-> g2_outer.fir#0: NoAlias
// CHECK-DAG: g1_p_tgt.fir#0 <-> g2_in_p.fir#0: NoAlias
// CHECK-DAG: g1_p_tgt.fir#0 <-> g2_p_tgt.fir#0: MayAlias

func.func @_QMmPtest_two_globals_outer() {
  %0 = fir.address_of(@_QMmEg1pair) : !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %g1:2 = hlfir.declare %0 {test.ptr = "g1_outer", uniq_name = "_QMmEg1pair"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>, !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>)
  %1 = fir.address_of(@_QMmEg2pair) : !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %g2:2 = hlfir.declare %1 {test.ptr = "g2_outer", uniq_name = "_QMmEg2pair"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>, !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>)
  %g1in = hlfir.designate %g1#0{"in"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>
  %g1p = hlfir.designate %g1in{"p"} {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "g1_in_p"} : (!fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %g1l = fir.load %g1p : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %g1t = fir.box_addr %g1l {test.ptr = "g1_p_tgt"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %g2in = hlfir.designate %g2#0{"in"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>
  %g2p = hlfir.designate %g2in{"p"} {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "g2_in_p"} : (!fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %g2l = fir.load %g2p : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %g2t = fir.box_addr %g2l {test.ptr = "g2_p_tgt"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  return
}

func.func @_QMmPtest_two_globals_outer_fir() {
  %0 = fir.address_of(@_QMmEg1pair) : !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %g1 = fir.declare %0 {test.ptr = "g1_outer.fir", uniq_name = "_QMmEg1pair"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %1 = fir.address_of(@_QMmEg2pair) : !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %g2 = fir.declare %1 {test.ptr = "g2_outer.fir", uniq_name = "_QMmEg2pair"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %g1in = fir.coordinate_of %g1, in : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>
  %g1p = fir.coordinate_of %g1in, p {test.ptr = "g1_in_p.fir"} : (!fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %g1l = fir.load %g1p : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %g1t = fir.box_addr %g1l {test.ptr = "g1_p_tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %g2in = fir.coordinate_of %g2, in : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>
  %g2p = fir.coordinate_of %g2in, p {test.ptr = "g2_in_p.fir"} : (!fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %g2l = fir.load %g2p : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %g2t = fir.box_addr %g2l {test.ptr = "g2_p_tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  return
}

// -----

// Two globals of the same outer type: one ordinary, one TARGET. Exercises
// disambiguation of static storage for composite objects vs pointer-component
// slots; pointee access stays MayAlias across globals (pointer vs target).

fir.global @_QMmEg_nt : !fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}> {
  %0 = fir.undefined !fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>
  fir.has_value %0 : !fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>
}
fir.global @_QMmEg_tgt_m target : !fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}> {
  %0 = fir.undefined !fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>
  fir.has_value %0 : !fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>
}

// CHECK-LABEL: Testing : "_QMmPtest_global_target_vs_ptr"
// CHECK-DAG: g_nt#0 <-> g_tg#0: NoAlias
// CHECK-DAG: g_nt_in_p#0 <-> g_tg_in_p#0: NoAlias
// CHECK-DAG: g_tg_in_p#0 <-> g_nt_p_tgt#0: NoAlias
// CHECK-DAG: g_nt_p_tgt#0 <-> g_tg_p_tgt#0: MayAlias
// CHECK-LABEL: Testing : "_QMmPtest_global_target_vs_ptr_fir"
// CHECK-DAG: g_nt.fir#0 <-> g_tg.fir#0: NoAlias
// CHECK-DAG: g_nt_in_p.fir#0 <-> g_tg_in_p.fir#0: NoAlias
// CHECK-DAG: g_tg_in_p.fir#0 <-> g_nt_p_tgt.fir#0: NoAlias
// CHECK-DAG: g_nt_p_tgt.fir#0 <-> g_tg_p_tgt.fir#0: MayAlias

func.func @_QMmPtest_global_target_vs_ptr() {
  %0 = fir.address_of(@_QMmEg_nt) : !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %nt:2 = hlfir.declare %0 {test.ptr = "g_nt", uniq_name = "_QMmEg_nt"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>, !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>)
  %1 = fir.address_of(@_QMmEg_tgt_m) : !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %tg:2 = hlfir.declare %1 {fortran_attrs = #fir.var_attrs<target>, test.ptr = "g_tg", uniq_name = "_QMmEg_tgt_m"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>, !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>)
  %ntin = hlfir.designate %nt#0{"in"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>
  %ntp = hlfir.designate %ntin{"p"} {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "g_nt_in_p"} : (!fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %tgin = hlfir.designate %tg#0{"in"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>
  %tgp = hlfir.designate %tgin{"p"} {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "g_tg_in_p"} : (!fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %ntl = fir.load %ntp : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %ntt = fir.box_addr %ntl {test.ptr = "g_nt_p_tgt"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %tgl = fir.load %tgp : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %tgt = fir.box_addr %tgl {test.ptr = "g_tg_p_tgt"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  return
}

func.func @_QMmPtest_global_target_vs_ptr_fir() {
  %0 = fir.address_of(@_QMmEg_nt) : !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %nt = fir.declare %0 {test.ptr = "g_nt.fir", uniq_name = "_QMmEg_nt"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %1 = fir.address_of(@_QMmEg_tgt_m) : !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %tg = fir.declare %1 {fortran_attrs = #fir.var_attrs<target>, test.ptr = "g_tg.fir", uniq_name = "_QMmEg_tgt_m"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %ntin = fir.coordinate_of %nt, in : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>
  %ntp = fir.coordinate_of %ntin, p {test.ptr = "g_nt_in_p.fir"} : (!fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %tgin = fir.coordinate_of %tg, in : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>
  %tgp = fir.coordinate_of %tgin, p {test.ptr = "g_tg_in_p.fir"} : (!fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %ntl = fir.load %ntp : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %ntt = fir.box_addr %ntl {test.ptr = "g_nt_p_tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %tgl = fir.load %tgp : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %tgt = fir.box_addr %tgl {test.ptr = "g_tg_p_tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  return
}

// -----

// Reproducer for pointer-deref vs TARGET aliasing.
// A local POINTER may be associated with a TARGET dummy argument at runtime.
// Accessing a component's descriptor slot or pointee data through the pointer
// must MayAlias the same access through the TARGET variable.
//
// subroutine test_ptr_target_slot(tgt)
//   type(outer), target :: tgt
//   type(outer), pointer :: ptr
//   ptr => tgt         ! legal per F18 8.5.7
//   ! ptr%in%p (box) may alias tgt%in%p (box)
//   ! ptr%in%p.tgt may alias tgt%in%p.tgt
// end subroutine

// CHECK-LABEL: Testing : "_QMmPtest_ptr_target_slot"
// Descriptor slots accessed through pointer vs TARGET: must MayAlias.
// CHECK-DAG: ptr_in_p#0 <-> tgt_in_p#0: MayAlias
// Data through pointer vs TARGET: must MayAlias.
// CHECK-DAG: ptr_in_p_tgt#0 <-> tgt_in_p_tgt#0: MayAlias
// Data through pointer vs local without TARGET: NoAlias.
// CHECK-DAG: slot_local#0 <-> ptr_in_p_tgt#0: NoAlias
// CHECK-LABEL: Testing : "_QMmPtest_ptr_target_slot_fir"
// CHECK-DAG: ptr_in_p.fir#0 <-> tgt_in_p.fir#0: MayAlias
// CHECK-DAG: ptr_in_p_tgt.fir#0 <-> tgt_in_p_tgt.fir#0: MayAlias
// CHECK-DAG: slot_local.fir#0 <-> ptr_in_p_tgt.fir#0: NoAlias

func.func @_QMmPtest_ptr_target_slot(%arg0: !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>> {fir.bindc_name = "tgt", fir.target}) {
  %0 = fir.alloca !fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>> {uniq_name = "_QMmFtest_ptr_target_slotEptr"}
  %ptr:2 = hlfir.declare %0 {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "ptr_op", uniq_name = "_QMmFtest_ptr_target_slotEptr"} : (!fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>)
  %tgt:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<target>, test.ptr = "tgt_op", uniq_name = "_QMmFtest_ptr_target_slotEtgt"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>, !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>)
  %local = fir.alloca f32 {bindc_name = "v", uniq_name = "_QMmFtest_ptr_target_slotEv"}
  %ldecl:2 = hlfir.declare %local {test.ptr = "slot_local", uniq_name = "_QMmFtest_ptr_target_slotEv"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)

  // Access through pointer: ptr%in%p (descriptor slot)
  %lptr = fir.load %ptr#0 : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>
  %ptr_addr = fir.box_addr %lptr : (!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>) -> !fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %ptr_in = hlfir.designate %ptr_addr{"in"} : (!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>
  %ptr_in_p = hlfir.designate %ptr_in{"p"} {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "ptr_in_p"} : (!fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %ptr_in_p_load = fir.load %ptr_in_p : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %ptr_in_p_tgt = fir.box_addr %ptr_in_p_load {test.ptr = "ptr_in_p_tgt"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>

  // Access through target: tgt%in%p (descriptor slot)
  %tgt_in = hlfir.designate %tgt#0{"in"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>
  %tgt_in_p = hlfir.designate %tgt_in{"p"} {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "tgt_in_p"} : (!fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %tgt_in_p_load = fir.load %tgt_in_p : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %tgt_in_p_tgt = fir.box_addr %tgt_in_p_load {test.ptr = "tgt_in_p_tgt"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  return
}

func.func @_QMmPtest_ptr_target_slot_fir(%arg0: !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>> {fir.bindc_name = "tgt", fir.target}) {
  %0 = fir.alloca !fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>> {uniq_name = "_QMmFtest_ptr_target_slotEptr"}
  %ptr = fir.declare %0 {fortran_attrs = #fir.var_attrs<pointer>, test.ptr = "ptr_op.fir", uniq_name = "_QMmFtest_ptr_target_slotEptr"} : (!fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>
  %tgt = fir.declare %arg0 {fortran_attrs = #fir.var_attrs<target>, test.ptr = "tgt_op.fir", uniq_name = "_QMmFtest_ptr_target_slotEtgt"} : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %local = fir.alloca f32 {bindc_name = "v", uniq_name = "_QMmFtest_ptr_target_slotEv"}
  %ldecl = fir.declare %local {test.ptr = "slot_local.fir", uniq_name = "_QMmFtest_ptr_target_slotEv"} : (!fir.ref<f32>) -> !fir.ref<f32>

  // Access through pointer: ptr%in%p (descriptor slot)
  %lptr = fir.load %ptr : !fir.ref<!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>>
  %ptr_addr = fir.box_addr %lptr : (!fir.box<!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>>) -> !fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>
  %ptr_in = fir.coordinate_of %ptr_addr, in : (!fir.ptr<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>
  %ptr_in_p = fir.coordinate_of %ptr_in, p {test.ptr = "ptr_in_p.fir"} : (!fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %ptr_in_p_load = fir.load %ptr_in_p : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %ptr_in_p_tgt = fir.box_addr %ptr_in_p_load {test.ptr = "ptr_in_p_tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>

  // Access through target: tgt%in%p (descriptor slot)
  %tgt_in = fir.coordinate_of %tgt, in : (!fir.ref<!fir.type<_QMmTouter{in:!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>}>>) -> !fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>
  %tgt_in_p = fir.coordinate_of %tgt_in, p {test.ptr = "tgt_in_p.fir"} : (!fir.ref<!fir.type<_QMmTinner{p:!fir.box<!fir.ptr<f32>>}>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %tgt_in_p_load = fir.load %tgt_in_p : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %tgt_in_p_tgt = fir.box_addr %tgt_in_p_load {test.ptr = "tgt_in_p_tgt.fir"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  return
}
