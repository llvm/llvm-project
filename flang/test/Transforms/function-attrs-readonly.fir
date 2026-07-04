// RUN: fir-opt --function-attr="set-readonly=true" %s | FileCheck %s --check-prefix=CHECK
// RUN: fir-opt --function-attr="set-readonly=true set-nocapture=true set-noalias=true" %s | FileCheck %s --check-prefix=ALL
// RUN: fir-opt --function-attr="set-readonly=false" %s | FileCheck %s --check-prefix=OFF

// Test translation of the FIR marker placed on INTENT(IN) by-reference dummy
// arguments to the LLVM dialect argument attribute.

// CHECK-LABEL: func.func private @test_ref(
// CHECK-SAME:    %{{.*}}: !fir.ref<i32> {fir.read_only, llvm.readonly}) {
// ALL-LABEL:   func.func private @test_ref(
// ALL-SAME:      %{{.*}}: !fir.ref<i32> {fir.read_only, llvm.noalias, llvm.nocapture, llvm.readonly}) {
// OFF-LABEL:   func.func private @test_ref(
// OFF-SAME:      %{{.*}}: !fir.ref<i32> {fir.read_only}) {
func.func private @test_ref(%arg0: !fir.ref<i32> {fir.read_only}) {
  return
}

// An unmarked reference is not read-only.
// CHECK-LABEL: func.func private @test_unmarked_ref(
// CHECK-SAME:    %{{.*}}: !fir.ref<i32>) {
func.func private @test_unmarked_ref(%arg0: !fir.ref<i32>) {
  return
}

// TARGET prevents noalias and nocapture, but it does not prevent readonly.
// Match the complete argument attribute dictionary under the ALL run so an
// accidental llvm.noalias or llvm.nocapture addition makes the test fail.
// CHECK-LABEL: func.func private @test_target(
// CHECK-SAME:    %{{.*}}: !fir.ref<i32> {fir.read_only, fir.target, llvm.readonly}) {
// ALL-LABEL:   func.func private @test_target(
// ALL-SAME:      %{{.*}}: !fir.ref<i32> {fir.read_only, fir.target, llvm.readonly}) {
func.func private @test_target(%arg0: !fir.ref<i32> {fir.read_only, fir.target}) {
  return
}

// A derived-type reference is handled like any other reference. Pointer or
// allocatable components do not affect the shallow LLVM readonly contract.
// CHECK-LABEL: func.func private @test_derived(
// CHECK-SAME:    %{{.*}}: !fir.ref<!fir.type<_QFtest_derivedTt{p:!fir.box<!fir.ptr<i32>>}>> {fir.read_only, llvm.readonly}) {
func.func private @test_derived(%arg0: !fir.ref<!fir.type<_QFtest_derivedTt{p:!fir.box<!fir.ptr<i32>>}>> {fir.read_only}) {
  return
}

// A Fortran BIND(C) definition must obey INTENT(IN), so readonly is valid.
// Unlike ordinary definitions, it must not receive noalias or nocapture.
// CHECK-LABEL: func.func @test_bindc_definition(
// CHECK-SAME:    %{{.*}}: !fir.ref<i32> {fir.read_only, llvm.readonly}) attributes
// ALL-LABEL:   func.func @test_bindc_definition(
// ALL-SAME:      %{{.*}}: !fir.ref<i32> {fir.read_only, llvm.readonly}) attributes
func.func @test_bindc_definition(%arg0: !fir.ref<i32> {fir.read_only}) attributes {fir.bindc_name = "test_bindc_definition", fir.proc_attrs = #fir.proc_attrs<bind_c>} {
  return
}

// Do not apply readonly to a declaration of an external C procedure: its body
// is not constrained by the Fortran INTENT appearing in the local interface.
// Use a module-style FIR name so this exercises the BIND(C)-declaration check,
// rather than the generic check that skips non-module declarations.
// CHECK-LABEL: func.func private @_QMreadonly_modPbindc_declaration(
// CHECK-SAME:    !fir.ref<i32> {fir.read_only}) attributes
func.func private @_QMreadonly_modPbindc_declaration(!fir.ref<i32> {fir.read_only}) attributes {fir.bindc_name = "bindc_declaration", fir.proc_attrs = #fir.proc_attrs<bind_c>}

// A declaration of a Fortran module procedure carries the Fortran contract.
// CHECK-LABEL: func.func private @_QMreadonly_modPfortran_declaration(
// CHECK-SAME:    !fir.ref<i32> {fir.read_only, llvm.readonly})
func.func private @_QMreadonly_modPfortran_declaration(!fir.ref<i32> {fir.read_only})

// fir.box is not a ReferenceType. The marker is intentionally inert until
// descriptor-level readonly is implemented.
// CHECK-LABEL: func.func private @test_box(
// CHECK-SAME:    %{{.*}}: !fir.box<!fir.array<?xi32>> {fir.read_only}) {
func.func private @test_box(%arg0: !fir.box<!fir.array<?xi32>> {fir.read_only}) {
  return
}
