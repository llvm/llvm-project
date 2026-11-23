// RUN: fir-opt -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis-modref))' \
// RUN:   --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=LOOKUP
// RUN: fir-opt -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis-modref{no-symbol-lookups=true}))' \
// RUN:   --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOLOOKUP

// With symbol lookups disabled, fir.call modref does not look at the callee or
// at the global of a saved local, and assumes the worst instead. A pass that
// runs on a function must not inspect these operations.

// The intent of the dummy is only known from the callee.
// LOOKUP-LABEL: Testing : "test_intent"
// LOOKUP: in_call -> in#0: Ref{{$}}
// NOLOOKUP-LABEL: Testing : "test_intent"
// NOLOOKUP: in_call -> in#0: ModRef{{$}}
func.func @test_intent() {
  %0 = fir.alloca f32 {bindc_name = "in", uniq_name = "_QFtest_intentEin"}
  %1 = fir.declare %0 uniq_name("_QFtest_intentEin") {test.ptr = "in"} : (!fir.ref<f32>) -> !fir.ref<f32>
  fir.call @_QPread_in(%1) {test.ptr = "in_call"} : (!fir.ref<f32>) -> ()
  return
}
func.func private @_QPread_in(!fir.ref<f32> {fir.fortran_attrs = #fir.var_attrs<intent_in>})

// The callee is only known to be a Fortran procedure from its
// fir.internal_name attribute.
// LOOKUP-LABEL: Testing : "test_internal_name_"
// LOOKUP: callee_call -> xx#0: NoModRef{{$}}
// NOLOOKUP-LABEL: Testing : "test_internal_name_"
// NOLOOKUP: callee_call -> xx#0: ModRef{{$}}
func.func @test_internal_name_() {
  %0 = fir.alloca f32 {bindc_name = "xx", uniq_name = "_QFtest_internal_nameExx"}
  %1 = fir.declare %0 uniq_name("_QFtest_internal_nameExx") {test.ptr = "xx"} : (!fir.ref<f32>) -> !fir.ref<f32>
  fir.call @callee_() {test.ptr = "callee_call"} : () -> ()
  return
}
func.func private @callee_() attributes {fir.internal_name = "_QPcallee"}

// A saved local cannot be captured by the call unless it has the TARGET
// attribute, which is only known from its fir.global.
// LOOKUP-LABEL: Testing : "test_saved"
// LOOKUP: external_call -> xsaved#0: NoModRef{{$}}
// NOLOOKUP-LABEL: Testing : "test_saved"
// NOLOOKUP: external_call -> xsaved#0: ModRef{{$}}
fir.global internal @_QFtest_savedExsaved : f32 {
  %0 = fir.zero_bits f32
  fir.has_value %0 : f32
}
func.func @test_saved() {
  %0 = fir.address_of(@_QFtest_savedExsaved) : !fir.ref<f32>
  %1 = fir.declare %0 uniq_name("_QFtest_savedExsaved") {test.ptr = "xsaved"} : (!fir.ref<f32>) -> !fir.ref<f32>
  fir.call @_QPexternal() {test.ptr = "external_call"} : () -> ()
  return
}
func.func private @_QPexternal()
