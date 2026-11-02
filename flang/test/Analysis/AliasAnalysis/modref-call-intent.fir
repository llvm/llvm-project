// RUN:  fir-opt -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis-modref))' \
// RUN:  --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s

// A direct call uses the callee dummy's Fortran intent when the argument
// aliases the queried variable. Intent is the fortran_attrs attribute on the
// func.func argument. A fir.declare in the callee body is not consulted, so
// the same result holds for a declaration with no body.
//   intent(in)    -> Ref
//   intent(out)   -> Mod for a trivial non-pointer, non-allocatable dummy
//   intent(out)   -> ModRef for an allocatable or a derived type
//   intent(inout) -> ModRef
// A dummy with no intent stays ModRef. A local that is not passed is NoModRef.

// CHECK-LABEL: Testing : "test_call_intent"
// CHECK-DAG: in_call -> in#0: Ref{{$}}
// CHECK-DAG: in_call -> out#0: NoModRef{{$}}
// CHECK-DAG: in_call -> other#0: NoModRef{{$}}
// CHECK-DAG: out_call -> out#0: Mod{{$}}
// CHECK-DAG: out_call -> in#0: NoModRef{{$}}
// CHECK-DAG: inout_call -> inout#0: ModRef{{$}}
// CHECK-DAG: inout_call -> other#0: NoModRef{{$}}
// CHECK-DAG: unknown_call -> unknown#0: ModRef{{$}}
// CHECK-DAG: unknown_call -> other#0: NoModRef{{$}}
// CHECK-DAG: mixed_call -> in#0: Ref{{$}}
// CHECK-DAG: mixed_call -> inout#0: ModRef{{$}}
// CHECK-DAG: mixed_call -> out#0: Mod{{$}}
// CHECK-DAG: mixed_call -> other#0: NoModRef{{$}}
// CHECK-DAG: alloc_out_call -> alloc_out#0: ModRef{{$}}
// CHECK-DAG: alloc_out_call -> other#0: NoModRef{{$}}
// CHECK-DAG: derived_out_call -> derived_out#0: ModRef{{$}}
// CHECK-DAG: derived_out_call -> other#0: NoModRef{{$}}
// CHECK-DAG: decl_out_call -> decl_out#0: Mod{{$}}
// CHECK-DAG: decl_out_call -> other#0: NoModRef{{$}}
// CHECK-DAG: body_only_call -> body_only#0: ModRef{{$}}
// CHECK-DAG: body_only_call -> other#0: NoModRef{{$}}
// CHECK-DAG: sig_wins_call -> sig_wins#0: Ref{{$}}
// CHECK-DAG: sig_wins_call -> other#0: NoModRef{{$}}
func.func @test_call_intent() {
  %in_alloca = fir.alloca f32 {bindc_name = "in", uniq_name = "_QFtest_call_intentEin"}
  %in = fir.declare %in_alloca uniq_name("_QFtest_call_intentEin") {test.ptr = "in"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %out_alloca = fir.alloca f32 {bindc_name = "out", uniq_name = "_QFtest_call_intentEout"}
  %out = fir.declare %out_alloca uniq_name("_QFtest_call_intentEout") {test.ptr = "out"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %inout_alloca = fir.alloca f32 {bindc_name = "inout", uniq_name = "_QFtest_call_intentEinout"}
  %inout = fir.declare %inout_alloca uniq_name("_QFtest_call_intentEinout") {test.ptr = "inout"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %unknown_alloca = fir.alloca f32 {bindc_name = "unknown", uniq_name = "_QFtest_call_intentEunknown"}
  %unknown = fir.declare %unknown_alloca uniq_name("_QFtest_call_intentEunknown") {test.ptr = "unknown"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %other_alloca = fir.alloca f32 {bindc_name = "other", uniq_name = "_QFtest_call_intentEother"}
  %other = fir.declare %other_alloca uniq_name("_QFtest_call_intentEother") {test.ptr = "other"} : (!fir.ref<f32>) -> !fir.ref<f32>

  fir.call @_QPread_in(%in) {test.ptr = "in_call"} : (!fir.ref<f32>) -> ()
  fir.call @_QPfill_out(%out) {test.ptr = "out_call"} : (!fir.ref<f32>) -> ()
  fir.call @_QPupdate_inout(%inout) {test.ptr = "inout_call"} : (!fir.ref<f32>) -> ()
  fir.call @_QPno_intent(%unknown) {test.ptr = "unknown_call"} : (!fir.ref<f32>) -> ()
  fir.call @_QPscratch(%in, %inout, %out) {test.ptr = "mixed_call"} : (!fir.ref<f32>, !fir.ref<f32>, !fir.ref<f32>) -> ()

  %alloc_out_alloca = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = "alloc_out", uniq_name = "_QFtest_call_intentEalloc_out"}
  %alloc_out = fir.declare %alloc_out_alloca uniq_name("_QFtest_call_intentEalloc_out") fortran_attrs<allocatable> {test.ptr = "alloc_out"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  fir.call @_QPfill_alloc_out(%alloc_out) {test.ptr = "alloc_out_call"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> ()

  %derived_out_alloca = fir.alloca !fir.type<_QTdt{i:i32}> {bindc_name = "derived_out", uniq_name = "_QFtest_call_intentEderived_out"}
  %derived_out = fir.declare %derived_out_alloca uniq_name("_QFtest_call_intentEderived_out") {test.ptr = "derived_out"} : (!fir.ref<!fir.type<_QTdt{i:i32}>>) -> !fir.ref<!fir.type<_QTdt{i:i32}>>
  fir.call @_QPfill_derived_out(%derived_out) {test.ptr = "derived_out_call"} : (!fir.ref<!fir.type<_QTdt{i:i32}>>) -> ()

  %decl_out_alloca = fir.alloca f32 {bindc_name = "decl_out", uniq_name = "_QFtest_call_intentEdecl_out"}
  %decl_out = fir.declare %decl_out_alloca uniq_name("_QFtest_call_intentEdecl_out") {test.ptr = "decl_out"} : (!fir.ref<f32>) -> !fir.ref<f32>
  fir.call @_QPfill_out_decl(%decl_out) {test.ptr = "decl_out_call"} : (!fir.ref<f32>) -> ()

  %body_only_alloca = fir.alloca f32 {bindc_name = "body_only", uniq_name = "_QFtest_call_intentEbody_only"}
  %body_only = fir.declare %body_only_alloca uniq_name("_QFtest_call_intentEbody_only") {test.ptr = "body_only"} : (!fir.ref<f32>) -> !fir.ref<f32>
  fir.call @_QPbody_only_out(%body_only) {test.ptr = "body_only_call"} : (!fir.ref<f32>) -> ()

  %sig_wins_alloca = fir.alloca f32 {bindc_name = "sig_wins", uniq_name = "_QFtest_call_intentEsig_wins"}
  %sig_wins = fir.declare %sig_wins_alloca uniq_name("_QFtest_call_intentEsig_wins") {test.ptr = "sig_wins"} : (!fir.ref<f32>) -> !fir.ref<f32>
  fir.call @_QPsig_in_body_out(%sig_wins) {test.ptr = "sig_wins_call"} : (!fir.ref<f32>) -> ()
  return
}

func.func private @_QPread_in(%arg0: !fir.ref<f32> {fir.fortran_attrs = #fir.var_attrs<intent_in>}) {
  %x = fir.declare %arg0 uniq_name("_QFread_inEx") fortran_attrs<intent_in> : (!fir.ref<f32>) -> !fir.ref<f32>
  return
}

func.func private @_QPfill_out(%arg0: !fir.ref<f32> {fir.fortran_attrs = #fir.var_attrs<intent_out>}) {
  %x = fir.declare %arg0 uniq_name("_QFfill_outEx") fortran_attrs<intent_out> : (!fir.ref<f32>) -> !fir.ref<f32>
  return
}

func.func private @_QPupdate_inout(%arg0: !fir.ref<f32> {fir.fortran_attrs = #fir.var_attrs<intent_inout>}) {
  %x = fir.declare %arg0 uniq_name("_QFupdate_inoutEx") fortran_attrs<intent_inout> : (!fir.ref<f32>) -> !fir.ref<f32>
  return
}

func.func private @_QPno_intent(%arg0: !fir.ref<f32>) {
  %x = fir.declare %arg0 uniq_name("_QFno_intentEx") : (!fir.ref<f32>) -> !fir.ref<f32>
  return
}

func.func private @_QPscratch(%arg0: !fir.ref<f32> {fir.fortran_attrs = #fir.var_attrs<intent_in>}, %arg1: !fir.ref<f32> {fir.fortran_attrs = #fir.var_attrs<intent_inout>}, %arg2: !fir.ref<f32> {fir.fortran_attrs = #fir.var_attrs<intent_out>}) {
  %x = fir.declare %arg0 uniq_name("_QFscratchEx") fortran_attrs<intent_in> : (!fir.ref<f32>) -> !fir.ref<f32>
  %a = fir.declare %arg1 uniq_name("_QFscratchEa") fortran_attrs<intent_inout> : (!fir.ref<f32>) -> !fir.ref<f32>
  %res = fir.declare %arg2 uniq_name("_QFscratchEres") fortran_attrs<intent_out> : (!fir.ref<f32>) -> !fir.ref<f32>
  return
}

func.func private @_QPfill_alloc_out(%arg0: !fir.ref<!fir.box<!fir.heap<f32>>> {fir.fortran_attrs = #fir.var_attrs<intent_out>}) {
  %x = fir.declare %arg0 uniq_name("_QFfill_alloc_outEx") fortran_attrs<allocatable, intent_out> : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  return
}

func.func private @_QPfill_derived_out(%arg0: !fir.ref<!fir.type<_QTdt{i:i32}>> {fir.fortran_attrs = #fir.var_attrs<intent_out>}) {
  %x = fir.declare %arg0 uniq_name("_QFfill_derived_outEx") fortran_attrs<intent_out> : (!fir.ref<!fir.type<_QTdt{i:i32}>>) -> !fir.ref<!fir.type<_QTdt{i:i32}>>
  return
}

// No body: the signature attribute is the only place intent can live.
func.func private @_QPfill_out_decl(!fir.ref<f32> {fir.fortran_attrs = #fir.var_attrs<intent_out>})

// Intent on the declare only. It must not be visible across functions.
func.func private @_QPbody_only_out(%arg0: !fir.ref<f32>) {
  %x = fir.declare %arg0 uniq_name("_QFbody_only_outEx") fortran_attrs<intent_out> : (!fir.ref<f32>) -> !fir.ref<f32>
  return
}

// The signature wins when it disagrees with the declare in the body.
func.func private @_QPsig_in_body_out(%arg0: !fir.ref<f32> {fir.fortran_attrs = #fir.var_attrs<intent_in>}) {
  %x = fir.declare %arg0 uniq_name("_QFsig_in_body_outEx") fortran_attrs<intent_out> : (!fir.ref<f32>) -> !fir.ref<f32>
  return
}
