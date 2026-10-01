// RUN:  fir-opt -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis-modref))' \
// RUN:  --mlir-disable-threading %s -o /dev/null 2>&1 | FileCheck %s

// A direct call uses the callee dummy's Fortran intent when the argument
// aliases the queried variable:
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
func.func @test_call_intent() {
  %in_alloca = fir.alloca f32 {bindc_name = "in", uniq_name = "_QFtest_call_intentEin"}
  %in = fir.declare %in_alloca {test.ptr = "in", uniq_name = "_QFtest_call_intentEin"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %out_alloca = fir.alloca f32 {bindc_name = "out", uniq_name = "_QFtest_call_intentEout"}
  %out = fir.declare %out_alloca {test.ptr = "out", uniq_name = "_QFtest_call_intentEout"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %inout_alloca = fir.alloca f32 {bindc_name = "inout", uniq_name = "_QFtest_call_intentEinout"}
  %inout = fir.declare %inout_alloca {test.ptr = "inout", uniq_name = "_QFtest_call_intentEinout"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %unknown_alloca = fir.alloca f32 {bindc_name = "unknown", uniq_name = "_QFtest_call_intentEunknown"}
  %unknown = fir.declare %unknown_alloca {test.ptr = "unknown", uniq_name = "_QFtest_call_intentEunknown"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %other_alloca = fir.alloca f32 {bindc_name = "other", uniq_name = "_QFtest_call_intentEother"}
  %other = fir.declare %other_alloca {test.ptr = "other", uniq_name = "_QFtest_call_intentEother"} : (!fir.ref<f32>) -> !fir.ref<f32>

  fir.call @_QPread_in(%in) {test.ptr = "in_call"} : (!fir.ref<f32>) -> ()
  fir.call @_QPfill_out(%out) {test.ptr = "out_call"} : (!fir.ref<f32>) -> ()
  fir.call @_QPupdate_inout(%inout) {test.ptr = "inout_call"} : (!fir.ref<f32>) -> ()
  fir.call @_QPno_intent(%unknown) {test.ptr = "unknown_call"} : (!fir.ref<f32>) -> ()
  fir.call @_QPscratch(%in, %inout, %out) {test.ptr = "mixed_call"} : (!fir.ref<f32>, !fir.ref<f32>, !fir.ref<f32>) -> ()

  %alloc_out_alloca = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = "alloc_out", uniq_name = "_QFtest_call_intentEalloc_out"}
  %alloc_out = fir.declare %alloc_out_alloca {fortran_attrs = #fir.var_attrs<allocatable>, test.ptr = "alloc_out", uniq_name = "_QFtest_call_intentEalloc_out"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  fir.call @_QPfill_alloc_out(%alloc_out) {test.ptr = "alloc_out_call"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> ()

  %derived_out_alloca = fir.alloca !fir.type<_QTdt{i:i32}> {bindc_name = "derived_out", uniq_name = "_QFtest_call_intentEderived_out"}
  %derived_out = fir.declare %derived_out_alloca {test.ptr = "derived_out", uniq_name = "_QFtest_call_intentEderived_out"} : (!fir.ref<!fir.type<_QTdt{i:i32}>>) -> !fir.ref<!fir.type<_QTdt{i:i32}>>
  fir.call @_QPfill_derived_out(%derived_out) {test.ptr = "derived_out_call"} : (!fir.ref<!fir.type<_QTdt{i:i32}>>) -> ()
  return
}

func.func private @_QPread_in(%arg0: !fir.ref<f32>) {
  %x = fir.declare %arg0 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFread_inEx"} : (!fir.ref<f32>) -> !fir.ref<f32>
  return
}

func.func private @_QPfill_out(%arg0: !fir.ref<f32>) {
  %x = fir.declare %arg0 {fortran_attrs = #fir.var_attrs<intent_out>, uniq_name = "_QFfill_outEx"} : (!fir.ref<f32>) -> !fir.ref<f32>
  return
}

func.func private @_QPupdate_inout(%arg0: !fir.ref<f32>) {
  %x = fir.declare %arg0 {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QFupdate_inoutEx"} : (!fir.ref<f32>) -> !fir.ref<f32>
  return
}

func.func private @_QPno_intent(%arg0: !fir.ref<f32>) {
  %x = fir.declare %arg0 {uniq_name = "_QFno_intentEx"} : (!fir.ref<f32>) -> !fir.ref<f32>
  return
}

func.func private @_QPscratch(%arg0: !fir.ref<f32>, %arg1: !fir.ref<f32>, %arg2: !fir.ref<f32>) {
  %x = fir.declare %arg0 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFscratchEx"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %a = fir.declare %arg1 {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QFscratchEa"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %res = fir.declare %arg2 {fortran_attrs = #fir.var_attrs<intent_out>, uniq_name = "_QFscratchEres"} : (!fir.ref<f32>) -> !fir.ref<f32>
  return
}

func.func private @_QPfill_alloc_out(%arg0: !fir.ref<!fir.box<!fir.heap<f32>>>) {
  %x = fir.declare %arg0 {fortran_attrs = #fir.var_attrs<allocatable, intent_out>, uniq_name = "_QFfill_alloc_outEx"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  return
}

func.func private @_QPfill_derived_out(%arg0: !fir.ref<!fir.type<_QTdt{i:i32}>>) {
  %x = fir.declare %arg0 {fortran_attrs = #fir.var_attrs<intent_out>, uniq_name = "_QFfill_derived_outEx"} : (!fir.ref<!fir.type<_QTdt{i:i32}>>) -> !fir.ref<!fir.type<_QTdt{i:i32}>>
  return
}
