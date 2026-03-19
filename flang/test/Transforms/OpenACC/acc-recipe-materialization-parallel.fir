// RUN: fir-opt %s -acc-recipe-materialization | FileCheck %s

// Test that the reduction recipes are correctly inlined when attached to a
// parallel construct without loop. Verify init and combine materialize in the region.
// CHECK-LABEL: func.func @par_reduction_clause_
// CHECK:       acc.parallel {
// CHECK:       [[PRIVATE:%.*]] = acc.reduction_init {{.*}} <add>
// CHECK-NEXT:  [[ZERO:%.*]] = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:  [[ALLOCA:%.*]] = fir.alloca f64
// CHECK-NEXT:  {{.*}} = fir.declare [[ALLOCA]] {{.*}}acc.reduction.init
// CHECK-NEXT:  fir.store [[ZERO]] to {{.*}}
// CHECK-NEXT:  acc.yield {{.*}}
// CHECK:       } {{.*}}acc.var_name = #acc.var_name<"tmp">
// CHECK:       fir.load [[PRIVATE]]
// CHECK:       fir.store {{.*}} to [[PRIVATE]]
// CHECK:       acc.reduction_combine_region [[PRIVATE]] into [[REDUCVAR:%.*]] :
// CHECK:       [[LOADVAR:%.*]] = fir.load [[REDUCVAR]]
// CHECK-NEXT:  [[LOADPRIV:%.*]] = fir.load [[PRIVATE]]
// CHECK-NEXT:  [[COMBINE:%.*]] = arith.addf [[LOADVAR]], [[LOADPRIV]]
// CHECK-NEXT:  fir.store [[COMBINE]] to [[REDUCVAR]]
// CHECK:       acc.yield

acc.reduction.recipe @reduction_add_ref_f64 : !fir.ref<f64> reduction_operator <add> init {
^bb0(%arg0: !fir.ref<f64>):
  %cst = arith.constant 0.000000e+00 : f64
  %0 = fir.alloca f64
  %1 = fir.declare %0 {uniq_name = "acc.reduction.init"} : (!fir.ref<f64>) -> !fir.ref<f64>
  fir.store %cst to %1 : !fir.ref<f64>
  acc.yield %1 : !fir.ref<f64>
} combiner {
^bb0(%arg0: !fir.ref<f64>, %arg1: !fir.ref<f64>):
  %0 = fir.load %arg0 : !fir.ref<f64>
  %1 = fir.load %arg1 : !fir.ref<f64>
  %2 = arith.addf %0, %1 fastmath<contract> : f64
  fir.store %2 to %arg0 : !fir.ref<f64>
  acc.yield %arg0 : !fir.ref<f64>
}
func.func @par_reduction_clause_(%arg0: !fir.ref<f64> {fir.bindc_name = "tmp"}) attributes {fir.internal_name = "_QPpar_reduction_clause"} {
  %cst = arith.constant 1.000000e+00 : f64
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "_QFpar_reduction_clauseEtmp"} : (!fir.ref<f64>, !fir.dscope) -> !fir.ref<f64>
  %2 = acc.reduction varPtr(%1 : !fir.ref<f64>) recipe(@reduction_add_ref_f64) -> !fir.ref<f64> {name = "tmp"}
  acc.parallel reduction(%2 : !fir.ref<f64>) {
    %3 = fir.load %2 : !fir.ref<f64>
    %4 = arith.addf %3, %cst fastmath<contract> : f64
    fir.store %4 to %2 : !fir.ref<f64>
    acc.yield
  }
  return
}
